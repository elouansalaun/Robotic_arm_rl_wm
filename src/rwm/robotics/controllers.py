"""Low-level Cartesian control and the scripted pick-and-place controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rwm.environments.state import SLICES, STATE_DIM, TASK_DIM
from rwm.robotics.kinematics import ArmKinematics, top_down_rotation, wrap_half_pi


class CartesianController:
    """TCP pose target -> joint position targets (a few DLS iterations per step)."""

    def __init__(self, kin: ArmKinematics, iterations: int = 3, damping: float = 0.05, rot_weight: float = 0.5):
        self.kin = kin
        self.iterations = iterations
        self.damping = damping
        self.rot_weight = rot_weight

    def solve(self, q: np.ndarray, p_target: np.ndarray, yaw_target: float) -> np.ndarray:
        q_cmd, _ = self.kin.inverse_dls(
            p_target,
            top_down_rotation(yaw_target),
            q,
            iters=self.iterations,
            damping=self.damping,
            rot_weight=self.rot_weight,
        )
        return q_cmd


def split_observation(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return obs[..., :STATE_DIM], obs[..., STATE_DIM:STATE_DIM + TASK_DIM]


def object_yaw(state: np.ndarray) -> float:
    """Yaw of the object's local x axis (the narrow side of boxes by construction)."""
    x_axis = state[SLICES["obj_rot"]][:3]
    return float(np.arctan2(x_axis[1], x_axis[0]))


def min_jerk(tau: float) -> float:
    tau = np.clip(tau, 0.0, 1.0)
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


@dataclass
class Segment:
    p0: np.ndarray
    p1: np.ndarray
    yaw0: float
    yaw1: float
    n_steps: int
    k: int = 0

    def reference(self, k: int) -> tuple[np.ndarray, float]:
        s = min_jerk(k / max(self.n_steps, 1))
        return self.p0 + s * (self.p1 - self.p0), self.yaw0 + s * (self.yaw1 - self.yaw0)

    @property
    def finished(self) -> bool:
        return self.k >= self.n_steps


class ScriptedPickPlace:
    """Classical controller: waypoints + minimum-jerk interpolation + IK.

    The phases are approach -> descend -> close -> lift -> transport -> lower
    -> release -> retreat. Each phase builds a straight-line Cartesian segment
    from the current TCP pose to its waypoint; the reference is tracked by
    sending the displacement to the next reference point as the action. A
    failed grasp (fingers closing on nothing, object dropped) sends the
    controller back to the approach phase.
    """

    PHASES = ("approach", "descend", "close", "lift", "transport", "lower", "release", "retreat", "done")

    def __init__(
        self,
        dt: float = 0.05,
        max_tcp_step: float = 0.025,
        max_yaw_step: float = 0.15,
        speed: float = 0.40,
        hover: float = 0.08,
        clearance: float = 0.07,
        close_steps: int = 6,
        release_steps: int = 4,
        z_min: float = 0.438,
        max_retries: int = 2,
    ):
        self.dt = dt
        self.max_tcp_step = max_tcp_step
        self.max_yaw_step = max_yaw_step
        self.speed = speed
        self.hover = hover
        self.clearance = clearance
        self.close_steps = close_steps
        self.release_steps = release_steps
        self.z_min = z_min
        self.max_retries = max_retries
        self.reset()

    @classmethod
    def from_env(cls, env, **kwargs) -> "ScriptedPickPlace":
        ctrl = env.unwrapped.cfg["control"]
        return cls(
            dt=env.unwrapped.dt,
            max_tcp_step=ctrl["max_tcp_step"],
            max_yaw_step=ctrl["max_yaw_step"],
            z_min=ctrl["workspace"]["z"][0],
            **kwargs,
        )

    def reset(self) -> None:
        self.phase = "approach"
        self.segment: Segment | None = None
        self.counter = 0
        self.retries = 0
        self.grasp_yaw = 0.0

    # ------------------------------------------------------------ planning
    def _waypoint(self, state, task) -> tuple[np.ndarray, float]:
        obj = state[SLICES["obj_pos"]]
        tcp = state[SLICES["tcp_pos"]]
        target, half_h = task[:3], task[3]
        carry_z = max(target[2], obj[2]) + half_h + self.clearance
        if self.phase == "approach":
            self.grasp_yaw = float(wrap_half_pi(object_yaw(state)))
            return np.array([obj[0], obj[1], obj[2] + self.hover]), self.grasp_yaw
        if self.phase == "descend":
            return np.array([obj[0], obj[1], max(obj[2] - 0.004, self.z_min)]), self.grasp_yaw
        if self.phase == "lift":
            return np.array([tcp[0], tcp[1], carry_z]), self.grasp_yaw
        if self.phase == "transport":
            return np.array([target[0], target[1], carry_z]), self.grasp_yaw
        if self.phase == "lower":
            # TCP height that puts the object (held at its current offset) just above the platform
            offset = tcp[2] - obj[2]
            return np.array([target[0], target[1], target[2] + half_h + offset + 0.004]), self.grasp_yaw
        if self.phase == "retreat":
            return tcp + np.array([0.0, 0.0, 0.10]), self.grasp_yaw
        return tcp.copy(), float(state[SLICES["tcp_yaw"]][0])

    def _new_segment(self, state, task) -> Segment:
        p0 = state[SLICES["tcp_pos"]].astype(float)
        yaw0 = float(state[SLICES["tcp_yaw"]][0])
        p1, yaw1 = self._waypoint(state, task)
        dist = np.linalg.norm(p1 - p0)
        n = int(np.ceil(max(dist / self.speed, abs(yaw1 - yaw0) / 2.0, 0.15) / self.dt))
        return Segment(p0, p1, yaw0, yaw1, n)

    def _advance(self, next_phase: str) -> None:
        self.phase = next_phase
        self.segment = None
        self.counter = 0

    # ------------------------------------------------------------- control
    def act(self, obs: np.ndarray) -> np.ndarray:
        state, task = split_observation(np.asarray(obs, dtype=float))
        grip_open = self.phase in ("approach", "descend", "release", "retreat", "done")
        grip = 1.0 if grip_open else -1.0

        if self.phase in ("close", "release"):
            self.counter += 1
            limit = self.close_steps if self.phase == "close" else self.release_steps
            if self.counter >= limit:
                if self.phase == "close" and not self._holding(state, task):
                    self._retry()
                else:
                    self._advance("lift" if self.phase == "close" else "retreat")
            return self._hold(state, grip)

        if self.phase == "done":
            return self._hold(state, grip)

        if self.phase in ("lift", "transport", "lower") and not self._holding(state, task):
            self._retry()
            return self._hold(state, 1.0)

        if self.segment is None:
            self.segment = self._new_segment(state, task)

        seg = self.segment
        seg.k += 1
        p_ref, yaw_ref = seg.reference(seg.k)
        tcp = state[SLICES["tcp_pos"]]
        yaw = state[SLICES["tcp_yaw"]][0]
        a = np.zeros(5)
        a[:3] = (p_ref - tcp) / self.max_tcp_step
        a[3] = (yaw_ref - yaw) / self.max_yaw_step
        a[4] = grip

        if seg.finished:
            err = np.linalg.norm(seg.p1 - tcp)
            if err < 0.006 or seg.k > seg.n_steps + 10:
                nxt = {"approach": "descend", "descend": "close", "lift": "transport",
                       "transport": "lower", "lower": "release", "retreat": "done"}[self.phase]
                self._advance(nxt)
        return np.clip(a, -1.0, 1.0)

    def _hold(self, state, grip: float) -> np.ndarray:
        a = np.zeros(5)
        a[4] = grip
        return a

    def _holding(self, state, task) -> bool:
        width = state[SLICES["grip"]][0]
        dist = np.linalg.norm(state[SLICES["tcp_pos"]] - state[SLICES["obj_pos"]])
        # fingers stopped by the object and the object still between them
        return width > 1.2 * task[4] and dist < 0.04

    def _retry(self) -> None:
        self.retries += 1
        self._advance("approach" if self.retries <= self.max_retries else "done")
