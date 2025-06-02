"""Gymnasium environment for the pick-and-place task.

Action (5,) in [-1, 1]:
    dx, dy, dz   TCP displacement, scaled by control.max_tcp_step
    dyaw         rotation of the finger axis around the vertical
    grip         absolute gripper command, -1 closed ... +1 open

The tool always points down; the low-level Cartesian controller converts the
commanded TCP pose into joint position targets with damped least squares IK.
All controllers in this project (scripted, SAC, MPC) act through this same
interface, so their results are directly comparable.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rwm.config import load_config, scenario_config
from rwm.environments.reward import pick_place_reward, placed
from rwm.environments.state import SLICES, STATE_DIM, TASK_DIM, read_state, read_task
from rwm.robotics.controllers import CartesianController
from rwm.simulation.scene import FINGER_TRAVEL, TABLE_Z, PickPlaceScene, Scenario, ScenarioSampler

OBS_DIM = STATE_DIM + TASK_DIM + 6


def build_observation(state: np.ndarray, task: np.ndarray) -> np.ndarray:
    """Policy input: state, task and the two vectors that matter most."""
    obj = state[..., SLICES["obj_pos"]]
    tcp = state[..., SLICES["tcp_pos"]]
    place = task[..., 0:3].copy()
    place[..., 2] += task[..., 3]
    return np.concatenate([state, task, obj - tcp, place - obj], axis=-1).astype(np.float32)


class PickPlaceEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        sim_cfg: dict | None = None,
        condition: str = "nominal",
        render_mode: str | None = None,
        state_estimator=None,
        seed: int | None = None,
    ):
        super().__init__()
        self.cfg = sim_cfg or load_config("simulation")
        ctrl = self.cfg["control"]
        self.frame_skip = ctrl["frame_skip"]
        self.max_steps = ctrl["max_steps"]
        self.max_tcp_step = ctrl["max_tcp_step"]
        self.max_yaw_step = ctrl["max_yaw_step"]
        ws = ctrl["workspace"]
        self.ws_low = np.array([ws["x"][0], ws["y"][0], ws["z"][0]])
        self.ws_high = np.array([ws["x"][1], ws["y"][1], ws["z"][1]])

        self.scene = PickPlaceScene(home_tcp=self.cfg["home_tcp"], home_yaw=self.cfg["home_yaw"])
        self.cartesian = CartesianController(self.scene.kin, **ctrl["ik"])
        self.sampler = ScenarioSampler(scenario_config(self.cfg, condition), seed=seed)
        self.render_mode = render_mode
        self._renderer = None
        # optional perception module: replaces the true object pose in the
        # observation by an estimate (see perception/estimator.py)
        self.state_estimator = state_estimator

        self.action_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.dt = self.scene.model.opt.timestep * self.frame_skip

    # ------------------------------------------------------------------ API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.sampler.seed(seed)
        options = options or {}
        scenario = options.get("scenario")
        if scenario is None:
            scenario = self.sampler.sample()
        elif isinstance(scenario, dict):
            scenario = Scenario.from_dict(scenario)
        self.scene.reset(scenario)

        self.steps = 0
        self.n_collision_steps = 0
        self.max_lift = 0.0
        self.first_success_step = None
        self.true_state = read_state(self.scene)
        self.task = read_task(self.scene)
        if self.state_estimator is not None:
            self.state_estimator.reset(self.scene)
        return self._observe(), self._info(contacts=self.scene.contacts(), success=False)

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        p_tcp, _ = self.scene.tcp_pose()
        yaw = self.true_state[SLICES["tcp_yaw"]][0]
        p_cmd = np.clip(p_tcp + a[:3] * self.max_tcp_step, self.ws_low, self.ws_high)
        yaw_cmd = np.clip(yaw + a[3] * self.max_yaw_step, -np.pi / 2, np.pi / 2)
        q_cmd = self.cartesian.solve(self.scene.arm_q, p_cmd, yaw_cmd)
        grip_cmd = 0.5 * (a[4] + 1.0) * FINGER_TRAVEL
        self.scene.step(np.append(q_cmd, grip_cmd), self.frame_skip)
        self.steps += 1

        self.true_state = read_state(self.scene)
        contacts = self.scene.contacts()
        self.n_collision_steps += int(contacts["collisions"] > 0)
        lift = self.true_state[SLICES["obj_pos"]][2] - (TABLE_Z + self.task[3])
        self.max_lift = max(self.max_lift, float(lift))

        success = self._is_success(contacts)
        if success and self.first_success_step is None:
            self.first_success_step = self.steps

        reward = float(pick_place_reward(self.true_state, self.task, action=a))
        obj_z = self.true_state[SLICES["obj_pos"]][2]
        terminated = bool(obj_z < TABLE_Z - 0.03)  # object fell off the table
        truncated = self.steps >= self.max_steps
        return self._observe(), reward, terminated, truncated, self._info(contacts, success)

    def render(self):
        import mujoco

        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.scene.model, 480, 640)
        self._renderer.update_scene(self.scene.data, camera="overview")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # -------------------------------------------------------------- helpers
    def state(self) -> np.ndarray:
        """Ground-truth state (what the world model is trained on)."""
        return self.true_state.copy()

    def _observe(self) -> np.ndarray:
        if self.state_estimator is None:
            return build_observation(self.true_state, self.task)
        # called once per step: the estimator keeps track of the object between frames
        estimated = self.state_estimator.estimate(self.scene, self.true_state)
        return build_observation(estimated, self.state_estimator.task())

    def _is_success(self, contacts: dict) -> bool:
        s = self.true_state
        resting = np.linalg.norm(s[SLICES["obj_linvel"]]) < 0.05
        released = not (contacts["finger_left"] or contacts["finger_right"])
        return bool(placed(s, self.task) and resting and released and self.max_lift > 0.01)

    def _info(self, contacts: dict, success: bool) -> dict:
        obj = self.true_state[SLICES["obj_pos"]]
        return {
            "is_success": success,
            "first_success_step": self.first_success_step,
            "collisions": contacts["collisions"],
            "collision_steps": self.n_collision_steps,
            "grasped": contacts["finger_left"] and contacts["finger_right"],
            "finger_contacts": (contacts["finger_left"], contacts["finger_right"]),
            "place_error": float(np.linalg.norm(obj[:2] - self.task[:2])),
            "max_lift": self.max_lift,
            "object_kind": self.scene.scenario.object_kind,
        }
