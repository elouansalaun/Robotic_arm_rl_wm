"""MuJoCo scene wrapper: model loading, episode configuration, state access."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from rwm.robotics.kinematics import ArmKinematics, top_down_rotation

ASSETS = Path(__file__).resolve().parents[3] / "assets"
SCENE_XML = ASSETS / "scenes" / "pick_place.xml"

TABLE_Z = 0.42
TARGET_HALF_HEIGHT = 0.01
OBJECT_KINDS = ("box", "cylinder", "sphere")
ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]
FINGER_JOINTS = ["finger_left", "finger_right"]
FINGER_TRAVEL = 0.045
PAD_OFFSET = 0.006  # inner face of a pad when the finger joint is at 0


@dataclass
class Scenario:
    """Everything that is randomised for one episode. Serialisable to JSON."""

    object_kind: str = "box"
    object_size: list[float] = field(default_factory=lambda: [0.022, 0.022, 0.022])
    object_mass: float = 0.15
    object_friction: float = 1.0
    object_xy: list[float] = field(default_factory=lambda: [0.5, -0.1])
    object_yaw: float = 0.0
    target_xy: list[float] = field(default_factory=lambda: [0.5, 0.15])
    distractors: list[list[float]] = field(default_factory=list)  # [x, y, yaw] per distractor
    seed: int | None = None

    @property
    def half_height(self) -> float:
        if self.object_kind == "box":
            return self.object_size[2]
        if self.object_kind == "cylinder":
            return self.object_size[1]
        return self.object_size[0]

    @property
    def half_width(self) -> float:
        """Half of the width seen by the fingers (box: smallest side)."""
        if self.object_kind == "box":
            return min(self.object_size[0], self.object_size[1])
        return self.object_size[0]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        return cls(**d)


def save_scenarios(scenarios: list[Scenario], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([s.to_dict() for s in scenarios], f, indent=1)


def load_scenarios(path: str | Path) -> list[Scenario]:
    with open(path) as f:
        return [Scenario.from_dict(d) for d in json.load(f)]


class ScenarioSampler:
    """Draws random scenarios from the ranges given in configs/simulation.yaml."""

    def __init__(self, cfg: dict, seed: int | None = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    def seed(self, seed: int | None) -> None:
        self.rng = np.random.default_rng(seed)

    def _uniform(self, bounds):
        lo, hi = bounds
        return float(self.rng.uniform(lo, hi))

    def sample(self) -> Scenario:
        c = self.cfg
        rng = self.rng
        kind = str(rng.choice(c["object_kinds"]))
        shape = c["shapes"][kind]
        if kind == "box":
            half = self._uniform(shape["half_size"])
            # boxes are close to cubes, a bit of aspect ratio variation
            size = [half, half * self._uniform(shape.get("aspect", [1.0, 1.0])), half]
        elif kind == "cylinder":
            size = [self._uniform(shape["radius"]), self._uniform(shape["half_height"])]
        else:
            size = [self._uniform(shape["radius"])]

        obj_xy = [self._uniform(c["object_x"]), self._uniform(c["object_y"])]
        for _ in range(100):
            target_xy = [self._uniform(c["target_x"]), self._uniform(c["target_y"])]
            if np.hypot(target_xy[0] - obj_xy[0], target_xy[1] - obj_xy[1]) > c["min_separation"]:
                break

        distractors = []
        for _ in range(int(c.get("n_distractors", 0))):
            ang = rng.uniform(-np.pi, np.pi)
            dist = self._uniform(c.get("distractor_distance", [0.08, 0.12]))
            distractors.append([obj_xy[0] + dist * np.cos(ang), obj_xy[1] + dist * np.sin(ang), rng.uniform(-np.pi, np.pi)])

        return Scenario(
            object_kind=kind,
            object_size=size,
            object_mass=self._uniform(c["mass"]),
            object_friction=self._uniform(c["friction"]),
            object_xy=obj_xy,
            object_yaw=self._uniform(c["yaw"]) if kind != "sphere" else 0.0,
            target_xy=target_xy,
            distractors=distractors,
            seed=int(rng.integers(2**31)),
        )


def box_inertia(m, hx, hy, hz):
    return m / 3.0 * np.array([hy**2 + hz**2, hx**2 + hz**2, hx**2 + hy**2])


def cylinder_inertia(m, r, hz):
    i_xy = m * (3 * r**2 + 4 * hz**2) / 12.0
    return np.array([i_xy, i_xy, 0.5 * m * r**2])


def sphere_inertia(m, r):
    return np.full(3, 0.4 * m * r**2)


class PickPlaceScene:
    """Thin layer over mjModel/mjData with the queries the rest of the code needs."""

    def __init__(self, xml_path: str | Path = SCENE_XML, home_tcp=(0.45, 0.0, 0.70), home_yaw=0.0):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.kin = ArmKinematics()
        m = self.model

        name2id = lambda kind, name: mujoco.mj_name2id(m, kind, name)
        self.arm_qpos = np.array([m.jnt_qposadr[name2id(mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ARM_JOINTS])
        self.arm_dof = np.array([m.jnt_dofadr[name2id(mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ARM_JOINTS])
        self.finger_qpos = np.array([m.jnt_qposadr[name2id(mujoco.mjtObj.mjOBJ_JOINT, j)] for j in FINGER_JOINTS])
        self.finger_dof = np.array([m.jnt_dofadr[name2id(mujoco.mjtObj.mjOBJ_JOINT, j)] for j in FINGER_JOINTS])
        self.tcp_site = name2id(mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.target_mocap = m.body_mocapid[name2id(mujoco.mjtObj.mjOBJ_BODY, "target")]
        self.target_geom = name2id(mujoco.mjtObj.mjOBJ_GEOM, "target_geom")
        self.table_geom = name2id(mujoco.mjtObj.mjOBJ_GEOM, "table")
        self.floor_geom = name2id(mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.pad_geoms = {name2id(mujoco.mjtObj.mjOBJ_GEOM, f"{f}_pad") for f in FINGER_JOINTS}
        self.left_pad = name2id(mujoco.mjtObj.mjOBJ_GEOM, "finger_left_pad")
        self.right_pad = name2id(mujoco.mjtObj.mjOBJ_GEOM, "finger_right_pad")

        robot_root = name2id(mujoco.mjtObj.mjOBJ_BODY, "base")
        self.robot_geoms = {g for g in range(m.ngeom) if self._is_descendant(m.geom_bodyid[g], robot_root)}

        self.objects = {}
        for kind in OBJECT_KINDS:
            body = name2id(mujoco.mjtObj.mjOBJ_BODY, f"obj_{kind}")
            joint = m.body_jntadr[body]
            self.objects[kind] = dict(
                body=body,
                geom=name2id(mujoco.mjtObj.mjOBJ_GEOM, f"obj_{kind}_geom"),
                qpos=m.jnt_qposadr[joint],
                dof=m.jnt_dofadr[joint],
            )
        self.distractors = []
        for i in range(2):
            body = name2id(mujoco.mjtObj.mjOBJ_BODY, f"distractor_{i}")
            joint = m.body_jntadr[body]
            self.distractors.append(
                dict(body=body, geom=name2id(mujoco.mjtObj.mjOBJ_GEOM, f"distractor_{i}_geom"),
                     qpos=m.jnt_qposadr[joint], dof=m.jnt_dofadr[joint])
            )
        self.parking = {k: m.body_pos[v["body"]].copy() for k, v in self.objects.items()}
        self.parking.update({f"d{i}": m.body_pos[d["body"]].copy() for i, d in enumerate(self.distractors)})

        self.home_q = self.kin.inverse(np.asarray(home_tcp), top_down_rotation(home_yaw), q_ref=np.zeros(6))
        if self.home_q is None:
            raise ValueError("home pose is not reachable")
        self.scenario: Scenario | None = None
        self.active: dict | None = None

    def _is_descendant(self, body: int, root: int) -> bool:
        while body != 0:
            if body == root:
                return True
            body = self.model.body_parentid[body]
        return False

    # ----------------------------------------------------------- episode setup
    def _set_object_params(self, sc: Scenario) -> None:
        m = self.model
        obj = self.objects[sc.object_kind]
        g, b = obj["geom"], obj["body"]
        mass = sc.object_mass
        if sc.object_kind == "box":
            hx, hy, hz = sc.object_size
            m.geom_size[g] = [hx, hy, hz]
            inertia = box_inertia(mass, hx, hy, hz)
            half_extent = np.array([hx, hy, hz])
            rbound = np.linalg.norm(half_extent)
        elif sc.object_kind == "cylinder":
            r, hz = sc.object_size
            m.geom_size[g, :2] = [r, hz]
            inertia = cylinder_inertia(mass, r, hz)
            half_extent = np.array([r, r, hz])
            rbound = np.hypot(r, hz)
        else:
            (r,) = sc.object_size
            m.geom_size[g, 0] = r
            inertia = sphere_inertia(mass, r)
            half_extent = np.array([r, r, r])
            rbound = r
        # keep the broad-phase data consistent with the new size
        m.geom_rbound[g] = rbound
        m.geom_aabb[g] = np.concatenate([np.zeros(3), half_extent])
        m.geom_friction[g, 0] = sc.object_friction
        m.body_mass[b] = mass
        m.body_inertia[b] = inertia
        m.body_ipos[b] = 0.0
        m.body_iquat[b] = [1, 0, 0, 0]

    def _geom_half_height(self, g: int) -> float:
        size = self.model.geom_size[g]
        gtype = self.model.geom_type[g]
        if gtype == mujoco.mjtGeom.mjGEOM_BOX:
            return float(size[2])
        if gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
            return float(size[1])
        return float(size[0])

    def _place_free_body(self, qadr: int, pos, yaw: float) -> None:
        self.data.qpos[qadr:qadr + 3] = pos
        self.data.qpos[qadr + 3:qadr + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]

    def reset(self, scenario: Scenario, settle_steps: int = 100, arm_q: np.ndarray | None = None) -> None:
        m, d = self.model, self.data
        self.scenario = scenario
        mujoco.mj_resetData(m, d)
        self._set_object_params(scenario)
        mujoco.mj_setConst(m, d)

        q = self.home_q if arm_q is None else arm_q
        d.qpos[self.arm_qpos] = q
        d.qpos[self.finger_qpos] = FINGER_TRAVEL
        d.ctrl[:6] = q
        d.ctrl[6] = FINGER_TRAVEL

        for kind, obj in self.objects.items():
            if kind == scenario.object_kind:
                z = TABLE_Z + scenario.half_height + 1e-3
                self._place_free_body(obj["qpos"], [*scenario.object_xy, z], scenario.object_yaw)
            else:
                self._place_free_body(obj["qpos"], self.parking[kind], 0.0)
        for i, dist in enumerate(self.distractors):
            if i < len(scenario.distractors):
                x, y, yaw = scenario.distractors[i]
                hz = self._geom_half_height(dist["geom"])
                self._place_free_body(dist["qpos"], [x, y, TABLE_Z + hz + 1e-3], yaw)
            else:
                self._place_free_body(dist["qpos"], self.parking[f"d{i}"], 0.0)

        d.mocap_pos[self.target_mocap] = [*scenario.target_xy, TABLE_Z + TARGET_HALF_HEIGHT]
        self.active = self.objects[scenario.object_kind]
        mujoco.mj_forward(m, d)
        for _ in range(settle_steps):
            mujoco.mj_step(m, d)

    # ------------------------------------------------------------- stepping
    def step(self, ctrl: np.ndarray, n_substeps: int) -> None:
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data, nstep=n_substeps)

    def get_state(self) -> np.ndarray:
        """Full simulator state (time, qpos, qvel, act, mocap...)."""
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(self.model, spec))
        mujoco.mj_getState(self.model, self.data, state, spec)
        return state

    def set_state(self, state: np.ndarray) -> None:
        mujoco.mj_setState(self.model, self.data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(self.model, self.data)

    def save_snapshot(self, path: str | Path) -> None:
        """Scenario + simulator state, enough to restart an episode exactly."""
        assert self.scenario is not None
        payload = {"scenario": self.scenario.to_dict(), "state": self.get_state().tolist()}
        with open(path, "w") as f:
            json.dump(payload, f)

    def load_snapshot(self, path: str | Path) -> None:
        with open(path) as f:
            payload = json.load(f)
        self.reset(Scenario.from_dict(payload["scenario"]), settle_steps=0)
        self.set_state(np.asarray(payload["state"]))

    # ------------------------------------------------------------ queries
    @property
    def arm_q(self) -> np.ndarray:
        return self.data.qpos[self.arm_qpos].copy()

    @property
    def arm_qd(self) -> np.ndarray:
        return self.data.qvel[self.arm_dof].copy()

    def tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.site_xpos[self.tcp_site].copy(), self.data.site_xmat[self.tcp_site].reshape(3, 3).copy()

    def gripper_opening(self) -> float:
        """Distance between the inner faces of the two pads (m)."""
        return float(2 * PAD_OFFSET + np.sum(self.data.qpos[self.finger_qpos]))

    def gripper_velocity(self) -> float:
        return float(np.sum(self.data.qvel[self.finger_dof]))

    def object_pose(self) -> tuple[np.ndarray, np.ndarray]:
        adr = self.active["qpos"]
        return self.data.qpos[adr:adr + 3].copy(), self.data.qpos[adr + 3:adr + 7].copy()

    def object_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        """Linear velocity (world) and angular velocity (local frame) of the object."""
        adr = self.active["dof"]
        return self.data.qvel[adr:adr + 3].copy(), self.data.qvel[adr + 3:adr + 6].copy()

    def object_rotation(self) -> np.ndarray:
        return self.data.xmat[self.active["body"]].reshape(3, 3).copy()

    def target_position(self) -> np.ndarray:
        """Centre of the top face of the target platform."""
        p = self.data.mocap_pos[self.target_mocap].copy()
        p[2] += TARGET_HALF_HEIGHT
        return p

    def contacts(self) -> dict:
        """Classify the active contacts.

        - finger_left / finger_right: pad touching the target object
        - collisions: robot geoms touching anything that is not the object
          through the pads (table, platform, distractors, object hit by a link)
        """
        obj_geom = self.active["geom"]
        left = right = False
        collisions = 0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            pair = {g1, g2}
            if obj_geom in pair and self.left_pad in pair:
                left = True
                continue
            if obj_geom in pair and self.right_pad in pair:
                right = True
                continue
            robot_in = (g1 in self.robot_geoms) or (g2 in self.robot_geoms)
            if robot_in and not pair <= self.robot_geoms:
                collisions += 1
        return {"finger_left": left, "finger_right": right, "collisions": collisions}
