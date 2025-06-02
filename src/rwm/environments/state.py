"""Layout of the low-dimensional state shared by the RL agent and the world model.

The state is meant to be Markov for the commanded system (arm + gripper +
object), so the dynamics model can be trained on (s, a, s') triplets:

    tcp_pos      3   TCP position (world)
    tcp_yaw      1   finger axis yaw, kept in [-pi/2, pi/2] by the controller
    grip         1   distance between the pads (m)
    grip_vel     1
    obj_pos      3
    obj_rot      6   first two columns of the object rotation matrix
    obj_linvel   3   world frame
    obj_angvel   3   world frame
    arm_q        6
    arm_qd       6

The object orientation uses the continuous 6D representation instead of a
quaternion or Euler angles: no double cover, no wrap-around, so deltas are
well defined. The task vector (target position, object half height and half
width) is constant during an episode and kept separate.
"""

from __future__ import annotations

import numpy as np

from rwm.robotics.kinematics import tool_yaw

FIELDS = [
    ("tcp_pos", 3),
    ("tcp_yaw", 1),
    ("grip", 1),
    ("grip_vel", 1),
    ("obj_pos", 3),
    ("obj_rot", 6),
    ("obj_linvel", 3),
    ("obj_angvel", 3),
    ("arm_q", 6),
    ("arm_qd", 6),
]

SLICES: dict[str, slice] = {}
_i = 0
for _name, _n in FIELDS:
    SLICES[_name] = slice(_i, _i + _n)
    _i += _n
STATE_DIM = _i
TASK_DIM = 5  # target_pos (3), object half height, object half width


def rot6_from_matrix(R: np.ndarray) -> np.ndarray:
    return np.concatenate([R[:, 0], R[:, 1]])


def matrix_from_rot6(r6):
    """Gram-Schmidt on the two predicted columns (works on batches, numpy or torch)."""
    a, b = r6[..., 0:3], r6[..., 3:6]
    if isinstance(r6, np.ndarray):
        x = a / np.linalg.norm(a, axis=-1, keepdims=True)
        b = b - np.sum(x * b, axis=-1, keepdims=True) * x
        y = b / np.linalg.norm(b, axis=-1, keepdims=True)
        z = np.cross(x, y)
        return np.stack([x, y, z], axis=-1)
    import torch

    x = torch.nn.functional.normalize(a, dim=-1)
    b = b - (x * b).sum(-1, keepdim=True) * x
    y = torch.nn.functional.normalize(b, dim=-1)
    z = torch.cross(x, y, dim=-1)
    return torch.stack([x, y, z], dim=-1)


def object_up_axis_z(r6):
    """z component of the object's local z axis (1 = upright)."""
    x, y = r6[..., 0:3], r6[..., 3:6]
    return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]


def read_state(scene) -> np.ndarray:
    p_tcp, R_tcp = scene.tcp_pose()
    obj_pos, _ = scene.object_pose()
    R_obj = scene.object_rotation()
    v_lin, w_local = scene.object_velocity()
    s = np.concatenate(
        [
            p_tcp,
            [tool_yaw(R_tcp)],
            [scene.gripper_opening()],
            [scene.gripper_velocity()],
            obj_pos,
            rot6_from_matrix(R_obj),
            v_lin,
            R_obj @ w_local,
            scene.arm_q,
            scene.arm_qd,
        ]
    )
    return s.astype(np.float32)


def read_task(scene) -> np.ndarray:
    sc = scene.scenario
    return np.array([*scene.target_position(), sc.half_height, sc.half_width], dtype=np.float32)


def get(state, name: str):
    return state[..., SLICES[name]]
