"""Kinematics of the 6-DOF arm described in assets/robot/arm6.xml.

The arm is an anthropomorphic manipulator with a spherical wrist:

    q1  base yaw        (z)
    q2  shoulder pitch  (y)
    q3  elbow pitch     (y)
    q4  forearm roll    (x)
    q5  wrist pitch     (y)
    q6  tool roll       (x)

Everything here is written from the MJCF chain directly (no DH table), so the
forward kinematics can be checked one-to-one against MuJoCo in the tests.
The TCP frame has its x axis along the approach direction and its y axis along
the finger closing direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def wrap_angle(a):
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def rotation_log(R: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle) of a rotation matrix."""
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if theta < 1e-6:
        return 0.5 * w
    if np.pi - theta < 1e-4:
        # close to pi the antisymmetric part vanishes, take the eigenvector instead
        vals, vecs = np.linalg.eig(R)
        axis = np.real(vecs[:, np.argmin(np.abs(vals - 1.0))])
        return theta * axis
    return theta * w / (2.0 * np.sin(theta))


def top_down_rotation(yaw: float) -> np.ndarray:
    """TCP orientation with the tool pointing down (-z) and the finger axis at `yaw`.

    yaw = 0 means the fingers close along the world x axis.
    """
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[0.0, c, s], [0.0, s, -c], [-1.0, 0.0, 0.0]])


def tool_yaw(R: np.ndarray) -> float:
    """Yaw of the finger axis for a (roughly) top-down TCP orientation."""
    return float(np.arctan2(R[1, 1], R[0, 1]))


def wrap_half_pi(a):
    """Fold an axis direction into [-pi/2, pi/2): the gripper is symmetric under a half turn."""
    return (np.asarray(a) + np.pi / 2) % np.pi - np.pi / 2


@dataclass(frozen=True)
class ArmGeometry:
    base: tuple[float, float, float] = (0.0, 0.0, 0.42)
    j1_height: float = 0.08     # mounting plate -> joint 1
    shoulder_height: float = 0.12  # joint 1 -> shoulder axis
    upper_arm: float = 0.40
    forearm: float = 0.30       # elbow -> joint 4
    wrist: float = 0.08         # joint 4 -> wrist centre
    tool: float = 0.17          # wrist centre -> TCP
    lower: tuple[float, ...] = field(
        default_factory=lambda: tuple(np.deg2rad([-170, -60, -120, -180, -125, -180]))
    )
    upper: tuple[float, ...] = field(
        default_factory=lambda: tuple(np.deg2rad([170, 115, 150, 180, 125, 180]))
    )

    @property
    def elbow_to_wrist(self) -> float:
        return self.forearm + self.wrist


class ArmKinematics:
    def __init__(self, geometry: ArmGeometry | None = None):
        self.geo = geometry or ArmGeometry()
        self.base = np.asarray(self.geo.base, dtype=float)
        self.lower = np.asarray(self.geo.lower)
        self.upper = np.asarray(self.geo.upper)
        self.shoulder = self.base + np.array([0.0, 0.0, self.geo.j1_height + self.geo.shoulder_height])

    # ------------------------------------------------------------------ FK
    def joint_frames(self, q: np.ndarray):
        """World position and axis of every joint, plus the TCP pose."""
        g = self.geo
        q1, q2, q3, q4, q5, q6 = q
        ex, ey, ez = np.eye(3)

        R1 = rot_z(q1)
        o1 = self.base + np.array([0.0, 0.0, g.j1_height])
        o2 = self.shoulder
        R2 = R1 @ rot_y(q2)
        o3 = o2 + R2 @ np.array([0.0, 0.0, g.upper_arm])
        R3 = R2 @ rot_y(q3)
        o4 = o3 + R3 @ np.array([g.forearm, 0.0, 0.0])
        R4 = R3 @ rot_x(q4)
        o5 = o4 + R4 @ np.array([g.wrist, 0.0, 0.0])
        R5 = R4 @ rot_y(q5)
        R6 = R5 @ rot_x(q6)
        p_tcp = o5 + R6 @ np.array([g.tool, 0.0, 0.0])

        origins = [o1, o2, o3, o4, o5, o5]
        axes = [ez, R1 @ ey, R2 @ ey, R3 @ ex, R4 @ ey, R5 @ ex]
        return origins, axes, p_tcp, R6

    def forward(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _, _, p, R = self.joint_frames(q)
        return p, R

    def wrist_centre(self, q: np.ndarray) -> np.ndarray:
        origins, _, _, _ = self.joint_frames(q)
        return origins[4]

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """Geometric Jacobian of the TCP, rows = (v, omega) in the world frame."""
        origins, axes, p, _ = self.joint_frames(q)
        J = np.zeros((6, 6))
        for i, (o, z) in enumerate(zip(origins, axes)):
            J[:3, i] = np.cross(z, p - o)
            J[3:, i] = z
        return J

    # ------------------------------------------------------------------ IK
    def within_limits(self, q: np.ndarray, tol: float = 1e-6) -> bool:
        return bool(np.all(q >= self.lower - tol) and np.all(q <= self.upper + tol))

    def inverse_all(self, p: np.ndarray, R: np.ndarray, q4_ref: float = 0.0) -> list[np.ndarray]:
        """All closed-form solutions (up to 8) inside the joint limits.

        Kinematic decoupling: the position of the wrist centre fixes q1..q3,
        the remaining orientation R36 = R03^T R is an X-Y-X Euler triplet.
        """
        g = self.geo
        a2, a3 = g.upper_arm, g.elbow_to_wrist
        pw = np.asarray(p) - g.tool * np.asarray(R)[:, 0]
        v = pw - self.shoulder

        solutions = []
        for flip in (False, True):
            q1 = np.arctan2(v[1], v[0]) + (np.pi if flip else 0.0)
            r = np.hypot(v[0], v[1]) * (-1.0 if flip else 1.0)
            s = v[2]
            sin_q3 = (a2**2 + a3**2 - r**2 - s**2) / (2.0 * a2 * a3)
            if abs(sin_q3) > 1.0:
                continue
            for q3 in (np.arcsin(sin_q3), np.pi - np.arcsin(sin_q3)):
                k1 = a2 - a3 * np.sin(q3)
                k2 = a3 * np.cos(q3)
                q2 = np.arctan2(r, s) - np.arctan2(k2, k1)
                R03 = rot_z(q1) @ rot_y(q2 + q3)
                for wrist in self._wrist_solutions(R03.T @ R, q4_ref):
                    q = wrap_angle(np.array([q1, q2, q3, *wrist]))
                    if self.within_limits(q):
                        solutions.append(q)
        return solutions

    @staticmethod
    def _wrist_solutions(R36: np.ndarray, q4_ref: float):
        # R36 = Rx(q4) Ry(q5) Rx(q6)
        c5 = np.clip(R36[0, 0], -1.0, 1.0)
        s5 = np.hypot(R36[0, 1], R36[0, 2])
        if s5 < 1e-6:
            # wrist singularity: only q4 + q6 is defined, keep q4 where it is
            q46 = np.arctan2(R36[2, 1], R36[1, 1])
            q5 = 0.0 if c5 > 0 else np.pi
            return [(q4_ref, q5, q46 - q4_ref)]
        out = []
        for sign in (1.0, -1.0):
            q5 = np.arctan2(sign * s5, c5)
            q4 = np.arctan2(sign * R36[1, 0], -sign * R36[2, 0])
            q6 = np.arctan2(sign * R36[0, 1], sign * R36[0, 2])
            out.append((q4, q5, q6))
        return out

    def inverse(self, p: np.ndarray, R: np.ndarray, q_ref: np.ndarray | None = None) -> np.ndarray | None:
        """Closed-form IK, returns the solution closest to q_ref (or None)."""
        q_ref = np.zeros(6) if q_ref is None else np.asarray(q_ref)
        sols = self.inverse_all(p, R, q4_ref=q_ref[3])
        if not sols:
            return None
        # the base and shoulder joints are the expensive ones to move
        weights = np.array([2.0, 2.0, 1.5, 1.0, 1.0, 0.5])
        costs = [np.sum(weights * wrap_angle(q - q_ref) ** 2) for q in sols]
        return sols[int(np.argmin(costs))]

    def inverse_dls(
        self,
        p: np.ndarray,
        R: np.ndarray,
        q0: np.ndarray,
        iters: int = 50,
        damping: float = 0.05,
        rot_weight: float = 0.5,
        tol: float = 1e-4,
    ) -> tuple[np.ndarray, bool]:
        """Damped least-squares IK starting from q0, clamped to the joint limits."""
        q = np.array(q0, dtype=float)
        W = np.diag([1.0, 1.0, 1.0, rot_weight, rot_weight, rot_weight])
        for _ in range(iters):
            p_cur, R_cur = self.forward(q)
            err = np.concatenate([p - p_cur, rotation_log(R @ R_cur.T)])
            if np.linalg.norm(err[:3]) < tol and np.linalg.norm(err[3:]) < 10 * tol:
                return q, True
            J = W @ self.jacobian(q)
            e = W @ err
            dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), e)
            q = np.clip(q + dq, self.lower, self.upper)
        p_cur, R_cur = self.forward(q)
        ok = np.linalg.norm(p - p_cur) < 10 * tol
        return q, bool(ok)

    def manipulability(self, q: np.ndarray) -> float:
        J = self.jacobian(q)
        return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))
