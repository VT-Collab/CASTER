"""Natural cubic splines for position and relative rotation trajectories."""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation


def build_spline_basis(num_steps: int, num_control_points: int) -> np.ndarray:
    knots = np.linspace(0, num_steps - 1, num_control_points)
    return CubicSpline(knots, np.eye(num_control_points), bc_type="natural")(
        np.arange(num_steps)
    )


def _fit_control_points(data: np.ndarray, basis: np.ndarray) -> np.ndarray:
    points = np.empty((basis.shape[1], data.shape[1]), dtype=np.float64)
    points[0], points[-1] = data[0], data[-1]
    if len(points) > 2:
        residual = data - basis[:, [0]] * points[[0]] - basis[:, [-1]] * points[[-1]]
        points[1:-1] = np.linalg.lstsq(basis[:, 1:-1], residual, rcond=None)[0]
    return points


class TrajectorySpline:

    def __init__(self, num_steps: int, K_pos: int, K_rot: int, start_quaternion):
        self.position_basis = build_spline_basis(num_steps, K_pos)
        self.rotation_basis = build_spline_basis(num_steps, K_rot)
        self.start_rotation = Rotation.from_quat(np.asarray(start_quaternion)[[1, 2, 3, 0]])

    def fit(self, trajectory: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        positions = _fit_control_points(trajectory[:, :3], self.position_basis)
        rotations = Rotation.from_quat(trajectory[:, [4, 5, 6, 3]])
        relative_vectors = (rotations * self.start_rotation.inv()).as_rotvec()
        return positions, _fit_control_points(relative_vectors, self.rotation_basis)

    def reconstruct(self, position_knots, rotation_knots) -> np.ndarray:
        positions = self.position_basis @ position_knots
        relative_rotations = Rotation.from_rotvec(self.rotation_basis @ rotation_knots)
        quaternions = (relative_rotations * self.start_rotation).as_quat()
        return np.hstack([positions, quaternions[:, [3, 0, 1, 2]]])
