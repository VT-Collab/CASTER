import numpy as np
from pathlib import Path
from typing import Any, Mapping
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as Rotation

from .obstacle import ObstacleCost
from ..rollout import SimulationRollout


@dataclass(frozen=True)
class CostTerm:
    """Configuration for one term in the objective function."""

    weight: float
    enabled: bool

    @property
    def effective_weight(self) -> float:
        return self.weight if self.enabled else 0.0


def _contact_force_costs(
    forces: np.ndarray,
    barrier_weight: float,
    force_threshold: float,
    force_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert forbidden contact forces to penalties, violations, and excess."""
    excess = np.maximum(0.0, forces - force_threshold)
    normalized = excess / force_scale
    collided = excess > 0.0
    costs = barrier_weight * np.where(
        collided, 1.0 + np.minimum(normalized, 10.0) ** 2, 0.0
    )
    return costs, collided.astype(np.float64), excess


class TrajectoryCostFunction(SimulationRollout):

    def __init__(
        self,
        *,
        cost_terms: Mapping[str, Any],
        w_disp_mean_pos: float = 0.2,
        w_disp_peak_pos: float = 0.5,
        w_disp_final_pos: float = 0.3,
        w_disp_rot: float = 0.5,
        tcp_radius: float = 0.04,
        safety_margin: float = 0.02,
        obstacle_temperature: float = 0.01,
        obstacle_query_workers: int = 8,
        obstacle_distance_backend: str = "grid",
        obstacle_grid_resolution: float = 0.003,
        obstacle_grid_padding: float | None = None,
        obstacle_grid_temperature_tail: float = 8.0,
        obstacle_max_grid_cells: int = 30_000_000,
        position_tolerance: float = 0.02,
        rotation_tolerance: float = np.deg2rad(10.0),
        displacement_tolerance: float = 0.01,
        minimum_tcp_z: float = 0.0,
        minimum_tcp_z_ramp_steps: int = 5,
        tcp_z_feasibility_tolerance: float = 0.002,
        maximum_tcp_speed: float | None = 0.60,
        enforce_speed_feasibility: bool = False,
        control_timestep: float = 1.0 / 60.0,
        position_loss_weight: float = 1.0,
        orientation_loss_weight: float = 1.0,
        position_worst_fraction: float = 0.10,
        position_worst_weight: float = 0.0,
        feasibility_tolerance: float = 1e-6,
        position_travel_scale: float = 0.10,
        position_search_radius: float = 0.40,
        rotation_search_radius: float = float(np.pi),
        demo_path: str | None = None,
        video_path: str | None = None,
        server_address: str = "tcp://localhost:15555",
        server_timeout: float = 3600,
        render: bool = False,
        execute_release: bool = True,
        evaluate_release: bool = True,
        post_grasp_settle_steps: int = 15,
        m2t2_hover_distance: float = 0.27,
        m2t2_grasp_depth_offset: float = 0.01,
        m2t2_grasp_height_offset: float = 0.0,
        robot_collision_enabled: bool = True,
        robot_collision_barrier_weight: float = 5000.0,
        robot_collision_force_threshold: float = 1.0,
        robot_collision_force_scale: float = 10.0,
        simulated_arm_collision_enabled: bool = True,
        simulated_arm_link_names: list[str],
        simulated_arm_segment_radii: list[float],
        simulated_arm_samples_per_segment: int = 3,
        joint_lower_limits: list[float],
        joint_upper_limits: list[float],
        feasible_dedup_position_tolerance: float = 0.002,
        feasible_dedup_rotation_tolerance: float = np.deg2rad(1.0),
        feasible_dedup_joint_tolerance: float = 0.02,
    ):
        self.cost_terms = self._build_cost_terms(cost_terms)
        self.w_traj = self.cost_terms["trajectory"].effective_weight
        self.w_endpoint = self.cost_terms["endpoint"].effective_weight
        self.w_displacement = self.cost_terms["displacement"].effective_weight
        self.w_obstacle = self.cost_terms["obstacle"].effective_weight
        self.w_position_travel = self.cost_terms["position_travel"].effective_weight
        self.w_speed = self.cost_terms["speed"].effective_weight
        self.w_workspace_constraint = self.cost_terms["workspace_constraint"].effective_weight

        self.w_disp_mean_pos = w_disp_mean_pos
        self.w_disp_peak_pos = w_disp_peak_pos
        self.w_disp_final_pos = w_disp_final_pos
        self.w_disp_rot = w_disp_rot

        self.position_tolerance = position_tolerance
        self.rotation_tolerance = rotation_tolerance
        self.displacement_tolerance = displacement_tolerance
        self.minimum_tcp_z = float(minimum_tcp_z)
        self.minimum_tcp_z_ramp_steps = int(minimum_tcp_z_ramp_steps)
        self.tcp_z_feasibility_tolerance = float(tcp_z_feasibility_tolerance)
        if self.tcp_z_feasibility_tolerance < 0.0:
            raise ValueError("tcp_z_feasibility_tolerance cannot be negative.")
        self.maximum_tcp_speed = (None if maximum_tcp_speed is None else float(maximum_tcp_speed))
        self.enforce_speed_feasibility = bool(enforce_speed_feasibility)
        self.control_timestep = float(control_timestep)
        if self.maximum_tcp_speed is not None and self.maximum_tcp_speed <= 0.0:
            raise ValueError("maximum_tcp_speed must be positive or None.")
        if self.control_timestep <= 0.0:
            raise ValueError("control_timestep must be positive.")
        self.position_loss_weight = float(position_loss_weight)
        self.orientation_loss_weight = float(orientation_loss_weight)
        self.position_worst_fraction = float(position_worst_fraction)
        self.position_worst_weight = float(position_worst_weight)
        self.feasibility_tolerance = float(feasibility_tolerance)
        if not 0.0 < self.position_worst_fraction <= 1.0:
            raise ValueError("position_worst_fraction must be in the interval (0, 1].")
        if self.feasibility_tolerance < 0.0:
            raise ValueError("feasibility_tolerance cannot be negative.")

        self.position_search_radius = float(position_search_radius)
        self.rotation_search_radius = float(rotation_search_radius)
        if self.position_search_radius <= 0.0:
            raise ValueError("position_search_radius must be positive.")
        if self.rotation_search_radius <= 0.0:
            raise ValueError("rotation_search_radius must be positive.")

        self._demo_path = Path(demo_path) if demo_path else None
        self.video_path = Path(video_path) if video_path else None
        
        self.optimization_output_dir: Path | None = None
        self.server_address = server_address
        self.server_timeout = server_timeout
        self.render = render
        self.execute_release = bool(execute_release)
        self.evaluate_release = bool(evaluate_release)
        self.post_grasp_settle_steps = int(post_grasp_settle_steps)
        self.m2t2_hover_distance = float(m2t2_hover_distance)
        self.m2t2_grasp_depth_offset = float(m2t2_grasp_depth_offset)
        self.m2t2_grasp_height_offset = float(m2t2_grasp_height_offset)
        self.robot_collision_enabled = bool(robot_collision_enabled)
        self.robot_collision_barrier_weight = float(robot_collision_barrier_weight)
        self.robot_collision_force_threshold = float(robot_collision_force_threshold)
        self.robot_collision_force_scale = float(robot_collision_force_scale)
        if self.robot_collision_force_threshold < 0.0:
            raise ValueError("robot_collision_force_threshold cannot be negative.")
        if self.robot_collision_force_scale <= 0.0:
            raise ValueError("robot_collision_force_scale must be positive.")
        self.simulated_arm_collision_enabled = bool(simulated_arm_collision_enabled)
        self.simulated_arm_link_names = list(simulated_arm_link_names)
        self.simulated_arm_segment_radii = np.asarray(simulated_arm_segment_radii, dtype=np.float64)
        self.simulated_arm_samples_per_segment = int(simulated_arm_samples_per_segment)
        if len(self.simulated_arm_link_names) < 2:
            raise ValueError("simulated_arm_link_names must contain at least two links.")
        if self.simulated_arm_segment_radii.shape != (len(self.simulated_arm_link_names) - 1,):
            raise ValueError(
                "simulated_arm_segment_radii must contain one radius for "
                "each consecutive pair in simulated_arm_link_names."
            )
        if np.any(self.simulated_arm_segment_radii <= 0.0):
            raise ValueError("simulated_arm_segment_radii must all be positive.")
        if self.simulated_arm_samples_per_segment < 2:
            raise ValueError("simulated_arm_samples_per_segment must be at least 2.")
        self.joint_lower_limits = np.asarray(joint_lower_limits, dtype=np.float64)
        self.joint_upper_limits = np.asarray(joint_upper_limits, dtype=np.float64)
        if (self.joint_lower_limits.shape != (7,) or self.joint_upper_limits.shape != (7,)):
            raise ValueError(
                "joint_lower_limits and joint_upper_limits must each "
                "contain exactly 7 values."
            )
        if np.any(self.joint_lower_limits >= self.joint_upper_limits):
            raise ValueError("Every joint lower limit must be below its upper limit.")
        self.feasible_dedup_position_tolerance = float(feasible_dedup_position_tolerance)
        self.feasible_dedup_rotation_tolerance = float(feasible_dedup_rotation_tolerance)
        self.feasible_dedup_joint_tolerance = float(feasible_dedup_joint_tolerance)
        self._feasible_candidates = []
        self._feasible_archive_num_steps = None
        self._feasible_archive_population_size = 0
        self._cached_sim_num_envs = None
        self._grasp_context = None # Metadata captured during the initial reset/grasp.
        self._grasp_context_target = None
        self._collect_object_trajectories = False # Enabled only by FeatureCostFunction. 

        self.obstacle_cost = ObstacleCost(
            demo_path=self._demo_path,
            tcp_radius=tcp_radius,
            safety_margin=safety_margin,
            temperature=obstacle_temperature,
            query_workers=obstacle_query_workers,
            distance_backend=obstacle_distance_backend,
            grid_resolution=obstacle_grid_resolution,
            grid_padding=obstacle_grid_padding,
            grid_temperature_tail=obstacle_grid_temperature_tail,
            max_grid_cells=obstacle_max_grid_cells,
        )

        # Checkpoint variables
        self.best_feasible_x = None
        self.best_feasible_cost = float('inf')
        self.best_violation_x = None
        self.best_violation_cost = float('inf')
        self.best_violation = float('inf')
        if position_travel_scale <= 0.0:
            raise ValueError("position_travel_scale must be positive.")
        self.position_travel_scale = float(position_travel_scale)
    @property
    def demo_path(self) -> Path | None:
        return self._demo_path

    @property
    def optimizer_policy(self) -> dict[str, float]:
        """Use shared CEM settings with configurable search bounds."""
        return {
            "position_init_sigma": 0.08,
            "rotation_init_sigma": 0.10,
            "position_search_radius": self.position_search_radius,
            "rotation_search_radius": self.rotation_search_radius,
        }

    @demo_path.setter
    def demo_path(self, value: str | Path | None) -> None:
        new_demo_path = Path(value) if value else None

        # Only invalidate the grasp metadata when the demo actually changes.
        if new_demo_path != self._demo_path:
            self._grasp_context = None
            self._grasp_context_target = None

        self._demo_path = new_demo_path

        self.obstacle_cost.set_demo_path(self._demo_path)

    @staticmethod
    def _build_cost_terms(config: Mapping[str, Any]) -> dict[str, CostTerm]:
        names = {"trajectory", "endpoint", "displacement", "obstacle",
            "position_travel", "speed", "workspace_constraint",
        }
        if set(config) != names:
            raise ValueError(f"cost_terms must contain exactly these terms: {sorted(names)}")
        return {
            name: CostTerm(weight=float(settings["weight"]), enabled=bool(settings["enabled"]))
            for name, settings in config.items()
        }

    def _weighted_total(
        self,
        *,
        trajectory: float,
        endpoint: float,
        displacement: float,
        obstacle: float,
        position_travel: float,
        speed: float,
        workspace_constraint: float,
    ) -> float:
        """Combine raw costs in the same order as the original implementation."""
        return (
            self.w_traj * trajectory
            + self.w_endpoint * endpoint
            + self.w_displacement * displacement
            + self.w_obstacle * obstacle
            + self.w_position_travel * position_travel
            + self.w_speed * speed
            + self.w_workspace_constraint * workspace_constraint
        )

    def _term_enabled(self, name: str) -> bool:
        return self.cost_terms[name].enabled


    # =========================Cost Terms=========================

    def _tracking_costs(
        self,
        actual_trajectory: np.ndarray,
        desired_trajectory: np.ndarray,
        sim_results: Mapping[str, Any],
    ) -> tuple[float, float]:
        """Return full-trajectory and endpoint position/orientation costs."""
        position_error = np.linalg.norm(actual_trajectory[:, :3] - desired_trajectory[:, :3], axis=1)
        num_steps = len(actual_trajectory)
        time_weights = np.ones(num_steps)
        tail_length = max(2, min(20, num_steps))
        time_weights[-tail_length:] = np.linspace(2.0, 5.0, tail_length)
        normalized_position_error = (position_error / self.position_tolerance) ** 2
        mean_position_cost = np.average(normalized_position_error, weights=time_weights)
        worst_count = max(1, int(np.ceil(self.position_worst_fraction * num_steps)))
        worst_position_cost = np.mean(
            np.partition(
                normalized_position_error,
                num_steps - worst_count,
            )[-worst_count:]
        )
        position_cost = (mean_position_cost + self.position_worst_weight * worst_position_cost)

        actual_quat = actual_trajectory[:, 3:7].copy()
        desired_quat = desired_trajectory[:, 3:7].copy()
        actual_quat /= np.linalg.norm(actual_quat, axis=1, keepdims=True)
        desired_quat /= np.linalg.norm(desired_quat, axis=1, keepdims=True)
        dots = np.clip(np.abs(np.sum(actual_quat * desired_quat, axis=1)), 0.0, 1.0)
        angular_error = 2.0 * np.arccos(dots)
        normalized_orientation_error = (angular_error / self.rotation_tolerance) ** 2
        orientation_cost = np.average(normalized_orientation_error, weights=time_weights)

        trajectory_cost = float(self.position_loss_weight * position_cost
            + self.orientation_loss_weight * orientation_cost
        )
        endpoint_cost = float(
            self.position_loss_weight * normalized_position_error[-1]
            + self.orientation_loss_weight * normalized_orientation_error[-1]
        )
        return trajectory_cost, endpoint_cost

    def _speed_cost(self, tcp_trajectory: np.ndarray) -> float:
        """Soft penalty for commanded TCP speeds above the configured cap."""
        if self.maximum_tcp_speed is None:
            return 0.0
        tcp = np.asarray(tcp_trajectory, dtype=np.float64)
        if len(tcp) < 2:
            return 0.0
        speeds = np.linalg.norm(np.diff(tcp[:, :3], axis=0), axis=1) / self.control_timestep
        excess_fraction = np.maximum(0.0, speeds / self.maximum_tcp_speed - 1.0)
        return float(np.mean(excess_fraction ** 2))

    def _speed_violation(self, tcp_trajectory: np.ndarray) -> float:
        """Maximum commanded TCP speed excess in m/s for feasibility checks."""
        if (not self.enforce_speed_feasibility or self.maximum_tcp_speed is None):
            return 0.0
        tcp = np.asarray(tcp_trajectory, dtype=np.float64)
        if len(tcp) < 2:
            return 0.0
        max_speed = float(np.max(np.linalg.norm(np.diff(tcp[:, :3], axis=0), axis=1) / self.control_timestep))
        return max(0.0, max_speed - self.maximum_tcp_speed)

    def _displacement_cost(
        self,
        object_displacements: Mapping[str, Mapping[str, float]],
        target_object: str,
    ) -> tuple[float, float]:
        """Return the aggregate distractor cost and maximum position change."""
        per_object_costs = []
        max_displacement = 0.0
        for object_name, metrics in object_displacements.items():
            if object_name == target_object:
                continue

            per_object_costs.append(
                self.w_disp_mean_pos
                * (metrics["mean_pos"] / self.displacement_tolerance) ** 2
                + self.w_disp_peak_pos
                * (metrics["peak_pos"] / self.displacement_tolerance) ** 2
                + self.w_disp_final_pos
                * (metrics["final_pos"] / self.displacement_tolerance) ** 2
                + self.w_disp_rot
                * (metrics["peak_rot"] / self.rotation_tolerance) ** 2
            )
            max_displacement = max(max_displacement, metrics["peak_pos"])

        if not per_object_costs:
            return 0.0, max_displacement

        aggregate = (0.1 * np.mean(per_object_costs) + 0.9 * np.max(per_object_costs))
        return float(aggregate), max_displacement

    def _position_travel_cost(self, trajectory: np.ndarray) -> float:
        """Normalized commanded-TCP path length used to discourage loops."""
        tcp = np.asarray(trajectory, dtype=np.float64)
        if len(tcp) < 2:
            return 0.0
        path_length = np.linalg.norm(np.diff(tcp[:, :3], axis=0), axis=1).sum()
        return float(path_length / self.position_travel_scale)

    def _minimum_tcp_z_profile(self, tcp_trajectory: np.ndarray) -> np.ndarray:
        """Return the phase-aware minimum TCP height for each rollout frame."""
        tcp = np.asarray(tcp_trajectory, dtype=np.float64)
        if tcp.ndim != 2 or tcp.shape[1] < 3 or len(tcp) == 0:
            raise ValueError("tcp_trajectory must have non-empty shape (N, >=3), "f"got {tcp.shape}.")

        floor = np.full(len(tcp), self.minimum_tcp_z, dtype=np.float64)
        ramp_steps = self.minimum_tcp_z_ramp_steps
        if ramp_steps > 0:
            start_floor = min(float(tcp[0, 2]), self.minimum_tcp_z)
            ramp = np.linspace(
                start_floor,
                self.minimum_tcp_z,
                ramp_steps + 1,
                dtype=np.float64,
            )
            ramp_length = min(len(tcp), len(ramp))
            floor[:ramp_length] = ramp[:ramp_length]
        return floor

    def _workspace_constraint_cost(
        self,
        tcp_trajectory: np.ndarray,
        joint_trajectory: np.ndarray | None = None,
    ) -> float:
        """Penalize TCP poses below minimum Z and Panda joint-limit excess."""
        tcp = np.asarray(tcp_trajectory, dtype=np.float64)
        minimum_z_profile = self._minimum_tcp_z_profile(tcp)
        z_violations = np.maximum(0.0, minimum_z_profile - tcp[:, 2])
        cost = np.mean((z_violations / 0.01) ** 2)

        if joint_trajectory is not None:
            joints = np.asarray(joint_trajectory, dtype=np.float64)
            if joints.ndim != 2 or joints.shape[1] != 7:
                raise ValueError(
                    "joint_trajectory must have shape (N, 7), "
                    f"got {joints.shape}."
                )
            lower_excess = np.maximum(0.0, self.joint_lower_limits - joints)
            upper_excess = np.maximum(0.0, joints - self.joint_upper_limits)
            cost += np.mean((lower_excess / 0.01) ** 2 + (upper_excess / 0.01) ** 2)

        return float(cost)

    def _constraint_violation_breakdown(
        self,
        max_displacement: float,
        tcp_trajectory: np.ndarray,
        joint_trajectory: np.ndarray | None,
        commanded_tcp_trajectory: np.ndarray | None = None,
        additional_violation: float = 0.0,
    ) -> dict[str, float]:
        """Return raw measurements and weighted hard-constraint components."""
        tcp = np.asarray(tcp_trajectory, dtype=np.float64)
        minimum_tcp_z = float(np.min(tcp[:, 2]))
        minimum_z_profile = self._minimum_tcp_z_profile(tcp)
        z_shortfalls = np.maximum(0.0, minimum_z_profile - tcp[:, 2])
        z_violation_frame = int(np.argmax(z_shortfalls))
        maximum_z_shortfall = float(z_shortfalls[z_violation_frame])
        z_violation = max(0.0, maximum_z_shortfall - self.tcp_z_feasibility_tolerance)

        if joint_trajectory is None:
            joint_violation = float("inf")
        else:
            joints = np.asarray(joint_trajectory, dtype=np.float64)
            lower_excess = np.maximum(0.0, self.joint_lower_limits - joints)
            upper_excess = np.maximum(0.0, joints - self.joint_upper_limits)
            joint_violation = float(max(np.max(lower_excess), np.max(upper_excess)))

        speed_violation = self._speed_violation(
            tcp_trajectory
            if commanded_tcp_trajectory is None
            else commanded_tcp_trajectory
        )

        components = {
            "max_displacement_m": float(max_displacement),
            "minimum_tcp_z_m": minimum_tcp_z,
            "maximum_tcp_z_shortfall_m": maximum_z_shortfall,
            "tcp_z_feasibility_tolerance_m": (
                self.tcp_z_feasibility_tolerance
            ),
            "tcp_z_violation_frame": float(z_violation_frame),
            "required_tcp_z_at_violation_m": float(
                minimum_z_profile[z_violation_frame]
            ),
            "max_joint_excess_rad": joint_violation,
            "displacement": 10.0 * max(0.0, max_displacement - 0.04),
            "tcp_z": 100.0 * z_violation,
            "joint": 100.0 * joint_violation,
            "speed": 100.0 * speed_violation,
            "additional": float(additional_violation),
        }
        components["total"] = float(
            components["displacement"]
            + components["tcp_z"]
            + components["joint"]
            + components["speed"]
            + components["additional"]
        )
        return components

    # =================================================================================


    def reset_feasible_archive(
        self,
        num_steps: int,
        population_size: int | None = None,
    ) -> None:
        """Discard candidates from earlier optimization runs."""
        self._feasible_candidates = []
        self._feasible_archive_num_steps = int(num_steps)
        self._feasible_archive_population_size = int(
            population_size or 0
        )

    @property
    def feasible_archive_population_size(self) -> int:
        """Number of candidates evaluated for the current archive."""
        return self._feasible_archive_population_size

    def _record_feasible_candidate(
        self,
        tcp_trajectory: np.ndarray,
        joint_trajectory: np.ndarray,
        cost: float,
        violation: float,
    ) -> None:
        if violation > self.feasibility_tolerance:
            return
        self._feasible_candidates.append(
            {
                "tcp": np.asarray(
                    tcp_trajectory,
                    dtype=np.float64,
                ).copy(),
                "joints": np.asarray(
                    joint_trajectory,
                    dtype=np.float64,
                ).copy(),
                "cost": float(cost),
                "violation": float(violation),
            }
        )

    def lowest_cost_recorded_feasible_tcp(
        self,
    ) -> tuple[np.ndarray, float, float] | None:
        """Return the cheapest candidate eligible for archive saving."""
        if not self._feasible_candidates:
            return None

        selected = min(
            self._feasible_candidates,
            key=lambda item: item["cost"],
        )
        return (
            selected["tcp"].copy(),
            float(selected["cost"]),
            float(selected["violation"]),
        )

    def save_feasible_archive(self, path: str | Path) -> tuple[int, int]:
        """Deduplicate, rank, and save final-population feasible candidates."""
        archive_path = Path(path)
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        raw_count = len(self._feasible_candidates)
        ordered = sorted(
            self._feasible_candidates,
            key=lambda item: item["cost"],
        )
        kept = []

        for candidate in ordered:
            if kept:
                kept_tcp = np.stack([item["tcp"] for item in kept], axis=0)
                kept_joints = np.stack([item["joints"] for item in kept], axis=0)
                position_rms = np.sqrt(np.mean(np.sum(
                            (kept_tcp[:, :, :3]
                            - candidate["tcp"][None, :, :3]
                            )** 2,
                            axis=2,
                        ),
                        axis=1,
                    )
                )

                kept_quat = kept_tcp[:, :, 3:7]
                candidate_quat = candidate["tcp"][None, :, 3:7]
                kept_quat = kept_quat / np.linalg.norm(kept_quat, axis=2, keepdims=True)
                candidate_quat = candidate_quat / np.linalg.norm(candidate_quat, axis=2, keepdims=True)
                quat_dots = np.clip(np.abs(np.sum(
                            kept_quat * candidate_quat,
                            axis=2,
                        )
                    ),
                    0.0, 1.0,
                )
                rotation_rms = np.sqrt(np.mean((2.0 * np.arccos(quat_dots)) ** 2, axis=1))
                joint_rms = np.sqrt(
                    np.mean(
                        (
                            kept_joints
                            - candidate["joints"][None, :, :]
                        )
                        ** 2,
                        axis=(1, 2),
                    )
                )
                is_near_duplicate = np.any(
                    (position_rms <= self.feasible_dedup_position_tolerance)
                    & (rotation_rms <= self.feasible_dedup_rotation_tolerance)
                    & (joint_rms <= self.feasible_dedup_joint_tolerance)
                )
                if is_near_duplicate:
                    continue

            kept.append(candidate)

        num_steps = int(self._feasible_archive_num_steps or 0)
        if kept:
            tcp_trajectories = np.stack([item["tcp"] for item in kept], axis=0)
            joint_trajectories = np.stack([item["joints"] for item in kept], axis=0)
            costs = np.asarray([item["cost"] for item in kept], dtype=np.float64)
            violations = np.asarray([item["violation"] for item in kept], dtype=np.float64)
        else:
            tcp_trajectories = np.empty((0, num_steps, 7), dtype=np.float64)
            joint_trajectories = np.empty((0, num_steps, 7), dtype=np.float64)
            costs = np.empty((0,), dtype=np.float64)
            violations = np.empty((0,), dtype=np.float64)

        np.savez_compressed(
            archive_path,
            tcp_trajectories=tcp_trajectories,
            joint_trajectories=joint_trajectories,
            costs=costs,
            violations=violations,
            raw_feasible_count=np.asarray(raw_count, dtype=np.int64),
            saved_count=np.asarray(len(kept), dtype=np.int64),
            evaluated_population_count=np.asarray(
                self._feasible_archive_population_size,
                dtype=np.int64,
            ),
        )
        return raw_count, len(kept)

    def reset_checkpoint(self):
        self.best_feasible_x = None
        self.best_feasible_cost = float('inf')
        self.best_violation_x = None
        self.best_violation_cost = float('inf')
        self.best_violation = float('inf')

    def update_checkpoint(self, x, cost, violation):
        """Update feasible and least-violation checkpoints from evaluated constraints."""
        if violation <= self.feasibility_tolerance:
            if (self.best_feasible_x is None or cost < self.best_feasible_cost
            ):
                self.best_feasible_x = x.copy()
                self.best_feasible_cost = float(cost)
            return violation

        lower_violation = (violation < self.best_violation - self.feasibility_tolerance)
        same_violation_lower_cost = (
            abs(violation - self.best_violation)
            <= self.feasibility_tolerance
            and cost < self.best_violation_cost
        )
        if (
            self.best_violation_x is None
            or lower_violation
            or same_violation_lower_cost
        ):
            self.best_violation_x = x.copy()
            self.best_violation_cost = float(cost)
            self.best_violation = float(violation)
        return violation


    def _simulation_contact_query(self, target_object: str) -> dict | None:
        """Check robot contacts; target-object contacts require anchor metadata."""
        if not self.robot_collision_enabled:
            return None
        return {
            "target_object": target_object,
            "include_target_contacts": False,
        }

    def _collision_cost_and_violation(
        self,
        sim_results_list: list[Mapping[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Penalize and reject robot contact with any non-target object."""
        count = len(sim_results_list)
        if not self.robot_collision_enabled:
            return np.zeros(count), np.zeros(count)
        forces = np.asarray([
            result.get("max_forbidden_contact_force", np.nan)
            for result in sim_results_list
        ], dtype=np.float64)
        if forces.shape != (count,) or not np.all(np.isfinite(forces)):
            raise RuntimeError(
                "Robot collision checking requires one finite forbidden "
                "contact-force maximum per simulated trajectory."
            )
        costs, violations, excess = _contact_force_costs(
            forces,
            self.robot_collision_barrier_weight,
            self.robot_collision_force_threshold,
            self.robot_collision_force_scale,
        )
        self._last_robot_collision_batch = {
            "max_forbidden_contact_force": forces.copy(),
            "force_excess": excess.copy(),
        }
        print(
            "[Robot Collision] "
            f"forbidden={int(np.count_nonzero(violations))}/{count} "
            f"| max_force={np.max(forces):.3f}N",
            flush=True,
        )
        return costs, violations

    def validation_collision_diagnostics(
        self,
        sim_results: Mapping[str, Any],
    ) -> dict[str, float]:
        """Report robot-only forbidden contacts for final validation."""
        if not self.robot_collision_enabled:
            return {"violation": 0.0}
        _, violations = self._collision_cost_and_violation([sim_results])
        last = self._last_robot_collision_batch
        return {
            "violation": float(violations[0]),
            "max_forbidden_contact_force": float(last["max_forbidden_contact_force"][0]),
            "force_excess": float(last["force_excess"][0]),
        }

    def _additional_batch_cost_and_violation(
        self,
        candidate_trajectories: np.ndarray,
        sim_results_list: list[Mapping[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply robot-contact penalties and feasibility checks."""
        return self._collision_cost_and_violation(sim_results_list)

    def evaluate_batch(
        self,
        candidate_trajectories: np.ndarray,
        target_object: str,
        desired_trajectory: np.ndarray,
        candidate_spline_params: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluates a batch of candidate trajectories in parallel.
        candidate_trajectories: np.ndarray of shape (num_envs, N, D).

        """
        num_envs = candidate_trajectories.shape[0]

        sim_results_list = self._run_actual_simulation_batched(target_object, candidate_trajectories)

        # Compute batched obstacle proximity loss (using full trajectories with robot proxies)
        if self._term_enabled("obstacle"):
            extra_sphere_positions = None
            extra_sphere_radii = None
            if self.simulated_arm_collision_enabled:
                link_positions = np.stack([result["arm_link_positions"]for result in sim_results_list], axis=0)
                segment_starts = link_positions[:, :, :-1, :]
                segment_ends = link_positions[:, :, 1:, :]
                interpolation = np.linspace(0.0, 1.0, self.simulated_arm_samples_per_segment, dtype=np.float64)
                capsule_points = (
                    segment_starts[:, :, :, None, :]
                    * (1.0 - interpolation[None, None, None, :, None])
                    + segment_ends[:, :, :, None, :]
                    * interpolation[None, None, None, :, None]
                )
                extra_sphere_positions = capsule_points.reshape(num_envs, candidate_trajectories.shape[1], -1, 3)
                extra_sphere_radii = np.repeat(self.simulated_arm_segment_radii, self.simulated_arm_samples_per_segment)
            obstacle_cost_values = self.obstacle_cost.evaluate_batch(
                candidate_trajectories,
                target_object,
                extra_sphere_positions=extra_sphere_positions,
                extra_sphere_radii=extra_sphere_radii,
            )
        else:
            obstacle_cost_values = np.zeros(num_envs)

        additional_costs, additional_violations = (
            self._additional_batch_cost_and_violation(
                candidate_trajectories, sim_results_list
            )
        )

        costs = np.zeros(num_envs)
        violation_breakdowns: list[dict[str, float]] = []
        max_displacement_objects: list[str] = []

        for env_idx in range(num_envs):
            actual_trajectory = sim_results_list[env_idx]["actual_trajectory"]
            object_displacements = sim_results_list[env_idx]["object_displacements"]

            trajectory_cost, endpoint_cost = self._tracking_costs(
                actual_trajectory,
                desired_trajectory,
                sim_results=sim_results_list[env_idx],
            )
            displacement_cost, max_disp_m = self._displacement_cost(
                object_displacements,
                target_object,
            )
            distractor_peaks = {
                name: float(metrics["peak_pos"])
                for name, metrics in object_displacements.items()
                if name != target_object
            }
            max_displacement_objects.append(
                max(distractor_peaks, key=distractor_peaks.get)
                if distractor_peaks else "none"
            )

            # 3. Obstacle proximity continuous loss
            obstacle_cost = obstacle_cost_values[env_idx]
            candidate_trajectory = candidate_trajectories[env_idx]
            position_travel_cost = self._position_travel_cost(candidate_trajectory)
            speed_cost = self._speed_cost(candidate_trajectory)
            workspace_cost = self._workspace_constraint_cost(
                sim_results_list[env_idx]["actual_tcp_trajectory"],
                sim_results_list[env_idx]["joint_trajectory"],
            )

            # Total cost
            costs[env_idx] = self._weighted_total(
                trajectory=trajectory_cost,
                endpoint=endpoint_cost,
                displacement=displacement_cost,
                obstacle=obstacle_cost,
                position_travel=position_travel_cost,
                speed=speed_cost,
                # Retain the existing cost-term key for config compatibility.
                workspace_constraint=workspace_cost,
            )
            costs[env_idx] += additional_costs[env_idx]

            # Update checkpoint
            x_params = candidate_spline_params[env_idx]
            joint_trajectory = sim_results_list[env_idx]["joint_trajectory"]
            violation_breakdowns.append(
                self._constraint_violation_breakdown(
                    max_disp_m,
                    sim_results_list[env_idx]["actual_tcp_trajectory"],
                    joint_trajectory,
                    commanded_tcp_trajectory=candidate_trajectory,
                    additional_violation=additional_violations[env_idx],
                )
            )
            violation = violation_breakdowns[-1]["total"]
            self.update_checkpoint(x_params, costs[env_idx], violation)
            self._record_feasible_candidate(
                candidate_trajectory,
                joint_trajectory,
                costs[env_idx],
                violation,
            )

        violation_totals = np.asarray(
            [details["total"] for details in violation_breakdowns],
            dtype=np.float64,
        )
        component_labels = {
            "displacement": "displacement",
            "tcp_z": "tcp_z",
            "joint": "joint_limits",
            "speed": "speed_hard",
            "additional": (
                "feature_collision"
                if hasattr(self, "feature_collision_enabled")
                else "robot_collision"
            ),
        }
        violation_counts = {
            label: int(np.count_nonzero([
                details[key] > self.feasibility_tolerance
                for details in violation_breakdowns
            ]))
            for key, label in component_labels.items()
        }
        feasible_count = int(np.count_nonzero(
            violation_totals <= self.feasibility_tolerance
        ))
        print(
            "[Constraint Check] "
            f"feasible={feasible_count}/{num_envs} | violated counts: "
            + ", ".join(
                f"{name}={count}"
                for name, count in violation_counts.items()
            ),
            flush=True,
        )
        least_index = int(np.argmin(violation_totals))
        least = violation_breakdowns[least_index]
        print(
            "[Constraint Check] "
            f"least-violation candidate={least_index}, "
            f"total={least['total']:.6g} | "
            f"displacement={least['displacement']:.6g} "
            f"({max_displacement_objects[least_index]} "
            f"peak={least['max_displacement_m']:.4f}m), "
            f"tcp_z={least['tcp_z']:.6g} "
            f"(min={least['minimum_tcp_z_m']:.4f}m, "
            f"max_shortfall={least['maximum_tcp_z_shortfall_m']:.4f}m "
            f"at frame {int(least['tcp_z_violation_frame'])}, "
            f"required={least['required_tcp_z_at_violation_m']:.4f}m, "
            f"tolerance={least['tcp_z_feasibility_tolerance_m']:.4f}m), "
            f"joint_limits={least['joint']:.6g} "
            f"(max_excess={least['max_joint_excess_rad']:.6g}rad), "
            f"speed_hard={least['speed']:.6g}, "
            f"{component_labels['additional']}={least['additional']:.6g}",
            flush=True,
        )

        return costs

