import logging
import json
import numpy as np
from pathlib import Path
from typing import Any, Mapping
from scipy.spatial.transform import Rotation as Rotation

from .trajectory import TrajectoryCostFunction, _contact_force_costs

logger = logging.getLogger(__name__)

class FeatureCostFunction(TrajectoryCostFunction):
    """Track weighted relative object features extracted from demonstrations.
    All non-tracking costs are inherited unchanged.  
    """

    def __init__(
        self,
        *,
        feature_collision_enabled: bool = True,
        feature_collision_barrier_weight: float = 5000.0,
        feature_collision_force_threshold: float = 1.0,
        feature_collision_force_scale: float = 10.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.feature_collision_enabled = bool(feature_collision_enabled)
        self.feature_collision_barrier_weight = float(feature_collision_barrier_weight)
        self.feature_collision_force_threshold = float(feature_collision_force_threshold)
        self.feature_collision_force_scale = float(feature_collision_force_scale)
        if self.feature_collision_barrier_weight < 0.0:
            raise ValueError("feature_collision_barrier_weight cannot be negative.")
        if self.feature_collision_force_threshold < 0.0:
            raise ValueError("feature_collision_force_threshold cannot be negative.")
        if self.feature_collision_force_scale <= 0.0:
            raise ValueError("feature_collision_force_scale must be positive.")
        if self.feature_collision_enabled and not self._term_enabled("obstacle"):
            raise ValueError("Feature collision enforcement requires the obstacle term.")
        self._collect_object_trajectories = True
        self.task_features: list[dict[str, Any]] = []
        self.required_feature_objects: set[str] = set()

    def _allowed_target_contact_objects(self, target_object: str) -> set[str]:
        """References named by this target may receive intentional contact."""
        return {
            feature["reference_object"]
            for feature in self.task_features
            if feature["target_object"] == target_object
        }

    def _simulation_contact_query(self, target_object: str) -> dict | None:
        if not self.feature_collision_enabled:
            return None
        return {
            "target_object": target_object,
            "allowed_target_contacts": sorted(
                self._allowed_target_contact_objects(target_object)
            ),
        }

    def _additional_batch_cost_and_violation(
        self,
        candidate_trajectories: np.ndarray,
        sim_results_list: list[Mapping[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Add fixed-weight rotation regularization and forbidden-contact costs."""
        collision_costs, violations = self._collision_cost_and_violation(sim_results_list)
        smoothness_costs = np.asarray([
            self._smoothness_cost(trajectory) for trajectory in candidate_trajectories
        ])
        return collision_costs + smoothness_costs, violations

    def _collision_cost_and_violation(
        self,
        sim_results_list: list[Mapping[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Make forbidden PhysX contacts infeasible for feature optimization."""
        batch_size = len(sim_results_list)
        if not self.feature_collision_enabled:
            return np.zeros(batch_size), np.zeros(batch_size)
        forces = np.asarray(
            [
                result.get("max_forbidden_contact_force", np.nan)
                for result in sim_results_list
            ],
            dtype=np.float64,
        )
        if forces.shape != (batch_size,) or not np.all(np.isfinite(forces)):
            raise RuntimeError(
                "Feature collision checking requires one finite forbidden "
                "contact-force maximum per simulated trajectory. Restart "
                "the simulation server after updating the code."
            )
        barrier_costs, violations, excess = _contact_force_costs(
            forces,
            self.feature_collision_barrier_weight,
            self.feature_collision_force_threshold,
            self.feature_collision_force_scale,
        )
        self._last_feature_collision_batch = {
            "max_forbidden_contact_force": forces.copy(),
            "force_excess": excess.copy(),
        }
        print(
            "[Feature Collision] "
            f"forbidden={int(np.count_nonzero(violations))}/{batch_size} "
            f"| max_force={np.max(forces):.3f}N",
            flush=True,
        )
        return barrier_costs, violations

    def validation_collision_diagnostics(
        self,
        sim_results: Mapping[str, Any],
    ) -> dict[str, float]:
        """Return feature-only collision feasibility for the final rollout."""
        if not self.feature_collision_enabled:
            return {"violation": 0.0}
        _, violations = self._collision_cost_and_violation([sim_results])
        last = self._last_feature_collision_batch
        return {
            "violation": float(violations[0]),
            "max_forbidden_contact_force": float(last["max_forbidden_contact_force"][0]),
            "force_excess": float(last["force_excess"][0]),
        }

    def set_feature_directory(self, cf_dir: str | Path) -> None:
        """Load every consensus ``tf_*_trajectory.json`` in ``cf_dir``."""
        directory = Path(cf_dir)
        paths = sorted(directory.glob("tf_*_trajectory.json"))
        if not paths:
            raise FileNotFoundError(f"No tf_*_trajectory.json files found in {directory}")

        features: list[dict[str, Any]] = []
        indices: set[int] = set()
        for path in paths:
            with open(path, "r") as file:
                feature = json.load(file)
            self._validate_task_feature(feature, path)
            index = int(feature["constraint_index"])
            if index in indices:
                raise ValueError(f"Duplicate constraint_index {index} in {directory}")
            indices.add(index)
            feature["_path"] = path
            features.append(feature)

        self.task_features = features
        self.required_feature_objects = {
            object_name
            for feature in features
            for object_name in (
                feature["target_object"], feature["reference_object"]
            )
        }
        logger.info(
            "Loaded %d weighted task feature(s) from %s for objects: %s",
            len(features),
            directory,
            sorted(self.required_feature_objects),
        )

    @property
    def has_position_only_features(self) -> bool:
        """Whether every loaded feature tracks position and no orientation."""
        return bool(self.task_features) and all(
            "relative_position" in feature["relation"]
            and "relative_orientation" not in feature["relation"]
            for feature in self.task_features
        )

    @property
    def has_orientation_only_features(self) -> bool:
        """Whether every loaded feature tracks orientation and no position."""
        return bool(self.task_features) and all(
            "relative_orientation" in feature["relation"]
            and "relative_position" not in feature["relation"]
            for feature in self.task_features
        )

    @staticmethod
    def _validate_task_feature(feature: Any, path: Path) -> None:
        if not isinstance(feature, dict):
            raise TypeError(f"Task feature at {path} must be a JSON object.")
        required = {
            "constraint_index", "target_object", "reference_object",
            "relation", "feature_dim", "weight_dim", "trajectory",
        }
        missing = required - set(feature)
        if missing:
            raise ValueError(f"Task feature at {path} missing keys: {sorted(missing)}")
        relation = feature["relation"]
        if not isinstance(relation, list):
            raise TypeError(f"Task feature at {path} must use a list of relations.")
        expected_dim = (3 * ("relative_position" in relation) + 4 * ("relative_orientation" in relation))
        expected_weight_dim = (3 * ("relative_position" in relation) + 3 * ("relative_orientation" in relation))
        if expected_dim == 0 or feature["feature_dim"] != expected_dim:
            raise ValueError(
                f"Task feature at {path} has feature_dim={feature['feature_dim']}; "
                f"expected {expected_dim} from relation={relation}."
            )
        if feature["weight_dim"] != expected_weight_dim:
            raise ValueError(
                f"Task feature at {path} has weight_dim={feature['weight_dim']}; "
                f"expected {expected_weight_dim} from relation={relation}."
            )
        trajectory = feature["trajectory"]
        if not isinstance(trajectory, list) or not trajectory:
            raise ValueError(f"Task feature at {path} must contain a non-empty trajectory.")
        for frame in trajectory:
            if not isinstance(frame, dict):
                raise TypeError(f"Task feature at {path} has a non-object trajectory frame.")
            if frame["valid"]:
                values, weights = frame.get("values"), frame.get("weights")
                if (
                    not isinstance(values, list)
                    or not isinstance(weights, list)
                    or len(values) != expected_dim
                    or len(weights) != expected_weight_dim
                ):
                    raise ValueError(f"Task feature at {path} has invalid values/weights dimensions.")

    @staticmethod
    def _relative_feature_values(
        target_poses: np.ndarray,
        reference_poses: np.ndarray,
        relation: list[str],
    ) -> np.ndarray:
        target = np.asarray(target_poses, dtype=np.float64)
        reference = np.asarray(reference_poses, dtype=np.float64)
        if target.ndim != 2 or target.shape[1] != 7 or target.shape != reference.shape:
            raise ValueError("Object trajectories must have matching shape (num_steps, 7).")
        values = []
        if "relative_position" in relation:
            values.append(target[:, :3] - reference[:, :3])
        if "relative_orientation" in relation:
            target_quat = target[:, 3:7]
            reference_quat = reference[:, 3:7]
            target_norm = np.linalg.norm(target_quat, axis=1)
            reference_norm = np.linalg.norm(reference_quat, axis=1)
            if np.any(target_norm <= 1e-12) or np.any(reference_norm <= 1e-12):
                raise ValueError("Object trajectory contains a zero-norm quaternion.")
            target_xyzw = target_quat[:, [1, 2, 3, 0]] / target_norm[:, None]
            reference_xyzw = reference_quat[:, [1, 2, 3, 0]] / reference_norm[:, None]
            target_rotation = Rotation.from_quat(target_xyzw)
            reference_rotation = Rotation.from_quat(reference_xyzw)
        
            target_motion = target_rotation * target_rotation[0].inv()
            reference_motion = reference_rotation * reference_rotation[0].inv()
            relative_xyzw = (reference_motion.inv() * target_motion).as_quat()
            values.append(relative_xyzw[:, [3, 0, 1, 2]])
        return np.hstack(values)

    @staticmethod
    def _feature_error_vector(
        observed: np.ndarray,
        desired: np.ndarray,
        relation: list[str],
    ) -> np.ndarray:
        """Return XYZ position and rotation-vector errors in weight order."""
        errors = []
        has_position = "relative_position" in relation
        if has_position:
            errors.append(observed[:3] - desired[:3])
        if "relative_orientation" in relation:
            orientation_start = 3 if has_position else 0
            observed_wxyz = observed[orientation_start:orientation_start + 4]
            desired_wxyz = desired[orientation_start:orientation_start + 4]
            if (
                np.linalg.norm(observed_wxyz) <= 1e-12
                or np.linalg.norm(desired_wxyz) <= 1e-12
            ):
                raise ValueError("Feature contains a zero-norm quaternion.")
            observed_rotation = Rotation.from_quat(observed_wxyz[[1, 2, 3, 0]])
            desired_rotation = Rotation.from_quat(desired_wxyz[[1, 2, 3, 0]])
            errors.append((desired_rotation.inv() * observed_rotation).as_rotvec())
        return np.hstack(errors)

    def _tracking_costs(
        self,
        actual_trajectory: np.ndarray,
        desired_trajectory: np.ndarray,
        sim_results: Mapping[str, Any],
    ) -> tuple[float, float]:
        """Return weighted full-trajectory and final-feature tracking costs."""
        if not self.task_features:
            raise RuntimeError("FeatureCostFunction has no loaded task features.")
        object_trajectories = sim_results["object_trajectories"]

        total_loss = 0.0
        total_active_dimensions = 0
        endpoint_loss = 0.0
        endpoint_active_dimensions = 0
        for feature in self.task_features:
            target_name = feature["target_object"]
            reference_name = feature["reference_object"]
            missing = {target_name, reference_name} - set(object_trajectories)
            if missing:
                raise KeyError(
                    f"Simulation result is missing feature object(s) {sorted(missing)} "
                    f"for {feature['_path']}"
                )
            actual = self._relative_feature_values(
                object_trajectories[target_name],
                object_trajectories[reference_name],
                feature["relation"],
            )
            frames = feature["trajectory"]
            if len(actual) != len(frames):
                raise ValueError(
                    f"Feature {feature['_path']} has {len(frames)} timesteps but "
                    f"the rollout has {len(actual)}. Timesteps must match exactly."
                )

            last_valid_endpoint_loss = None
            last_valid_endpoint_count = 0
            for timestep, frame in enumerate(frames):
                if not frame["valid"]:
                    continue
                desired = np.asarray(frame["values"], dtype=np.float64)
                weights = np.asarray(frame["weights"], dtype=np.float64)
                error = self._feature_error_vector(
                    actual[timestep], desired, feature["relation"]
                )
                tolerances = np.full(feature["weight_dim"], self.position_tolerance)
                if "relative_orientation" in feature["relation"]:
                    tolerances[-3:] = self.rotation_tolerance
                squared_error = (error / tolerances) ** 2

                finite_endpoint = np.isfinite(squared_error)
                if np.any(finite_endpoint):
                    last_valid_endpoint_loss = float(np.sum(squared_error[finite_endpoint]))
                    last_valid_endpoint_count = int(np.count_nonzero(finite_endpoint))

                active = np.isfinite(weights) & (weights > 0.0)
                if not np.any(active):
                    continue
                frame_loss = float(np.sum(weights[active] * squared_error[active]))
                frame_count = int(np.count_nonzero(active))
                total_loss += frame_loss
                total_active_dimensions += frame_count

            if last_valid_endpoint_loss is not None:
                endpoint_loss += last_valid_endpoint_loss
                endpoint_active_dimensions += last_valid_endpoint_count

        trajectory_cost = total_loss / max(total_active_dimensions, 1)
        endpoint_cost = endpoint_loss / max(endpoint_active_dimensions, 1)
        return float(trajectory_cost), float(endpoint_cost)

    def log_tracking_breakdown(
        self,
        sim_results: Mapping[str, Any],
    ) -> None:
        """Print final weighted TF contributions for comparison runs."""
        object_trajectories = sim_results["object_trajectories"]
        for feature in self.task_features:
            actual = self._relative_feature_values(
                object_trajectories[feature["target_object"]],
                object_trajectories[feature["reference_object"]],
                feature["relation"],
            )
            relation = feature["relation"]
            contributions = np.zeros(feature["weight_dim"], dtype=np.float64)
            active_counts = np.zeros(feature["weight_dim"], dtype=np.int64)
            for timestep, frame in enumerate(feature["trajectory"]):
                if not frame["valid"]:
                    continue
                desired = np.asarray(frame["values"], dtype=np.float64)
                weights = np.asarray(frame["weights"], dtype=np.float64)
                error = self._feature_error_vector(
                    actual[timestep], desired, relation
                )
                tolerances = np.full(feature["weight_dim"], self.position_tolerance)
                if "relative_orientation" in relation:
                    tolerances[-3:] = self.rotation_tolerance
                active = np.isfinite(weights) & (weights > 0.0)
                contributions[active] += (
                    weights[active]
                    * (error[active] / tolerances[active]) ** 2
                )
                active_counts[active] += 1
            per_dimension = np.divide(
                contributions,
                active_counts,
                out=np.zeros_like(contributions),
                where=active_counts > 0,
            )
            print(
                "[TF Tracking] constraint="
                f"{feature['constraint_index']} "
                f"({feature['target_object']} relative to "
                f"{feature['reference_object']}): "
                f"per-dimension={per_dimension.tolist()} | "
                f"active-frame-counts={active_counts.tolist()}",
                flush=True,
            )

    def _smoothness_cost(self, trajectory: np.ndarray) -> float:
        """Penalize TCP rotational travel with weight 1.0 when orientation is untracked."""
        if not self.has_position_only_features:
            return 0.0
        poses = np.asarray(trajectory, dtype=np.float64)
        if poses.ndim != 2 or poses.shape[1] < 7 or len(poses) < 2:
            return 0.0
        quaternions = poses[:, 3:7]
        norms = np.linalg.norm(quaternions, axis=1)
        if np.any(norms <= 1e-12):
            raise ValueError("TCP trajectory contains a zero-norm quaternion.")
        quaternions = quaternions / norms[:, None]
        dots = np.clip(np.abs(np.sum(quaternions[1:] * quaternions[:-1], axis=1)), 0.0, 1.0)
        angular_steps = 2.0 * np.arccos(dots)
        return float(np.sum(angular_steps / self.rotation_tolerance))
