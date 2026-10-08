import json
import logging
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


logger = logging.getLogger(__name__)


def compute_timestep_feature_weights(
    trajectories: list[list[dict]],
    feature_dim: int,
    feature_blocks: list[tuple[int, int]],
    orientation_slice: tuple[int, int] | None = None,
    dimension_mask: list[bool] | None = None,
    variance_percentile: float = 0.25,
    weight_scale_divisor: float = 3.0,
    minimum_valid_samples: int = 2,
    trajectory_change_boost: float = 1.0,
    initial_weight_fraction: float = 0.15,
    initial_weight_start: float = 0.1,
) -> tuple[list[dict], dict]:
    """Outputs CF trajectory and unnormalized confidence weights.

    Position variance is computed per Cartesian dimension. 
    Orientation targets remain quaternions, while orientation variance and weights use the three components of the rotation-vector residual around the reference rotation.
    """
    if feature_dim < 0:
        raise ValueError(f"feature_dim must be non-negative, got {feature_dim}")
    weight_dim = feature_dim - (1 if orientation_slice is not None else 0) # orientation is saved as quaterrnion, but only three components are used for variance and weights
    if dimension_mask is None:
        dimension_mask = [True] * weight_dim
    if len(dimension_mask) != weight_dim or not all(isinstance(value, bool) for value in dimension_mask):
        raise ValueError(
            "dimension_mask must be a boolean list with one entry per feature dimension "
            f"({weight_dim}), got {dimension_mask}"
        )
    if not trajectories:
        raise ValueError("At least one trajectory is required to compute feature weights")

    num_frames = min(len(trajectory) for trajectory in trajectories)
    variances = np.full((num_frames, weight_dim), np.nan, dtype=np.float64)
    sample_counts = np.zeros(num_frames, dtype=np.int64)
    reference_steps: list[dict] = []

    # extract task features from multiple trajectories
    for frame in range(num_frames):
        samples = []
        for trajectory in trajectories:
            step = trajectory[frame]
            values = step.get("values")
            if step.get("valid", False) and isinstance(values, list) and len(values) == feature_dim:
                samples.append(np.asarray(values, dtype=np.float64))

        sample_counts[frame] = len(samples)
        if not samples:
            reference_steps.append({"frame": frame, "valid": False, "values": None, "weights": [0.0] * weight_dim, "sample_count": 0})
            continue

        sample_array = np.stack(samples)
        if orientation_slice is not None:
            start, end = orientation_slice
            reference_quaternion = sample_array[0, start:end]
            for sample_index in range(1, len(sample_array)):
                if np.dot(sample_array[sample_index, start:end], reference_quaternion) < 0:
                    sample_array[sample_index, start:end] *= -1

        reference_values = np.median(sample_array, axis=0)
        if orientation_slice is not None:
            start, end = orientation_slice
            sample_xyzw = sample_array[:, start:end][:, [1, 2, 3, 0]]
            reference_xyzw = Rotation.from_quat(sample_xyzw).mean().as_quat()
            reference_values[start:end] = reference_xyzw[[3, 0, 1, 2]]

        if len(sample_array) >= minimum_valid_samples:
            position_dim = orientation_slice[0] if orientation_slice is not None else feature_dim
            if position_dim:
                variances[frame, :position_dim] = np.var(
                    sample_array[:, :position_dim], axis=0
                )
            if orientation_slice is not None:
                start, end = orientation_slice
                reference_xyzw = reference_values[start:end][[1, 2, 3, 0]]
                sample_xyzw = sample_array[:, start:end][:, [1, 2, 3, 0]]
                reference_rotation = Rotation.from_quat(reference_xyzw)
                residual_rotvecs = (
                    reference_rotation.inv() * Rotation.from_quat(sample_xyzw)
                ).as_rotvec()
                variances[frame, start:start + 3] = np.var(
                    residual_rotvecs, axis=0
                )
        reference_steps.append({
            "frame": frame,
            "valid": True,
            "values": reference_values.tolist(),
            "weights": [0.0] * weight_dim,
            "sample_count": int(len(sample_array)),
        })

    # calculate the weight cutoff based on the variance_percentile
    # produces cutoffs and scales
    variance_cutoffs: list[float | None] = [None] * weight_dim
    scales: list[float | None] = [None] * weight_dim
    block_cutoffs = []
    epsilon = float(np.finfo(np.float64).eps)
    for start, end in feature_blocks:
        candidates = [
            dimension for dimension in range(start, end)
            if dimension_mask[dimension]
        ]
        if not candidates:
            continue
        block_variances = variances[:, candidates]
        valid_variances = block_variances[np.isfinite(block_variances)]
        if not valid_variances.size:
            block_cutoffs.append({"index_range": [start, end], "cutoff": None, "scale": None})
            continue
        cutoff = float(np.percentile(valid_variances, variance_percentile * 100.0))
        scale = max(cutoff / weight_scale_divisor, epsilon)
        block_cutoffs.append({"index_range": [start, end], "cutoff": cutoff, "scale": scale})
        for dimension in candidates:
            variance_cutoffs[dimension] = cutoff
            scales[dimension] = scale

    # if an axis has larger change, the movement is probably more significant. produces trajectory_change
    # Position uses per-axis total variation. 
    # Orientation uses the XYZ components of the incremental rotation vector.
    trajectory_change = np.zeros(weight_dim, dtype=np.float64)
    for start, end in feature_blocks:
        is_orientation = (
            orientation_slice is not None
            and start == orientation_slice[0]
            and end == orientation_slice[0] + 3
        )
        if is_orientation:
            previous_rotation = None
            for step in reference_steps:
                if not step["valid"] or step["sample_count"] < minimum_valid_samples:
                    previous_rotation = None
                    continue
                q_start, q_end = orientation_slice
                quaternion_wxyz = np.asarray(
                    step["values"][q_start:q_end], dtype=np.float64
                )
                rotation = Rotation.from_quat(quaternion_wxyz[[1, 2, 3, 0]])
                if previous_rotation is not None:
                    increment = (previous_rotation.inv() * rotation).as_rotvec()
                    trajectory_change[start:end] += np.abs(increment)
                previous_rotation = rotation
        else:
            for dimension in range(start, end):
                previous_value = None
                for step in reference_steps:
                    if not step["valid"] or step["sample_count"] < minimum_valid_samples:
                        previous_value = None
                        continue
                    value = float(step["values"][dimension])
                    if previous_value is not None:
                        trajectory_change[dimension] += abs(value - previous_value)
                    previous_value = value

    # compute trajectory_change_multiplier
    normalized_trajectory_change = np.zeros(weight_dim, dtype=np.float64)
    for start, end in feature_blocks:
        candidates = [dimension for dimension in range(start, end) if dimension_mask[dimension]]
        if not candidates:
            continue
        block_change = trajectory_change[candidates]
        min_change = float(np.min(block_change))
        max_change = float(np.max(block_change))
        if max_change > min_change:
            normalized_trajectory_change[candidates] = (block_change - min_change) / (max_change - min_change)
        elif max_change > epsilon:
            normalized_trajectory_change[candidates] = 1.0
    trajectory_change_multipliers = 1.0 + trajectory_change_boost * normalized_trajectory_change
    # not important, but during starting the variance is almost always low, so the weight can be gradually increased
    warmup_frames = int(np.ceil(num_frames * initial_weight_fraction))
    warmup_multipliers = np.ones(num_frames, dtype=np.float64)
    if warmup_frames:
        warmup_multipliers[:warmup_frames] = np.linspace(initial_weight_start, 1.0, warmup_frames)

    for frame, step in enumerate(reference_steps):
        if sample_counts[frame] < minimum_valid_samples:
            continue
        weights = np.zeros(weight_dim, dtype=np.float64)
        for dimension in range(weight_dim):
            variance = variances[frame, dimension]
            cutoff = variance_cutoffs[dimension]
            if cutoff is not None and np.isfinite(variance) and variance <= cutoff:
                weights[dimension] = (np.exp(-variance / scales[dimension])* trajectory_change_multipliers[dimension]* warmup_multipliers[frame])
        step["weights"] = weights.tolist()

    metadata = {
        "dimension_mask": dimension_mask,
        "weight_dim": weight_dim,
        "orientation_weight_representation": (
            "rotation_vector_xyz" if orientation_slice is not None else None
        ),
        "variance_cutoffs": variance_cutoffs,
        "weight_scales": scales,
        "block_variance_cutoffs": block_cutoffs,
        "variance_gate_percentile": variance_percentile * 100.0,
        "weight_scale_divisor": weight_scale_divisor,
        "minimum_valid_samples": minimum_valid_samples,
        "trajectory_change": trajectory_change.tolist(),
        "normalized_trajectory_change": normalized_trajectory_change.tolist(),
        "trajectory_change_boost": trajectory_change_boost,
        "initial_weight_fraction": initial_weight_fraction,
        "initial_weight_start": initial_weight_start,
        "warmup_frames": warmup_frames,
    }
    return reference_steps, metadata

def _feature_layout(relation: list[str]):
    if "relative_position" in relation and "relative_orientation" in relation:
        return 7, 6, [0, 7], [(0, 3), (3, 6)], (3, 7)
    if "relative_position" in relation:
        return 3, 3, [0, 3], [(0, 3)], None
    if "relative_orientation" in relation:
        return 4, 3, [3, 7], [(0, 3)], (0, 4)
    raise ValueError(f"No supported feature relation: {relation}")


def _trajectory_path(demo: Path, object_name: str) -> Path | None:
    for name in (object_name.replace(" ", "_"), object_name):
        path = demo / name / "trajectory.json"
        if path.is_file():
            return path
    return None


def _relative_trajectory(target: list[dict], reference: list[dict], relation: list[str]) -> list[dict]:
    steps = []
    for frame, (target_step, reference_step) in enumerate(zip(target, reference)):
        step = {"frame": frame, "valid": False, "values": None}
        steps.append(step)
        if not (target_step.get("valid", False) and reference_step.get("valid", False)):
            continue
        values = []
        if "relative_position" in relation:
            values.extend((
                np.asarray(target_step["center_robot"], dtype=np.float64)
                - np.asarray(reference_step["center_robot"], dtype=np.float64)
            ).tolist())
        if "relative_orientation" in relation:
            if not (
                target_step.get("rotation_valid", True)
                and reference_step.get("rotation_valid", True)
                and "rotation_robot" in target_step
                and "rotation_robot" in reference_step
            ):
                continue
            rotation = (
                Rotation.from_matrix(reference_step["rotation_robot"]).inv()
                * Rotation.from_matrix(target_step["rotation_robot"])
            )
            quaternion = rotation.as_quat()
            values.extend(quaternion[[3, 0, 1, 2]].tolist())
        step.update(valid=True, values=values)
    return steps


def extract_task_features(
    feature_dir: str | Path,
    demo_dirs: list[Path],
    weighting: dict | None = None,
) -> list[Path]:
    feature_dir = Path(feature_dir)
    constraints = json.loads((feature_dir / "task_feature.json").read_text())
    if not isinstance(constraints, list):
        raise TypeError("task_feature.json must contain a top-level list")
    demos = sorted(Path(demo) for demo in demo_dirs)
    if not demos:
        raise ValueError("At least one demonstration is required")
    if any(demo.parent != feature_dir for demo in demos):
        raise ValueError("All demonstrations must be direct children of the feature directory")
    outputs = []
    for index, constraint in enumerate(constraints):
        target = constraint.get("target_object")
        reference = constraint.get("reference_object")
        if not target or not reference:
            raise ValueError(f"Constraint {index} requires target_object and reference_object")
        relation = constraint.get("relation", [])
        if isinstance(relation, str):
            relation = [relation]
        if not isinstance(relation, list):
            raise TypeError(f"Constraint {index} has invalid relation: {relation}")
        feature_dim, weight_dim, index_range, blocks, orientation = _feature_layout(relation)
        trajectories, skipped = [], []
        for demo in demos:
            target_path = _trajectory_path(demo, target)
            reference_path = _trajectory_path(demo, reference)
            if target_path is None or reference_path is None:
                skipped.append(demo.name)
                continue
            trajectories.append(_relative_trajectory(
                json.loads(target_path.read_text()),
                json.loads(reference_path.read_text()),
                relation,
            ))
        if not trajectories:
            raise FileNotFoundError(f"No complete trajectories for constraint {index} ({target}, {reference})")
        feature_reference, metadata = compute_timestep_feature_weights(
            trajectories, feature_dim, blocks, orientation,
            constraint.get("dimension_mask"), **(weighting or {}),
        )
        data = {
            "constraint_index": index,
            "target_object": target,
            "reference_object": reference,
            "relation": relation,
            "feature_dim": feature_dim,
            "weight_dim": weight_dim,
            "index_range": index_range,
            "num_demos_used": len(trajectories),
            "skipped_demos": skipped,
            "weighting": metadata,
            "trajectory": feature_reference,
        }
        path = feature_dir / f"tf_{index}_trajectory.json"
        path.write_text(json.dumps(data, indent=4))
        _plot_weights(data, path.with_name(f"tf_{index}_trajectory_weights.png"))
        logger.info("Saved task feature: %s", path)
        outputs.append(path)
    return outputs


def _plot_weights(data: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    has_position = "relative_position" in data["relation"]
    has_orientation = "relative_orientation" in data["relation"]
    labels = []
    if has_position:
        labels.extend(f"relative position {axis}" for axis in "xyz")
    if has_orientation:
        labels.extend(f"relative rotation {axis}" for axis in "xyz")
    frames, values, weights = [], [], []
    for step in data["trajectory"]:
        if not step["valid"]:
            continue
        value = step["values"][:3] if has_position else []
        if has_orientation:
            start = 3 if has_position else 0
            quaternion = np.asarray(step["values"][start:start + 4])
            value = value + Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_rotvec().tolist()
        frames.append(step["frame"])
        values.append(value)
        weights.append(step["weights"])
    figure, axes = plt.subplots(
        data["weight_dim"], 1, sharex=True, squeeze=False,
        figsize=(12, max(2.5 * data["weight_dim"], 3)),
    )
    try:
        scatter = None
        for dimension, axis in enumerate(axes[:, 0]):
            axis.set_ylabel(labels[dimension])
            axis.grid(True, alpha=0.25)
            if frames:
                axis.plot(frames, np.asarray(values)[:, dimension], color="0.65", linewidth=1, zorder=1)
                scatter = axis.scatter(
                    frames, np.asarray(values)[:, dimension],
                    c=np.clip(np.asarray(weights)[:, dimension], 0, 1),
                    cmap="Blues", vmin=0, vmax=1, s=22, zorder=2,
                )
        axes[-1, 0].set_xlabel("frame")
        figure.suptitle(f"Task-feature weights: {data['target_object']} relative to {data['reference_object']}", y=0.995)
        if scatter is not None:
            colorbar = figure.colorbar(scatter, ax=axes[:, 0].tolist(), pad=0.015)
            colorbar.set_label("optimization weight (light = low, dark = high)")
        figure.savefig(path, dpi=160, bbox_inches="tight")
    finally:
        plt.close(figure)
