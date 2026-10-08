"""Validate simulator rollouts and export optimization artifacts."""

from pathlib import Path

import numpy as np
from termcolor import colored


def _validate_arrays(commanded, results) -> dict[str, np.ndarray]:
    arrays = {
        "tcp_trajectory": np.asarray(commanded, dtype=np.float64),
        "actual_tcp_trajectory": np.asarray(results["actual_tcp_trajectory"], dtype=np.float64),
        "object_trajectory": np.asarray(results["actual_trajectory"], dtype=np.float64),
        "joint_trajectory": np.asarray(results["joint_trajectory"], dtype=np.float64),
    }
    num_steps = len(commanded)
    for name, array in arrays.items():
        widths = (3, 7) if name == "actual_tcp_trajectory" else (7,)
        if array.ndim != 2 or array.shape[0] != num_steps or array.shape[1] not in widths:
            raise ValueError(f"Invalid final {name} shape {array.shape}; expected {num_steps} rows and {widths} columns.")
        if not np.isfinite(array).all():
            raise ValueError(f"The final {name} contains non-finite values.")
    return arrays


def _report_constraints(breakdown, displacements, target_object, has_feature_collision):
    peaks = {
        name: float(metrics["peak_pos"])
        for name, metrics in displacements.items() if name != target_object
    }
    displaced_object = max(peaks, key=peaks.get) if peaks else "none"
    collision_label = "feature_collision" if has_feature_collision else "robot_collision"
    print(
        "[Validation Constraints] "
        f"total={breakdown['total']:.6g} | "
        f"displacement={breakdown['displacement']:.6g} "
        f"({displaced_object} peak={breakdown['max_displacement_m']:.4f}m), "
        f"tcp_z={breakdown['tcp_z']:.6g} "
        f"(min={breakdown['minimum_tcp_z_m']:.4f}m, "
        f"max_shortfall={breakdown['maximum_tcp_z_shortfall_m']:.4f}m "
        f"at frame {int(breakdown['tcp_z_violation_frame'])}, "
        f"required={breakdown['required_tcp_z_at_violation_m']:.4f}m, "
        f"tolerance={breakdown['tcp_z_feasibility_tolerance_m']:.4f}m), "
        f"joint_limits={breakdown['joint']:.6g} "
        f"(max_excess={breakdown['max_joint_excess_rad']:.6g}rad), "
        f"speed_hard={breakdown['speed']:.6g}, "
        f"{collision_label}={breakdown['additional']:.6g}",
        flush=True,
    )


def validate_and_export(cost_function, target_object, trajectory, output_dir: Path):
    """Render the final rollout and save it only if all constraints pass.
    """
    original_render, original_video_path = cost_function.render, cost_function.video_path
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        cost_function.render = True
        cost_function.video_path = output_dir / "optimized_sim_rollout.mp4"
        print("\n--- [Validation] Running final validation rollout with rendering ---", flush=True)
        results = cost_function._run_actual_simulation(target_object, trajectory)
        if hasattr(cost_function, "log_tracking_breakdown"):
            cost_function.log_tracking_breakdown(results)
        arrays = _validate_arrays(trajectory, results)
        _, max_displacement = cost_function._displacement_cost(results["object_displacements"], target_object)
        has_feature_collision = hasattr(cost_function, "feature_collision_enabled")
        collision_diagnostics = None
        if hasattr(cost_function, "validation_collision_diagnostics"):
            collision_diagnostics = cost_function.validation_collision_diagnostics(results)
        additional_violation = collision_diagnostics["violation"] if collision_diagnostics is not None else 0.0
        
        breakdown = cost_function._constraint_violation_breakdown(
            max_displacement,
            arrays["actual_tcp_trajectory"],
            arrays["joint_trajectory"],
            arrays["tcp_trajectory"] if has_feature_collision else None,
            additional_violation,
        )
        _report_constraints(breakdown, results["object_displacements"], target_object, has_feature_collision)
        violation = breakdown["total"]
        tolerance = float(cost_function.feasibility_tolerance)
        if not np.isfinite(violation) or violation > tolerance:
            raise RuntimeError(
                f"The final validation rollout is infeasible: constraint violation={violation:.6g}, "
                f"tolerance={tolerance:.6g}. No optimized trajectory archive was saved."
            )
        payload = dict(
            **arrays,
            constraint_violation=np.asarray(violation, dtype=np.float64),
            feasibility_tolerance=np.asarray(tolerance, dtype=np.float64),
            feasible=np.asarray(True, dtype=np.bool_),
            target_object=np.asarray(target_object),
        )
        if collision_diagnostics is not None:
            collision_prefix = "feature_collision" if has_feature_collision else "robot_collision"
            payload.update({
                f"{collision_prefix}_{name}": np.asarray(value, dtype=np.float64)
                for name, value in collision_diagnostics.items()
            })
        archive_path = output_dir / "optimized_trajectory.npz"
        np.savez_compressed(archive_path, **payload)
        print(f"[Validation] Final rollout render saved to {cost_function.video_path}", flush=True)
        print(f"[Validation] Feasible commanded/measured TCP, object, and joint trajectories saved to {archive_path}", flush=True)
    finally:
        cost_function.render, cost_function.video_path = original_render, original_video_path


def export_feasible_archive(cost_function, output_dir: Path):
    archive_path = output_dir / "feasible_trajectories.npz"
    raw_count, saved_count = cost_function.save_feasible_archive(archive_path)
    print(
        colored(
            f"[Feasible Archive] Total saved: {saved_count} unique feasible trajectories "
            f"({raw_count} feasible before near-duplicate removal) from "
            f"{cost_function.feasible_archive_population_size} final-population candidates to {archive_path}",
            "green" if saved_count else "yellow",
        ),
        flush=True,
    )
