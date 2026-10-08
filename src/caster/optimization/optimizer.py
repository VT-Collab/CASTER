import logging
import time
from pathlib import Path

import numpy as np
from termcolor import colored
from typing_extensions import Literal

from .costs.trajectory import TrajectoryCostFunction
from .spline import TrajectorySpline
from .validation import export_feasible_archive, validate_and_export

logger = logging.getLogger(__name__)


def _bounds_around(values: np.ndarray, radius: float) -> list[tuple[float, float]]:
    """Create symmetric box constraints around a flat parameter vector."""
    return [(value - radius, value + radius) for value in values]


def run_cem_refinement(
    cost_func,
    target_object,
    desired_object_trajectory,
    x_start,
    reconstruct_fn,
    bounds,
    pop_size=128,
    num_generations=15,
    elite_fraction=0.15,
    init_sigma: float | np.ndarray = 0.03,
    checkpoint_policy: Literal["lowest_cost", "feasible_first"] = "lowest_cost",
):
    """Refine spline parameters with the cross-entropy method and simulator costs."""
    if checkpoint_policy not in {"lowest_cost", "feasible_first"}:
        raise ValueError(
            "checkpoint_policy must be 'lowest_cost' or 'feasible_first'."
        )

    num_params = len(x_start)
    mu = x_start.copy()
    try:
        sigma = np.broadcast_to(
            np.asarray(init_sigma, dtype=np.float64),
            (num_params,),
        ).copy()
    except ValueError as error:
        raise ValueError(
            "init_sigma must be a scalar or contain one value "
            "per optimization parameter."
        ) from error
    if np.any(sigma <= 0.0):
        raise ValueError("Every initial sigma must be positive.")

    best_cost = float('inf')
    best_x = x_start.copy()

    cost_func.reset_checkpoint()
    lower, upper = np.asarray(bounds).T

    for gen in range(num_generations):
        gen_start_time = time.time()
        samples = np.random.normal(mu, sigma, size=(pop_size, num_params))

        samples[-1] = best_x

        samples = np.clip(samples, lower, upper)

        candidate_trajectories = np.array([reconstruct_fn(sample) for sample in samples])

        # Export feasible candidates from the final population, including non-elites.
        cost_func.reset_feasible_archive(
            candidate_trajectories.shape[1],
            population_size=len(candidate_trajectories),
        )

        print(f"\n--- [CEM Gen {gen+1}/{num_generations}] Evaluating population of size {pop_size} ---", flush=True)
        costs = cost_func.evaluate_batch(
            candidate_trajectories,
            target_object,
            desired_object_trajectory,
            candidate_spline_params=samples,
        )

        sort_indices = np.argsort(costs)
        num_elites = max(1, int(pop_size * elite_fraction))
        elites = samples[sort_indices[:num_elites]]
        elite_costs = costs[sort_indices[:num_elites]]

        current_best_cost = elite_costs[0]
        if best_cost == float('inf'):
            relative_improvement = 1.0
        else:
            relative_improvement = (best_cost - current_best_cost) / max(abs(best_cost), 1.0)

        if relative_improvement > 0.10:
            best_cost = current_best_cost
            best_x = elites[0].copy()
            print(f"[CEM] New generation best cost: {best_cost:.4f} (relative improvement: {relative_improvement:.2%})", flush=True)

        else:
            if current_best_cost < best_cost:
                best_cost = current_best_cost
                best_x = elites[0].copy()

        # Update distribution parameters
        mu = np.mean(elites, axis=0)
        sigma = np.std(elites, axis=0) + 1e-4  # regularizer to prevent variance collapse

        print(f"[CEM Gen {gen+1}/{num_generations}] Time taken: {time.time() - gen_start_time:.4f}s", flush=True)

        # Convergence checks
        if np.max(sigma) < 1e-3:
            print("[CEM] Stop: variance collapsed.", flush=True)
            break

    if checkpoint_policy == "lowest_cost":
        print(
            f"[CEM] Returning lowest-cost candidate with cost "
            f"{best_cost:.4f}.",
            flush=True,
        )
        return best_x

    recorded_feasible = cost_func.lowest_cost_recorded_feasible_tcp()
    if recorded_feasible is not None:
        selected_tcp, selected_cost, selected_violation = recorded_feasible
        matching_candidates = np.flatnonzero(
            np.all(
                np.isclose(
                    candidate_trajectories,
                    selected_tcp[None, :, :],
                    rtol=0.0,
                    atol=1e-12,
                ),
                axis=(1, 2),
            )
        )
        if matching_candidates.size:
            selected_x = samples[int(matching_candidates[0])].copy()
            print(
                "[CEM] Returning the lowest-cost candidate from the "
                "final population's feasible archive with cost "
                f"{selected_cost:.4f} and violation "
                f"{selected_violation:.6g}.",
                flush=True,
            )
            return selected_x

        print(
            "[CEM] Warning: the lowest-cost recorded feasible TCP could "
            "not be matched to the final population. Falling back to the "
            "previous checkpoint-selection behavior.",
            flush=True,
        )
    else:
        print(
            "[CEM] Warning: the final population's feasible archive is empty. "
            "Falling back to the previous checkpoint-selection behavior.",
            flush=True,
        )

    if cost_func.best_feasible_x is not None:
        print(
            "[CEM] Returning the best feasible checkpoint from all "
            f"generations with cost {cost_func.best_feasible_cost:.4f}.",
            flush=True,
        )
        return cost_func.best_feasible_x

    if cost_func.best_violation_x is not None:
        print(
            colored(
                "[CEM] NO FEASIBLE CANDIDATE EXISTS. Returning an "
                "INFEASIBLE least-violation fallback "
                f"(violation={cost_func.best_violation:.6g}, "
                f"cost={cost_func.best_violation_cost:.4f}).",
                "red",
                attrs=["bold"],
            ),
            flush=True,
        )
        return cost_func.best_violation_x

    return best_x


class TrajectoryOptimizer:
    """
    Jointly optimize position and orientation spline control points.

    Fit the initial spline and refine its position/orientation parameters
    with a joint CEM distribution and simulator rollouts.
    """

    def __init__(
        self,
        cost_function: TrajectoryCostFunction,
        checkpoint_policy: Literal["lowest_cost", "feasible_first"] = "lowest_cost",
        K_pos: int = 15,
        K_rot: int = 5,
    ):
        self.cost_function = cost_function
        if checkpoint_policy not in {"lowest_cost", "feasible_first"}:
            raise ValueError(
                "checkpoint_policy must be 'lowest_cost' or "
                "'feasible_first'."
            )
        self.checkpoint_policy = checkpoint_policy
        self.K_pos = int(K_pos)
        self.K_rot = int(K_rot)
        if self.K_pos < 2:
            raise ValueError("K_pos must be at least 2.")
        if self.K_rot < 2:
            raise ValueError("K_rot must be at least 2.")

    def optimize(
        self,
        target_object: str,
        desired_trajectory: np.ndarray | list,
        desired_object_trajectory: np.ndarray | list,
        demo_path: str | Path,
    ) -> np.ndarray:
        """
        Fit a spline, refine it with simulator rollouts, and export validated results.
        """
        desired_trajectory = np.asarray(desired_trajectory, dtype=np.float64)
        desired_object_trajectory = np.asarray(desired_object_trajectory, dtype=np.float64)

        if desired_trajectory.ndim != 2 or desired_trajectory.shape[1] != 7:
            raise ValueError(f"Spline optimization requires shape (N, 7) for desired_trajectory, got {desired_trajectory.shape}")

        self.cost_function.demo_path = Path(demo_path)

        N = len(desired_trajectory)
        gripper_pos = desired_trajectory[0, :3]
        gripper_quat = desired_trajectory[0, 3:7]

        K_pos = self.K_pos
        K_rot = self.K_rot
        if K_pos > N:
            raise ValueError(
                f"K_pos ({K_pos}) cannot exceed trajectory length ({N})."
            )
        if K_rot > N:
            raise ValueError(
                f"K_rot ({K_rot}) cannot exceed trajectory length ({N})."
            )

        spline = TrajectorySpline(N, K_pos, K_rot, gripper_quat)

        # Start untracked modalities at the post-grasp pose and leave them free to search.
        position_only_features = bool(
            getattr(self.cost_function, "has_position_only_features", False)
        )
        orientation_only_features = bool(
            getattr(self.cost_function, "has_orientation_only_features", False)
        )
        initialization_trajectory = desired_trajectory.copy()
        if orientation_only_features:
            initialization_trajectory[:, :3] = gripper_pos
        if position_only_features:
            initialization_trajectory[:, 3:7] = gripper_quat
        if position_only_features or orientation_only_features:
            omitted = []
            if orientation_only_features:
                omitted.append("position")
            if position_only_features:
                omitted.append("orientation")
            logger.info(
                "FeatureCostFunction: initializing free %s from the "
                "post-grasp pose, without demo-modality trust.",
                " and ".join(omitted),
            )

        init_pos_knots, init_rot_knots = spline.fit(initialization_trajectory)

        # Define flat initial parameters for optimization (excluding the first fixed knots)
        x0_pos = (init_pos_knots[1:] - gripper_pos).flatten()
        x0_rot = init_rot_knots[1:].flatten()

        feature_policy = self.cost_function.optimizer_policy
        logger.info("Optimizer policy: %s", feature_policy)
        # Search relative to the post-grasp pose, with the fitted demo inside the bounds.
        pos_radius = max(float(feature_policy["position_search_radius"]), float(np.max(np.abs(x0_pos))))
        rot_radius = max(float(feature_policy["rotation_search_radius"]), float(np.max(np.abs(x0_rot))))
        pos_bounds = _bounds_around(np.zeros_like(x0_pos), radius=pos_radius)
        rot_bounds = _bounds_around(np.zeros_like(x0_rot), radius=rot_radius)
        logger.info("Search bounds: +/- %.3fm position, +/- %.3frad rotation.", pos_radius, rot_radius)

        num_position_parameters = x0_pos.size
        x0_joint = np.concatenate([x0_pos, x0_rot])
        joint_bounds = pos_bounds + rot_bounds

        joint_init_sigma = np.concatenate([
            np.full(
                x0_pos.shape,
                float(feature_policy["position_init_sigma"]),
                dtype=np.float64,
            ),
            np.full(
                x0_rot.shape,
                float(feature_policy["rotation_init_sigma"]),
                dtype=np.float64,
            ),
        ])

        logger.info(
            "Starting joint position-orientation trajectory optimization "
            "with %d position and %d orientation parameters.",
            len(x0_pos),
            len(x0_rot),
        )

        def reconstruct_joint(x_joint):
            x_joint = np.asarray(x_joint, dtype=np.float64)
            x_pos = x_joint[:num_position_parameters]
            x_rot = x_joint[num_position_parameters:]

            position_knots = np.zeros((K_pos, 3))
            position_knots[0] = gripper_pos
            position_knots[1:] = gripper_pos + x_pos.reshape(K_pos - 1, 3)

            rotation_knots = np.zeros((K_rot, 3))
            rotation_knots[1:] = x_rot.reshape(K_rot - 1, 3)

            return spline.reconstruct(position_knots, rotation_knots)

        print(
            "\n================== JOINT POSITION-ORIENTATION "
            "OPTIMIZATION ==================",
            flush=True,
        )
        print("\n--- Running joint CEM simulator refinement ---", flush=True)
        x_joint_opt = run_cem_refinement(
            cost_func=self.cost_function,
            target_object=target_object,
            desired_object_trajectory=desired_object_trajectory,
            x_start=x0_joint,
            reconstruct_fn=reconstruct_joint,
            bounds=joint_bounds,
            pop_size=1024,
            num_generations=10,
            elite_fraction=0.10,
            init_sigma=joint_init_sigma,
            checkpoint_policy=self.checkpoint_policy,
        )

        optimized_trajectory = reconstruct_joint(x_joint_opt)

        output_dir = Path(self.cost_function.optimization_output_dir)
        try:
            validate_and_export(self.cost_function, target_object, optimized_trajectory, output_dir)
        finally:
            export_feasible_archive(self.cost_function, output_dir)

        return optimized_trajectory
