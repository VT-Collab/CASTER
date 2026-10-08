"""Optimize selected demonstrations with trajectory or task-feature costs."""

import json
import re
from pathlib import Path

import hydra
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf

from caster.optimization.replay import _demo_inputs

_RENDERED_GRASP_VIDEOS: set[Path] = set()


def _cost_function_output_name(cost_function):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", cost_function.__class__.__name__).lower()


def _optimize_demo(config, demo_path, target_object, trajectory_path):
    if config.cost_function not in {"feature", "trajectory"}:
        raise ValueError("cost_function must be 'feature' or 'trajectory'")
    settings = OmegaConf.create(OmegaConf.to_container(config.optimizer, resolve=True))
    cost_path = Path(__file__).resolve().parents[1] / "config/optimization/cost_function.yaml"
    settings.cost_function = OmegaConf.merge(OmegaConf.load(cost_path), settings.cost_function)
    class_name = {"feature": "FeatureCostFunction", "trajectory": "TrajectoryCostFunction"}[config.cost_function]
    settings.cost_function._target_ = f"caster.optimization.costs.{config.cost_function}.{class_name}"
    collision = settings.cost_function.pop("collision")
    collision_prefix = "feature" if config.cost_function == "feature" else "robot"
    for name, value in collision.items():
        settings.cost_function[f"{collision_prefix}_collision_{name}"] = value
    optimizer = instantiate(settings)
    feature_dir = demo_path.parent
    with trajectory_path.open() as file:
        trajectory_data = json.load(file)
    optimizer.cost_function.demo_path = demo_path

    if hasattr(optimizer.cost_function, "set_feature_directory"):
        optimizer.cost_function.set_feature_directory(feature_dir)
        print(f"Using task features from: {feature_dir}")

    cost_function_name = _cost_function_output_name(
        optimizer.cost_function
    )
    output_dir = (
        feature_dir
        / "optimization"
        / cost_function_name
        / demo_path.name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer.cost_function.optimization_output_dir = output_dir
    optimizer.cost_function.video_path = output_dir / "optimized_sim_rollout.mp4"
    print(f"Saving optimization artifacts to: {output_dir}")

    # Keep one shared grasp video for the complete optimization run, rather
    # than writing the identical scene-level grasp inside every demo folder.
    grasp_output_dir = output_dir.parent
    grasp_video_path = grasp_output_dir / "grasp_sim_rollout.mp4"
    grasp_video_key = grasp_video_path.resolve()
    if grasp_video_key not in _RENDERED_GRASP_VIDEOS:
        optimizer.cost_function.render_grasp_video(
            target_object,
            grasp_video_path,
        )
        _RENDERED_GRASP_VIDEOS.add(grasp_video_key)
    else:
        print(
            "Using existing optimization-level grasp video: "
            f"{grasp_video_path}"
        )


    # Obtain the reset-time target pose and cached post-grasp gripper pose.
    grasp_context = (
        optimizer.cost_function
        .initialize_grasp_context(target_object)
    )

    target_pose = grasp_context["target_pose"]
    gripper_pose = grasp_context["gripper_pose"]

    target_pos = target_pose["position"]
    if isinstance(target_pos[0], list):
        target_pos = target_pos[0]
    target_quat = target_pose["orientation"]
    if isinstance(target_quat[0], list):
        target_quat = target_quat[0]
    gripper_pos = gripper_pose["position"]
    if isinstance(gripper_pos[0], list):
        gripper_pos = gripper_pos[0]
    gripper_quat = gripper_pose["orientation"]
    if isinstance(gripper_quat[0], list):
        gripper_quat = gripper_quat[0]
    print(
        "[Optimizer Setup] Captured object starting pose "
        f"(pre-grasp): {target_pos}, rot: {target_quat}"
    )
    print(
        "[Optimizer Setup] Captured gripper TCP starting pose "
        f"(post-grasp): {gripper_pos}, rot: {gripper_quat}"
    )

    q_init_target_xyzw = [target_quat[1], target_quat[2], target_quat[3], target_quat[0]]
    R_init_target = Rot.from_quat(q_init_target_xyzw).as_matrix()

    q_init_gripper_xyzw = [gripper_quat[1], gripper_quat[2], gripper_quat[3], gripper_quat[0]]
    R_init_gripper = Rot.from_quat(q_init_gripper_xyzw).as_matrix()

    for frame_data in trajectory_data:
        if frame_data["valid"]:
            R_first = np.array(frame_data["rotation_robot"])
            pos_first = np.array(frame_data["center_robot"])
            break
    else:
        raise ValueError(f"No valid robot-frame pose in {trajectory_path}")

    # Load raw trajectory as desired gripper trajectory (for spline fitting/optimization)
    desired_trajectory = []
    # Load raw trajectory as desired object trajectory (for cost tracking)
    desired_object_trajectory = []
    
    for frame_data in trajectory_data:
        pos = np.array(frame_data["center_robot"])
        pos_rel = pos - pos_first
        
        # Gripper position target
        pos_desired_gripper = (pos_rel + np.array(gripper_pos)).tolist()
        # Object position target
        pos_desired_object = (pos_rel + np.array(target_pos)).tolist()
        
        rot_mat = frame_data["rotation_robot"]
        R_t = np.array(rot_mat)
        R_rel = R_t @ R_first.T
        
        # Gripper orientation target
        R_des_gripper = R_rel @ R_init_gripper
        r_grip = Rot.from_matrix(R_des_gripper)
        q_xyzw_grip = r_grip.as_quat()
        q_wxyz_grip = [q_xyzw_grip[3], q_xyzw_grip[0], q_xyzw_grip[1], q_xyzw_grip[2]]
        
        # Object orientation target
        R_des_obj = R_rel @ R_init_target
        r_obj = Rot.from_matrix(R_des_obj)
        q_xyzw_obj = r_obj.as_quat()
        q_wxyz_obj = [q_xyzw_obj[3], q_xyzw_obj[0], q_xyzw_obj[1], q_xyzw_obj[2]]

        desired_trajectory.append(pos_desired_gripper + q_wxyz_grip)
        desired_object_trajectory.append(pos_desired_object + q_wxyz_obj)
        
    desired_trajectory = np.array(desired_trajectory)
    desired_object_trajectory = np.array(desired_object_trajectory)

    # Configure rendering on the cost function (enabled to visualize steps)
    optimizer.cost_function.render = config.render

    # Run the optimization, passing demo_path directly to the optimizer
    optimized_traj = optimizer.optimize(target_object, desired_trajectory, desired_object_trajectory, demo_path=demo_path)

    print("Desired trajectory sample (first 3 steps):")
    print(desired_trajectory[:3])
    print("Optimized trajectory sample (first 3 steps):")
    print(optimized_traj[:3])

    # Save
    output_trajectory_data = []
    for ti in range(len(optimized_traj)):
        pos = optimized_traj[ti, :3].tolist()
        quat_wxyz = optimized_traj[ti, 3:7]
        quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        R = Rot.from_quat(quat_xyzw).as_matrix().tolist()

        output_trajectory_data.append({
            "frame": ti,
            "target_object": target_object,
            "center_robot": pos,
            "rotation_robot": R
        })

    opt_traj_path = output_dir / "optimized_trajectory.json"
    with open(opt_traj_path, "w") as f:
        json.dump(output_trajectory_data, f, indent=4)
    print(f"Saved optimized trajectory to {opt_traj_path}")
    return output_dir


def run_optimization(config: DictConfig) -> list[Path]:
    scene = Path(to_absolute_path(config.scene_dir))
    indices = list(config.demo_indices)
    if not indices or any(type(index) is not int or index < 1 for index in indices):
        raise ValueError("demo_indices must contain positive integer indices")
    if len(set(indices)) != len(indices):
        raise ValueError("demo_indices must not contain duplicates")
    demos = [_demo_inputs(scene, config.task_name, config.cf_index, index) for index in indices]
    transforms = scene / "assets/meshes/transforms.json"
    if not transforms.is_file():
        raise FileNotFoundError(f"Reconstruct the scene before optimization: {transforms}")
    outputs = []
    for demo, target, trajectory in demos:
        outputs.append(_optimize_demo(config, demo, target, trajectory))
    return outputs


@hydra.main(version_base=None, config_path="../config", config_name="optimization/optimization")
def main(config: DictConfig):
    run_optimization(config)


if __name__ == "__main__":
    main()
