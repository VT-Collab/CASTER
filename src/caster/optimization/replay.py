"""Convert tracked object motion to TCP poses and replay it in simulation."""

import json
import logging
import re
from pathlib import Path

import hydra
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation, Slerp


from caster.sim_env.sim_client import SimClient


logger = logging.getLogger(__name__)


def _proper_rotation(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Rotation must be a finite 3x3 matrix")
    u, _, vt = np.linalg.svd(matrix)
    result = u @ vt
    if np.linalg.det(result) < 0.0:
        u[:, -1] *= -1.0
        result = u @ vt
    return result


def load_object_trajectory(path):
    """Load all frames, interpolating tracker-invalid positions/orientations."""
    with path.open("r", encoding="utf-8") as stream:
        frames = json.load(stream)
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{path} must contain a non-empty list")

    count = len(frames)
    positions = np.full((count, 3), np.nan, dtype=np.float64)
    rotation_indices = []
    rotations = []
    for index, frame in enumerate(frames):
        center = np.asarray(frame["center_robot"], dtype=np.float64)
        if frame["valid"] and center.shape == (3,) and np.isfinite(center).all():
            positions[index] = center
        matrix = frame["rotation_robot"]
        if frame["rotation_valid"]:
            try:
                rotations.append(_proper_rotation(matrix))
                rotation_indices.append(index)
            except (ValueError, np.linalg.LinAlgError):
                pass

    valid_position = np.flatnonzero(np.isfinite(positions).all(axis=1))
    if not len(valid_position):
        raise ValueError(f"{path} has no valid robot-frame positions")
    timeline = np.arange(count)
    for axis in range(3):
        positions[:, axis] = np.interp(
            timeline, valid_position, positions[valid_position, axis]
        )
    if not rotation_indices:
        raise ValueError(f"{path} has no valid robot-frame orientations")
    if len(rotation_indices) == 1:
        rotation_trajectory = np.repeat(rotations, count, axis=0)
    else:
        indices = np.asarray(rotation_indices)
        clamped_timeline = np.clip(timeline, indices[0], indices[-1])
        rotation_trajectory = Slerp(
            indices, Rotation.from_matrix(np.stack(rotations))
        )(clamped_timeline).as_matrix()

    result = np.repeat(np.eye(4, dtype=np.float64)[None], count, axis=0)
    result[:, :3, :3] = rotation_trajectory
    result[:, :3, 3] = positions
    return result


def object_to_tcp_trajectory(object_transforms, grasp_tcp_transform):
    objects = np.asarray(object_transforms, dtype=np.float64)
    grasp_tcp = np.asarray(grasp_tcp_transform, dtype=np.float64)
    if objects.ndim != 3 or objects.shape[1:] != (4, 4) or len(objects) < 2:
        raise ValueError(f"Expected object transforms shaped (T, 4, 4), got {objects.shape}")
    if grasp_tcp.shape != (4, 4):
        raise ValueError(f"Expected grasp TCP transform shaped (4, 4), got {grasp_tcp.shape}")
    object_to_tcp = np.linalg.inv(objects[0]) @ grasp_tcp
    tcp_transforms = objects @ object_to_tcp
    quaternions_xyzw = Rotation.from_matrix(tcp_transforms[:, :3, :3]).as_quat()
    # Keep WXYZ at the file/API boundary and remove harmless quaternion sign flips.
    for index in range(1, len(quaternions_xyzw)):
        if np.dot(quaternions_xyzw[index - 1], quaternions_xyzw[index]) < 0.0:
            quaternions_xyzw[index] *= -1.0
    tcp = np.empty((len(objects), 7), dtype=np.float64)
    tcp[:, :3] = tcp_transforms[:, :3, 3]
    tcp[:, 3] = quaternions_xyzw[:, 3]
    tcp[:, 4:] = quaternions_xyzw[:, :3]
    return tcp, object_to_tcp


def _first_environment(values):
    values = np.asarray(values, dtype=float)
    if values.ndim == 2 and len(values):
        values = values[0]
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError(f"Invalid simulation response shape: {values.shape}")
    return values


def pose_values(pose):
    position = _first_environment(pose["position"])
    orientation = _first_environment(pose["orientation"])
    if position.shape != (3,) or orientation.shape != (4,):
        raise ValueError("Simulation poses must contain XYZ positions and WXYZ orientations")
    values = np.concatenate((position, orientation))
    if np.linalg.norm(orientation) == 0:
        raise ValueError(f"Invalid simulation pose: {pose}")
    return values


def pose_transform(pose):
    values = pose_values(pose)
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat(values[[4, 5, 6, 3]]).as_matrix()
    transform[:3, 3] = values[:3]
    return transform


def replay_trajectory(client, objects, target, output_dir, grasp, substeps, render,
                      source_trajectory, post_grasp_settle_steps):
    grasp_info = client.execute_grasp(
        target, hover_distance=grasp.hover_distance,
        grasp_depth_offset=grasp.depth_offset,
        grasp_height_offset=grasp.height_offset,
    )
    grasp_tcp = pose_transform(grasp_info["gripper_pose"])
    tcp, attachment = object_to_tcp_trajectory(objects, grasp_tcp)
    np.savez_compressed(
        output_dir / "grasp_trajectory.npz",
        joint_trajectory=np.asarray(grasp_info["joint_trajectory"], dtype=float),
        source_hz=np.asarray(grasp_info["source_hz"], dtype=float),
        grasp_tcp_transform=grasp_tcp,
        target_object=np.asarray(target),
    )
    # Retain the closed gripper and final arm target while the grasp settles.
    for _ in range(post_grasp_settle_steps):
        client.step(None, record=False, return_positions=False)
    actual_tcp, object_poses, joints = [], [], []
    for frame, action in enumerate(tcp):
        for substep in range(substeps):
            last = substep == substeps - 1
            info = client.step(action.tolist(), record=render and last,
                               return_positions=last)
        measured_tcp = _first_environment(
            info["actual_tcp_pose"],
        )
        if measured_tcp.shape != (7,):
            raise ValueError(f"Invalid measured TCP: {measured_tcp}")
        actual_tcp.append(measured_tcp)
        object_poses.append(pose_values(info["object_poses"][target]))
        joint_positions = _first_environment(info["joint_positions"])
        if joint_positions.shape != (7,):
            raise ValueError("Simulation must return seven Panda arm joints")
        joints.append(joint_positions)
        error = np.linalg.norm(action[:3] - measured_tcp[:3])
        logger.info("Frame %d/%d: TCP error %.6f m", frame + 1, len(tcp), error)
    if grasp.execute_release:
        client.execute_release()
    client.save_video()
    measured_objects = np.asarray(object_poses)
    np.savez_compressed(
        output_dir / "trajectory.npz",
        tcp_trajectory=tcp,
        actual_tcp_trajectory=np.asarray(actual_tcp),
        object_trajectory=measured_objects,
        joint_trajectory=np.asarray(joints),
        object_to_tcp_transform=attachment,
        target_object=np.asarray(target),
        source_trajectory=np.asarray(str(source_trajectory)),
    )
    rotations = Rotation.from_quat(measured_objects[:, [4, 5, 6, 3]]).as_matrix()
    frames = [dict(frame=index, target_object=target, center_robot=pose[:3].tolist(),
                   rotation_robot=rotation.tolist())
              for index, (pose, rotation) in enumerate(zip(measured_objects, rotations))]
    (output_dir / "trajectory.json").write_text(json.dumps(frames, indent=2) + "\n")


def _demo_inputs(scene: Path, task: str, cf_index: int, demo_index: int):
    folder = scene / f"cf_{cf_index}_{task}"
    features = json.loads((folder / "task_feature.json").read_text())
    if not isinstance(features, list) or not features:
        raise ValueError("task_feature.json must contain a non-empty constraint list")
    targets = {item["target_object"] for item in features}
    if len(targets) != 1:
        raise ValueError(f"Simulation requires one moving target, found {sorted(targets)}")
    target = targets.pop()
    demos = sorted(path for path in folder.glob(f"demo_{demo_index}_seed_*") if path.is_dir())
    if len(demos) != 1:
        raise ValueError(f"Expected one demo with index {demo_index} in {folder}, found {demos}")
    demo = demos[0]
    names = (target, target.replace(" ", "_"), re.sub(r"[^a-zA-Z0-9_-]", "_", target))
    for name in names:
        trajectory = demo / name / "trajectory.json"
        if trajectory.is_file():
            return demo, target, trajectory
    raise FileNotFoundError(f"No trajectory for {target!r} in {demo}")


def replay(config: DictConfig) -> list[Path]:
    scene = Path(to_absolute_path(config.scene_dir))
    indices = list(config.demo_indices)
    if not indices or any(type(index) is not int or index < 1 for index in indices):
        raise ValueError("demo_indices must contain positive integer indices")
    if len(set(indices)) != len(indices):
        raise ValueError("demo_indices must not contain duplicates")
    if config.output_path and len(indices) > 1:
        raise ValueError("output_path is only supported when replaying one demonstration")
    demos = [_demo_inputs(scene, config.task_name, config.cf_index, index) for index in indices]
    transforms = scene / "assets/meshes/transforms.json"
    if not transforms.is_file():
        raise FileNotFoundError(f"Reconstruct the scene before simulation: {transforms}")
    if config.substeps < 1:
        raise ValueError("substeps must be positive")
    if config.post_grasp_settle_steps < 0:
        raise ValueError("post_grasp_settle_steps must not be negative")
    outputs = []
    with SimClient(config.server_address, config.server_timeout) as client:
        for demo, target, trajectory in demos:
            objects = load_object_trajectory(trajectory)
            if len(objects) < 2:
                raise ValueError(f"Replay requires at least two frames: {trajectory}")
            output_dir = demo.parent / "optimization/replay" / demo.name
            output_dir.mkdir(parents=True, exist_ok=True)
            video = Path(to_absolute_path(config.output_path)) if config.output_path else output_dir / "sim_render.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            client.reset(transforms, video, config.render)
            replay_trajectory(client, objects, target, output_dir, config.grasp,
                              config.substeps, config.render, trajectory,
                              config.post_grasp_settle_steps)
            logger.info("Replay saved: %s", output_dir)
            outputs.append(output_dir)
    return outputs


@hydra.main(version_base=None, config_path="../config", config_name="optimization/replay")
def main(config: DictConfig):
    replay(config)


if __name__ == "__main__":
    main()
