"""Simulation rollouts and cached grasp initialization shared by both costs."""

import logging

import numpy as np

from .grasp import GraspPreparation, resolve_transforms

logger = logging.getLogger(__name__)


def _object_displacements(grasp_poses, pos_history, quat_history, env_index=0):
    displacements = {}
    for name, pose in grasp_poses.items():
        ref_pos = pose["position"]
        ref_quat = pose["orientation"]
        if isinstance(ref_pos[0], list):
            ref_pos = ref_pos[env_index]
        if isinstance(ref_quat[0], list):
            ref_quat = ref_quat[env_index]
        positions = np.array(pos_history[name])
        quaternions = np.array(quat_history[name])
        position_diffs = np.linalg.norm(positions - np.array(ref_pos), axis=1)
        quaternion_norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
        quaternion_norms = np.where(quaternion_norms > 1e-6, quaternion_norms, 1.0)
        normalized_quaternions = quaternions / quaternion_norms
        reference = np.array(ref_quat)
        reference_norm = np.linalg.norm(reference)
        reference /= reference_norm if reference_norm > 1e-6 else 1.0
        dots = np.abs(np.sum(normalized_quaternions * reference, axis=1))
        rotation_diffs = 2.0 * np.arccos(np.clip(dots, 0.0, 1.0))
        displacements[name] = {
            "mean_pos": float(np.mean(position_diffs)),
            "peak_pos": float(np.max(position_diffs)),
            "final_pos": float(position_diffs[-1]),
            "peak_rot": float(np.max(rotation_diffs)),
        }
    return displacements


def _append_release_poses(grasp_poses, release_poses, pos_history, quat_history, env_index=0):
    # Release affects displacement statistics, but not the tracking endpoint.
    for name in grasp_poses:
        pose = release_poses[name]
        position = pose["position"]
        orientation = pose["orientation"]
        if isinstance(position[0], list):
            position = position[env_index]
        if isinstance(orientation[0], list):
            orientation = orientation[env_index]
        pos_history[name].append(position)
        quat_history[name].append(orientation)


def _trajectory_result(actual, tcp, joints, displacements):
    return {
        "actual_trajectory": np.array(actual),
        "actual_tcp_trajectory": np.array(tcp),
        "joint_trajectory": np.array(joints),
        "object_displacements": displacements,
    }


def _object_trajectories(pos_history, quat_history, num_steps):
    return {
        name: np.hstack((
            np.asarray(pos_history[name][:num_steps], dtype=np.float64),
            np.asarray(quat_history[name][:num_steps], dtype=np.float64),
        ))
        for name in pos_history
    }


def _environment_history(history, env_index):
    return {name: values[env_index] for name, values in history.items()}


class SimulationRollout(GraspPreparation):
    def _run_actual_simulation(self, target_object: str, candidate_trajectory: np.ndarray) -> dict:
        """
        Runs the candidate trajectory in the actual simulation.
        Connects to the SimServer, rolls out the trajectory, and returns the actual
        trajectory of the target object (position + orientation) and final object displacements.
        """
        from caster.sim_env.sim_client import SimClient

        if not self.demo_path:
            raise ValueError("demo_path must be set to run actual simulation.")

        transforms_path = resolve_transforms(self.demo_path)

        video_path = self.video_path

        client = SimClient(self.server_address, self.server_timeout)
        actual_trajectory = []
        actual_tcp_trajectory = []
        joint_trajectory = []
        contact_query_fn = getattr(self, "_simulation_contact_query", None)
        contact_query = (
            contact_query_fn(target_object)
            if contact_query_fn is not None
            else None
        )
        max_forbidden_contact_force = 0.0

        try:
            grasp_poses = self._prepare_rollout(client, target_object, transforms_path, video_path)

            # Initialize pose histories to compute detailed displacements
            pos_history = {obj_name: [] for obj_name in grasp_poses.keys()}
            quat_history = {obj_name: [] for obj_name in grasp_poses.keys()}

            # Play back the candidate trajectory.
            for waypoint_index, way_point in enumerate(candidate_trajectory):
                action = way_point.tolist()
                step_contact_query = (
                    dict(
                        contact_query,
                        return_result=(
                            waypoint_index == len(candidate_trajectory) - 1
                        ),
                    )
                    if contact_query is not None
                    else None
                )
                info = client.step(
                    action,
                    record=self.render,
                    return_positions=True,
                    contact_query=step_contact_query,
                )
                if (
                    contact_query is not None
                    and waypoint_index == len(candidate_trajectory) - 1
                ):
                    force = info.get("max_forbidden_contact_force")
                    if force is None:
                        raise RuntimeError(
                            "Simulation server did not return forbidden-contact data."
                        )
                    if isinstance(force, list):
                        force = force[0]
                    max_forbidden_contact_force = max(
                        max_forbidden_contact_force,
                        float(force),
                    )

                poses = info["object_poses"]
                target_pose = poses[target_object]
                target_pos = target_pose["position"]
                target_quat = target_pose["orientation"]
                actual_tcp = info["actual_tcp"]
                joint_positions = info.get("joint_positions")
                if joint_positions is None:
                    raise RuntimeError(
                        "The simulator step response must include "
                        "'joint_positions' for the Panda joint-limit cost."
                    )

                if isinstance(target_pos[0], list):
                    actual_trajectory.append(target_pos[0] + target_quat[0])
                else:
                    actual_trajectory.append(target_pos + target_quat)

                if isinstance(actual_tcp[0], list):
                    actual_tcp_trajectory.append(actual_tcp[0])
                else:
                    actual_tcp_trajectory.append(actual_tcp)

                if isinstance(joint_positions[0], list):
                    joint_trajectory.append(joint_positions[0][:7])
                else:
                    joint_trajectory.append(joint_positions[:7])

                # Collect history of poses for all objects
                for obj_name in grasp_poses:
                    current_pose = poses[obj_name]
                    current_pos = current_pose["position"]
                    current_quat = current_pose["orientation"]
                    if isinstance(current_pos[0], list):
                        current_pos = current_pos[0]
                    if isinstance(current_quat[0], list):
                        current_quat = current_quat[0]
                    pos_history[obj_name].append(current_pos)
                    quat_history[obj_name].append(current_quat)

            # Release the target object.
            if self.execute_release:
                client.execute_release()
            if self.evaluate_release:
                release_info = client.step(
                    None,
                    record=self.render,
                    return_positions=True,
                )
                release_poses = release_info["object_poses"]

                _append_release_poses(grasp_poses, release_poses, pos_history, quat_history)

            # Save the recorded rollout.
            if self.render:
                client.save_video()
                print(f"[Validation] Rollout video saved to {video_path}", flush=True)

            object_displacements = _object_displacements(grasp_poses, pos_history, quat_history)

        except Exception:
            logger.exception("Error during actual simulation run")
            raise
        finally:
            client.disconnect()

        result = _trajectory_result(
            actual_trajectory, actual_tcp_trajectory, joint_trajectory,
            object_displacements,
        )
        if contact_query is not None:
            result["max_forbidden_contact_force"] = max_forbidden_contact_force
        if self._collect_object_trajectories:
            result["object_trajectories"] = _object_trajectories(
                pos_history, quat_history, len(actual_trajectory)
            )
        return result

    def _run_actual_simulation_batched(self, target_object: str, candidate_trajectories: np.ndarray) -> list[dict]:
        """
        Runs a batch of candidate trajectories in parallel in simulation,
        automatically chunking if the simulator has fewer environments than requested.
        """
        from caster.sim_env.sim_client import SimClient

        if not self.demo_path:
            raise ValueError("demo_path must be set to run actual simulation.")

        transforms_path = resolve_transforms(self.demo_path)

        video_path = self.video_path

        total_trajectories, num_steps, _ = candidate_trajectories.shape
        collect_arm_links = (
            self._term_enabled("obstacle")
            and self.simulated_arm_collision_enabled
        )
        contact_query_fn = getattr(self, "_simulation_contact_query", None)
        contact_query = (
            contact_query_fn(target_object)
            if contact_query_fn is not None
            else None
        )

        # The server environment count is stable for the optimizer lifetime.
        # Detect it once instead of performing an extra reset every generation.
        if self._cached_sim_num_envs is None:
            probe_client = SimClient(self.server_address, self.server_timeout)
            try:
                reset_response = probe_client.reset(
                    str(transforms_path),
                    str(video_path),
                    self.render,
                )
                initial_poses = reset_response["initial_poses"]
                any_obj = next(iter(initial_poses.values()))
                positions = any_obj["position"]
                self._cached_sim_num_envs = (
                    len(positions)
                    if isinstance(positions[0], list)
                    else 1
                )
            finally:
                probe_client.disconnect()

        actual_num_envs = self._cached_sim_num_envs

        total_chunks = (total_trajectories + actual_num_envs - 1) // actual_num_envs
        print(f"[Batched Sim] Detected {actual_num_envs} environment(s) on simulation server.", flush=True)
        print(f"[Batched Sim] Evaluating {total_trajectories} trajectories in {total_chunks} chunk(s)...", flush=True)

        # We will split total_trajectories into chunks of size actual_num_envs
        results = []
        chunk_idx = 0
        client = SimClient(self.server_address, self.server_timeout)
        try:
            for start_idx in range(0, total_trajectories, actual_num_envs):
                chunk_idx += 1
                end_idx = min(start_idx + actual_num_envs, total_trajectories)
                chunk_size = end_idx - start_idx
                print(f"[Batched Sim] Running chunk {chunk_idx}/{total_chunks} (evaluating trajectories {start_idx} to {end_idx - 1})...", flush=True)

                # Extract chunk
                chunk_trajectories = candidate_trajectories[start_idx:end_idx]

                grasp_poses = self._prepare_rollout(client, target_object, transforms_path, video_path)

                actual_trajectories_chunk = [[] for _ in range(chunk_size)]
                actual_tcp_trajectories_chunk = [[] for _ in range(chunk_size)]
                joint_trajectories_chunk = [[] for _ in range(chunk_size)]
                object_displacements_chunk = [{} for _ in range(chunk_size)]
                max_forbidden_contact_force_chunk = np.zeros(
                    chunk_size,
                    dtype=np.float64,
                )
                arm_link_positions_chunk = [
                    {
                        link_name: []
                        for link_name in self.simulated_arm_link_names
                    }
                    for _ in range(chunk_size)
                ]

                # Initialize history dictionaries for this chunk
                pos_history_chunk = {
                    obj_name: [[] for _ in range(chunk_size)]
                    for obj_name in grasp_poses.keys()
                }
                quat_history_chunk = {
                    obj_name: [[] for _ in range(chunk_size)]
                    for obj_name in grasp_poses.keys()
                }

                for step_idx in range(num_steps):
                    action_list = chunk_trajectories[:, step_idx, :].tolist()
                    if chunk_size < actual_num_envs:
                        padding_action = action_list[-1]
                        action_list += [padding_action] * (actual_num_envs - chunk_size)

                    step_contact_query = (
                        dict(
                            contact_query,
                            return_result=(step_idx == num_steps - 1),
                        )
                        if contact_query is not None
                        else None
                    )
                    info = client.step(
                        action_list,
                        record=self.render,
                        return_positions=True,
                        return_robot_links=(
                            self.simulated_arm_link_names
                            if collect_arm_links
                            else None
                        ),
                        contact_query=step_contact_query,
                    )
                    if contact_query is not None and step_idx == num_steps - 1:
                        forces = info.get("max_forbidden_contact_force")
                        if forces is None:
                            raise RuntimeError(
                                "Simulation server did not return forbidden-contact data."
                            )
                        forces = np.asarray(forces, dtype=np.float64).reshape(-1)
                        if len(forces) < chunk_size:
                            raise RuntimeError(
                                "Simulation server returned too few contact-force values."
                            )
                        max_forbidden_contact_force_chunk = np.maximum(
                            max_forbidden_contact_force_chunk,
                            forces[:chunk_size],
                        )

                    poses = info["object_poses"]
                    target_pose = poses[target_object]
                    target_pos = target_pose["position"]
                    target_quat = target_pose["orientation"]
                    actual_tcp = info["actual_tcp"]
                    joint_positions = info.get("joint_positions")
                    if joint_positions is None:
                        raise RuntimeError(
                            "The simulator step response must include "
                            "'joint_positions' for the Panda joint-limit cost."
                        )
                    robot_link_positions = info["robot_link_positions"] if collect_arm_links else None

                    for env_idx in range(chunk_size):
                        if isinstance(target_pos[0], list):
                            actual_trajectories_chunk[env_idx].append(target_pos[env_idx] + target_quat[env_idx])
                        else:
                            actual_trajectories_chunk[env_idx].append(target_pos + target_quat)

                        if isinstance(actual_tcp[0], list):
                            actual_tcp_trajectories_chunk[env_idx].append(actual_tcp[env_idx])
                        else:
                            actual_tcp_trajectories_chunk[env_idx].append(actual_tcp)

                        if isinstance(joint_positions[0], list):
                            joint_trajectories_chunk[env_idx].append(
                                joint_positions[env_idx][:7]
                            )
                        else:
                            joint_trajectories_chunk[env_idx].append(
                                joint_positions[:7]
                            )

                        if collect_arm_links:
                            for link_name in self.simulated_arm_link_names:
                                link_positions = robot_link_positions[link_name]
                                if isinstance(link_positions[0], list):
                                    link_position = link_positions[env_idx]
                                else:
                                    link_position = link_positions
                                arm_link_positions_chunk[env_idx][
                                    link_name
                                ].append(link_position)

                        # Accumulate irrelevant object poses at each waypoint step
                        for obj_name in grasp_poses:
                            current_pose = poses[obj_name]
                            current_pos = current_pose["position"]
                            current_quat = current_pose["orientation"]
                            if isinstance(current_pos[0], list):
                                current_pos_env = current_pos[env_idx]
                            else:
                                current_pos_env = current_pos
                            if isinstance(current_quat[0], list):
                                current_quat_env = current_quat[env_idx]
                            else:
                                current_quat_env = current_quat

                            pos_history_chunk[obj_name][env_idx].append(current_pos_env)
                            quat_history_chunk[obj_name][env_idx].append(current_quat_env)

                if self.execute_release:
                    client.execute_release()
                if self.evaluate_release:
                    release_info = client.step(
                        None,
                        record=self.render,
                        return_positions=True,
                    )
                    release_poses = release_info["object_poses"]

                    for env_idx in range(chunk_size):
                        _append_release_poses(
                            grasp_poses, release_poses,
                            _environment_history(pos_history_chunk, env_idx),
                            _environment_history(quat_history_chunk, env_idx),
                            env_idx,
                        )

                for env_idx in range(chunk_size):
                    object_displacements_chunk[env_idx] = _object_displacements(
                        grasp_poses,
                        _environment_history(pos_history_chunk, env_idx),
                        _environment_history(quat_history_chunk, env_idx),
                        env_idx,
                    )

                for env_idx in range(chunk_size):
                    if collect_arm_links:
                        arm_link_positions = np.stack(
                            [
                                np.asarray(
                                    arm_link_positions_chunk[env_idx][link_name],
                                    dtype=np.float64,
                                )
                                for link_name in self.simulated_arm_link_names
                            ],
                            axis=1,
                        )
                    else:
                        arm_link_positions = None
                    result = _trajectory_result(
                        actual_trajectories_chunk[env_idx], actual_tcp_trajectories_chunk[env_idx],
                        joint_trajectories_chunk[env_idx], object_displacements_chunk[env_idx],
                    )
                    result["arm_link_positions"] = arm_link_positions
                    if contact_query is not None:
                        result["max_forbidden_contact_force"] = float(
                            max_forbidden_contact_force_chunk[env_idx]
                        )
                    if self._collect_object_trajectories:
                        result["object_trajectories"] = _object_trajectories(
                            _environment_history(pos_history_chunk, env_idx),
                            _environment_history(quat_history_chunk, env_idx),
                            num_steps,
                        )
                    results.append(result)
        finally:
            client.disconnect()
        return results
