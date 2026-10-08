"""Prepare cached grasp states and export grasp videos and joint trajectories."""

from pathlib import Path

import numpy as np


def resolve_transforms(demo_path: Path) -> Path:
    path = demo_path.parent.parent / "assets" / "meshes" / "transforms.json"
    if not path.exists():
        raise FileNotFoundError(f"Could not find transforms.json for {demo_path}")
    return path


class GraspPreparation:
    def initialize_grasp_context(self, target_object: str) -> dict:
        from caster.sim_env.sim_client import SimClient

        if (
            self._grasp_context is not None
            and self._grasp_context_target == target_object
        ):
            return self._grasp_context

        if not self.demo_path:
            raise ValueError(
                "demo_path must be set before initializing the grasp context."
            )

        transforms_path = resolve_transforms(self.demo_path)

        video_path = self.demo_path / "temp_grasp_query.mp4"
        client = SimClient(self.server_address, self.server_timeout)

        try:
            reset_info = client.reset(
                str(transforms_path),
                str(video_path),
                render=False,
            )

            # Returns both object_poses and gripper_pose. If a matching server-side
            # state exists, restore it instead of executing the grasp again.
            grasp_info = client.restore_or_execute_grasp_get_all(
                target_object,
                hover_distance=self.m2t2_hover_distance,
                grasp_depth_offset=self.m2t2_grasp_depth_offset,
                grasp_height_offset=self.m2t2_grasp_height_offset,
            )

            if not grasp_info:
                raise ValueError(
                    "The simulator did not return grasp information."
                )

            object_poses = grasp_info["object_poses"]
            gripper_pose = grasp_info["gripper_pose"]

            if not gripper_pose:
                raise ValueError(
                    "The simulator did not return the post-grasp gripper pose."
                )

            obstacle_poses = {
                name: pose
                for name, pose in object_poses.items()
                if name != target_object
            }
            self.obstacle_cost.set_post_grasp_poses(obstacle_poses)

            target_pose = reset_info["initial_poses"][target_object]

            self._grasp_context = {
                "target_pose": target_pose,
                "gripper_pose": gripper_pose,
                "object_poses": object_poses,
            }
            self._grasp_context_target = target_object

            print(
                "[Cost Setup] Cached grasp context: "
                f"target='{target_object}', "
                f"obstacles={list(obstacle_poses)}",
                flush=True,
            )

            return self._grasp_context
        finally:
            client.disconnect()

    def render_grasp_video(
        self,
        target_object: str,
        video_path: str | Path,
    ) -> Path:
        """Execute and render one configured grasp, caching its post-grasp state."""
        from caster.sim_env.sim_client import SimClient

        if not self.demo_path:
            raise ValueError(
                "demo_path must be set before rendering the grasp."
            )

        transforms_path = resolve_transforms(self.demo_path)

        output_path = Path(video_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        client = SimClient(self.server_address, self.server_timeout)
        try:
            client.reset(
                str(transforms_path),
                str(output_path),
                render=True,
            )
            try:
                grasp_info = client.execute_grasp_get_all(
                    target_object,
                    hover_distance=self.m2t2_hover_distance,
                    grasp_depth_offset=self.m2t2_grasp_depth_offset,
                    grasp_height_offset=self.m2t2_grasp_height_offset,
                )
            except Exception:
                # The simulator records the approach and stabilization attempt
                # before rejecting an inaccurate grasp. Preserve those frames
                # for diagnosis, while still propagating the grasp failure.
                try:
                    client.save_video()
                    print(
                        "[Grasp Render] Failed grasp attempt saved to "
                        f"{output_path}",
                        flush=True,
                    )
                except Exception as save_error:
                    print(
                        "[Grasp Render] Could not save failed grasp attempt: "
                        f"{save_error}",
                        flush=True,
                    )
                raise
            if not grasp_info or not grasp_info.get("gripper_pose"):
                raise RuntimeError(
                    "The simulator did not return a completed grasp."
                )
            grasp_joints = np.asarray(
                grasp_info["joint_trajectory"],
                dtype=np.float64,
            )
            if (
                grasp_joints.ndim != 2
                or grasp_joints.shape[1] != 7
                or len(grasp_joints) < 2
            ):
                raise RuntimeError(
                    "The simulator did not return a valid grasp joint "
                    "trajectory."
                )
            grasp_trajectory_path = output_path.with_name(
                "grasp_trajectory.npz"
            )
            np.savez_compressed(
                grasp_trajectory_path,
                joint_trajectory=grasp_joints,
                source_hz=np.asarray(
                    float(grasp_info["source_hz"]),
                    dtype=np.float64,
                ),
            )
            client.save_video()
        finally:
            client.disconnect()

        print(
            f"[Grasp Render] Grasp video saved to {output_path}",
            flush=True,
        )
        print(
            "[Grasp Render] Grasp joint trajectory saved to "
            f"{grasp_trajectory_path}",
            flush=True,
        )
        return output_path


    def _settle_post_grasp(self, client) -> None:
        """Advance closed-gripper physics before the first optimized command."""
        for _ in range(self.post_grasp_settle_steps):
            # ``action=None`` intentionally retains the grasp controller's
            # closed-finger target and last arm target.
            client.step(None, record=False, return_positions=False)

    def _prepare_rollout(self, client, target_object, transforms_path, video_path):
        """Reset, restore the grasp, settle physics, and cache obstacle poses."""
        client.reset(str(transforms_path), str(video_path), self.render)
        grasp_poses = client.restore_or_execute_grasp(
            target_object,
            hover_distance=self.m2t2_hover_distance,
            grasp_depth_offset=self.m2t2_grasp_depth_offset,
            grasp_height_offset=self.m2t2_grasp_height_offset,
        )
        if not grasp_poses:
            raise ValueError("grasp_poses is empty or was not returned by the simulator client.")
        self._settle_post_grasp(client)
        if not self.obstacle_cost.has_post_grasp_poses:
            self.obstacle_cost.set_post_grasp_poses({
                name: pose for name, pose in grasp_poses.items() if name != target_object
            })
        return grasp_poses
