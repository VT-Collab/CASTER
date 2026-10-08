"""Mesh conversion, table geometry, and grasp control for Isaac simulation."""

import json
import os
from pathlib import Path

import numpy as np
import torch
import isaaclab.utils.math as math_utils
from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, MassPropertiesCfg, RigidBodyPropertiesCfg


def as_tensor(value):
    """Read Tensor-backed or ProxyArray-backed Isaac Lab data."""
    return getattr(value, "torch", value)


def _set_collision_approximation(usd_path, approximation):
    """Sets physics collision approximation in Usd Stage."""
    from pxr import Usd, UsdPhysics
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        return 0
    n = 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_api.CreateApproximationAttr().Set(approximation)
            n += 1
    if n:
        stage.GetRootLayer().Save()
    return n

def _bake_friction_into_usd(usd_path, static_friction, dynamic_friction):
    """Bake a physics material with custom friction parameters into the USD model."""
    from pxr import Usd, UsdPhysics, UsdShade
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        return
    mat = UsdShade.Material.Define(stage, "/World/ObjectFrictionMaterial")
    phys_api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    phys_api.CreateStaticFrictionAttr().Set(static_friction)
    phys_api.CreateDynamicFrictionAttr().Set(dynamic_friction)
    phys_api.CreateRestitutionAttr().Set(0.0)
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                mat, materialPurpose="physics"
            )
    stage.GetRootLayer().Save()

def convert_mesh(
    mesh_path,
    scale,
    out_name,
    usd_dir,
    mass_kg,
    collision_approx,
    static_friction,
    dynamic_friction,
    torsional_patch_radius=0.0,
    min_torsional_patch_radius=0.0,
):
    """Convert OBJ/GLB mesh to USD asset using Isaac Lab MeshConverter."""
    if not os.path.isfile(mesh_path):
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    obj_usd_dir = os.path.join(usd_dir, str(out_name).replace(" ", "_"))
    s = float(scale)

    sqrt_half = float(np.sqrt(0.5))

    cfg = MeshConverterCfg(
        asset_path=mesh_path,
        usd_dir=obj_usd_dir,
        force_usd_conversion=True,
        translation=(0.0, 0.0, 0.0),
        rotation=(sqrt_half, 0.0, 0.0, sqrt_half),
        scale=(s, s, s),
        make_instanceable=False,
        rigid_props=RigidBodyPropertiesCfg(
            rigid_body_enabled=True, kinematic_enabled=False, disable_gravity=False,
        ),
        collision_props=CollisionPropertiesCfg(
            collision_enabled=True,
            torsional_patch_radius=torsional_patch_radius,
            min_torsional_patch_radius=min_torsional_patch_radius,
        ),
        mass_props=MassPropertiesCfg(mass=mass_kg),
    )

    converter = MeshConverter(cfg)
    _set_collision_approximation(converter.usd_path, collision_approx)
    _bake_friction_into_usd(converter.usd_path, static_friction, dynamic_friction)
    return converter.usd_path

def add_table_to_scene_cfg(scene_cfg, table_translation, table):
    """Keep the top surface at table_translation.z in every environment."""
    from isaaclab.sim import CuboidCfg, PreviewSurfaceCfg, CollisionPropertiesCfg
    from isaaclab.assets import AssetBaseCfg
    table_size = np.asarray(table["size"], dtype=float)
    if table_size.shape != (3,) or not np.isfinite(table_size).all() or np.any(table_size <= 0):
        raise ValueError("table.size must contain three positive finite dimensions")
    table_center = np.asarray(table_translation, dtype=float)
    leg_height = float(table_center[2] - table_size[2])
    leg_width, inset = float(table["leg_width"]), float(table["leg_inset"])
    if not np.isfinite([leg_height, leg_width, inset]).all() or min(leg_height, leg_width) <= 0 or inset < leg_width / 2 or inset >= min(table_size[:2]) / 2:
        raise ValueError("Invalid table leg geometry")
    scene_cfg.table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=CuboidCfg(
            size=tuple(table_size),
            collision_props=CollisionPropertiesCfg(collision_enabled=True),
            visual_material=PreviewSurfaceCfg(
                diffuse_color=tuple(table["top_color"]), roughness=float(table.get("top_roughness", 1.0)), metallic=0.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(
            float(table_center[0]), float(table_center[1]),
            float(table_center[2] - table_size[2] / 2))),
    )
    for index, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        leg = AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/TableLeg{index}",
            spawn=CuboidCfg(
                size=(leg_width, leg_width, leg_height),
                visual_material=PreviewSurfaceCfg(
                    diffuse_color=tuple(table["leg_color"]), roughness=0.5, metallic=0.5)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(
                float(table_center[0] + sx * (table_size[0] / 2 - inset)),
                float(table_center[1] + sy * (table_size[1] / 2 - inset)),
                leg_height / 2)),
        )
        setattr(scene_cfg, f"table_leg_{index}", leg)


class GraspController:
    def _actual_tcp_pose(self):
        """Return actual TCP positions and hand quaternions in robot frame."""
        body_pos_w = as_tensor(self.robot_controller.robot.data.body_pos_w)
        body_quat_w = as_tensor(self.robot_controller.robot.data.body_quat_w)
        ee_pos_w = body_pos_w[:, self.robot_controller.ee_body_idx]
        ee_quat_w = body_quat_w[:, self.robot_controller.ee_body_idx]
        root_pos_w = as_tensor(
            self.robot_controller.robot.data.root_pos_w
        )
        root_quat_w = as_tensor(
            self.robot_controller.robot.data.root_quat_w
        )
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
            root_pos_w,
            root_quat_w,
            ee_pos_w,
            ee_quat_w,
        )
        local_offset = torch.tensor(
            [0.0, 0.0, self.tcp_offset_m],
            dtype=ee_pos_b.dtype,
            device=ee_pos_b.device,
        ).expand(self.num_envs, -1)
        tcp_pos_b = ee_pos_b + math_utils.quat_apply(
            ee_quat_b,
            local_offset,
        )
        return (
            tcp_pos_b.detach().cpu().numpy(),
            ee_quat_b.detach().cpu().numpy(),
        )

    def _hold_grasp_pose_until_stable(
        self,
        target_pos,
        target_quat,
        *,
        phase_name,
        joint_trajectory=None,
        frame_offset=0,
        maximum_steps=180,
        required_stable_steps=12,
        position_tolerance=0.02,
        orientation_tolerance_deg=5.0,
        position_motion_tolerance=0.001,
        orientation_motion_tolerance_deg=1.0,
    ):
        """Hold a grasp pose until it is accurate and no longer moving."""
        target_position = np.asarray(target_pos, dtype=np.float64)
        target_orientation = np.asarray(target_quat, dtype=np.float64)
        target_orientation /= np.linalg.norm(target_orientation)
        action = target_position.tolist() + target_orientation.tolist()
        orientation_tolerance = np.deg2rad(orientation_tolerance_deg)
        orientation_motion_tolerance = np.deg2rad(
            orientation_motion_tolerance_deg
        )

        previous_positions = None
        previous_orientations = None
        stable_steps = 0
        max_position_error = float("inf")
        max_orientation_error = float("inf")
        max_position_motion = float("inf")
        max_orientation_motion = float("inf")

        for step_i in range(maximum_steps):
            self.robot_controller.step(action)
            self.scene.write_data_to_sim()
            for _ in range(self.decimation):
                self.sim.step(render=False)
                self.scene.update(self.physics_dt)
            if joint_trajectory is not None:
                joint_trajectory.append(
                    self.robot_controller.robot.data.joint_pos[0, :7]
                    .detach()
                    .cpu()
                    .tolist()
                )
            if (
                getattr(self, "enable_render", True)
                and self.camera is not None
                and self.camera.enabled
                and step_i % self.camera.record_frequency == 0
            ):
                self.sim.render()
                self.camera.record_frame(
                    frame_offset + step_i,
                    force=True,
                )

            positions, orientations = self._actual_tcp_pose()
            max_position_error = float(np.max(np.linalg.norm(
                positions - target_position[None, :],
                axis=1,
            )))
            target_dots = np.clip(
                np.abs(orientations @ target_orientation),
                0.0,
                1.0,
            )
            max_orientation_error = float(np.max(
                2.0 * np.arccos(target_dots)
            ))

            if previous_positions is None:
                max_position_motion = float("inf")
                max_orientation_motion = float("inf")
            else:
                max_position_motion = float(np.max(np.linalg.norm(
                    positions - previous_positions,
                    axis=1,
                )))
                motion_dots = np.clip(
                    np.abs(np.sum(
                        orientations * previous_orientations,
                        axis=1,
                    )),
                    0.0,
                    1.0,
                )
                max_orientation_motion = float(np.max(
                    2.0 * np.arccos(motion_dots)
                ))

            accurate = (
                max_position_error <= position_tolerance
                and max_orientation_error <= orientation_tolerance
            )
            stationary = (
                max_position_motion <= position_motion_tolerance
                and max_orientation_motion <= orientation_motion_tolerance
            )
            stable_steps = stable_steps + 1 if accurate and stationary else 0
            if stable_steps >= required_stable_steps:
                print(
                    f"[IsaacSim] {phase_name} stable after {step_i + 1} "
                    f"hold steps: position_error={max_position_error:.4f}m, "
                    f"orientation_error={np.degrees(max_orientation_error):.2f}deg",
                    flush=True,
                )
                return step_i + 1

            previous_positions = positions.copy()
            previous_orientations = orientations.copy()

        raise RuntimeError(
            f"{phase_name} did not stabilize; refusing to close the gripper. "
            f"position_error={max_position_error:.4f}m, "
            f"orientation_error={np.degrees(max_orientation_error):.2f}deg, "
            f"position_motion={max_position_motion:.4f}m/step, "
            f"orientation_motion={np.degrees(max_orientation_motion):.2f}deg/step"
        )

    def execute_grasp(
        self,
        object_name,
        hover_distance=None,
        grasp_depth_offset=None,
        grasp_height_offset=None,
    ):
        """Move the robot using the saved M2T2 grasp."""
        if not getattr(self, "initialized", False):
            raise RuntimeError("Environment not constructed! Call initialize_scene_once() first.")

        effective_hover_distance = (
            0.27 if hover_distance is None else float(hover_distance)
        )
        if effective_hover_distance <= 0.0:
            raise ValueError("hover_distance must be positive.")
        effective_grasp_depth_offset = (
            0.01
            if grasp_depth_offset is None
            else float(grasp_depth_offset)
        )
        if not np.isfinite(effective_grasp_depth_offset):
            raise ValueError("grasp_depth_offset must be finite.")
        effective_grasp_height_offset = (
            0.0
            if grasp_height_offset is None
            else float(grasp_height_offset)
        )
        if not np.isfinite(effective_grasp_height_offset):
            raise ValueError("grasp_height_offset must be finite.")

        slot = getattr(self, "object_name_to_slot", {}).get(object_name)
        if slot is None:
            raise KeyError(f"Unknown grasp target: {object_name!r}")
        if not hasattr(self, "robot_controller"):
            raise RuntimeError("The robot controller is unavailable")

        # Record the measured arm path only through the final stable
        # pre-grasp pose. Gripper closure is intentionally represented by
        # the gripper closure, not as an arm waypoint.
        grasp_joint_trajectory = [
            self.robot_controller.robot.data.joint_pos[0, :7]
            .detach()
            .cpu()
            .tolist()
        ]
        root_pos_w = as_tensor(self.robot_controller.robot.data.root_pos_w)
        root_quat_w = as_tensor(self.robot_controller.robot.data.root_quat_w)

        from scipy.spatial.transform import Rotation as Rot

        transforms_path = Path(self._active_transforms_path)
        grasp_json_path = transforms_path.parent / "grasp.json"
        if not grasp_json_path.is_file():
            raise FileNotFoundError(f"M2T2 grasp file not found: {grasp_json_path}")

        with open(grasp_json_path) as f:
            grasp_data = json.load(f)

        if grasp_data.get("server") != "m2t2_grasp":
            raise ValueError(
                f"Expected an M2T2 grasp file at {grasp_json_path}, "
                f"found server={grasp_data.get('server')!r}"
            )
        slot_to_name = {v: k for k, v in getattr(self, "object_name_to_slot", {}).items()}
        actual_name = slot_to_name.get(slot, object_name)
        object_grasps = grasp_data.get("objects")
        if isinstance(object_grasps, dict):
            grasp_entry = next(
                (
                    object_grasps.get(name)
                    for name in (object_name, actual_name, slot)
                    if isinstance(object_grasps.get(name), dict)
                ),
                None,
            )
        elif grasp_data.get("object") in (
            object_name,
            actual_name,
            slot,
        ):
            # A target-specific M2T2 request writes the selected
            # object directly at the top level.
            grasp_entry = grasp_data
        else:
            grasp_entry = None

        if grasp_entry is None:
            raise KeyError(
                f"No M2T2 grasp for '{actual_name}' in "
                f"{grasp_json_path}"
            )
        if grasp_entry.get("frame") != "scene":
            raise ValueError(
                f"M2T2 grasp for '{actual_name}' must be in the "
                f"scene frame, got {grasp_entry.get('frame')!r}"
            )
        if "transform_matrix" not in grasp_entry:
            raise KeyError(
                f"M2T2 grasp for '{actual_name}' has no "
                "transform_matrix"
            )

        # M2T2 reconstructs and predicts directly in the scene,
        # which is the robot-base frame in this pipeline.
        G_base = np.asarray(
            grasp_entry["transform_matrix"], dtype=np.float64
        ).reshape(4, 4)

        # Get current starting TCP position and EE orientation (XYZW) in robot base frame
        ee_pos_w = as_tensor(self.robot_controller.robot.data.body_pos_w[:, self.robot_controller.ee_body_idx])
        ee_quat_w = as_tensor(self.robot_controller.robot.data.body_quat_w[:, self.robot_controller.ee_body_idx])
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w,
            ee_pos_w, ee_quat_w
        )

        q_start_xyzw = ee_quat_b[0].cpu().numpy()
        R_start = Rot.from_quat(q_start_xyzw).as_matrix()
        start_tcp = (
            ee_pos_b[0].cpu().numpy()
            + self.tcp_offset_m * R_start[:, 2]
        )

        # M2T2 stores its internal gripper-base pose 0.1034 m
        # behind the contact point. Convert it to the controller
        # TCP convention with the same -90 degree local-Z axis
        # mapping used by M2T2's gripper_pose_to_rlbench().
        depth_offset = effective_grasp_depth_offset
        R_grasp_tcp = Rot.from_euler("z", -90.0, degrees=True).as_matrix()

        T_grasp_tcp = np.eye(4)
        T_grasp_tcp[:3, :3] = R_grasp_tcp
        T_grasp_tcp[:3, 3] = np.array([0.0, 0.0, depth_offset], dtype=np.float64)

        T_base_tcp = G_base @ T_grasp_tcp

        print(f"[IsaacSim] Grasp depth offset: {depth_offset:.3f} m", flush=True)
        print(f"[IsaacSim] Approach direction in base: {G_base[:3, 2]}", flush=True)
        print(f"[IsaacSim] Original grasp position: {G_base[:3, 3]}", flush=True)
        print(f"[IsaacSim] Deeper TCP base position: {T_base_tcp[:3, 3]}", flush=True)

        # The parallel-jaw gripper is symmetric under 180
        # degrees about M2T2's approach axis.
        R_grasp_1 = G_base[:3, :3]
        R_grasp_2 = R_grasp_1 @ np.diag([-1.0, -1.0, 1.0])

        q1_xyzw = Rot.from_matrix(R_grasp_1 @ R_grasp_tcp).as_quat()
        q2_xyzw = Rot.from_matrix(R_grasp_2 @ R_grasp_tcp).as_quat()

        dot1 = np.dot(q1_xyzw, q_start_xyzw)
        dot2 = np.dot(q2_xyzw, q_start_xyzw)

        selected_candidate = 1 if abs(dot1) >= abs(dot2) else 2
        if selected_candidate == 1:
            selected_q_xyzw = q1_xyzw
            selected_dot = dot1
            R_tcp_selected = R_grasp_1 @ R_grasp_tcp
        else:
            selected_q_xyzw = q2_xyzw
            selected_dot = dot2
            R_tcp_selected = R_grasp_2 @ R_grasp_tcp

        # Ensure positive dot product to avoid quaternion 360-deg hemisphere wrapping in IK
        if selected_dot < 0:
            selected_q_xyzw = -selected_q_xyzw

        target_quat = selected_q_xyzw.tolist()

        z_approach = G_base[:3, 2]
        # Recover the world-space contact point represented
        # by M2T2's stock-Franka gripper model.  This stays
        # 0.1034 m even for UMI; Panda.step() separately
        # converts that contact target using the installed
        # gripper's physical TCP offset.
        grasp_tcp = T_base_tcp[:3, 3] + 0.1034 * z_approach
        # Keep the hover pose tied to the original M2T2
        # grasp. The height offset applies only to the final
        # grasp target in robot-base-frame vertical (+Z).
        target_pos_above = (
            grasp_tcp
            - effective_hover_distance * z_approach
        ).tolist()
        final_grasp_tcp = grasp_tcp.copy()
        final_grasp_tcp[2] += effective_grasp_height_offset
        target_pos = final_grasp_tcp.tolist()

        print(f"[IsaacSim] Executing M2T2 grasp for '{actual_name}' (conf={grasp_entry.get('confidence', 0.0):.3f})...", flush=True)
        print(
            "[IsaacSim] M2T2 hover distance: "
            f"{effective_hover_distance:.3f}m",
            flush=True,
        )
        print(
            "[IsaacSim] M2T2 final grasp height offset "
            "(robot base Z): "
            f"{effective_grasp_height_offset:.3f}m",
            flush=True,
        )

        # Phase 1a: Move to hover position using 3D position-only weighted IK with quintic smoothstep
        distance = np.linalg.norm(np.asarray(target_pos_above) - start_tcp)
        control_dt = self.physics_dt * self.decimation
        max_tcp_speed = 0.10  # m/s

        num_steps_1a = max(
            40,
            int(np.ceil(distance / (max_tcp_speed * control_dt))),
        )

        for step_i in range(num_steps_1a):
            u = (step_i + 1) / float(num_steps_1a)
            # Quintic smoothstep: zero velocity and acceleration at both ends
            t = 10.0 * (u**3) - 15.0 * (u**4) + 6.0 * (u**5)
            pos_step = (1.0 - t) * start_tcp + t * np.asarray(target_pos_above)

            # Send 3D action only: leverages weighted position-only IK solver
            self.robot_controller.step(pos_step.tolist())
            self.scene.write_data_to_sim()
            should_render = getattr(self, "enable_render", True)
            for _ in range(self.decimation):
                self.sim.step(render=False)
                self.scene.update(self.physics_dt)
            grasp_joint_trajectory.append(
                self.robot_controller.robot.data.joint_pos[0, :7]
                .detach()
                .cpu()
                .tolist()
            )
            if should_render and self.camera is not None and self.camera.enabled:
                if step_i % self.camera.record_frequency == 0:
                    self.sim.render()
                    # Sampling was already applied above;
                    # bypass Camera.record_frame's second
                    # modulo filter.
                    self.camera.record_frame(step_i, force=True)

        # Capture actual end-effector orientation at hover position before reorienting
        ee_pos_w = as_tensor(self.robot_controller.robot.data.body_pos_w[:, self.robot_controller.ee_body_idx])
        ee_quat_w = as_tensor(self.robot_controller.robot.data.body_quat_w[:, self.robot_controller.ee_body_idx])
        root_pos_w = as_tensor(self.robot_controller.robot.data.root_pos_w)
        root_quat_w = as_tensor(self.robot_controller.robot.data.root_quat_w)
        _, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        q_hover_actual_xyzw = ee_quat_b[0].detach().cpu().numpy()

        # Phase 1b: Change orientation at hover position (target_pos_above)
        from scipy.spatial.transform import Slerp
        key_rots = Rot.from_quat([q_hover_actual_xyzw, selected_q_xyzw])
        slerp = Slerp([0.0, 1.0], key_rots)

        orientation_dot = float(np.clip(
            abs(np.dot(
                q_hover_actual_xyzw,
                selected_q_xyzw,
            )),
            0.0,
            1.0,
        ))
        orientation_angle = 2.0 * np.arccos(
            orientation_dot
        )
        max_angular_speed = np.deg2rad(45.0)
        # Quintic smoothstep has a peak derivative of
        # 1.875, so account for that when enforcing the
        # angular-speed limit.
        num_steps_1b = max(
            60,
            int(np.ceil(
                1.875 * orientation_angle
                / (max_angular_speed * control_dt)
            )),
        )

        for step_i in range(num_steps_1b):
            u = (step_i + 1) / float(num_steps_1b)
            t = (
                10.0 * (u**3)
                - 15.0 * (u**4)
                + 6.0 * (u**5)
            )
            q_step_xyzw = slerp([t])[0].as_quat()
            action_step = target_pos_above + q_step_xyzw.tolist()
            self.robot_controller.step(action_step)
            self.scene.write_data_to_sim()
            should_render = getattr(self, "enable_render", True)
            for _ in range(self.decimation):
                self.sim.step(render=False)
                self.scene.update(self.physics_dt)
            grasp_joint_trajectory.append(
                self.robot_controller.robot.data.joint_pos[0, :7]
                .detach()
                .cpu()
                .tolist()
            )
            if should_render and self.camera is not None and self.camera.enabled:
                if step_i % self.camera.record_frequency == 0:
                    self.sim.render()
                    self.camera.record_frame(
                        num_steps_1a + step_i,
                        force=True,
                    )

        hover_hold_steps = self._hold_grasp_pose_until_stable(
            target_pos_above,
            target_quat,
            phase_name="M2T2 hover pose",
            joint_trajectory=grasp_joint_trajectory,
            frame_offset=num_steps_1a + num_steps_1b,
        )

        # Phase 2: Smooth and controlled descent from hover pose (target_pos_above) to final grasp pose (target_pos)
        descent_distance = np.linalg.norm(np.asarray(target_pos) - np.asarray(target_pos_above))
        descent_speed = 0.1  # m/s (slow, controlled descent)
        control_dt = self.physics_dt * self.decimation

        num_steps_2 = max(
            60,
            int(np.ceil(descent_distance / (descent_speed * control_dt))),
        )

        for step_i in range(num_steps_2):
            u = (step_i + 1) / float(num_steps_2)
            # Quintic smoothstep: soft start from hover and soft touchdown at target grasp pose
            t = 10.0 * (u**3) - 15.0 * (u**4) + 6.0 * (u**5)
            pos_step = (1.0 - t) * np.asarray(target_pos_above) + t * np.asarray(target_pos)

            action_step = pos_step.tolist() + target_quat
            self.robot_controller.step(action_step)
            self.scene.write_data_to_sim()
            should_render = getattr(self, "enable_render", True)
            for _ in range(self.decimation):
                self.sim.step(render=False)
                self.scene.update(self.physics_dt)
            grasp_joint_trajectory.append(
                self.robot_controller.robot.data.joint_pos[0, :7]
                .detach()
                .cpu()
                .tolist()
            )
            if should_render and self.camera is not None and self.camera.enabled:
                if step_i % self.camera.record_frequency == 0:
                    self.sim.render()
                    self.camera.record_frame(
                        num_steps_1a
                        + num_steps_1b
                        + hover_hold_steps
                        + step_i,
                        force=True,
                    )

        # Diagnostic logging: compare commanded rotation against actual converged EE rotation
        ee_pos_w = as_tensor(self.robot_controller.robot.data.body_pos_w[:, self.robot_controller.ee_body_idx])
        ee_quat_w = as_tensor(self.robot_controller.robot.data.body_quat_w[:, self.robot_controller.ee_body_idx])
        root_pos_w = as_tensor(self.robot_controller.robot.data.root_pos_w)
        root_quat_w = as_tensor(self.robot_controller.robot.data.root_quat_w)
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)

        q_actual_xyzw = as_tensor(ee_quat_b)[0].detach().cpu().numpy()
        R_actual = Rot.from_quat(q_actual_xyzw).as_matrix()

        R_tracking_error = R_tcp_selected.T @ R_actual
        tracking_rot = Rot.from_matrix(R_tracking_error)
        angle_deg = np.degrees(tracking_rot.magnitude())
        rotvec = tracking_rot.as_rotvec()
        axis_local = rotvec / np.linalg.norm(rotvec) if np.linalg.norm(rotvec) > 1e-8 else np.zeros(3)

        print(
            f"[IsaacSim] IK tracking error: {angle_deg:.2f} deg, local axis={axis_local}",
            flush=True,
        )

        # Never close on a time-only assumption. Hold the final pose until
        # position, orientation, and residual TCP motion are all stable.
        self._hold_grasp_pose_until_stable(
            target_pos,
            target_quat,
            phase_name="final pre-grasp pose",
            joint_trajectory=grasp_joint_trajectory,
        )

        # Close the gripper only after the convergence gate above passes.
        self.robot_controller.set_gripper(0.0)
            
        for step_i in range(30):
            # We don't call robot_controller.step() here so we don't overwrite the gripper target
            self.scene.write_data_to_sim()
            should_render = getattr(self, "enable_render", True)
            for _ in range(self.decimation):
                self.sim.step(render=False)
                self.scene.update(self.physics_dt)
            if should_render and self.camera is not None and self.camera.enabled:
                if step_i % self.camera.record_frequency == 0:
                    self.sim.render()
                    self.camera.record_frame(step_i)

        object_poses = {}
        slot_to_name = {v: k for k, v in getattr(self, "object_name_to_slot", {}).items()}
        for obj in self.object_names:
            if obj in self.scene.keys():
                rigid_obj = self.scene[obj]
                actual_name = slot_to_name.get(obj, obj)
                pos_list, quat_list = self._to_robot_frame(rigid_obj.data.root_pos_w, rigid_obj.data.root_quat_w)
                if self.num_envs == 1:
                    pos_list = pos_list[0]
                    quat_list = quat_list[0]
                object_poses[actual_name] = {
                    "position": pos_list,
                    "orientation": quat_list
                }

        # Get the controller TCP position and hand orientation in the base
        # frame. Optimizer 7D actions interpret their position component as
        # TCP, so returning the hand-body origin here would apply the
        # hand-to-TCP offset a second time.
        gripper_pose = {}
        if hasattr(self, "robot_controller"):
            ee_pos_w = as_tensor(
                self.robot_controller.robot.data.body_pos_w[
                    :, self.robot_controller.ee_body_idx
                ]
            )
            ee_quat_w = as_tensor(self.robot_controller.robot.data.body_quat_w[:, self.robot_controller.ee_body_idx])
            root_pos_w = as_tensor(self.robot_controller.robot.data.root_pos_w)
            root_quat_w = as_tensor(self.robot_controller.robot.data.root_quat_w)
            _, ee_quat_b = math_utils.subtract_frame_transforms(
                root_pos_w,
                root_quat_w,
                ee_pos_w,
                ee_quat_w,
            )
            pos_list = self.robot_controller.get_tcp_pos()
            # Isaac Lab reports XYZW; expose the WXYZ
            # convention at the simulation API boundary.
            quat_list = ee_quat_b[:, [3, 0, 1, 2]].cpu().numpy().tolist()
            if self.num_envs == 1:
                quat_list = quat_list[0]
            gripper_pose = {
                "position": pos_list,
                "orientation": quat_list
            }
        grasp_info = {
            "object_poses": object_poses,
            "gripper_pose": gripper_pose,
            "joint_trajectory": grasp_joint_trajectory,
            "source_hz": 1.0 / self.sim_dt,
        }
        self.cache_post_grasp_state(object_name, grasp_info)
        return grasp_info

    def execute_release(self):
        """Open the gripper to release the object."""
        if not getattr(self, "initialized", False):
            raise RuntimeError("Environment not constructed! Call initialize_scene_once() first.")
            
        if hasattr(self, "robot_controller"):
            try:
                self.robot_controller.set_gripper(0.04)
            except Exception as e:
                print(f"[IsaacSim] Failed to set gripper target: {e}")
                
            for step_i in range(30):
                self.scene.write_data_to_sim()
                should_render = getattr(self, "enable_render", True)
                for _ in range(self.decimation):
                    self.sim.step(render=False)
                    self.scene.update(self.physics_dt)
                if should_render and self.camera is not None and self.camera.enabled:
                    if step_i % self.camera.record_frequency == 0:
                        self.sim.render()
                        self.camera.record_frame(step_i)
