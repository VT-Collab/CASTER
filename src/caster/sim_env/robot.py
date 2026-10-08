"""Franka joint control and TCP tracking in Isaac Lab."""

import torch
import numpy as np
import isaaclab.utils.math as math_utils
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg

def as_torch(value):
    """Compatible with both older Tensor-backed and newer ProxyArray-backed Isaac Lab."""
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "torch"):
        return value.torch
    import warp as wp
    return wp.to_torch(value)

class Panda:
    def __init__(self, scene, name="robot", tcp_offset_m=0.1033):
        """
        Wrapper class for the Franka Panda articulation in IsaacLab.
        """
        self.robot = scene[name]
        self.device = self.robot.device
        self.num_envs = scene.num_envs
        self.tcp_offset_m = float(tcp_offset_m)
        if not np.isfinite(self.tcp_offset_m) or self.tcp_offset_m <= 0.0:
            raise ValueError("tcp_offset_m must be a positive finite distance")
        
        # End effector is the hand
        try:
            self.ee_body_idx = self.robot.find_bodies("panda_hand")[0][0]
        except Exception:
            self.ee_body_idx = self.robot.find_bodies("panda_link8")[0][0]
            
        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
        )
        self.ik_controller = DifferentialIKController(cfg=ik_cfg, num_envs=self.num_envs, device=self.device)
        
        try:
            self.finger_joint_ids = self.robot.find_joints("panda_finger.*")[0]
        except Exception:
            self.finger_joint_ids = None

        if self.finger_joint_ids is not None:
            self.gripper_target = torch.full(
                (self.num_envs, len(self.finger_joint_ids)),
                0.04,
                device=self.device,
                dtype=torch.float32
            )
        else:
            self.gripper_target = None
            
        self.last_target_quat = None
        self._using_xyz_only_last_step = False
        self.xyz_reference_joint_pos = None
        self._step_count = 0
        self.previous_tcp_target = None
        self.previous_tcp_position = None

    def reset(self):
        """
        Reset the robot to its default joint positions.
        """
        self.last_target_quat = None
        self._using_xyz_only_last_step = False
        self.xyz_reference_joint_pos = None
        self._step_count = 0
        self.previous_tcp_target = None
        self.previous_tcp_position = None
        joint_pos = as_torch(self.robot.data.default_joint_pos).clone()
        joint_vel = as_torch(self.robot.data.default_joint_vel).clone()
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)
        self.robot.set_joint_position_target(joint_pos)
        self.ik_controller.reset()
        if self.finger_joint_ids is not None:
            self.gripper_target = as_torch(self.robot.data.default_joint_pos)[:, self.finger_joint_ids].clone()

        
    def _compute_jacobian(self):
        J_w = as_torch(self.robot.root_physx_view.get_jacobians())
        
        num_cols = J_w.shape[-1]
        num_joints = self.robot.num_joints
        col_offset = num_cols - num_joints
        
        # PhysX excludes the fixed root from the Jacobian body dimension.
        if self.robot.is_fixed_base:
            jacobian_body_idx = self.ee_body_idx - 1
        else:
            jacobian_body_idx = self.ee_body_idx

        J_ee_w = J_w[:, jacobian_body_idx, :, col_offset:]
        
        base_rot = as_torch(self.robot.data.root_quat_w)
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
        
        J_b = J_ee_w.clone()
        J_b[:, :3, :] = torch.bmm(base_rot_matrix, J_ee_w[:, :3, :])
        J_b[:, 3:, :] = torch.bmm(base_rot_matrix, J_ee_w[:, 3:, :])
        return J_b

    def set_gripper(self, gripper_pos):
        """
        Set target gripper position.
        gripper_pos: float, list, or tensor
        """
        if self.finger_joint_ids is not None:
            if isinstance(gripper_pos, (int, float)):
                self.gripper_target = torch.full_like(self.gripper_target, gripper_pos)
            elif isinstance(gripper_pos, torch.Tensor):
                self.gripper_target = gripper_pos.to(self.device).clone()
            else:
                self.gripper_target = torch.tensor(gripper_pos, device=self.device, dtype=torch.float32)
            
            # Apply immediately
            self.robot.set_joint_position_target(self.gripper_target, joint_ids=self.finger_joint_ids)

    def step(self, action):
        """
        Apply an action to the robot.
        action: list, numpy array, or torch tensor. Can be 1D (shape [3], [4], [7], [8]) or 2D (shape [num_envs, D]).
        """
        if action is None:
            return
            
        # Convert action to a tensor on the correct device
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32, device=self.device)
        else:
            action = action.to(dtype=torch.float32, device=self.device)
            
        if action.ndim == 1:
            # If 1D action, tile/repeat it to match self.num_envs
            action = action.unsqueeze(0).repeat(self.num_envs, 1)
            
        if action.shape[1] < 3:
            return
            
        center = action[:, :3]
        
        # Get current ee pose in base frame
        body_pos_w = as_torch(self.robot.data.body_pos_w)
        body_quat_w = as_torch(self.robot.data.body_quat_w)
        ee_pos_w = body_pos_w[:, self.ee_body_idx]
        ee_quat_w = body_quat_w[:, self.ee_body_idx]
        root_pos_w = as_torch(self.robot.data.root_pos_w)
        root_quat_w = as_torch(self.robot.data.root_quat_w)
        
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        joint_pos = as_torch(self.robot.data.joint_pos)
        joint_pos_arm = joint_pos[:, :7]

        # Parse gripper target if provided (col 3 for shape [num_envs, 4] or col 7 for shape [num_envs, 8])
        if action.shape[1] == 4 or action.shape[1] == 8:
            gripper_pos = action[:, -1]
            if self.finger_joint_ids is not None:
                self.gripper_target = gripper_pos.unsqueeze(-1).repeat(1, len(self.finger_joint_ids))

        # --- MODE 1: Full 7-DoF Pose Commands (Grasping Phase) ---
        if action.shape[1] >= 7:
            target_quat = action[:, 3:7].clone()
            target_quat = torch.nn.functional.normalize(target_quat, dim=-1)

            if (
                self.last_target_quat is not None
                and self.last_target_quat.shape == target_quat.shape
            ):
                quat_dot = torch.sum(
                    target_quat * self.last_target_quat,
                    dim=-1,
                    keepdim=True,
                )
                target_quat = torch.where(
                    quat_dot < 0.0,
                    -target_quat,
                    target_quat,
                )

            self.last_target_quat = target_quat.detach().clone()
            self._using_xyz_only_last_step = False
            self.xyz_reference_joint_pos = None

            local_offset = torch.tensor(
                [[0.0, 0.0, self.tcp_offset_m]],
                device=self.device,
                dtype=torch.float32,
            ).repeat(self.num_envs, 1)
            offset_b = math_utils.quat_apply(target_quat, local_offset)
            target_pos = center - offset_b

            cmd = torch.zeros((self.num_envs, 7), device=self.device, dtype=torch.float32)
            cmd[:, 0:3] = target_pos
            cmd[:, 3:7] = target_quat

            self.ik_controller.set_command(cmd)
            jacobian = self._compute_jacobian()
            joint_pos_des = self.ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        # --- MODE 2: Position-Only Trajectory Commands (XYZ Playback) ---
        else:
            if not self._using_xyz_only_last_step:
                self.xyz_reference_joint_pos = joint_pos_arm.detach().clone()

            self._using_xyz_only_last_step = True

            # Construct TCP position and translational TCP Jacobian J_tcp
            local_offset = torch.tensor(
                [0.0, 0.0, self.tcp_offset_m],
                dtype=ee_pos_b.dtype,
                device=self.device,
            ).expand(self.num_envs, -1)
            offset_b = math_utils.quat_apply(ee_quat_b, local_offset)
            tcp_pos_b = ee_pos_b + offset_b
            position_error = center - tcp_pos_b

            jacobian = self._compute_jacobian()
            J_linear = jacobian[:, 0:3, :7]
            J_angular = jacobian[:, 3:6, :7]

            rx, ry, rz = offset_b.unbind(dim=-1)
            zeros = torch.zeros_like(rx)
            skew_offset = torch.stack(
                (
                    zeros, -rz,   ry,
                    rz,    zeros, -rx,
                    -ry,   rx,    zeros,
                ),
                dim=-1,
            ).reshape(-1, 3, 3)

            J_tcp = J_linear - skew_offset @ J_angular

            # Franka Panda arm joint limits and velocity limits (rad/s)
            lower_limits = torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], device=self.device, dtype=J_tcp.dtype)
            upper_limits = torch.tensor([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973], device=self.device, dtype=J_tcp.dtype)
            velocity_limits = torch.tensor([2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610], device=self.device, dtype=J_tcp.dtype)

            margin = 0.15
            distance_lower = joint_pos_arm - lower_limits
            distance_upper = upper_limits - joint_pos_arm
            distance_limit = torch.minimum(distance_lower, distance_upper)

            limit_weight = 1.0 + 20.0 * torch.clamp(
                (margin - distance_limit) / margin,
                min=0.0,
                max=1.0,
            ) ** 2

            joint_weights = limit_weight
            joint_weights[:, 1] *= 10.0  # panda_joint2
            joint_weights[:, 4] *= 10.0  # panda_joint5

            W_inv = torch.diag_embed(1.0 / joint_weights)

            damping = 0.05
            posture_gain = 0.05

            eye3 = torch.eye(3, dtype=J_tcp.dtype, device=self.device).expand(self.num_envs, -1, -1)
            eye7 = torch.eye(7, dtype=J_tcp.dtype, device=self.device).expand(self.num_envs, -1, -1)

            A = J_tcp @ W_inv @ J_tcp.transpose(1, 2) + (damping**2) * eye3
            J_pinv = W_inv @ J_tcp.transpose(1, 2) @ torch.linalg.solve(A, eye3)

            delta_q_task = (J_pinv @ position_error.unsqueeze(-1)).squeeze(-1)
            nullspace = eye7 - J_pinv @ J_tcp

            posture_error = self.xyz_reference_joint_pos - joint_pos_arm
            delta_q_posture = (nullspace @ (posture_gain * posture_error).unsqueeze(-1)).squeeze(-1)

            delta_q = delta_q_task + delta_q_posture

            # Control DT: 10 substeps * (1/120s physics_dt) = 0.0833s
            control_dt = 0.0833
            max_delta_q = velocity_limits * control_dt
            delta_q = torch.maximum(torch.minimum(delta_q, max_delta_q), -max_delta_q)

            joint_pos_des = joint_pos.clone()
            joint_pos_des[:, :7] = joint_pos_arm + delta_q

        # Override gripper joints in joint_pos_des
        if self.finger_joint_ids is not None and self.gripper_target is not None:
            joint_pos_des[:, self.finger_joint_ids] = self.gripper_target
            
        self.robot.set_joint_position_target(joint_pos_des)

    def get_tcp_pos(self):
        """Get actual TCP position in robot base frame."""
        body_pos_w = as_torch(self.robot.data.body_pos_w)
        body_quat_w = as_torch(self.robot.data.body_quat_w)
        ee_pos_w = body_pos_w[:, self.ee_body_idx]
        ee_quat_w = body_quat_w[:, self.ee_body_idx]
        root_pos_w = as_torch(self.robot.data.root_pos_w)
        root_quat_w = as_torch(self.robot.data.root_quat_w)
        
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        
        # TCP lies in front of the hand along its local Z-axis.  Its distance
        # depends on the installed physical finger geometry.
        local_offset = torch.tensor(
            [[0.0, 0.0, self.tcp_offset_m]],
            device=self.device,
            dtype=torch.float32,
        ).repeat(self.num_envs, 1)
        offset_b = math_utils.quat_apply(ee_quat_b, local_offset)
        tcp_pos_b = ee_pos_b + offset_b
        pos_list = tcp_pos_b.detach().cpu().numpy().tolist()
        if self.num_envs == 1:
            pos_list = pos_list[0]
        return pos_list
