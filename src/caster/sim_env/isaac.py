"""Build and step a Franka tabletop scene after Isaac Lab's app has launched."""

import copy
import json
import os
from pathlib import Path

import numpy as np
import torch
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationContext
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

from caster.sim_env.sim_utils import as_tensor, convert_mesh
from caster.sim_env.sim_utils import GraspController
from caster.sim_env.robot import Panda


class IsaacSim(GraspController):
    def __init__(self,
                 robot_base_pos=(0.0, 0.0, 1.05),
                 table_translation=(0.55, 0.0, 1.05),
                 env_spacing=3.0,
                 sim_dt=0.016666667,
                 usd_dir="outputs/sim_usd",
                 num_envs=1,
                 object_mass_kg=0.08,
                 collision_approx="convexDecomposition",
                 static_friction=2.0,
                 dynamic_friction=2.0,
                 torsional_patch_radius=0.007,
                 min_torsional_patch_radius=0.002,
                 arm_stiffness_scale=4.0,
                 gripper_stiffness=4_000.0,
                 gripper_damping=200.0,
                 gripper_effort_limit=200.0,
                 gripper_velocity_limit=0.2,
                 camera=None,
                 headless=True,
                 gripper_variant="umi",
                 umi_asset_dir=None,
                 table=None):
        """
        Initialize the simulation environment config parameters and calibration values.
        """
        self.headless = headless
        self.gripper_variant = str(gripper_variant).strip().lower()
        if self.gripper_variant not in {"stock", "umi"}:
            raise ValueError(
                "gripper_variant must be either 'stock' or 'umi', got "
                f"{gripper_variant!r}."
            )
        self.table = table
        self.umi_asset_dir = Path(umi_asset_dir) if umi_asset_dir else Path(__file__).parent / "assets/umi"
        self.tcp_offset_m = 0.1033
        if self.gripper_variant == "umi":
            summary_path = (
                self.umi_asset_dir
                / "build_summary.json"
            )
            if not summary_path.exists():
                raise FileNotFoundError(
                    f"UMI build summary not found at {summary_path}. "
                    "Set umi_asset_dir to the directory containing the UMI USD and build_summary.json."
                )
            with summary_path.open("r", encoding="utf-8") as stream:
                summary = json.load(stream)
            self.tcp_offset_m = float(summary["hand_to_tcp_m"])
        print(
            f"[IsaacSim] Controller TCP offset: {self.tcp_offset_m:.4f} m "
            f"({self.gripper_variant})",
            flush=True,
        )
        self.robot_base_pos = tuple(robot_base_pos)
        self.table_translation = tuple(table_translation)
        self.env_spacing = float(env_spacing)
        self.sim_dt = float(sim_dt)
        self.decimation = 2
        self.physics_dt = self.sim_dt / self.decimation
        
        self.usd_dir = str(Path(usd_dir).resolve())
        os.makedirs(self.usd_dir, exist_ok=True)
        
        self.num_envs = int(num_envs)
        self.object_mass_kg = float(object_mass_kg)
        self.collision_approx = collision_approx
        self.static_friction = float(static_friction)
        self.dynamic_friction = float(dynamic_friction)
        self.torsional_patch_radius = float(torsional_patch_radius)
        self.min_torsional_patch_radius = float(
            min_torsional_patch_radius
        )
        if (
            not np.isfinite(self.torsional_patch_radius)
            or not np.isfinite(self.min_torsional_patch_radius)
            or self.torsional_patch_radius < 0.0
            or self.min_torsional_patch_radius < 0.0
            or self.min_torsional_patch_radius
            > self.torsional_patch_radius
        ):
            raise ValueError(
                "Torsional patch radii must be finite and satisfy "
                "0 <= min_torsional_patch_radius <= "
                "torsional_patch_radius."
            )
        self.arm_stiffness_scale = float(arm_stiffness_scale)
        self.gripper_stiffness = float(gripper_stiffness)
        self.gripper_damping = float(gripper_damping)
        self.gripper_effort_limit = float(gripper_effort_limit)
        self.gripper_velocity_limit = float(gripper_velocity_limit)
        if min(
            self.arm_stiffness_scale,
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_effort_limit,
            self.gripper_velocity_limit,
        ) <= 0.0:
            raise ValueError("Arm and gripper actuator settings must be positive.")
        
        self.sim_cfg = sim_utils.SimulationCfg(
            dt=self.physics_dt,
            render=sim_utils.RenderCfg(
                rendering_mode=None,
                antialiasing_mode="FXAA",
                carb_settings={
                    "/rtx/rendermode": "PathTracing",
                },
            ),
        )
        self.sim = None
        self.scene = None

        self.camera = camera
        self.initialized = False
        self.object_names = []
        self._active_transforms_path = None
        self._post_grasp_transforms_path = None
        self._post_grasp_object_name = None
        self._post_grasp_scene_state = None
        self._post_grasp_info = None
        self._robot_link_body_indices = {}
        self._robot_contact_sensor_keys = []
        self._object_contact_sensor_keys = {}
        self._contact_filter_objects = {}
        self._forbidden_contact_force_accumulator = None


    @staticmethod
    def _clone_state(value):
        """Deep-clone a nested Isaac Lab state without sharing tensor storage."""
        if torch.is_tensor(value):
            return value.clone()
        if isinstance(value, dict):
            return {key: IsaacSim._clone_state(item) for key, item in value.items()}
        if isinstance(value, list):
            return [IsaacSim._clone_state(item) for item in value]
        if isinstance(value, tuple):
            return tuple(IsaacSim._clone_state(item) for item in value)
        return copy.deepcopy(value)


    def _clear_post_grasp_state(self):
        self._post_grasp_transforms_path = None
        self._post_grasp_object_name = None
        self._post_grasp_scene_state = None
        self._post_grasp_info = None


    def cache_post_grasp_state(self, object_name, grasp_info):
        """Capture the complete scene immediately after a successful grasp."""
        if not hasattr(self.scene, "get_state"):
            raise RuntimeError(
                "This Isaac Lab version does not provide InteractiveScene.get_state()."
            )

        state = self.scene.get_state(is_relative=False)
        self._post_grasp_scene_state = self._clone_state(state)
        self._post_grasp_transforms_path = self._active_transforms_path
        self._post_grasp_object_name = object_name
        self._post_grasp_info = copy.deepcopy(grasp_info)
        print(
            f"[IsaacSim] Cached full post-grasp scene state for '{object_name}'.",
            flush=True,
        )


    def restore_post_grasp_state(self, object_name):
        """Restore a previously captured post-grasp scene without replaying the grasp."""
        cache_matches = (
            self._post_grasp_scene_state is not None
            and self._post_grasp_object_name == object_name
            and self._post_grasp_transforms_path == self._active_transforms_path
        )
        if not cache_matches:
            return None
        if not hasattr(self.scene, "reset_to"):
            raise RuntimeError(
                "This Isaac Lab version does not provide InteractiveScene.reset_to()."
            )

        # reset_to writes root/joint poses, velocities, and PD targets for
        # every environment.
        self.scene.reset_to(
            self._clone_state(self._post_grasp_scene_state),
            env_ids=None,
            is_relative=False,
        )

        if hasattr(self, "robot_controller"):
            self.robot_controller.set_gripper(0.0)
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(self.physics_dt)

        print(
            f"[IsaacSim] Restored cached post-grasp scene state for '{object_name}'.",
            flush=True,
        )
        return copy.deepcopy(self._post_grasp_info)


    def _get_robot_link_positions(self, link_names):
        """Return requested articulation-link origins in the robot-base frame."""
        robot = self.robot_controller.robot
        result = {}
        for link_name in link_names:
            if link_name not in self._robot_link_body_indices:
                indices = robot.find_bodies(link_name)[0]
                if len(indices) != 1:
                    raise ValueError(
                        f"Expected exactly one robot body named '{link_name}', "
                        f"found {len(indices)}."
                    )
                self._robot_link_body_indices[link_name] = int(indices[0])

            body_idx = self._robot_link_body_indices[link_name]
            positions_w = as_tensor(robot.data.body_pos_w[:, body_idx])
            # Orientation is not needed for capsules defined between link origins.
            dummy_quaternions = as_tensor(robot.data.body_quat_w[:, body_idx])
            positions_b, _ = self._to_robot_frame(
                positions_w,
                dummy_quaternions,
            )
            result[link_name] = (
                positions_b[0] if self.num_envs == 1 else positions_b
            )
        return result


    def _forbidden_contact_forces(
        self,
        target_object,
        allowed_target_contacts=None,
        *,
        include_target_contacts=True,
    ):
        """Return one maximum forbidden PhysX contact force per environment."""
        allowed_target_contacts = set(allowed_target_contacts or ())
        maximum = torch.zeros(self.num_envs, device=self.sim.device)

        target_filter_index = None
        ordered_objects = list(self.object_name_to_slot)
        if target_object in ordered_objects:
            target_filter_index = ordered_objects.index(target_object)

        # Robot contact with the grasped target is intentional. Every other
        # robot-object pair remains forbidden, including the TF reference.
        for sensor_key in self._robot_contact_sensor_keys:
            matrix = self.scene[sensor_key].data.force_matrix_w
            if matrix is None:
                continue
            forces = as_tensor(matrix)
            if forces.ndim == 4:
                forces = forces[:, 0]
            magnitudes = torch.linalg.vector_norm(forces, dim=-1)
            if target_filter_index is not None:
                magnitudes = magnitudes.clone()
                magnitudes[:, target_filter_index] = 0.0
            maximum = torch.maximum(maximum, torch.amax(magnitudes, dim=1))

        # The grasped target may contact its TF reference objects. Contacts
        # with every other reconstructed object are forbidden.
        sensor_key = (
            self._object_contact_sensor_keys.get(target_object)
            if include_target_contacts else None
        )
        if sensor_key is not None:
            matrix = self.scene[sensor_key].data.force_matrix_w
            if matrix is not None:
                forces = as_tensor(matrix)
                if forces.ndim == 4:
                    forces = forces[:, 0]
                magnitudes = torch.linalg.vector_norm(forces, dim=-1)
                filter_names = self._contact_filter_objects[target_object]
                magnitudes = magnitudes.clone()
                for index, name in enumerate(filter_names):
                    if name in allowed_target_contacts:
                        magnitudes[:, index] = 0.0
                maximum = torch.maximum(
                    maximum,
                    torch.amax(magnitudes, dim=1),
                )

        return maximum


    def initialize_scene_once(self, transforms_json_path, mesh_dir=None):
        """
        Build the static scene and USD assets exactly once.
        """
        if getattr(self, "initialized", False):
            return

        transforms_json_path = Path(transforms_json_path)
        if mesh_dir is None:
            mesh_dir = transforms_json_path.parent
        else:
            mesh_dir = Path(mesh_dir)

        print(f"[IsaacSim] Building scene once from {transforms_json_path}...")

        if not transforms_json_path.exists():
            raise FileNotFoundError(f"Transforms file not found: {transforms_json_path}")
            
        with open(transforms_json_path, "r") as f:
            transforms_data = json.load(f)
            
        object_specs = []
        for i, obj_entry in enumerate(transforms_data.get("objects", [])):
            name = obj_entry.get("name")
            
            # Select GLB or OBJ mesh file
            mesh_file = mesh_dir / obj_entry.get("glb")
            obj_file = mesh_dir / (Path(obj_entry.get("glb")).stem + ".obj")
            if obj_file.exists():
                mesh_file = obj_file
                
            scale = obj_entry.get("scale", [1.0, 1.0, 1.0])
            if isinstance(scale, (list, tuple)):
                uniform_scale = scale[0]
            else:
                uniform_scale = float(scale)
                
            translation = obj_entry.get("translation", [0.0, 0.0, 0.0])
            rotation_wxyz = obj_entry.get("rotation_wxyz")
            if rotation_wxyz is None:
                rotation_wxyz = obj_entry.get("rotation", [1.0, 0.0, 0.0, 0.0])

            w, x, y, z = map(float, rotation_wxyz)
            rotation_xyzw = (x, y, z, w)

            is_robot_frame = "rotation_wxyz" in obj_entry

            if is_robot_frame:
                env_pos = (self.robot_base_pos[0] + translation[0],
                           self.robot_base_pos[1] + translation[1],
                           self.robot_base_pos[2] + translation[2])
                env_rot = rotation_xyzw
            else:
                raise NotImplementedError("Camera-frame to environment-frame conversion not implemented in this version.")

            # Convert OBJ/GLB mesh to USD
            usd_path = convert_mesh(
                mesh_path=str(mesh_file),
                scale=uniform_scale,
                out_name=name,
                usd_dir=self.usd_dir,
                mass_kg=self.object_mass_kg,
                collision_approx=self.collision_approx,
                static_friction=self.static_friction,
                dynamic_friction=self.dynamic_friction,
                torsional_patch_radius=self.torsional_patch_radius,
                min_torsional_patch_radius=(
                    self.min_torsional_patch_radius
                ),
            )
            
            # Decorative scene objects may be fixed while retaining collisions.
            kinematic_enabled = obj_entry.get("kinematic_enabled", False)
            if not isinstance(kinematic_enabled, bool):
                raise ValueError(f"kinematic_enabled for {name} must be a boolean")
            object_specs.append({
                "kinematic_enabled": kinematic_enabled,
                "name": name,
                "usd_path": usd_path,
                "env_pos": env_pos,
                "env_rot": env_rot
            })
            
        if self.gripper_variant == "umi":
            robot_usd_path = (
                self.umi_asset_dir
                / "franka_umi_fingers.usda"
            )
            if not robot_usd_path.exists():
                raise FileNotFoundError(
                    f"UMI Panda asset not found at {robot_usd_path}. "
                    "Set umi_asset_dir to the directory containing the UMI USD and build_summary.json."
                )
            robot_usd_path = str(robot_usd_path)
        else:
            robot_usd_path = (
                f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/"
                "FrankaPanda/franka.usd"
            )
        print(
            f"[IsaacSim] Franka gripper variant: {self.gripper_variant} "
            f"({robot_usd_path})",
            flush=True,
        )

        # Dynamically build scene configuration
        @configclass
        class DynamicSceneCfg(InteractiveSceneCfg):
            ground = AssetBaseCfg(
                prim_path="/World/GroundPlane",
                spawn=sim_utils.GroundPlaneCfg(size=(50.0, 50.0)),
            )
            robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
                prim_path="{ENV_REGEX_NS}/Robot",
            )
            robot.spawn.usd_path = robot_usd_path
            robot.spawn.activate_contact_sensors = True

        scene_cfg = DynamicSceneCfg(num_envs=self.num_envs, env_spacing=self.env_spacing)
        from caster.sim_env.sim_utils import add_table_to_scene_cfg
        add_table_to_scene_cfg(scene_cfg, self.table_translation, self.table)

        # Cover the centered environment grid, including the offset tabletop
        # in the outermost rows. Keep five metres of floor beyond each edge.
        grid_side = int(np.ceil(np.sqrt(self.num_envs)))
        grid_span = (grid_side - 1) * self.env_spacing
        table_size = scene_cfg.table.spawn.size
        scene_cfg.ground.spawn.size = tuple(
            max(50.0, grid_span + 2 * abs(self.table_translation[axis])
                + table_size[axis] + 10.0)
            for axis in (0, 1)
        )

        if self.gripper_variant == "umi":
            # The longer rigid fingers need stronger articulation solving for
            # reliable convergence across all batched environments. Both the
            # stock 8/0 and intermediate 12/2 settings failed repeated grasps.
            scene_cfg.robot.spawn.articulation_props.solver_position_iteration_count = 16
            scene_cfg.robot.spawn.articulation_props.solver_velocity_iteration_count = 4

        scene_cfg.dome_light = AssetBaseCfg(
            prim_path="/World/DomeLight",
            spawn=sim_utils.DomeLightCfg(
                intensity=1500.0,
                color=(1.0, 1.0, 1.0),
            ),
        )

        scene_cfg.sun = AssetBaseCfg(
            prim_path="/World/Sun",
            spawn=sim_utils.DistantLightCfg(
                intensity=4000.0,
                color=(1.0, 0.98, 0.95),
                angle=0.0,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                # -45° around X, WXYZ
                rot=(0.9238795, -0.3826834, 0.0, 0.0),
            ),
        )
        
        # Preserve high-gain arm tracking without multiplying the hand
        # stiffness and leaving its damping unchanged.  That previously turned
        # Panda hand gains from (2000, 100) into (8000, 100), which makes
        # fingertip contact underdamped and prone to object oscillation.
        for name in ("panda_shoulder", "panda_forearm"):
            actuator = scene_cfg.robot.actuators[name]
            if isinstance(actuator.stiffness, dict):
                for key in actuator.stiffness:
                    actuator.stiffness[key] *= self.arm_stiffness_scale
            else:
                actuator.stiffness *= self.arm_stiffness_scale
        hand_actuator = scene_cfg.robot.actuators["panda_hand"]
        if self.gripper_variant == "umi":
            # NVIDIA's panda_finger_joint2 is already a PhysX mimic of joint1.
            # Driving it independently over-constrains the longer UMI finger
            # and can destabilize the mimic constraint under grasp load.
            hand_actuator.joint_names_expr = ["panda_finger_joint1"]
        hand_actuator.stiffness = self.gripper_stiffness
        hand_actuator.damping = self.gripper_damping
        hand_actuator.effort_limit_sim = self.gripper_effort_limit
        hand_actuator.velocity_limit_sim = self.gripper_velocity_limit
        print(
            "[IsaacSim] Actuator gains: "
            f"arm stiffness scale={self.arm_stiffness_scale:.2f}, "
            f"hand stiffness={self.gripper_stiffness:.1f}, "
            f"hand damping={self.gripper_damping:.1f}, "
            f"hand effort limit={self.gripper_effort_limit:.1f}, "
            f"hand velocity limit={self.gripper_velocity_limit:.3f}, "
            "torsional patch radius="
            f"{self.torsional_patch_radius:.4f}m, "
            "minimum torsional patch radius="
            f"{self.min_torsional_patch_radius:.4f}m",
            flush=True,
        )

        scene_cfg.robot.init_state.pos = self.robot_base_pos

        # Inject camera dynamically to DynamicSceneCfg
        if self.camera is not None and self.camera.enabled:
            self.camera.add_to_scene_cfg(scene_cfg)

        # Dynamically attach each object configuration
        from isaaclab.assets import RigidObjectCfg
        from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, CollisionPropertiesCfg, MassPropertiesCfg
        
        self.object_names = []
        self.object_name_to_slot = {}
        for i, spec in enumerate(object_specs):
            slot = f"obj_{i}"
            self.object_names.append(slot)
            self.object_name_to_slot[spec["name"]] = slot
            setattr(scene_cfg, slot, RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + slot,
                spawn=sim_utils.UsdFileCfg(
                    usd_path=spec["usd_path"],
                    activate_contact_sensors=True,
                    rigid_props=RigidBodyPropertiesCfg(
                        rigid_body_enabled=True,
                        kinematic_enabled=spec["kinematic_enabled"],
                        disable_gravity=spec["kinematic_enabled"],
                    ),
                    collision_props=CollisionPropertiesCfg(
                        collision_enabled=True,
                        torsional_patch_radius=(
                            self.torsional_patch_radius
                        ),
                        min_torsional_patch_radius=(
                            self.min_torsional_patch_radius
                        ),
                    ),
                    mass_props=MassPropertiesCfg(mass=self.object_mass_kg),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=spec["env_pos"], rot=spec["env_rot"]),
            ))

        # Contact sensors run inside PhysX and expose only compact force
        # matrices. One sensor per robot link avoids unsupported many-to-many
        # filtering while still distinguishing every scene object.
        object_filter_paths = [
            "{ENV_REGEX_NS}/" + self.object_name_to_slot[name]
            for name in self.object_name_to_slot
        ]
        contact_robot_links = [
            "panda_link0", "panda_link1", "panda_link2", "panda_link3",
            "panda_link4", "panda_link5", "panda_link6", "panda_link7",
            "panda_hand", "panda_leftfinger", "panda_rightfinger",
        ]
        self._robot_contact_sensor_keys = []
        for index, link_name in enumerate(contact_robot_links):
            sensor_key = f"robot_contact_{index}"
            setattr(
                scene_cfg,
                sensor_key,
                ContactSensorCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/{link_name}",
                    update_period=0.0,
                    history_length=0,
                    debug_vis=False,
                    filter_prim_paths_expr=object_filter_paths,
                ),
            )
            self._robot_contact_sensor_keys.append(sensor_key)

        self._object_contact_sensor_keys = {}
        self._contact_filter_objects = {}
        ordered_object_names = list(self.object_name_to_slot)
        for index, object_name in enumerate(ordered_object_names):
            sensor_key = f"object_contact_{index}"
            other_names = [
                name for name in ordered_object_names if name != object_name
            ]
            setattr(
                scene_cfg,
                sensor_key,
                ContactSensorCfg(
                    prim_path=(
                        "{ENV_REGEX_NS}/" + self.object_name_to_slot[object_name]
                    ),
                    update_period=0.0,
                    history_length=0,
                    debug_vis=False,
                    filter_prim_paths_expr=[
                        "{ENV_REGEX_NS}/" + self.object_name_to_slot[name]
                        for name in other_names
                    ],
                ),
            )
            self._object_contact_sensor_keys[object_name] = sensor_key
            self._contact_filter_objects[object_name] = other_names

        # Instantiate simulation context and scene
        if SimulationContext.instance() is None:
            self.sim = SimulationContext(self.sim_cfg)
        else:
            self.sim = SimulationContext.instance()

        import carb

        self.scene = InteractiveScene(scene_cfg)

        if self.camera is not None and self.camera.enabled:
            self.camera.resolve_sensor(self.scene)

        self.sim.reset()

        # Renderer initialization/reset can overwrite RTX settings,
        # so apply the production configuration after reset.
        settings = carb.settings.get_settings()

        settings.set("/rtx/rendermode", "PathTracing")
        settings.set("/rtx/pathtracing/spp", 8)
        settings.set("/rtx/pathtracing/totalSpp", 8)
        settings.set("/rtx/pathtracing/maxBounces", 2)
        settings.set(
            "/rtx/pathtracing/maxSpecularAndTransmissionBounces",
            1,
        )

        settings.set("/rtx/pathtracing/fireflyFilter/enabled", True)
        settings.set(
            "/rtx/pathtracing/fireflyFilter/maxIntensityPerSample",
            500.0,
        )
        settings.set(
            "/rtx/pathtracing/fireflyFilter/maxIntensityPerSampleDiffuse",
            500.0,
        )

        settings.set("/rtx/pathtracing/optixDenoiser/enabled", True)
        settings.set(
            "/rtx/pathtracing/optixDenoiser/temporalMode/enabled",
            False,
        )
        settings.set("/rtx/pathtracing/optixDenoiser/blendFactor", 0.0)

        self.robot_controller = Panda(
            self.scene,
            tcp_offset_m=self.tcp_offset_m,
        )

        # Settle physics without expensive rendering during warm-up
        for _ in range(20):
            self.sim.step(render=False)
            self.scene.update(self.physics_dt)

        # Populate the initial camera/render buffer
        if self.camera is not None and self.camera.enabled:
            self.sim.render()


        self.initialized = True
        print(f"[IsaacSim] Built scene: Franka Panda + table + {len(object_specs)} dynamic objects.", flush=True)

    def _to_robot_frame(self, pos_w, quat_w):
        """Return robot-frame positions and WXYZ quaternions."""
        pos_w = as_tensor(pos_w)
        quat_w = as_tensor(quat_w)
        device_str = str(pos_w.device)
        origins = self.scene.env_origins.to(device=device_str)
        robot_base = torch.tensor(self.robot_base_pos, dtype=torch.float32, device=device_str)
        
        if pos_w.ndim == 2:
            pos_b = pos_w - (origins + robot_base.unsqueeze(0))
            # Isaac Lab stores quaternions as XYZW; saved trajectories
            # use WXYZ.
            quat_b = quat_w[:, [3, 0, 1, 2]]
            return pos_b.cpu().numpy().tolist(), quat_b.cpu().numpy().tolist()
        else:
            pos_b = pos_w - (origins[0] + robot_base)
            quat_b = quat_w[[3, 0, 1, 2]]
            return pos_b.cpu().numpy().tolist(), quat_b.cpu().numpy().tolist()

    def reset_from_transforms(self, transforms_json_path, video_path=None, mesh_dir=None, render=True):
        """
        Reset robot and object poses for a new generation loop using the pre-built scene.
        """
        self.enable_render = render
        print(f"[IsaacSim] Teleporting objects to new poses from {transforms_json_path}...")
        transforms_json_path = Path(transforms_json_path)
        normalized_transforms_path = str(transforms_json_path.resolve())
        self._forbidden_contact_force_accumulator = None
        if (
            self._post_grasp_transforms_path is not None
            and self._post_grasp_transforms_path != normalized_transforms_path
        ):
            self._clear_post_grasp_state()
        self._active_transforms_path = normalized_transforms_path
        
        with open(transforms_json_path, "r") as f:
            transforms_data = json.load(f)

        self.sim.reset()
        
        # Reset Franka pose
        if hasattr(self, "robot_controller"):
            self.robot_controller.reset()

        # Reset object poses
        for i, obj_entry in enumerate(transforms_data.get("objects", [])):
            if i >= len(self.object_names):
                break
                
            obj_name = self.object_names[i]

            translation = obj_entry.get("translation", [0.0, 0.0, 0.0])
            rotation_wxyz = obj_entry.get("rotation_wxyz")
            if rotation_wxyz is None:
                rotation_wxyz = obj_entry.get("rotation", [1.0, 0.0, 0.0, 0.0])

            w, x, y, z = map(float, rotation_wxyz)
            rotation_xyzw = (x, y, z, w)

            is_robot_frame = "rotation_wxyz" in obj_entry

            if is_robot_frame:
                env_pos = (self.robot_base_pos[0] + translation[0],
                            self.robot_base_pos[1] + translation[1],
                            self.robot_base_pos[2] + translation[2])
                env_rot = rotation_xyzw
            else:
                raise NotImplementedError("Camera-frame to environment-frame conversion not implemented in this version.")
            
            rigid_obj = self.scene[obj_name]
            root_state = as_tensor(rigid_obj.data.default_root_state).clone()
            dev_str = str(root_state.device)
            origins = self.scene.env_origins.to(device=dev_str)
            pos_offset = torch.tensor(env_pos, dtype=torch.float32, device=dev_str)
            root_state[:, 0:3] = origins + pos_offset
            root_state[:, 3:7] = torch.tensor(env_rot, dtype=torch.float32, device=dev_str)
            root_state[:, 7:13] = 0.0
            rigid_obj.write_root_state_to_sim(root_state)

        # Write state to physics engine
        self.scene.write_data_to_sim()
        
        # Settle the physics engine
        for _ in range(20 * self.decimation):
            self.sim.step(render=False)
            self.scene.update(self.physics_dt)

        initial_poses = {}
        slot_to_name = {v: k for k, v in getattr(self, "object_name_to_slot", {}).items()}
        for obj_name in self.object_names:
            if obj_name in self.scene.keys():
                rigid_obj = self.scene[obj_name]
                actual_name = slot_to_name.get(obj_name, obj_name)
                pos_list, quat_list = self._to_robot_frame(rigid_obj.data.root_pos_w, rigid_obj.data.root_quat_w)
                if self.num_envs == 1:
                    pos_list = pos_list[0]
                    quat_list = quat_list[0]
                initial_poses[actual_name] = {
                    "position": pos_list,
                    "orientation": quat_list
                }
        return initial_poses

    def reset_video(self, video_path):
        """Clear camera buffers and update the output path."""
        if self.camera is not None and self.camera.enabled:
            self.camera.frames = []
            if video_path:
                self.camera.save_dir = os.path.dirname(video_path)
                self.camera.save_path = Path(self.camera.save_dir) / os.path.basename(video_path)

    def step(
        self,
        action=None,
        frame_idx=0,
        record=True,
        return_positions=False,
        return_robot_links=None,
        contact_query=None,
    ):
        """
        Step physics and update the scene states.
        """
        if not getattr(self, "initialized", False):
            raise RuntimeError("Environment not constructed! Call initialize_scene_once() first.")
            
        if hasattr(self, "robot_controller") and action is not None:
            controller_action = np.asarray(action, dtype=np.float32)
            if controller_action.shape[-1] >= 7:
                controller_action = controller_action.copy()
                # Optimizer trajectories are WXYZ; Isaac Lab's controller
                # requires XYZW. Internal grasp commands bypass this method
                # and therefore remain in their native Isaac XYZW convention.
                controller_action[..., 3:7] = np.asarray(action)[
                    ..., [4, 5, 6, 3]
                ]
            self.robot_controller.step(controller_action)
            
        # Flush the new joint targets to the PhysX backend
        self.scene.write_data_to_sim()
        
        # Only render if globally enabled for this sequence AND requested for this step
        should_render = getattr(self, "enable_render", True) and record
        
        # Step the physics simulator decimation times without rendering during physics update
        for _ in range(self.decimation):
            self.sim.step(render=False)
            self.scene.update(self.physics_dt)
            if contact_query:
                current_forces = self._forbidden_contact_forces(
                    contact_query.get("target_object"),
                    contact_query.get("allowed_target_contacts", []),
                    include_target_contacts=contact_query.get("include_target_contacts", True),
                )
                if self._forbidden_contact_force_accumulator is None:
                    self._forbidden_contact_force_accumulator = current_forces
                else:
                    self._forbidden_contact_force_accumulator = torch.maximum(
                        self._forbidden_contact_force_accumulator,
                        current_forces,
                    )
        
        if should_render and self.camera is not None and self.camera.enabled:
            self.sim.render()
            self.camera.record_frame(frame_idx, force=True)

        # Get tracking information after step
        info = {}
        if hasattr(self, "robot_controller"):
            actual_tcp = self.robot_controller.get_tcp_pos()
            info["actual_tcp"] = actual_tcp
            positions, orientations = self._actual_tcp_pose()
            poses = np.concatenate((positions, orientations[:, [3, 0, 1, 2]]), axis=1)
            info["actual_tcp_pose"] = poses[0].tolist() if self.num_envs == 1 else poses.tolist()
            joint_positions = (
                self.robot_controller.robot.data.joint_pos[:, :7]
                .detach()
                .cpu()
                .tolist()
            )
            info["joint_positions"] = (
                joint_positions[0]
                if self.num_envs == 1
                else joint_positions
            )
            if action is not None:
                action_np = np.array(action)
                if action_np.ndim == 2:
                    target_tcp = action_np[0, :3]
                else:
                    target_tcp = action_np[:3]
                error = np.linalg.norm(target_tcp - np.array(actual_tcp))
                info["error"] = float(error)

        if return_positions:
            object_poses = {}
            slot_to_name = {v: k for k, v in getattr(self, "object_name_to_slot", {}).items()}
            for obj_name in self.object_names:
                if obj_name in self.scene.keys():
                    rigid_obj = self.scene[obj_name]
                    actual_name = slot_to_name.get(obj_name, obj_name)
                    pos_list, quat_list = self._to_robot_frame(rigid_obj.data.root_pos_w, rigid_obj.data.root_quat_w)
                    if self.num_envs == 1:
                        pos_list = pos_list[0]
                        quat_list = quat_list[0]
                    object_poses[actual_name] = {
                        "position": pos_list,
                        "orientation": quat_list
                    }
            info["object_poses"] = object_poses
        if return_robot_links:
            info["robot_link_positions"] = self._get_robot_link_positions(
                return_robot_links
            )
        if contact_query:
            if contact_query.get("return_result", False):
                values = (
                    self._forbidden_contact_force_accumulator
                    .detach()
                    .cpu()
                    .tolist()
                )
                info["max_forbidden_contact_force"] = (
                    values[0] if self.num_envs == 1 else values
                )
        return info



    def save_video(self):
        """Compile and save recorded frames."""
        if getattr(self, "enable_render", True) and self.camera is not None and self.camera.enabled:
            self.camera.save_video()
