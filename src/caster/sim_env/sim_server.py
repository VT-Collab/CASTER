"""Launch Isaac Lab and serve scene reset, grasp, step, and video requests."""

import argparse
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


class SimulationSession:
    def __init__(self, config, headless=True):
        self.config = config
        self.headless = headless
        self.sim = None
        self.scene_path = None

    def reset(self, transforms_path, video_path, render=True):
        transforms = Path(transforms_path).resolve()
        video = Path(video_path).resolve()
        if not transforms.is_file():
            raise FileNotFoundError(f"Transforms not visible to the server: {transforms}")
        if self.scene_path is not None and transforms != self.scene_path:
            raise ValueError("Restart the simulation server before loading a different scene")
        if self.sim is None:
            from caster.sim_env.camera import Camera
            from caster.sim_env.isaac import IsaacSim

            settings = dict(self.config)
            camera_settings = dict(settings.pop("camera"))
            camera_settings.update(save_dir=str(video.parent), video_filename=video.name)
            camera = Camera(**camera_settings)
            self.sim = IsaacSim(**settings, camera=camera, headless=self.headless)
            self.scene_path = transforms
        if not self.sim.initialized:
            self.sim.initialize_scene_once(transforms)
        poses = self.sim.reset_from_transforms(transforms, render=render)
        self.sim.reset_video(str(video))
        return {"status": "ready", "initial_poses": poses}

    def handle(self, message):
        command = message.get("command")
        if command == "health":
            return {"status": "ready", "service": "caster_simulation",
                    "initialized": self.sim is not None and self.sim.initialized}
        if command == "shutdown":
            return {"status": "shutdown"}
        if command == "reset":
            return self.reset(message["transforms_path"], message["video_path"],
                              message.get("render", True))
        if self.sim is None or not self.sim.initialized:
            raise RuntimeError("Reset the simulation before sending commands")
        if command == "step":
            info = self.sim.step(
                message.get("action"), message.get("frame_idx", 0),
                message.get("record", True),
                return_positions=message.get("return_positions", False),
                return_robot_links=message.get("return_robot_links", []),
                contact_query=message.get("contact_query"),
            )
            return {"status": "success", "info": info}
        if command == "restore_post_grasp":
            info = self.sim.restore_post_grasp_state(message["object_name"])
            if info is None:
                return {"status": "not_cached"}
            return {"status": "success", "info": info}
        if command == "execute_grasp":
            info = self.sim.execute_grasp(
                message["object_name"],
                hover_distance=message.get("hover_distance"),
                grasp_depth_offset=message.get("grasp_depth_offset"),
                grasp_height_offset=message.get("grasp_height_offset"),
            )
            return {"status": "success", "info": info}
        if command == "execute_release":
            self.sim.execute_release()
        elif command == "save_video":
            self.sim.save_video()
        else:
            raise ValueError(f"Unknown simulation command: {command}")
        return {"status": "success"}


def serve(session, application, address):
    import zmq

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    try:
        socket.bind(address)
        logger.info("Simulation server ready at %s", address)
        while application.is_running():
            if not socket.poll(1000, zmq.POLLIN):
                continue
            try:
                message = socket.recv_json()
                response = session.handle(message)
            except Exception as error:
                logger.exception("Simulation request failed")
                response = {"status": "error", "error": f"{type(error).__name__}: {error}"}
            socket.send_json(response)
            if response["status"] == "shutdown":
                break
    finally:
        socket.close(linger=0)
        context.term()


def main():
    from isaaclab.app import AppLauncher
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent.parent / "config/simulation.yaml")
    parser.add_argument("--camera-config", type=Path,
                        help="Camera settings to merge into the simulation configuration")
    parser.add_argument("--address", default="tcp://0.0.0.0:15555")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Override a simulation setting; repeat for multiple settings")
    AppLauncher.add_app_launcher_args(parser)
    arguments = parser.parse_args()
    config = OmegaConf.load(arguments.config)
    if arguments.camera_config:
        config.camera = OmegaConf.merge(config.camera, OmegaConf.load(arguments.camera_config))
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(arguments.set))
    settings = OmegaConf.to_container(config, resolve=True)
    if settings["num_envs"] < 1:
        raise ValueError("num_envs must be positive")
    arguments.enable_cameras = bool(settings["camera"]["enabled"])
    arguments.renderer = "PathTracing"
    arguments.kit_args = (
        f"{arguments.kit_args} --/persistent/rtx/modes/pt/enabled=true "
        "--/rtx/rendermode=PathTracing"
    ).strip()
    logging.basicConfig(level=logging.INFO)
    launcher = AppLauncher(arguments)
    try:
        # Isaac Sim 6.0 moved the tensors API; Isaac Lab may still use the old path.
        import sys
        import omni.physics.tensors as tensors
        import omni.physics.tensors.api as tensors_api
        tensors.impl = tensors
        tensors_api.api = tensors_api
        sys.modules.setdefault("omni.physics.tensors.impl", tensors)
        sys.modules.setdefault("omni.physics.tensors.impl.api", tensors_api)
        serve(SimulationSession(settings, arguments.headless), launcher.app, arguments.address)
    finally:
        launcher.app.close()


if __name__ == "__main__":
    main()
