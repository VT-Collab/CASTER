"""Record the first simulation environment to an MP4 file."""

import os
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import isaaclab.sim as sim_utils


class Camera:
    def __init__(self,
                 prim_path="/World/envs/env_0/camera",
                 width=640,
                 height=480,
                 pos=(3.5, 0.0, 4.0),
                 rot=(0.65328, 0.27059, 0.27059, 0.65328),
                 data_types=("rgb",),
                 record_fps=30,
                 video_filename="sim_render.mp4",
                 save_dir="outputs/sim_videos",
                 record_frequency=2,
                 enabled=True):
        """
        Hydra-instantiable Camera sensor manager.
        """
        self.enabled = enabled
        self.prim_path = prim_path
        self.width = int(width)
        self.height = int(height)
        self.pos = tuple(pos)
        self.rot = tuple(rot)
        self.data_types = list(data_types)
        self.fps = int(record_fps)
        self.filename = video_filename
        self.record_frequency = int(record_frequency)
        
        self.save_dir = Path(save_dir).resolve()
        self.save_path = self.save_dir / self.filename
        
        self.frames = []
        self.camera_sensor = None

    def add_to_scene_cfg(self, scene_cfg):
        from isaaclab.sensors import CameraCfg

        # Configuration stores WXYZ; Isaac Lab 3.x expects XYZW.
        w, x, y, z = self.rot
        rot_xyzw = (x, y, z, w)

        sensor_cfg = CameraCfg(
            prim_path=self.prim_path,
            update_period=0.0,
            offset=CameraCfg.OffsetCfg(
                pos=self.pos,
                rot=rot_xyzw,
                convention="opengl",
            ),
            data_types=self.data_types,
            width=self.width,
            height=self.height,
            spawn=sim_utils.PinholeCameraCfg(),
        )

        setattr(scene_cfg, "camera", sensor_cfg)

    def resolve_sensor(self, scene):
        if "camera" in scene.keys():
            self.camera_sensor = scene["camera"]

    def record_frame(self, step_idx, force=False):
        if not self.enabled or self.camera_sensor is None:
            return

        if not force and step_idx % self.record_frequency != 0:
            return

        rgb = self.camera_sensor.data.output.get("rgb")
        if rgb is None:
            return

        # Isaac Lab 3.x output may be ProxyArray.
        if hasattr(rgb, "torch"):
            rgb = rgb.torch

        rgb_np = rgb[0].detach().cpu().numpy()

        if rgb_np.shape[-1] == 4:
            bgr_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGBA2BGR)
        else:
            bgr_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)



        self.frames.append(np.ascontiguousarray(bgr_np).copy())
        
    def save_video(self):
        """Compile BGR frames and save to file as MP4."""
        if not self.enabled or not self.frames:
            return
            
        os.makedirs(self.save_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        tmp_path = str(self.save_path).replace(".mp4", "_raw.mp4")
        out = cv2.VideoWriter(tmp_path, fourcc, self.fps, (self.width, self.height))
        
        for frame in self.frames:
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
            out.write(frame)
            
        out.release()
        
        try:
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            # Re-encode with ffmpeg and yuv420p pixel format to ensure VSCode compatibility
            subprocess.run([ffmpeg_exe, "-y", "-i", tmp_path, "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(self.save_path)], 
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.remove(tmp_path)
        except Exception as e:
            print(f"[Camera] ffmpeg re-encode failed: {e}. Keeping raw file.")
            if os.path.exists(tmp_path):
                os.rename(tmp_path, self.save_path)
                
        print(f"[Camera] Video recorded successfully and saved to: {self.save_path}")
