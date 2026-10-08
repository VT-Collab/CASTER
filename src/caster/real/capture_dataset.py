"""Capture RGB-D scenes as RGB images, metric point clouds, and calibrated C2R transforms."""

import cv2
import zmq
import time
import json
import yaml
import hydra
import viser
import shutil
import threading
import numpy as np
from pathlib import Path
from omegaconf import DictConfig



REPO_ROOT = Path(__file__).resolve().parents[3]


class CameraClient:
    def __init__(self, address):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, 2500)
        self.socket.setsockopt(zmq.SNDTIMEO, 2500)
        self.socket.connect(address)

    def close(self):
        self.socket.close(linger=0)
        self.context.term()

    def capture(self):
        self.socket.send(b"capture")
        parts = self.socket.recv_multipart()
        header = json.loads(parts[0])
        if not header.get("ok"):
            raise RuntimeError(header.get("error", "camera capture failed"))
        if len(parts) != 3:
            raise RuntimeError(f"Expected 3 capture response parts, received {len(parts)}")
        bgr = cv2.imdecode(np.frombuffer(parts[1], dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("Could not decode camera JPEG")
        depth = np.frombuffer(parts[2], dtype=np.dtype(header["depth_dtype"]))
        depth = depth.reshape(header["depth_shape"])
        if depth.dtype != np.uint16:
            raise ValueError(f"Expected uint16 depth, received {depth.dtype}")
        if bgr.shape[:2] != tuple(header["color_shape"]):
            raise ValueError("Decoded RGB shape does not match the camera response header")
        return bgr, depth.copy(), header


def next_scene_assets_dir(dataset_root):
    dataset_root.mkdir(parents=True, exist_ok=True)
    numbers = [
        int(path.name.removeprefix("scene_"))
        for path in dataset_root.glob("scene_*")
        if path.is_dir() and path.name.removeprefix("scene_").isdigit()
    ]
    assets = dataset_root / f"scene_{max(numbers, default=0) + 1}" / "assets"
    assets.mkdir(parents=True)
    return assets


def load_intrinsics(path):
    with path.open() as file:
        data = yaml.safe_load(file)
    fx, fy, cx, cy = (float(data[key]) for key in ("fx", "fy", "cx", "cy"))
    width, height = int(data["width"]), int(data["height"])
    if not np.isfinite([fx, fy, cx, cy]).all() or fx <= 0 or fy <= 0:
        raise ValueError(f"Invalid focal lengths in {path}")
    return fx, fy, cx, cy, width, height


def depth_to_ply_bytes(depth, fx, fy, cx, cy, depth_scale_mm=1.0):
    depth_m = np.asarray(depth, dtype=np.float32) * depth_scale_mm / 1000.0
    rows, cols = np.indices(depth_m.shape)
    valid = depth_m > 0
    z = depth_m[valid]
    x = (cols[valid].astype(np.float32) - cx) * z / fx
    y = (rows[valid].astype(np.float32) - cy) * z / fy
    points = np.column_stack((x, y, z)).astype("<f4", copy=False)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\nend_header\n"
    )
    return header.encode("ascii") + points.tobytes()


def save_capture(bgr, depth, assets, intrinsics, c2r_path, depth_scale_mm=1.0):
    fx, fy, cx, cy, width, height = intrinsics
    if bgr.shape[:2] != (height, width) or depth.shape != (height, width):
        raise ValueError(
            f"Frame resolution differs from intrinsics: RGB={bgr.shape[:2]}, "
            f"depth={depth.shape}, expected={(height, width)}"
        )
    c2r = np.load(c2r_path, allow_pickle=False)
    if c2r.shape != (4, 4) or not np.isfinite(c2r).all():
        raise ValueError(f"C2R must be a finite 4x4 matrix: {c2r_path}")
    if not cv2.imwrite(str(assets / "rgb.png"), bgr):
        raise RuntimeError("Could not write rgb.png")
    (assets / "pointcloud.ply").write_bytes(
        depth_to_ply_bytes(depth, fx, fy, cx, cy, depth_scale_mm)
    )
    shutil.copyfile(c2r_path, assets / "C2R.npy")


@hydra.main(version_base=None, config_path="config", config_name="capture")
def main(config: DictConfig):
    dataset_root = REPO_ROOT / Path(config.out).expanduser()
    intrinsics = load_intrinsics(REPO_ROOT / "src/caster/config/camera_intrinsics.yaml")
    c2r_path = REPO_ROOT / Path(config.c2r).expanduser()
    if not c2r_path.is_file():
        raise FileNotFoundError(f"C2R calibration not found: {c2r_path}")

    client = CameraClient(config.connect)
    server = None
    capture_requested = threading.Event()
    stop_requested = threading.Event()
    try:
        server = viser.ViserServer(port=config.preview_port, label="Scene Capture")
        server.gui.add_button("Capture scene").on_click(lambda _: capture_requested.set())
        server.gui.add_button("Quit").on_click(lambda _: stop_requested.set())
        print(f"Open http://localhost:{server.get_port()} and click Capture scene; Ctrl+C also quits.")
        preview = None
        while not stop_requested.is_set():
            bgr, depth, header = client.capture()
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if preview is None:
                preview = server.gui.add_image(rgb, label="Current Scene", format="jpeg")
            else:
                preview.image = rgb
            if capture_requested.is_set():
                capture_requested.clear()
                assets = next_scene_assets_dir(dataset_root)
                save_capture(
                    bgr, depth, assets, intrinsics, c2r_path,
                    float(header.get("depth_scale_mm", 1.0)),
                )
                print(f"Captured {assets.parent}", flush=True)
            time.sleep(1.0 / 30.0)
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.stop()
        client.close()


if __name__ == "__main__":
    main()
