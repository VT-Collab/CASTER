"""Replay recovered scene trajectories in Viser.

Required Hydra override: scene_num.
"""

import json
import logging
import re
import threading
import time
from pathlib import Path

import cv2
import hydra
import numpy as np
import open3d as o3d
import viser
import yaml
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation


logger = logging.getLogger(__name__)
DEMO = re.compile(r"demo_(\d+)_seed_([^_]+)_(.+)")
COLORS = [(255, 205, 40), (60, 220, 255), (255, 90, 210), (120, 255, 100)]


def _cloud(depth: np.ndarray, rgb: np.ndarray, intrinsics: dict, c2r: np.ndarray, limit: int):
    height, width = depth.shape
    rgb = cv2.cvtColor(cv2.resize(rgb, (width, height)), cv2.COLOR_BGR2RGB)
    ys, xs = np.where(np.isfinite(depth) & (depth > 0))
    if len(xs) > limit:
        keep = np.linspace(0, len(xs) - 1, limit, dtype=int)
        xs, ys = xs[keep], ys[keep]
    scale_x = width / intrinsics["width"]
    scale_y = height / intrinsics["height"]
    z = depth[ys, xs]
    points = np.column_stack((
        (xs - intrinsics["cx"] * scale_x) * z / (intrinsics["fx"] * scale_x),
        (ys - intrinsics["cy"] * scale_y) * z / (intrinsics["fy"] * scale_y),
        z,
    ))
    points = points @ c2r[:3, :3].T + c2r[:3, 3]
    return points.astype(np.float32), rgb[ys, xs]


def _add_scene(server: viser.ViserServer, assets: Path, intrinsics: dict, c2r: np.ndarray, config: DictConfig):
    scene_cloud = o3d.io.read_point_cloud(str(assets / "pointcloud.ply"))
    points = np.asarray(scene_cloud.points)
    if len(points) > config.max_points:
        points = points[np.linspace(0, len(points) - 1, config.max_points, dtype=int)]
    points = points @ c2r[:3, :3].T + c2r[:3, 3]
    rgb = cv2.imread(str(assets / "rgb.png"))
    if rgb is not None:
        # The point cloud is camera-frame; project onto the scene RGB image.
        camera_points = (points - c2r[:3, 3]) @ c2r[:3, :3]
        z = camera_points[:, 2]
        u = np.rint(intrinsics["fx"] * camera_points[:, 0] / np.maximum(z, 1e-6) + intrinsics["cx"]).astype(int)
        v = np.rint(intrinsics["fy"] * camera_points[:, 1] / np.maximum(z, 1e-6) + intrinsics["cy"]).astype(int)
        colors = np.full((len(points), 3), 180, dtype=np.uint8)
        valid = (z > 0) & (u >= 0) & (u < rgb.shape[1]) & (v >= 0) & (v < rgb.shape[0])
        colors[valid] = rgb[v[valid], u[valid], ::-1]
    else:
        colors = np.full((len(points), 3), 180, dtype=np.uint8)
    server.scene.add_frame(
        "/robot_base", axes_length=config.robot_axes_length, axes_radius=config.robot_axes_radius,
    )
    server.scene.add_point_cloud(
        "/scene", points=points.astype(np.float32), colors=colors,
        point_size=config.point_size, point_shape="rounded",
    )

    transforms_path = assets / "meshes/transforms.json"
    if transforms_path.is_file():
        for item in json.loads(transforms_path.read_text()).get("objects", []):
            mesh_path = transforms_path.parent / f"{Path(item['glb']).stem}.obj"
            if not mesh_path.is_file():
                mesh_path = transforms_path.parent / item["glb"]
            mesh = o3d.io.read_triangle_mesh(str(mesh_path))
            rotation = Rotation.from_quat(np.asarray(item.get("rotation_wxyz", [1, 0, 0, 0]))[[1, 2, 3, 0]])
            rotation *= Rotation.from_euler("x", 90, degrees=True)
            q = rotation.as_quat()
            scale = item.get("scale", 1.0)
            if isinstance(scale, list):
                scale = scale[0]
            server.scene.add_mesh_simple(
                f"/meshes/{item['name']}", vertices=np.asarray(mesh.vertices),
                faces=np.asarray(mesh.triangles), color=(190, 190, 190), opacity=0.65,
                side="double", scale=scale,
                position=item.get("translation", [0, 0, 0]), wxyz=q[[3, 0, 1, 2]],
            )


def _replay(demo: Path, video: Path, tracks: list[Path], assets: Path, config: DictConfig,
            intrinsics: dict, c2r: np.ndarray, labels: list[str], index: int) -> str:
    depths = np.load(demo / "video_gen/aligned_depth.npy", mmap_mode="r")
    server = viser.ViserServer(port=config.port)
    stop = threading.Event()
    paused = threading.Event()
    selection = {"value": "stop"}
    requested = {"frame": None}
    worker = None
    capture = cv2.VideoCapture(str(video))
    try:
        _add_scene(server, assets, intrinsics, c2r, config)
        ok, first = capture.read()
        if not ok:
            raise ValueError(f"Cannot decode {video}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 24
        stride = max(1, round(fps / 24))
        cloud = server.scene.add_point_cloud(
            "/video", *_cloud(depths[0], first, intrinsics, c2r, config.max_points),
            point_size=config.point_size, point_shape="rounded",
        )
        layers = []
        for number, path in enumerate(tracks):
            with np.load(path) as data:
                points = data["tracks_robot"].astype(np.float32)
                indices = data["indices"].reshape(-1)
            lookup = {int(frame): step for step, frame in enumerate(indices)}
            name = path.parent.name
            handle = server.scene.add_point_cloud(
                f"/trajectories/{name}", points=np.empty((0, 3), dtype=np.float32),
                colors=COLORS[number % len(COLORS)],
                point_size=config.track_point_size, point_shape="rounded",
            )
            poses = {}
            trajectory = path.with_name("trajectory.json")
            if trajectory.is_file():
                for step in json.loads(trajectory.read_text()):
                    if step.get("valid") and step.get("rotation_valid"):
                        q = Rotation.from_matrix(step["rotation_robot"]).as_quat()
                        poses[int(step["frame"])] = (step["center_robot"], q[[3, 0, 1, 2]])
            frame_handle = server.scene.add_frame(
                f"/frames/{name}", axes_length=config.object_axes_length,
                axes_radius=config.object_axes_radius, visible=False,
            )
            layers.append((handle, points, lookup, frame_handle, poses))

        server.gui.add_markdown(f"### {demo.name}")
        previous = server.gui.add_button("Previous demo", disabled=index == 0)
        next_button = server.gui.add_button("Next demo", disabled=index == len(labels) - 1)
        close = server.gui.add_button("Close viewer")
        picker = server.gui.add_dropdown("Jump to demo", options=labels, initial_value=labels[index])
        pause = server.gui.add_button("Stop / resume replay")
        slider = server.gui.add_slider("Video frame", min=0, max=len(depths) - 1, step=1, initial_value=0, disabled=True)
        changing_slider = threading.Event()

        def navigate(action):
            selection["value"] = action
            stop.set()

        @previous.on_click
        def _previous(_): navigate("previous")

        @next_button.on_click
        def _next(_): navigate("next")

        @close.on_click
        def _close(_): navigate("stop")

        @picker.on_update
        def _pick(event):
            if event.target.value != labels[index]:
                navigate(f"demo:{event.target.value}")

        @pause.on_click
        def _pause(_):
            if paused.is_set():
                paused.clear()
                slider.disabled = True
            else:
                paused.set()
                slider.disabled = False

        @slider.on_update
        def _scrub(event):
            if paused.is_set() and not changing_slider.is_set():
                requested["frame"] = int(event.target.value)

        def show(frame: int, image: np.ndarray):
            cloud.points, cloud.colors = _cloud(depths[frame], image, intrinsics, c2r, config.max_points)
            for handle, points, lookup, frame_handle, poses in layers:
                step = lookup.get(frame)
                visible_points = points[step] if step is not None else np.empty((0, 3))
                handle.points = visible_points[np.isfinite(visible_points).all(axis=1)].astype(np.float32)
                pose = poses.get(frame)
                frame_handle.visible = pose is not None
                if pose:
                    frame_handle.position, frame_handle.wxyz = pose
            changing_slider.set()
            slider.value = frame
            changing_slider.clear()

        def play():
            frame = 0
            while not stop.is_set():
                wanted = requested["frame"]
                if wanted is not None:
                    requested["frame"] = None
                    frame = wanted
                if paused.is_set() and wanted is None:
                    stop.wait(0.05)
                    continue
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame * stride)
                ok, image = capture.read()
                if ok:
                    show(frame, image)
                frame = (frame + 1) % len(depths)
                if paused.is_set():
                    continue
                stop.wait(stride / fps)

        worker = threading.Thread(target=play, daemon=True)
        worker.start()
        logger.info("Viser replay: http://localhost:%s", server.get_port())
        while not stop.wait(0.25):
            pass
    except KeyboardInterrupt:
        return "stop"
    finally:
        stop.set()
        if worker:
            worker.join(timeout=2)
        capture.release()
        server.stop()
    return selection["value"]


def visualize_dataset(config: DictConfig) -> None:
    scene = Path(to_absolute_path(f"dataset/scene_{config.scene_num}"))
    assets = scene / "assets"
    intrinsics = yaml.safe_load(Path(to_absolute_path(config.camera_intrinsics)).read_text())
    c2r = np.load(assets / "C2R.npy")
    demos = []
    for demo in scene.glob("demo_*"):
        match = DEMO.fullmatch(demo.name)
        if match and int(match[1]) >= config.start_demo:
            filename = f"output_seed_{match[2]}.mp4"
            video = next((path for path in (
                scene / match[3] / filename,
                demo / "video_gen" / filename,
                demo / filename,
            ) if path.is_file()), None)
            depth = demo / "video_gen/aligned_depth.npy"
            if video and depth.is_file():
                tracks = sorted(scene.glob(f"cf_*_{match[3]}/demo_{match[1]}_seed_{match[2]}/*/dense_tracks.npz"))
                demos.append((int(match[1]), demo, video, tracks))
    demos.sort(key=lambda item: item[0])
    if not demos:
        raise FileNotFoundError(f"No completed demos found in {scene}")
    labels = [demo.name for _, demo, _, _ in demos]
    index = 0
    while 0 <= index < len(demos):
        _, demo, video, tracks = demos[index]
        action = _replay(demo, video, tracks, assets, config, intrinsics, c2r, labels, index)
        if action == "stop":
            break
        if action == "previous":
            index -= 1
        elif action.startswith("demo:"):
            index = labels.index(action.removeprefix("demo:"))
        else:
            index += 1
        time.sleep(0.25)


@hydra.main(version_base=None, config_path="config", config_name="visualize_dataset")
def main(config: DictConfig) -> None:
    visualize_dataset(config)


if __name__ == "__main__":
    main()
