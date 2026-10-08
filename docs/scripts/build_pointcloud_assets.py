#!/usr/bin/env python3
"""Build compact, browser-ready point-cloud replays from the research bundle."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


TRACK_COLORS = (
    (255, 205, 40),
    (60, 220, 255),
    (255, 90, 210),
    (120, 255, 100),
    (255, 135, 55),
    (175, 120, 255),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, default=Path("replay_depth_align_examples")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("static/resources/pointcloud")
    )
    parser.add_argument("--frame-step", type=int, default=4)
    parser.add_argument("--pixel-step", type=int, default=8)
    parser.add_argument("--scene-points", type=int, default=35_000)
    return parser.parse_args()


def load_ply_xyz(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header_lines: list[bytes] = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Missing PLY header terminator in {path}")
            header_lines.append(line)
            if line.strip() == b"end_header":
                break
        header = b"".join(header_lines).decode("ascii")
        if "format binary_little_endian 1.0" not in header:
            raise ValueError(f"Only binary little-endian PLY is supported: {path}")
        vertex_line = next(
            line for line in header.splitlines() if line.startswith("element vertex ")
        )
        vertex_count = int(vertex_line.split()[-1])
        properties = [
            line for line in header.splitlines() if line.startswith("property ")
        ]
        if properties[:3] != [
            "property float x",
            "property float y",
            "property float z",
        ]:
            raise ValueError(f"Unexpected PLY vertex layout in {path}")
        return np.fromfile(handle, dtype="<f4", count=vertex_count * 3).reshape(-1, 3)


def camera_to_robot(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def project_colors(
    points_camera: np.ndarray,
    image: np.ndarray,
    intrinsics: dict[str, float],
) -> np.ndarray:
    colors = np.full((len(points_camera), 3), (82, 135, 190), dtype=np.uint8)
    z = points_camera[:, 2]
    u = np.rint(intrinsics["fx"] * points_camera[:, 0] / z + intrinsics["cx"]).astype(int)
    v = np.rint(intrinsics["fy"] * points_camera[:, 1] / z + intrinsics["cy"]).astype(int)
    valid = (
        (z > 1e-8)
        & (u >= 0)
        & (u < image.shape[1])
        & (v >= 0)
        & (v < image.shape[0])
    )
    colors[valid] = image[v[valid], u[valid], :3]
    return colors


def read_selected_video_frames(
    path: Path, width: int, height: int, frame_step: int
) -> np.ndarray:
    select = f"select=not(mod(n\\,{frame_step})),scale={width}:{height}"
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-vf",
        select,
        "-vsync",
        "vfr",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    result = subprocess.run(command, check=True, capture_output=True)
    frame_bytes = width * height * 3
    if len(result.stdout) % frame_bytes:
        raise ValueError(f"Unexpected decoded video size for {path}")
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, height, width, 3)


def humanize_task(task: str) -> str:
    return task.replace("_", " ").title()


class BinaryBundle:
    def __init__(self) -> None:
        self.data = bytearray()

    def add(self, array: np.ndarray, *, align: int = 1) -> dict[str, int | str]:
        while len(self.data) % align:
            self.data.append(0)
        contiguous = np.ascontiguousarray(array)
        offset = len(self.data)
        payload = contiguous.tobytes()
        self.data.extend(payload)
        return {
            "offset": offset,
            "bytes": len(payload),
            "dtype": str(contiguous.dtype),
        }


def quantize(points: np.ndarray, minimum: np.ndarray, extent: np.ndarray) -> np.ndarray:
    normalized = np.clip((points - minimum) / extent, 0.0, 1.0)
    return np.rint(normalized * 65535.0).astype("<u2")


def build_example(
    source: Path,
    output: Path,
    example: dict,
    intrinsics: dict[str, float],
    frame_step: int,
    pixel_step: int,
    scene_point_limit: int,
) -> dict:
    identifier = f"scene-{example['scene']}-demo-{example['demo']}"
    files = example["files"]
    transform = np.load(source / files["camera_to_robot"]).astype(np.float32)

    ply_camera_all = load_ply_xyz(source / files["pointcloud"])
    finite = np.isfinite(ply_camera_all).all(axis=1) & (ply_camera_all[:, 2] > 0)
    ply_camera_all = ply_camera_all[finite]
    ply_robot_all = camera_to_robot(ply_camera_all, transform)
    lower = np.percentile(ply_robot_all, 0.1, axis=0)
    upper = np.percentile(ply_robot_all, 99.9, axis=0)
    workspace = np.all((ply_robot_all >= lower) & (ply_robot_all <= upper), axis=1)
    candidate_indices = np.flatnonzero(workspace)
    sample_positions = np.linspace(
        0, len(candidate_indices) - 1, min(scene_point_limit, len(candidate_indices)), dtype=int
    )
    selected = candidate_indices[sample_positions]
    scene_camera = ply_camera_all[selected]
    scene_robot = ply_robot_all[selected].astype(np.float32)
    reference_rgb = np.asarray(Image.open(source / files["reference_rgb"]).convert("RGB"))
    scene_colors = project_colors(scene_camera, reference_rgb, intrinsics)

    depths = np.load(source / files["aligned_depth"], mmap_mode="r")
    frame_indices = np.arange(0, len(depths), frame_step, dtype=np.int32)
    height, width = depths.shape[1:]
    scale_x = width / float(intrinsics["width"])
    scale_y = height / float(intrinsics["height"])
    fx = float(intrinsics["fx"]) * scale_x
    fy = float(intrinsics["fy"]) * scale_y
    cx = float(intrinsics["cx"]) * scale_x
    cy = float(intrinsics["cy"]) * scale_y
    grid_u, grid_v = np.meshgrid(
        np.arange(pixel_step // 2, width, pixel_step),
        np.arange(pixel_step // 2, height, pixel_step),
    )
    u = grid_u.reshape(-1)
    v = grid_v.reshape(-1)
    video_points = []
    for frame in frame_indices:
        z = np.asarray(depths[frame, v, u], dtype=np.float32)
        camera = np.column_stack(((u - cx) * z / fx, (v - cy) * z / fy, z))
        video_points.append(camera_to_robot(camera, transform).astype(np.float32))
    video_points_array = np.stack(video_points)
    valid_pixels = np.isfinite(video_points_array).all(axis=(0, 2))
    expanded_lower = lower - 0.2
    expanded_upper = upper + 0.2
    in_workspace = np.all(
        (video_points_array >= expanded_lower)
        & (video_points_array <= expanded_upper),
        axis=(0, 2),
    )
    keep_pixels = valid_pixels & in_workspace
    video_points_array = video_points_array[:, keep_pixels]
    u = u[keep_pixels]
    v = v[keep_pixels]

    decoded_frames = read_selected_video_frames(
        source / files["generated_video"], width, height, frame_step
    )
    if len(decoded_frames) != len(frame_indices):
        raise ValueError(
            f"Depth/video frame mismatch for {identifier}: "
            f"{len(frame_indices)} vs {len(decoded_frames)}"
        )
    video_colors = decoded_frames[:, v, u, :].astype(np.uint8)

    track_entries = []
    all_track_points = []
    for track_number, track_relative in enumerate(example.get("object_tracks", [])):
        track_path = source / track_relative
        with np.load(track_path, allow_pickle=True) as track_data:
            tracks = np.asarray(track_data["tracks_robot"], dtype=np.float32)
            indices = np.asarray(track_data["indices"], dtype=np.int32).reshape(-1)
        frame_lookup = {int(frame): index for index, frame in enumerate(indices)}
        selected_tracks = np.stack(
            [tracks[frame_lookup.get(int(frame), 0)] for frame in frame_indices]
        )
        finite_points = np.isfinite(selected_tracks).all(axis=(0, 2))
        selected_tracks = selected_tracks[:, finite_points]
        object_name = track_path.parent.name.replace("_", " ")
        all_track_points.append(selected_tracks)
        track_entries.append(
            {
                "name": object_name.title(),
                "pointCount": int(selected_tracks.shape[1]),
                "color": list(TRACK_COLORS[track_number % len(TRACK_COLORS)]),
            }
        )

    bounds_sources = [scene_robot.reshape(-1, 3), video_points_array.reshape(-1, 3)]
    bounds_sources.extend(track.reshape(-1, 3) for track in all_track_points)
    combined = np.concatenate(bounds_sources)
    minimum = np.min(combined, axis=0).astype(np.float32)
    maximum = np.max(combined, axis=0).astype(np.float32)
    extent = np.maximum(maximum - minimum, 1e-6).astype(np.float32)

    bundle = BinaryBundle()
    scene_position = bundle.add(quantize(scene_robot, minimum, extent), align=2)
    scene_color = bundle.add(scene_colors)
    video_position = bundle.add(
        quantize(video_points_array, minimum, extent), align=2
    )
    video_color = bundle.add(video_colors)
    for entry, points in zip(track_entries, all_track_points):
        entry["position"] = bundle.add(quantize(points, minimum, extent), align=2)

    output.mkdir(parents=True, exist_ok=True)
    data_name = f"{identifier}.bin"
    video_name = f"{identifier}.mp4"
    (output / data_name).write_bytes(bundle.data)
    shutil.copyfile(source / files["generated_video"], output / video_name)

    return {
        "id": identifier,
        "sceneNumber": example["scene"],
        "demo": example["demo"],
        "seed": example["seed"],
        "task": example["task"],
        "label": humanize_task(example["task"]),
        "data": data_name,
        "video": video_name,
        "bytes": len(bundle.data),
        "camera": {"cameraToRobot": transform.tolist(), "intrinsics": intrinsics},
        "bounds": {"min": minimum.tolist(), "extent": extent.tolist()},
        "frameIndices": frame_indices.tolist(),
        "sourceFrameCount": int(len(depths)),
        "sourceFps": 24,
        "scene": {
            "pointCount": int(len(scene_robot)),
            "position": scene_position,
            "color": scene_color,
        },
        "videoCloud": {
            "frameCount": int(len(frame_indices)),
            "pointCount": int(video_points_array.shape[1]),
            "position": video_position,
            "color": video_color,
        },
        "tracks": track_entries,
    }


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    source_manifest = json.loads((source / "manifest.json").read_text())

    intrinsics = {}
    for line in (source / source_manifest["intrinsics"]).read_text().splitlines():
        if ":" in line and not line.lstrip().startswith("#"):
            key, value = line.split(":", 1)
            intrinsics[key.strip()] = float(value.strip())

    examples = []
    for example in source_manifest["examples"]:
        print(f"Building scene {example['scene']} / {example['task']}...", flush=True)
        examples.append(
            build_example(
                source,
                output,
                example,
                intrinsics,
                args.frame_step,
                args.pixel_step,
                args.scene_points,
            )
        )

    manifest = {
        "format": "caster.pointcloud.v1",
        "description": "Quantized, lazy-loaded point-cloud replays for the CASTER project page.",
        "examples": examples,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    total_bytes = sum(example["bytes"] for example in examples)
    print(
        f"Built {len(examples)} examples ({total_bytes / 1024 / 1024:.1f} MiB binary)."
    )


if __name__ == "__main__":
    main()
