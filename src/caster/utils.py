"""Service requests, depth, geometry, and visualization helpers for Caster."""

import json
import logging
import tempfile
import time
import zipfile
from pathlib import Path

import cv2
import imageio
import matplotlib
import numpy as np
import requests
import trimesh
import yaml
from PIL import Image
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from plyfile import PlyData
from scipy.spatial.transform import Rotation
from termcolor import colored

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class InferenceError(RuntimeError):
    """An inference service rejected or failed a request."""


def request_inference(url: str, endpoint: str, payload: dict, timeout: float) -> dict:
    response = requests.post(
        f"{url.rstrip('/')}/{endpoint}", json=payload, timeout=(10, timeout)
    )
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise InferenceError(f"{response.url} returned HTTP {response.status_code}: {detail}")
    return response.json()


def check_service(url: str, name: str) -> None:
    response = requests.get(f"{url.rstrip('/')}/health", timeout=2)
    response.raise_for_status()
    if response.json().get("service") != name:
        raise RuntimeError(f"Expected the {name} service at {url}")

def render_masks(video_path: Path, mask_paths: list[Path], output_path: Path) -> Path:
    if output_path.is_file():
        return output_path
    masks = [np.load(path, allow_pickle=True).item()["masks"] for path in mask_paths]
    capture = cv2.VideoCapture(str(video_path))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 24
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output_path), fps=fps, codec="libx264", pixelformat="yuv420p")
    colors = [(40, 80, 240), (50, 200, 90), (220, 150, 30), (180, 70, 180)]
    for frame_index in range(min(len(mask) for mask in masks)):
        ok, frame = capture.read()
        if not ok:
            break
        for index, mask_frames in enumerate(masks):
            mask = mask_frames[frame_index]
            if mask.shape != frame.shape[:2]:
                mask = cv2.resize(
                    mask, (width, height), interpolation=cv2.INTER_NEAREST
                )
            region = mask > 0
            frame[region] = (
                0.55 * frame[region] + 0.45 * np.array(colors[index % len(colors)])
            ).astype(np.uint8)
        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    writer.close()
    return output_path


def render_depth(depth_path: Path, output_path: Path, fps: int = 24) -> Path:
    if output_path.is_file():
        return output_path
    depth = np.load(depth_path)
    height, width = depth.shape[1:]
    sample = depth[:, ::16, ::16]
    valid = sample[np.isfinite(sample) & (sample > 0)]
    low, high = np.percentile(valid, [2, 98])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output_path), fps=fps, codec="libx264", pixelformat="yuv420p")
    for frame in depth:
        scaled = np.clip((frame - low) / max(high - low, 1e-6), 0, 1)
        colored_frame = cv2.applyColorMap(
            (255 * scaled).astype(np.uint8), cv2.COLORMAP_TURBO
        )
        colored_frame[~np.isfinite(frame)] = 0
        writer.append_data(cv2.cvtColor(colored_frame, cv2.COLOR_BGR2RGB))
    writer.close()
    return output_path


def generate_depth(video_path: Path, output_dir: Path, server_url: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{video_path.stem}_pred.npy"
    if result_path.is_file():
        return result_path

    with tempfile.TemporaryDirectory(prefix="caster_da3_") as scratch:
        scratch_dir = Path(scratch)
        frames_dir = scratch_dir / "frames"
        frames_dir.mkdir()
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        interval = max(1, int(capture.get(cv2.CAP_PROP_FPS) / 24))
        frames = []
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % interval == 0:
                path = frames_dir / f"{len(frames):06d}.png"
                cv2.imwrite(str(path), frame)
                frames.append(str(path))
            index += 1
        capture.release()
        logger.info("%s %d frames", colored("DA3 input:", "cyan"), len(frames))

        response = requests.post(
            f"{server_url}/inference",
            json={
                "image_paths": frames,
                "export_dir": str(scratch_dir),
                "export_format": "mini_npz",
                "process_res": 504,
                "process_res_method": "upper_bound_resize",
                "export_feat_layers": [],
                "align_to_input_ext_scale": True,
                "conf_thresh_percentile": 40.0,
                "num_max_points": 1000000,
                "show_cameras": True,
                "feat_vis_fps": 15,
            },
            timeout=30,
        )
        response.raise_for_status()
        job = response.json()
        if not job.get("success"):
            raise RuntimeError(job.get("message", "DA3 request failed"))

        while True:
            status_response = requests.get(
                f"{server_url}/task/{job['task_id']}", timeout=30
            )
            status_response.raise_for_status()
            status = status_response.json()
            if status["status"] == "completed":
                break
            if status["status"] == "failed":
                raise RuntimeError(status.get("message", "DA3 inference failed"))
            time.sleep(5)

        archive = scratch_dir / "exports/mini_npz/results.npz"
        deadline = time.monotonic() + 120
        while True:
            try:
                with np.load(archive) as result:
                    depth = result["depth"].copy()
                    extrinsics = (
                        result["extrinsics"].copy() if "extrinsics" in result else None
                    )
                    intrinsics = (
                        result["intrinsics"].copy() if "intrinsics" in result else None
                    )
                break
            except (
                FileNotFoundError,
                zipfile.BadZipFile,
                EOFError,
                OSError,
                ValueError,
            ):
                if time.monotonic() > deadline:
                    raise TimeoutError(f"DA3 result was not ready: {archive}")
                time.sleep(0.5)

        np.save(result_path, depth)
        if extrinsics is not None:
            if extrinsics.shape[-2:] == (3, 4):
                bottom = np.broadcast_to([0, 0, 0, 1], (len(extrinsics), 1, 4))
                extrinsics = np.concatenate([extrinsics, bottom], axis=1)
            np.save(output_dir / f"{video_path.stem}_extrinsics.npy", extrinsics)
        if intrinsics is not None:
            np.save(output_dir / f"{video_path.stem}_intrinsics.npy", intrinsics)
    logger.info("%s %s", colored("Saved depth:", "green"), result_path)
    return result_path


def real_depth_gen(
    depth_npy_path: str | Path,
    first_frame_ply: str | Path,
    camera_intrinsics_yaml: str | Path,
    output_path: str | Path,
    alignment_config: dict | None = None,
) -> str:
    """
    Aligns the predicted depth sequence with a real depth map from a PLY file.

    Args:
        depth_npy_path: Path to the predicted depth .npy file (T, H, W).
        first_frame_ply: Path to the PLY file for the first frame.
        camera_intrinsics_yaml: Path to the YAML file with camera intrinsics.
        output_path: Path to save the aligned real depth .npy file.
        alignment_config: Optional DA3 alignment controls. The default method
            anchors the tabletop and learns a ray-aware residual from reference
            PLY points inside the configured robot-frame ROI.

    Returns:
        str: Path to the saved real depth .npy file.
    """
    output_path = Path(output_path)
    if output_path.is_dir() or not output_path.suffix:
        output_path = output_path / "aligned_depth.npy"

    logging.info(
        colored("[Alignment] ", "cyan")
        + f"Loading predicted depth from {depth_npy_path}"
    )
    pred_depths = np.load(depth_npy_path)
    T, H, W = pred_depths.shape

    with open(camera_intrinsics_yaml, "r") as f:
        intr = yaml.safe_load(f)
    try:
        orig_H, orig_W = int(intr["height"]), int(intr["width"])
        fx_native, fy_native = float(intr["fx"]), float(intr["fy"])
        cx_native, cy_native = float(intr["cx"]), float(intr["cy"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "camera_intrinsics_yaml must define numeric fx, fy, cx, cy, width, and height"
        ) from error
    if orig_H <= 0 or orig_W <= 0 or fx_native <= 0 or fy_native <= 0:
        raise ValueError(
            "camera intrinsics focal lengths and resolution must be positive"
        )
    scale_h = H / orig_H
    scale_w = W / orig_W
    fx = fx_native * scale_w
    fy = fy_native * scale_h
    cx = cx_native * scale_w
    cy = cy_native * scale_h

    # Load PLY and project to first frame
    logging.info(
        colored("[Alignment] ", "cyan") + f"Loading PLY from {first_frame_ply}"
    )
    plydata = PlyData.read(first_frame_ply)
    v = plydata["vertex"]
    pts_3d = np.stack([v["x"], v["y"], v["z"]], axis=-1)

    # Project: u = fx*X/Z + cx, v = fy*Y/Z + cy
    z = pts_3d[:, 2]
    valid_z = z > 0.1  # Avoid division by zero and near-camera points
    pts_3d = pts_3d[valid_z]
    z = z[valid_z]

    u = (fx * pts_3d[:, 0] / z) + cx
    v_coords = (fy * pts_3d[:, 1] / z) + cy

    u_idx = np.round(u).astype(int)
    v_idx = np.round(v_coords).astype(int)

    # Filter points within frame
    in_frame = (u_idx >= 0) & (u_idx < W) & (v_idx >= 0) & (v_idx < H)
    u_idx = u_idx[in_frame]
    v_idx = v_idx[in_frame]
    z = z[in_frame]

    if len(z) == 0:
        raise ValueError("No points from PLY project onto the frame.")

    # Create real depth map with occlusion handling (min depth)
    real_depth_map = np.full((H, W), np.inf)
    # Sort by z descending so that when we assign, the closest point (smallest z) is assigned last
    sort_idx = np.argsort(z)[::-1]
    real_depth_map[v_idx[sort_idx], u_idx[sort_idx]] = z[sort_idx]
    valid_mask = np.isfinite(real_depth_map)

    # Fit a shared depth scale and shift from the reference point cloud.
    from sklearn.linear_model import RANSACRegressor, LinearRegression

    valid_ys, valid_xs = np.where(valid_mask)
    num_valid = len(valid_ys)
    logging.info(
        colored("[Alignment] ", "cyan")
        + f"Aligning with {num_valid} valid reference points"
    )

    real_depths = np.zeros(pred_depths.shape, dtype=np.float32)

    from sklearn.ensemble import HistGradientBoostingRegressor

    config = alignment_config or {}
    method = str(config.get("method", "table_anchored_ray_residual"))
    table_band_m = float(config.get("table_band_m", 0.05))
    reference_roi_min_robot_x = float(config.get("reference_roi_min_robot_x", 0.10))
    residual_max_iter = int(config.get("residual_max_iter", 100))
    residual_max_leaf_nodes = int(config.get("residual_max_leaf_nodes", 15))
    residual_l2_regularization = float(
        config.get("residual_l2_regularization", 1.0)
    )
    max_correction_m = float(config.get("max_correction_m", 0.25))
    minimum_samples = int(config.get("minimum_samples", 100))
    maximum_fit_samples = int(config.get("maximum_fit_samples", 20_000))
    if method not in {"table_anchored_ray_residual", "global_affine"}:
        raise ValueError(
            "DA3 depth alignment method must be "
            "'table_anchored_ray_residual' or 'global_affine', got "
            f"{method!r}."
        )
    if table_band_m < 0:
        raise ValueError("table_band_m must be nonnegative.")
    if (
        residual_max_iter < 1
        or residual_max_leaf_nodes < 2
        or residual_l2_regularization < 0
        or max_correction_m <= 0
    ):
        raise ValueError(
            "Residual model iteration/leaf counts and max_correction_m must "
            "be positive; residual_l2_regularization must be nonnegative."
        )
    if minimum_samples < 2 or maximum_fit_samples < minimum_samples:
        raise ValueError(
            "minimum_samples must be at least 2 and no larger than "
            "maximum_fit_samples."
        )

    c2r_path = Path(first_frame_ply).parent / "C2R.npy"
    if not c2r_path.exists():
        raise FileNotFoundError(
            f"C2R.npy not found in {c2r_path.parent}. Ensure the robot-to-camera transformation is available."
        )
    C2R = np.asarray(np.load(c2r_path), dtype=np.float64)
    if C2R.shape != (4, 4) or not np.all(np.isfinite(C2R)):
        raise ValueError(f"Expected a finite 4x4 C2R matrix at {c2r_path}.")

    # Reference ROI membership must come from the PLY, never from DA3. This
    # ensures that an object whose DA3 reconstruction incorrectly crosses
    # robot X=0.10 m remains part of calibration.
    reference_y = valid_ys
    reference_x = valid_xs
    reference_depth = real_depth_map[reference_y, reference_x]
    ray_u = (reference_x - cx) / fx
    ray_v = (reference_y - cy) / fy
    reference_camera = np.stack(
        [
            ray_u * reference_depth,
            ray_v * reference_depth,
            reference_depth,
            np.ones_like(reference_depth),
        ],
        axis=0,
    )
    reference_robot = C2R @ reference_camera
    reference_robot_x = reference_robot[0]
    reference_robot_z = reference_robot[2]
    raw_reference = pred_depths[0][reference_y, reference_x]

    rng = np.random.default_rng(42)

    def limited_indices(mask: np.ndarray) -> np.ndarray:
        indices = np.flatnonzero(mask)
        if len(indices) > maximum_fit_samples:
            indices = rng.choice(indices, size=maximum_fit_samples, replace=False)
        return indices

    table_mask = np.abs(reference_robot_z) <= table_band_m
    table_indices = limited_indices(table_mask)
    base_description = "table"
    if len(table_indices) < minimum_samples:
        table_indices = limited_indices(np.ones_like(table_mask, dtype=bool))
        base_description = "all valid PLY points (table fallback)"

    base_ransac = RANSACRegressor(
        estimator=LinearRegression(),
        min_samples=minimum_samples,
        max_trials=100,
        random_state=42,
    )
    base_ransac.fit(
        raw_reference[table_indices].reshape(-1, 1),
        reference_depth[table_indices],
    )
    global_scale = float(base_ransac.estimator_.coef_[0])
    global_shift = float(base_ransac.estimator_.intercept_)

    logging.info(
        colored("[Alignment] ", "cyan")
        + f"DA3 base RANSAC using {len(table_indices)} {base_description} samples: "
        f"scale={global_scale:.4f}, shift={global_shift:.4f}"
    )

    elevated_reference_roi_mask = (
        reference_robot_x > reference_roi_min_robot_x
    ) & (reference_robot_z > table_band_m)
    roi_indices = limited_indices(elevated_reference_roi_mask)
    use_ray_residual = (
        method == "table_anchored_ray_residual"
        and len(roi_indices) >= minimum_samples
    )

    def ray_coordinates(
        raw_depth: np.ndarray,
        normalized_u: np.ndarray,
        normalized_v: np.ndarray,
    ) -> np.ndarray:
        return np.column_stack([raw_depth, normalized_u, normalized_v])

    residual_model = None
    if use_ray_residual:
        baseline_reference = global_scale * raw_reference + global_shift
        residual_targets = reference_depth - baseline_reference
        residual_training_indices = np.concatenate([table_indices, roi_indices])
        residual_model = HistGradientBoostingRegressor(
            max_iter=residual_max_iter,
            max_leaf_nodes=residual_max_leaf_nodes,
            l2_regularization=residual_l2_regularization,
            random_state=42,
        )
        residual_model.fit(
            ray_coordinates(raw_reference, ray_u, ray_v)[residual_training_indices],
            residual_targets[residual_training_indices],
        )
        logging.info(
            colored("[Alignment] ", "cyan")
            + f"Fitted bounded ray-aware residual with {len(table_indices)} table "
            f"anchors and {len(roi_indices)} reference PLY ROI samples satisfying "
            f"robot X > {reference_roi_min_robot_x:.3f} m and "
            f"Z > {table_band_m:.3f} m."
        )
    elif method == "table_anchored_ray_residual":
        logging.warning(
            colored("[Alignment] ", "yellow")
            + f"Only {len(roi_indices)} elevated reference-ROI samples were "
            f"available (minimum {minimum_samples}); using the base affine fit."
        )

    pixel_v, pixel_u = np.indices((H, W), dtype=np.float64)
    normalized_u = (pixel_u - cx) / fx
    normalized_v = (pixel_v - cy) / fy
    flat_u = normalized_u.ravel()
    flat_v = normalized_v.ravel()
    clipped_values = 0
    predicted_values = 0

    for t in range(T):
        raw_t = np.asarray(pred_depths[t], dtype=np.float64)
        baseline_t = global_scale * raw_t + global_shift
        aligned_t = baseline_t
        if residual_model is not None:
            raw_flat = raw_t.ravel()
            predicted_residual = residual_model.predict(
                ray_coordinates(raw_flat, flat_u, flat_v)
            ).reshape(H, W)
            clipped_values += int(
                np.count_nonzero(np.abs(predicted_residual) > max_correction_m)
            )
            predicted_values += predicted_residual.size
            predicted_residual = np.clip(
                predicted_residual, -max_correction_m, max_correction_m
            )

            # Both table preservation and ROI importance are learned from
            # reference PLY supervision. There is deliberately no gate
            # based on DA3-predicted robot X or Z.
            aligned_t = baseline_t + predicted_residual

        invalid = ~np.isfinite(aligned_t) | (aligned_t <= np.finfo(np.float32).eps)
        if np.any(invalid):
            aligned_t = aligned_t.copy()
            valid_baseline = np.isfinite(baseline_t) & (
                baseline_t > np.finfo(np.float32).eps
            )
            aligned_t[invalid & valid_baseline] = baseline_t[
                invalid & valid_baseline
            ]
            aligned_t[invalid & ~valid_baseline] = np.finfo(np.float32).eps
        real_depths[t] = aligned_t.astype(np.float32)

    if predicted_values:
        logging.info(
            colored("[Alignment] ", "cyan") + f"Residual clipping rate: "
            f"{100.0 * clipped_values / predicted_values:.3f}% at "
            f"+/-{max_correction_m:.3f} m."
        )

    # Report errors only where the reference PLY defines the requested ROI.
    first_aligned = real_depths[0][reference_y, reference_x]
    depth_error = first_aligned - reference_depth
    robot_ray_x = C2R[0, 0] * ray_u + C2R[0, 1] * ray_v + C2R[0, 2]
    robot_x_error = robot_ray_x * depth_error
    table_eval = table_mask
    roi_eval = elevated_reference_roi_mask
    for label, mask in (
        ("table", table_eval),
        ("elevated reference ROI", roi_eval),
    ):
        if np.any(mask):
            logging.info(
                colored("[Alignment] ", "cyan")
                + f"First-frame {label}: samples={np.sum(mask)}, "
                f"depth_MAE={np.mean(np.abs(depth_error[mask])):.4f} m, "
                f"robot_X_MAE={np.mean(np.abs(robot_x_error[mask])):.4f} m, "
                f"robot_X_median_bias={np.median(robot_x_error[mask]):.4f} m"
            )

    np.save(output_path, real_depths)
    logging.info(colored("[Alignment] ", "cyan") + f"Real depth saved to {output_path}")
    return str(output_path)


def _scene_transform(item: dict) -> np.ndarray:
    w, x, y, z = item["rotation_wxyz"]
    correction = Rotation.from_quat([np.sqrt(0.5), 0, 0, np.sqrt(0.5)]).as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix() @ correction
    transform[:3, 3] = item["translation"]
    return transform


def _draw_mesh(ax, path: Path, item: dict, color: str) -> np.ndarray:
    mesh = trimesh.load(path, force="mesh")
    mesh.apply_scale(float(item["scale"]))
    faces = mesh.faces
    if len(faces) > 25000:
        faces = faces[np.linspace(0, len(faces) - 1, 25000, dtype=int)]

    transform = _scene_transform(item)
    vertices = mesh.vertices @ transform[:3, :3].T + transform[:3, 3]
    triangles = vertices[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    light = np.array([0.4, 0.4, 1.0])
    light /= np.linalg.norm(light)
    shade = np.clip(normals @ light, 0.35, 1.0)
    colors = np.column_stack(
        (shade[:, None] * to_rgb(color), np.full(len(faces), 0.85))
    )
    ax.add_collection3d(
        Poly3DCollection(triangles, facecolors=colors, edgecolors="none")
    )
    return vertices


def _gripper_segments() -> list[tuple[np.ndarray, np.ndarray]]:
    palm, tip = 0.054, 0.1034
    width, pad = 0.08, 0.04
    return [
        ([0, 0, 0], [0, 0, palm]),
        ([-width / 2, 0, palm], [width / 2, 0, palm]),
        ([-width / 2, 0, palm], [-width / 2, 0, tip]),
        ([width / 2, 0, palm], [width / 2, 0, tip]),
        ([-width / 2, 0, tip], [-pad / 2, 0, tip]),
        ([width / 2, 0, tip], [pad / 2, 0, tip]),
    ]


def _draw_gripper(ax, pose: np.ndarray, color: str, chosen: bool) -> None:
    for start, end in _gripper_segments():
        points = np.stack((start, end)) @ pose[:3, :3].T + pose[:3, 3]
        ax.plot(
            *points.T,
            color=color,
            alpha=1.0 if chosen else 0.18,
            linewidth=2.5 if chosen else 0.7,
        )


def render_grasps(
    transforms_path: Path,
    image_path: Path,
    results: dict[str, dict],
    output: Path | None = None,
) -> Path:
    fig = plt.figure(figsize=(15, 7))
    image_ax = fig.add_subplot(121)
    scene_ax = fig.add_subplot(122, projection="3d")
    image_ax.imshow(Image.open(image_path))
    image_ax.set_title("VLM and chosen grasp contacts")
    image_ax.axis("off")

    objects = json.loads(transforms_path.read_text())["objects"]
    palette = ["#DC3545", "#9B59B6", "#E67E22", "#17A2B8", "#2ECC71"]
    all_vertices = []
    for index, item in enumerate(objects):
        mesh_path = transforms_path.parent / f"{item['name']}.obj"
        if not mesh_path.exists():
            mesh_path = transforms_path.parent / item["glb"]
        all_vertices.append(
            _draw_mesh(scene_ax, mesh_path, item, palette[index % len(palette)])
        )

    rendered = 0
    for index, (name, result) in enumerate(results.items()):
        color = palette[index % len(palette)]
        candidates = np.asarray(result["all_candidate_transforms"])
        confidence = np.asarray(result["all_candidate_confidences"])
        chosen = int(result["chosen_index"])
        ranked = list(np.argsort(confidence)[-50:])
        if chosen not in ranked:
            ranked[-1] = chosen
        for candidate in ranked:
            if candidate != chosen:
                _draw_gripper(scene_ax, candidates[candidate], color, chosen=False)
        _draw_gripper(scene_ax, candidates[chosen], color, chosen=True)
        rendered += len(ranked)

        scene_ax.scatter(
            *result["chosen_contact_3d"],
            color=color,
            edgecolors="black",
            s=100,
            marker="X",
            label=f"{name}: chosen contact",
        )
        if result["vlm_pinpoint_2d"] is not None:
            image_ax.scatter(
                *result["vlm_pinpoint_2d"],
                color=color,
                edgecolors="white",
                s=130,
                marker="*",
                label=f"{name}: VLM",
            )
        if result["chosen_contact_2d"] is not None:
            image_ax.scatter(
                *result["chosen_contact_2d"],
                color=color,
                edgecolors="black",
                s=85,
                marker="X",
                label=f"{name}: chosen",
            )

    vertices = np.vstack(all_vertices)
    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    half_span = np.max(vertices.max(axis=0) - vertices.min(axis=0)) / 2
    scene_ax.set_xlim(center[0] - half_span, center[0] + half_span)
    scene_ax.set_ylim(center[1] - half_span, center[1] + half_span)
    scene_ax.set_zlim(center[2] - half_span, center[2] + half_span)
    scene_ax.set(
        xlabel="X (m)",
        ylabel="Y (m)",
        zlabel="Z (m)",
        title=f"M2T2 grasp proposals ({rendered} shown)",
    )
    if image_ax.get_legend_handles_labels()[0]:
        image_ax.legend(loc="upper right", fontsize=8)
    scene_ax.legend(loc="upper right", fontsize=8)

    output = output or transforms_path.parent / "grasp_viz.png"
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output
