"""TrackCraft3R dense tracking service for Caster."""

import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from caster.servers.http_service import InferenceRequest, create_app, run_inference, serve


class TrackRequest(InferenceRequest):
    video_path: str
    masks_path: str
    depth_path: str
    first_frame_ply: str
    camera_intrinsics_yaml: str
    c2r_path: str
    output_dir: str


def create_track_app(track):
    app = create_app("trackcraft")

    @app.post("/track")
    def track_request(request: TrackRequest) -> dict:
        output = run_inference(track, **request.model_dump())
        return {"output_path": output}

    return app


_PREDICTOR = None


@dataclass
class PreparedTrackingData:
    """Video, calibration, depth, and masks prepared for tracking."""

    video_path: Path
    first_frame_ply: Path
    output_dir: Path
    C2R: np.ndarray
    num_frames_video: int
    K_video: np.ndarray
    raw_depth: np.ndarray
    aligned_depth: np.ndarray
    scale: float
    fixed_extrinsics: np.ndarray
    model_height: int
    model_width: int
    resize_mode: str
    masks_original: np.ndarray
    target_object: str
    video_list: list
    intrinsics: np.ndarray
    extrinsics_w2c: np.ndarray


@dataclass
class WindowTrackingResult:
    """Window predictions and the canonical geometry from frame zero."""

    all_window_tracks: list[dict]
    canonical_points_0: np.ndarray | None
    canonical_radius_0: float
    canonical_offsets: np.ndarray | None
    ys_0: np.ndarray | None
    xs_0: np.ndarray | None


@dataclass
class RecoveredPoses:
    """Smoothed camera poses and the existing frame diagnostics."""

    rotations_filtered: list[np.ndarray]
    centers_smoothed: np.ndarray
    rotation_valid_final: list[bool]
    pose_residual: list[float]
    inlier_counts: list[int]
    point_counts: list[int]
    inlier_ratios: list[float]
    rejection_reasons: list[str]


def blend_rotations(R_list, weights=None):
    if len(R_list) == 1:
        return R_list[0]
    quats = np.zeros((len(R_list), 4))
    for i, R in enumerate(R_list):
        q = Rotation.from_matrix(R).as_quat()
        if i > 0 and np.dot(q, quats[0]) < 0:
            q = -q
        quats[i] = q
    if weights is None:
        q_avg = np.mean(quats, axis=0)
    else:
        q_avg = np.average(quats, axis=0, weights=weights)
    q_avg /= np.linalg.norm(q_avg)
    return Rotation.from_quat(q_avg).as_matrix()

def fit_rigid(P, Q):
    p_mean, q_mean = np.mean(P, axis=0), np.mean(Q, axis=0)
    P0, Q0 = P - p_mean, Q - q_mean
    U, _, Vt = np.linalg.svd(P0.T @ Q0)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return R, q_mean - R @ p_mean

def fit_rigid_weighted(P, Q, W):
    W_sum = np.sum(W)
    if W_sum < 1e-8:
        W = np.ones(len(P)) / len(P)
    else:
        W = W / W_sum
    p_mean = np.sum(P * W[:, None], axis=0)
    q_mean = np.sum(Q * W[:, None], axis=0)
    P0, Q0 = P - p_mean, Q - q_mean
    H = (P0 * W[:, None]).T @ Q0
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return R, q_mean - R @ p_mean

def refine_rigid_irls(P, Q, init_W=None, num_iters=4):
    if init_W is None:
        W = np.ones(len(P))
    else:
        W = init_W.copy()
    R, t = fit_rigid_weighted(P, Q, W)
    for _ in range(num_iters):
        errors = np.linalg.norm(Q - (P @ R.T + t), axis=1)
        delta = max(0.005, np.median(errors))
        huber_weights = np.where(errors < delta, 1.0, delta / (errors + 1e-8))
        if init_W is not None:
            W = init_W * huber_weights
        else:
            W = huber_weights
        W = np.maximum(W, 1e-6)
        R, t = fit_rigid_weighted(P, Q, W)
    return R, t

def smooth_rotations_gaussian(rotations, window_size=9, sigma=3.0):
    T_len = len(rotations)
    smoothed = []
    half_w = window_size // 2
    x = np.arange(-half_w, half_w + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    for t in range(T_len):
        cands = []
        weights = []
        for offset in range(-half_w, half_w + 1):
            t_neighbor = min(max(0, t + offset), T_len - 1)
            cands.append(rotations[t_neighbor])
            weights.append(kernel[offset + half_w])
        smoothed.append(blend_rotations(cands, weights=weights))
    return smoothed

def grid_subsample(uv_pts, max_pts=500):
    if len(uv_pts) <= max_pts:
        return uv_pts
    xmin, ymin = uv_pts.min(axis=0)
    xmax, ymax = uv_pts.max(axis=0)
    w_box, h_box = xmax - xmin + 1, ymax - ymin + 1
    r = float(w_box) / max(1, h_box)
    cell_h = int(np.round(np.sqrt(max_pts / r)))
    cell_h = max(2, min(cell_h, int(h_box)))
    cell_w = int(np.round(max_pts / cell_h))
    cell_w = max(2, min(cell_w, int(w_box)))
    cell_x = ((uv_pts[:, 0] - xmin) / w_box * cell_w).astype(np.int32)
    cell_y = ((uv_pts[:, 1] - ymin) / h_box * cell_h).astype(np.int32)
    cell_indices = cell_y * cell_w + cell_x
    _, idxs = np.unique(cell_indices, return_index=True)
    subsampled = uv_pts[idxs]
    if len(subsampled) > max_pts:
        subsampled = subsampled[
            np.random.choice(len(subsampled), max_pts, replace=False)
        ]
    return subsampled

def ransac_rigid(P, Q, iterations=80):
    finite = np.isfinite(P).all(axis=1) & np.isfinite(Q).all(axis=1)
    P, Q = P[finite], Q[finite]

    min_inliers = max(20, int(0.15 * len(P)))
    if len(P) < min_inliers:
        return None

    center = np.median(P, axis=0)
    radius = np.quantile(
        np.linalg.norm(P - center, axis=1),
        0.9,
    )
    threshold = max(0.005, 0.10 * radius)

    best_inliers = None

    for _ in range(iterations):
        sample = np.random.choice(len(P), 3, replace=False)
        sample_centered = P[sample] - P[sample].mean(axis=0)

        # Cheaper degeneracy test than matrix_rank/SVD.
        if (
            np.linalg.norm(np.cross(sample_centered[1], sample_centered[2]))
            < 1e-8
        ):
            continue

        R, t = fit_rigid(P[sample], Q[sample])
        errors = np.linalg.norm(
            Q - (P @ R.T + t),
            axis=1,
        )
        inliers = errors < threshold

        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers

        if inliers.mean() >= 0.90:
            break

    if best_inliers is None:
        return None

    return {
        "success": int(best_inliers.sum()) >= min_inliers,
        "inlier_count": int(best_inliers.sum()),
    }

def center_from_mask_depth(mask_t, depth_t, K, metric_scale=1.0):
    import cv2

    if depth_t.shape != mask_t.shape:
        depth_t = cv2.resize(
            depth_t,
            (mask_t.shape[1], mask_t.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    mask = (mask_t > 0).astype(np.uint8)
    mask = cv2.erode(mask, np.ones((7, 7), np.uint8), iterations=1)
    depth_metric = metric_scale * depth_t
    valid = (mask > 0) & np.isfinite(depth_metric) & (depth_metric > 0.01)
    v, u = np.where(valid)
    if len(u) < 20:
        return None
    z = depth_metric[v, u]
    z_med = np.median(z)
    z_mad = np.median(np.abs(z - z_med)) + 1e-6
    keep = np.abs(z - z_med) < 3.0 * z_mad
    u, v, z = u[keep], v[keep], z[keep]
    if len(z) < 20:
        return None
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    points_camera = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], axis=1)
    center_camera = np.median(points_camera, axis=0)
    return center_camera

def _prepare_tracking_data(
    video_path, masks_path, depth_path, first_frame_ply,
    camera_intrinsics_yaml, c2r_path, output_dir, trackcraft_root, load_npz_data,
) -> PreparedTrackingData:
    """Prepare calibrated video inputs and the upstream TrackCraft NPZ."""
    import subprocess
    import cv2
    from termcolor import colored

    video_path = Path(video_path).resolve()
    masks_path = Path(masks_path).resolve()
    depth_path = Path(depth_path).resolve()  # aligned_depth.npy
    first_frame_ply = Path(first_frame_ply).resolve()
    camera_intrinsics_yaml = Path(camera_intrinsics_yaml).resolve()
    c2r_path = Path(c2r_path).resolve()
    C2R = np.asarray(np.load(str(c2r_path)), dtype=np.float64)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine video resolution and frame count early
    cap = cv2.VideoCapture(str(video_path))
    width_video = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_video = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if width_video <= 0 or height_video <= 0 or num_frames_video <= 0:
        raise ValueError(
            f"Could not read a valid video from {video_path}: "
            f"width={width_video}, height={height_video}, frames={num_frames_video}"
        )

    # Use the shared physical camera calibration for every TrackCraft
    # operation. Scale it once from its calibration resolution to the
    # generated video's resolution.
    with open(str(camera_intrinsics_yaml), "r") as f:
        intrinsics_data = yaml.safe_load(f)

    required_intrinsics = ("fx", "fy", "cx", "cy", "width", "height")
    missing_intrinsics = [
        key for key in required_intrinsics if key not in intrinsics_data
    ]
    if missing_intrinsics:
        raise ValueError(
            f"Camera intrinsics file {camera_intrinsics_yaml} is missing "
            f"required fields: {missing_intrinsics}"
        )

    calibration_width = float(intrinsics_data["width"])
    calibration_height = float(intrinsics_data["height"])
    if calibration_width <= 0 or calibration_height <= 0:
        raise ValueError(
            f"Camera intrinsics file {camera_intrinsics_yaml} has invalid "
            f"resolution: width={calibration_width}, height={calibration_height}"
        )

    scale_w = width_video / calibration_width
    scale_h = height_video / calibration_height
    K_video = np.array(
        [
            [
                float(intrinsics_data["fx"]) * scale_w,
                0.0,
                float(intrinsics_data["cx"]) * scale_w,
            ],
            [
                0.0,
                float(intrinsics_data["fy"]) * scale_h,
                float(intrinsics_data["cy"]) * scale_h,
            ],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    video_stem = video_path.stem
    video_output_dir = depth_path.parent

    raw_depth_path = video_output_dir / f"{video_stem}_pred.npy"

    if not raw_depth_path.exists():
        raise FileNotFoundError(f"Raw depth not found at {raw_depth_path}")

    # Fit scale and shift
    raw_depth = np.load(str(raw_depth_path))
    aligned_depth = np.load(str(depth_path))

    H_orig_d, W_orig_d = raw_depth.shape[1], raw_depth.shape[2]

    # If the depth map resolution does not match the video frames, resize it.
    if (H_orig_d, W_orig_d) != (height_video, width_video):
        logging.info(
            colored("[TrackCraft Server] ", "cyan")
            + f"Resizing raw and aligned depth from {(H_orig_d, W_orig_d)} to {(height_video, width_video)} to match video..."
        )
        resized_raw = []
        resized_aligned = []
        for t in range(raw_depth.shape[0]):
            resized_raw.append(
                cv2.resize(
                    raw_depth[t],
                    (width_video, height_video),
                    interpolation=cv2.INTER_LINEAR,
                )
            )
            resized_aligned.append(
                cv2.resize(
                    aligned_depth[t],
                    (width_video, height_video),
                    interpolation=cv2.INTER_LINEAR,
                )
            )
        raw_depth = np.stack(resized_raw)
        aligned_depth = np.stack(resized_aligned)
        np.save(str(raw_depth_path), raw_depth)
        np.save(str(depth_path), aligned_depth)

    valid_pixels = (
        np.isfinite(raw_depth)
        & np.isfinite(aligned_depth)
        & (raw_depth > 0.01)
        & (aligned_depth > 0.01)
    )
    if np.any(valid_pixels):
        scale = float(
            np.median(aligned_depth[valid_pixels] / raw_depth[valid_pixels])
        )
    else:
        scale = 1.0

    logging.info(
        colored("[TrackCraft Server] ", "cyan")
        + f"Robust similarity scale fitted over all frames: scale={scale:.6f}"
    )

    # Fixed camera assumption: create identity extrinsics for all frames
    fixed_extrinsics = np.repeat(
        np.eye(4, dtype=np.float64)[None],
        raw_depth.shape[0],
        axis=0,
    )
    fixed_extrinsics_path = output_dir / f"{video_stem}_fixed_extrinsics.npy"
    np.save(str(fixed_extrinsics_path), fixed_extrinsics)

    physical_intrinsics_path = output_dir / f"{video_stem}_physical_intrinsics.npy"
    np.save(str(physical_intrinsics_path), K_video)
    logging.info(
        colored("[TrackCraft Server] ", "cyan")
        + f"Using physical intrinsics at video resolution "
        f"{(height_video, width_video)}: {K_video.tolist()}"
    )

    user_npz_path = output_dir / f"{video_stem}_user.npz"

    # Build user NPZ using the fixed identity extrinsics.
    # The saved physical intrinsics already correspond to the video/depth
    # resolution, so build_user_npz must preserve them without rescaling.
    build_cmd = [
        sys.executable,
        str(trackcraft_root / "scripts" / "build_user_npz.py"),
        "--video_path",
        str(video_path),
        "--depth_npy",
        str(raw_depth_path),
        "--extrinsics_npy",
        str(fixed_extrinsics_path),
        "--intrinsics_npy",
        str(physical_intrinsics_path),
        "--output_npz",
        str(user_npz_path),
        "--depth_convention",
        "z",
        "--extrinsics_convention",
        "w2c",
        "--intrinsics_resolution",
        str(height_video),
        str(width_video),
    ]
    logging.info(
        colored("[TrackCraft Server] ", "cyan")
        + "Running build_user_npz with fixed identity extrinsics..."
    )
    subprocess.run(build_cmd, cwd=str(trackcraft_root), check=True)

    # Inject dummy fields tracks_XYZ and visibility so load_npz_data can read it without KeyError
    npz_dict = dict(np.load(str(user_npz_path), allow_pickle=True))
    T_frames = npz_dict["images_jpeg_bytes"].shape[0]
    npz_dict["tracks_XYZ"] = np.zeros((T_frames, 1, 3), dtype=np.float32)
    npz_dict["visibility"] = np.ones((T_frames, 1), dtype=bool)
    np.savez_compressed(user_npz_path, **npz_dict)
    model_height = 480
    model_width = 832

    # Check aspect ratio to choose appropriate resize mode
    aspect_ratio = width_video / height_video
    model_aspect = model_width / model_height
    if abs(aspect_ratio - model_aspect) < 0.1:
        resize_mode = "stretch"
    else:
        resize_mode = "pad"

    # Load masks and prompt
    masks_data = np.load(str(masks_path), allow_pickle=True).item()
    masks_original = masks_data["masks"]
    target_object = masks_data.get(
        "target_object", masks_data.get("object_prompt", "unknown")
    )

    # Load NPZ data in TrackCraft3R format
    (video_list, _, _, intrinsics, _, _, _, extrinsics_w2c) = load_npz_data(
        str(user_npz_path), num_frames=num_frames_video, frame_stride=1
    )

    expected_intrinsics = np.array(
        [K_video[0, 0], K_video[1, 1], K_video[0, 2], K_video[1, 2]],
        dtype=np.float64,
    )
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if intrinsics.shape != (4,) or not np.allclose(
        intrinsics, expected_intrinsics, rtol=1e-7, atol=1e-7
    ):
        raise ValueError(
            "TrackCraft NPZ intrinsics do not match the shared physical "
            f"calibration: loaded={intrinsics.tolist()}, "
            f"expected={expected_intrinsics.tolist()}"
        )


    return PreparedTrackingData(
        video_path=video_path,
        first_frame_ply=first_frame_ply,
        output_dir=output_dir,
        C2R=C2R,
        num_frames_video=num_frames_video,
        K_video=K_video,
        raw_depth=raw_depth,
        aligned_depth=aligned_depth,
        scale=scale,
        fixed_extrinsics=fixed_extrinsics,
        model_height=model_height,
        model_width=model_width,
        resize_mode=resize_mode,
        masks_original=masks_original,
        target_object=target_object,
        video_list=video_list,
        intrinsics=intrinsics,
        extrinsics_w2c=extrinsics_w2c,
    )


def _get_predictor(
    data: PreparedTrackingData, trackcraft_root, hf_hub_download, WanSceneFlowPredictor,
):
    """Load the predictor once using the existing checkpoint and resize settings."""
    from termcolor import colored

    model_height = data.model_height
    model_width = data.model_width
    resize_mode = data.resize_mode


    global _PREDICTOR
    if _PREDICTOR is None:
        local_checkpoint = (
            trackcraft_root / "checkpoints/trackcraft3r/model.safetensors"
        )
        checkpoint = (
            local_checkpoint
            if local_checkpoint.is_file()
            else Path(
                hf_hub_download("trackcraft3r/checkpoint", "model.safetensors")
            )
        )
        local_wan = (
            trackcraft_root / "checkpoints/wan_models/Wan-AI/Wan2.1-T2V-1.3B"
        )
        if local_wan.is_dir():
            os.environ["MODELSCOPE_CACHE"] = str(
                trackcraft_root / "checkpoints/wan_models"
            )
        logging.info(
            colored("[TrackCraft Server] ", "cyan")
            + "Pre-loading WanSceneFlowPredictor..."
        )
        _PREDICTOR = WanSceneFlowPredictor(
            checkpoint_path=str(checkpoint),
            model_id="Wan-AI/Wan2.1-T2V-1.3B",
            lora_rank=1024,
            lora_target_modules="q,k,v,o,ffn.0,ffn.2",
            height=model_height,
            width=model_width,
            device="cuda",
            resize_mode=resize_mode,
        )
        logging.info(
            colored("[TrackCraft Server] ", "green")
            + "WanSceneFlowPredictor loaded."
        )

    predictor = _PREDICTOR


    return predictor


def _track_windows(data: PreparedTrackingData, predictor) -> WindowTrackingResult:
    """Track overlapping windows and estimate their relative rotations."""
    import cv2

    num_frames_video = data.num_frames_video
    raw_depth = data.raw_depth
    masks_original = data.masks_original
    video_list = data.video_list
    intrinsics = data.intrinsics
    extrinsics_w2c = data.extrinsics_w2c


    # Run consecutive local window tracking with step = 9 (smaller overlaps)
    win_size = 12
    win_step = 9
    anchor = 0
    all_window_tracks = []

    canonical_points_0 = None
    canonical_center_0 = None
    canonical_radius_0 = 1.0
    canonical_offsets = None
    ys_0 = None
    xs_0 = None

    while anchor < num_frames_video:
        win_size_actual = min(win_size, num_frames_video - anchor)
        if win_size_actual < 4:
            break

        indices = list(range(anchor, anchor + win_size_actual))

        # Seeding queries from anchor mask
        mask_a = masks_original[anchor]
        mask_a = cv2.erode(
            (mask_a > 0).astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1
        )
        ys_a, xs_a = np.where(mask_a > 0)
        query_uv_w = np.stack([xs_a, ys_a], axis=-1).astype(np.float32)

        query_uv_w = grid_subsample(query_uv_w, max_pts=500)

        if len(query_uv_w) < 10:
            anchor += win_step
            continue

        logging.info(
            f"[TrackCraft Server] Window at anchor {anchor}: tracking {len(query_uv_w)} query points for {win_size_actual} frames..."
        )
        video_list_run = [video_list[i] for i in indices]
        extrinsics_run = extrinsics_w2c[indices]
        depth_run = raw_depth[indices]
        vis_run = np.ones((len(indices), len(query_uv_w)), dtype=bool)

        t_inf_start = time.time()
        pred_run = predictor.predict(
            video_list_run,
            query_uv_w,
            vis_run,
            intrinsics,
            depth_map=depth_run,
            extrinsics_w2c=extrinsics_run,
        )
        t_inf_end = time.time()
        logging.info(
            f"[TC Profile] Predictor.predict took {t_inf_end - t_inf_start:.3f}s"
        )

        if pred_run is not None:
            P = pred_run[0]

            # Save canonical points from anchor 0
            if anchor == 0:
                canonical_points_0 = P.copy()
                canonical_center_0 = np.median(canonical_points_0, axis=0)
                canonical_radius_0 = np.quantile(
                    np.linalg.norm(canonical_points_0 - canonical_center_0, axis=1),
                    0.9,
                )
                canonical_offsets = canonical_points_0 - canonical_center_0

                ys_0 = query_uv_w[:, 1].copy()
                xs_0 = query_uv_w[:, 0].copy()

            # Get visibility weights for this window
            vis_w = np.ones((len(indices), len(query_uv_w)), dtype=np.float32)
            if getattr(predictor, "_last_vis_dense", None) is not None:
                vis_dense = predictor._last_vis_dense
                oob_mask = getattr(predictor, "_last_oob_mask", None)
                query_uv_scaled = getattr(predictor, "_last_query_uv_model", None)
                if oob_mask is not None and query_uv_scaled is not None:
                    u_q = query_uv_scaled[:, 0].astype(int)
                    v_q = query_uv_scaled[:, 1].astype(int)
                    for t_idx in range(len(indices)):
                        u_q_clamped = np.clip(u_q, 0, vis_dense.shape[2] - 1)
                        v_q_clamped = np.clip(v_q, 0, vis_dense.shape[1] - 1)
                        vis_w[t_idx] = vis_dense[t_idx, v_q_clamped, u_q_clamped]

            relative_rotations = {}
            relative_weights = {}

            t_ransac_start = time.time()
            last_valid_local_R = np.eye(3)
            for idx_t, t in enumerate(indices):
                Q_t = pred_run[idx_t]
                res = ransac_rigid(P, Q_t)
                if res is not None and res["success"]:
                    init_W = vis_w[idx_t].copy()
                    R_local, _ = refine_rigid_irls(
                        P, Q_t, init_W=init_W, num_iters=4
                    )
                    relative_rotations[t] = R_local
                    relative_weights[t] = float(
                        res["inlier_count"] * np.mean(init_W)
                    )
                    last_valid_local_R = R_local.copy()
                else:
                    relative_rotations[t] = last_valid_local_R.copy()
                    relative_weights[t] = 1e-4
            t_ransac_end = time.time()
            logging.info(
                f"[TC Profile] RANSAC & IRLS rigid fits ({len(indices)} frames) took {t_ransac_end - t_ransac_start:.3f}s"
            )

            # Save raw window tracks and relative geometry
            all_window_tracks.append(
                {
                    "anchor": anchor,
                    "indices": indices,
                    "tracks": pred_run,
                    "query_uv": query_uv_w,
                    "relative_rotations": relative_rotations,
                    "relative_weights": relative_weights,
                }
            )

        # Select next anchor dynamically from the overlap region
        if anchor + win_size_actual >= num_frames_video:
            break

        overlap_start = anchor + 8
        overlap_end = min(num_frames_video, anchor + 11)

        best_t = anchor + 9
        best_q = -1.0
        for t in range(overlap_start, overlap_end):
            area_r = masks_original[t].sum() / masks_original[0].sum()
            q_score = area_r * 0.5
            if q_score > best_q:
                best_q = q_score
                best_t = t

        next_anchor = best_t if best_q > 0.15 else anchor + 9
        anchor = next_anchor


    return WindowTrackingResult(
        all_window_tracks=all_window_tracks,
        canonical_points_0=canonical_points_0,
        canonical_radius_0=canonical_radius_0,
        canonical_offsets=canonical_offsets,
        ys_0=ys_0,
        xs_0=xs_0,
    )


def _recover_poses(data: PreparedTrackingData, tracking: WindowTrackingResult) -> RecoveredPoses:
    """Register the initial pose, blend windows, and smooth rotations and centers."""
    from termcolor import colored

    first_frame_ply = data.first_frame_ply
    num_frames_video = data.num_frames_video
    K_video = data.K_video
    aligned_depth = data.aligned_depth
    masks_original = data.masks_original
    all_window_tracks = tracking.all_window_tracks
    canonical_points_0 = tracking.canonical_points_0


    # 1. Initialize global_anchor_R using the PLY-aligned starting pose
    # Load PLY point cloud and compute initial alignment
    R_align = np.eye(3)
    try:
        import open3d as o3d

        first_frame_ply_path = Path(first_frame_ply).resolve()
        if os.path.exists(str(first_frame_ply_path)):
            pcd_ply = o3d.io.read_point_cloud(str(first_frame_ply_path))
            if not pcd_ply.is_empty():
                ply_pts = np.asarray(pcd_ply.points)
                # We will register canonical_points_0 (which is in frame 0 camera coordinates)
                # to the PLY point cloud to find the starting orientation
                pcd_ply_tree = o3d.geometry.KDTreeFlann(pcd_ply)
                correspondences = []
                valid_corr = []
                for i, p in enumerate(canonical_points_0):
                    [k, idx, _] = pcd_ply_tree.search_knn_vector_3d(p, 1)
                    if k > 0 and len(idx) > 0:
                        dist = np.linalg.norm(p - ply_pts[idx[0]])
                        if dist < 0.05:  # 5cm threshold
                            correspondences.append(ply_pts[idx[0]])
                            valid_corr.append(i)
                if len(valid_corr) >= 20:
                    P_tc = canonical_points_0[valid_corr]
                    Q_ply = np.array(correspondences)
                    R_align, _ = refine_rigid_irls(Q_ply, P_tc, num_iters=4)
                    logging.info(
                        colored("[TrackCraft Server] ", "green")
                        + "Successfully registered frame 0 against first_frame_ply."
                    )
    except Exception as e:
        logging.warning(
            f"Failed to load/register first_frame_ply: {e}. Using identity rotation for frame 0."
        )

    global_anchor_R = {0: R_align}
    visited_anchors = []

    rotations_by_frame_candidates = [[] for _ in range(num_frames_video)]
    weights_by_frame_candidates = [[] for _ in range(num_frames_video)]

    for w in all_window_tracks:
        anchor = w["anchor"]
        indices = w["indices"]
        rel_rots = w["relative_rotations"]
        rel_weights = w["relative_weights"]

        if anchor not in global_anchor_R:
            # Average of candidates at anchor from previous windows
            prev_cands = rotations_by_frame_candidates[anchor]
            prev_weights = weights_by_frame_candidates[anchor]
            if prev_cands:
                global_anchor_R[anchor] = blend_rotations(
                    prev_cands, weights=prev_weights
                )
            else:
                prev_anchor = visited_anchors[-2] if len(visited_anchors) > 1 else 0
                global_anchor_R[anchor] = global_anchor_R.get(
                    prev_anchor, np.eye(3)
                ).copy()

        R_global_anchor = global_anchor_R[anchor]
        visited_anchors.append(anchor)

        for t in indices:
            R_cand = rel_rots[t] @ R_global_anchor
            rotations_by_frame_candidates[t].append(R_cand)
            weights_by_frame_candidates[t].append(rel_weights[t])

    # Initialize rotations_global
    rotations_global = []
    rotation_valid_final = []
    pose_residual = []
    inlier_counts = []
    point_counts = []
    inlier_ratios = []
    rejection_reasons = []

    for t in range(num_frames_video):
        if rotations_by_frame_candidates[t]:
            R_blended = blend_rotations(
                rotations_by_frame_candidates[t],
                weights=weights_by_frame_candidates[t],
            )
            rotations_global.append(R_blended)
            rotation_valid_final.append(True)
            rejection_reasons.append("ok")
        else:
            rotations_global.append(np.eye(3))
            rotation_valid_final.append(False)
            rejection_reasons.append("no_window_coverage")

        # Fill dummy variables for stats/logging compatibility
        pose_residual.append(0.0)
        inlier_counts.append(
            len(canonical_points_0) if canonical_points_0 is not None else 100
        )
        point_counts.append(
            len(canonical_points_0) if canonical_points_0 is not None else 100
        )
        inlier_ratios.append(1.0)

    # Optimize pose-graph jointly using coordinate descent relaxation (5 iterations)
    for opt_iter in range(5):
        anchor_rotations = {}
        for w in all_window_tracks:
            anchor = w["anchor"]
            anchor_rotations[anchor] = rotations_global[anchor]

        for t in range(num_frames_video):
            cands_t = []
            weights_t = []
            for w in all_window_tracks:
                anchor = w["anchor"]
                indices = w["indices"]
                if t in indices:
                    R_cand = w["relative_rotations"][t] @ anchor_rotations[anchor]
                    cands_t.append(R_cand)
                    weights_t.append(w["relative_weights"][t])
            if cands_t:
                rotations_global[t] = blend_rotations(cands_t, weights=weights_t)

    # Pre-compute object centers directly in metric space using aligned depth
    t_centers_start = time.time()
    logging.info(
        "[TrackCraft Server] Pre-computing object translation centers from aligned depth..."
    )
    centers_ref = []
    for ti in range(num_frames_video):
        c_ref = center_from_mask_depth(
            masks_original[ti], aligned_depth[ti], K_video
        )
        if c_ref is None:
            c_ref = centers_ref[-1].copy() if len(centers_ref) > 0 else np.zeros(3)
        centers_ref.append(c_ref)
    centers_ref = np.array(centers_ref)
    t_centers_end = time.time()
    logging.info(
        f"[TC Profile] Pre-computing object translation centers took {t_centers_end - t_centers_start:.3f}s"
    )

    # Zero-lag bi-directional Gaussian SO(3) trajectory smoothing
    t_smooth_start = time.time()
    logging.info(
        "[TrackCraft Server] Applying zero-lag bi-directional Gaussian temporal pose smoothing..."
    )
    centers_smoothed = centers_ref.copy()
    wl = min(15, (num_frames_video // 2) * 2 - 1)
    if wl >= 5:
        try:
            centers_smoothed = savgol_filter(
                centers_ref,
                window_length=wl,
                polyorder=2,
                axis=0,
            )
        except Exception as e:
            logging.warning(f"Savgol filter failed: {e}. Using raw centers.")

    # Smooth optimized rotations
    rotations_filtered = smooth_rotations_gaussian(
        rotations_global, window_size=9, sigma=3.0
    )
    t_smooth_end = time.time()
    logging.info(
        f"[TC Profile] Zero-lag Gaussian temporal pose smoothing took {t_smooth_end - t_smooth_start:.3f}s"
    )


    return RecoveredPoses(
        rotations_filtered=rotations_filtered,
        centers_smoothed=centers_smoothed,
        rotation_valid_final=rotation_valid_final,
        pose_residual=pose_residual,
        inlier_counts=inlier_counts,
        point_counts=point_counts,
        inlier_ratios=inlier_ratios,
        rejection_reasons=rejection_reasons,
    )


def _align_window_clouds(
    data: PreparedTrackingData, tracking: WindowTrackingResult, poses: RecoveredPoses,
):
    """Add metric robot-frame clouds to the window records for export."""

    C2R = data.C2R
    scale = data.scale
    all_window_tracks = tracking.all_window_tracks
    centers_smoothed = poses.centers_smoothed


    # Post-process TrackCraft clouds without modifying the raw predictions.
    # Each cloud is translated so its centroid exactly matches the
    # aligned-depth center saved in trajectory.json.
    t_vectorized_start = time.time()
    C2R_f32 = np.asarray(C2R, dtype=np.float32)

    if not np.allclose(
        C2R_f32[3],
        np.array([0, 0, 0, 1], dtype=np.float32),
    ):
        raise ValueError("C2R must be an affine rigid transform.")

    R_c2r = C2R_f32[:3, :3]
    t_c2r = C2R_f32[:3, 3]

    for w in all_window_tracks:
        tracks_raw = np.asarray(w["tracks"], dtype=np.float32)
        tracks_metric = tracks_raw * np.float32(scale)

        valid = np.isfinite(tracks_metric).all(axis=2) & (
            tracks_metric[:, :, 2] > 0.01
        )

        counts = valid.sum(axis=1)
        safe_counts = np.maximum(counts, 1).astype(np.float32)

        source_centers = (
            np.where(valid[:, :, None], tracks_metric, 0.0).sum(axis=1)
            / safe_counts[:, None]
        )

        frame_indices = np.asarray(w["indices"], dtype=np.int64)
        target_centers = np.asarray(
            centers_smoothed[frame_indices],
            dtype=np.float32,
        )

        translations = target_centers - source_centers
        translations[counts == 0] = np.nan

        tracks_aligned = tracks_metric + translations[:, None, :]
        tracks_aligned[~valid] = np.nan

        # Vectorized affine camera-to-robot transformation.
        tracks_robot = tracks_aligned @ R_c2r.T + t_c2r[None, None, :]
        tracks_robot[~valid] = np.nan

        # Raw tracks remain untouched for the RGB visualization.
        w["tracks_robot"] = tracks_robot.astype(
            np.float32,
            copy=False,
        )
        w["center_alignment_translation"] = translations.astype(
            np.float32,
            copy=False,
        )
    t_vectorized_end = time.time()
    logging.info(
        f"[TC Profile] Vectorized centroid-alignment post-processing took {t_vectorized_end - t_vectorized_start:.3f}s"
    )



def _export_tracking_results(
    data: PreparedTrackingData, tracking: WindowTrackingResult, poses: RecoveredPoses,
) -> str:
    """Reconstruct rigid tracks and save dense tracks and trajectory poses."""
    import json
    from termcolor import colored

    output_dir = data.output_dir
    C2R = data.C2R
    num_frames_video = data.num_frames_video
    K_video = data.K_video
    scale = data.scale
    fixed_extrinsics = data.fixed_extrinsics
    masks_original = data.masks_original
    target_object = data.target_object
    intrinsics = data.intrinsics
    all_window_tracks = tracking.all_window_tracks
    canonical_points_0 = tracking.canonical_points_0
    canonical_radius_0 = tracking.canonical_radius_0
    canonical_offsets = tracking.canonical_offsets
    ys_0 = tracking.ys_0
    xs_0 = tracking.xs_0
    rotations_filtered = poses.rotations_filtered
    centers_smoothed = poses.centers_smoothed
    rotation_valid_final = poses.rotation_valid_final
    pose_residual = poses.pose_residual
    inlier_counts = poses.inlier_counts
    point_counts = poses.point_counts
    inlier_ratios = poses.inlier_ratios
    rejection_reasons = poses.rejection_reasons


    # Reconstruct rigid tracks
    N_points = len(canonical_points_0)
    rigid_tracks = np.zeros((num_frames_video, N_points, 3), dtype=np.float32)
    for ti in range(num_frames_video):
        rigid_tracks[ti] = (
            canonical_offsets @ rotations_filtered[ti].T + centers_smoothed[ti]
        )

    # Transform rigid tracks to robot space
    pts_tracks_robot_rigid = np.zeros_like(rigid_tracks)
    for ti in range(num_frames_video):
        points_ref = rigid_tracks[ti]
        points_h = np.concatenate([points_ref, np.ones((N_points, 1))], axis=-1)
        pts_robot_h = (C2R @ points_h.T).T
        pts_tracks_robot_rigid[ti] = pts_robot_h[:, :3] / pts_robot_h[:, 3:]

    # Log frame-by-frame details
    for ti in range(num_frames_video):
        if ti % 20 != 0 and ti != num_frames_video - 1 and rotation_valid_final[ti]:
            continue

        mask_t = (masks_original[ti] > 0).astype(np.uint8)
        logging.info(
            f"frame={ti}, "
            f"mask_area={int(mask_t.sum())}, "
            f"rotation_valid={rotation_valid_final[ti]}, "
            f"inliers={inlier_counts[ti]}/{point_counts[ti]}, "
            f"inlier_ratio={inlier_ratios[ti]:.3f}, "
            f"residual_ratio={pose_residual[ti] / canonical_radius_0:.3f}, "
            f"reason={rejection_reasons[ti]}"
        )

    # Generate trajectory for all frames
    trajectory_full = []
    indices = np.array(list(range(num_frames_video)))
    for ti in range(num_frames_video):
        r_valid = rotation_valid_final[ti]
        c_cam = centers_smoothed[ti].copy()
        R_camera = rotations_filtered[ti].copy()

        # Compute robot rotation/translation rigidly using C2R
        c_ref_h = np.append(c_cam, 1.0)
        c_robot_h = C2R @ c_ref_h
        c_rob = c_robot_h[:3] / c_robot_h[3]

        R_C2R = C2R[:3, :3]
        R_rob_clean = R_C2R @ R_camera @ R_C2R.T

        trajectory_full.append(
            {
                "frame": ti,
                "valid": True,
                "rotation_valid": bool(r_valid),
                "target_object": target_object,
                "center_robot": c_rob.tolist(),
                "center_camera": c_cam.tolist(),
                "rotation_robot": R_rob_clean.tolist(),
                "rotation_camera": R_camera.tolist(),
                "K": K_video.tolist(),
            }
        )

    # Save full dense tracks and trajectory
    np.savez_compressed(
        output_dir / "dense_tracks.npz",
        tracks_robot=pts_tracks_robot_rigid,
        tracks_reference=rigid_tracks,
        canonical_points=canonical_points_0,
        pose_residual=np.asarray(pose_residual),
        rotation_valid=np.asarray(rotation_valid_final),
        extrinsics_full=fixed_extrinsics,
        mask_pixels=np.stack([ys_0, xs_0], axis=-1),
        indices=indices,
        scale=scale,
        vis_map=np.ones((num_frames_video, N_points), dtype=np.float32),
        window_tracks=all_window_tracks,
        fx_fy_cx_cy=intrinsics,
    )

    traj_output_path = output_dir / "trajectory.json"
    with open(traj_output_path, "w") as f:
        json.dump(trajectory_full, f, indent=4)

    logging.info(
        colored("[TrackCraft Server] ", "green") + f"Saved results to {output_dir}"
    )
    return str(traj_output_path)


def trackcraft_api(
    video_path,
    masks_path,
    depth_path,
    first_frame_ply,
    camera_intrinsics_yaml,
    c2r_path,
    output_dir,
):
    from termcolor import colored

    logging.info(
        colored("[TrackCraft Server] ", "cyan")
        + f"Processing track request for video: {video_path}"
    )

    # Inject TrackCraft3r path to sys.path for evaluation imports
    trackcraft_root = Path(__file__).resolve().parent.parent.parent / "TrackCraft3r"
    if str(trackcraft_root) not in sys.path:
        sys.path.insert(0, str(trackcraft_root))

    from huggingface_hub import hf_hub_download

    from evaluation.dust3r_eval_utils import load_npz_data
    from evaluation.wan_scene_flow_predictor import WanSceneFlowPredictor

    data = _prepare_tracking_data(
        video_path, masks_path, depth_path, first_frame_ply,
        camera_intrinsics_yaml, c2r_path, output_dir, trackcraft_root, load_npz_data,
    )
    predictor = _get_predictor(
        data, trackcraft_root, hf_hub_download, WanSceneFlowPredictor,
    )
    tracking = _track_windows(data, predictor)
    poses = _recover_poses(data, tracking)
    _align_window_clouds(data, tracking, poses)
    return _export_tracking_results(data, tracking, poses)


def serve_trackcraft(port=29004):
    serve(create_track_app(trackcraft_api), "localhost", port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    serve_trackcraft()
