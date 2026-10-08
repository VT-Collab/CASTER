# sam3d_server.py
#
# SAM-3D reconstruction server. This is a DEPLOY ARTIFACT: it runs inside the
# SAM-3D repo on the GPU box (tc-gpu003), not in semantic_grasp — it imports
# `notebook.inference` and reads `checkpoints/`, which live there. Copy it over
# as that repo's `server.py`.
#
# What changed vs the old server: the transform it returns is now the object's
# rigid pose in the ROBOT-BASE frame (translation, rotation_wxyz, scale), with
# the camera->robot C2R extrinsics baked in HERE. Downstream (semantic_grasp)
# consumes it as-is — no service re-applies C2R anymore, so C2R lives in exactly
# one place. Point C2R_PATH (env var) at the same C2R.npy the pipeline uses.
import cmd
import json
import os
import io
import sys
import tempfile
import zipfile
import traceback
from pathlib import Path
import yaml

# Automatically change directory to sam-3d-objects root so relative paths
# and relative imports (like notebook.inference) resolve correctly
server_dir = Path(__file__).resolve().parent
sam3d_root = Path(__file__).resolve().parents[2] / "sam-3d-objects"
os.chdir(str(sam3d_root))
sys.path.insert(0, str(sam3d_root))
sys.path.insert(0, str(sam3d_root / "notebook")) # Add the notebook folder directly to sys.path

import numpy as np
import open3d as o3d
import torch
import zmq
from scipy.spatial.transform import Rotation
from pytorch3d.transforms import quaternion_to_matrix

from inference import Inference, load_image, load_mask

# ── Config ─────────────────────────────────────────────────────────────────────
SAM3D_CONFIG = "checkpoints/hf/pipeline.yaml"
SEED = 42
BIND_ADDR = "tcp://0.0.0.0:5561"   # for IPC transport use e.g. "ipc:///tmp/sam3d.ipc"

# ── Coordinate-frame conversion ───────────────────────────────────────────────
# SAM-3D camera pose -> "blender" axis convention.
_C = np.array([[-1, 0, 0],
               [ 0, 0, 1],
               [ 0, 1, 0]], dtype=np.float64)

# ── C2R: "blender" camera frame -> robot base frame ───────────────────────────
# The calibrated 4x4 camera->robot extrinsics. Its rotation block is right-
# multiplied by R_BL_TO_CV so it consumes poses in the blender convention that
# `_C` above produces (blender -> CV -> robot). This is exactly the chain the
# pipeline's downstream services used to each run themselves; it now runs once,
# here, so the emitted pose is directly robot-frame usable.
# C2R_PATH = os.environ.get("C2R_PATH", "C2R.npy")
_R_BL_TO_CV = np.array([[1, 0, 0],
                        [0, 0, -1],
                        [0, 1, 0]], dtype=np.float64)
# _C2R_RAW = np.load(C2R_PATH)
# _C2R = _C2R_RAW.copy()
# _C2R[:3, :3] = _C2R_RAW[:3, :3] @ _R_BL_TO_CV
# print(f"Loaded C2R from {C2R_PATH}")

# ── Load model once at startup ─────────────────────────────────────────────────
print("Loading SAM3D model...")
model = Inference(SAM3D_CONFIG, compile=False)
print("Model ready.")


def load_intrinsics(intrinsics_yaml_bytes: bytes) -> dict[str, float | int]:
    """Parse the pipeline-supplied calibration; no server-local file is used."""
    try:
        intrinsics = yaml.safe_load(intrinsics_yaml_bytes.decode("utf-8"))
        result = {
            "fx": float(intrinsics["fx"]),
            "fy": float(intrinsics["fy"]),
            "cx": float(intrinsics["cx"]),
            "cy": float(intrinsics["cy"]),
            "width": int(intrinsics["width"]),
            "height": int(intrinsics["height"]),
        }
    except (UnicodeDecodeError, KeyError, TypeError, ValueError, yaml.YAMLError) as error:
        raise ValueError(
            "SAM-3D request calibration must define numeric fx, fy, cx, cy, width, and height"
        ) from error
    if (
        not np.isfinite([result["fx"], result["fy"], result["cx"], result["cy"]]).all()
        or result["fx"] <= 0
        or result["fy"] <= 0
        or result["width"] <= 0
        or result["height"] <= 0
    ):
        raise ValueError("SAM-3D request calibration has invalid focal lengths or resolution")
    return result


def ply_to_pointmap(ply_path: str, intrinsics: dict[str, float | int]) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points, dtype=np.float32)
    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])
    width, height = int(intrinsics["width"]), int(intrinsics["height"])
    pointmap = np.zeros((height, width, 3), dtype=np.float32)

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    valid = z > 0
    x, y, z = x[valid], y[valid], z[valid]

    u = np.round(fx * x / z + cx).astype(np.int32)
    v = np.round(fy * y / z + cy).astype(np.int32)

    in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v = u[in_bounds], v[in_bounds]
    x, y, z = x[in_bounds], y[in_bounds], z[in_bounds]

    pointmap[v, u] = np.stack([-x, -y, z], axis=1)

    filled = np.count_nonzero(pointmap[:, :, 2])
    print(f"[pointmap] {len(points):,} pts → {filled:,}/{width*height:,} pixels "
          f"({100*filled/(width*height):.1f}% filled)")
    return pointmap


def transform_to_robot(result: dict, c2r_raw: np.ndarray) -> dict:
    """SAM-3D result -> object rigid pose in the ROBOT-BASE frame.
    """
    q_sam = result["rotation"].squeeze().cpu()
    t_cam = result["translation"].squeeze().cpu().numpy().astype(np.float64)
    s_sam = result["scale"].squeeze().cpu().numpy().astype(np.float64)

    R_row = quaternion_to_matrix(q_sam).numpy().astype(np.float64)
    R_col = R_row.T

    # 1) camera -> blender
    R_blender = _C @ R_col
    t_blender = _C @ t_cam

    # 2) blender -> robot base (apply dynamic c2r with rotation offset)
    _C2R = c2r_raw.copy()
    _C2R[:3, :3] = c2r_raw[:3, :3] @ _R_BL_TO_CV

    R_robot = _C2R[:3, :3] @ R_blender
    t_robot = (_C2R @ np.array([t_blender[0], t_blender[1], t_blender[2], 1.0]))[:3]

    q_xyzw = Rotation.from_matrix(R_robot).as_quat()
    q_wxyz = [float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])]

    scale = float(s_sam.mean())
    return {
        "translation": t_robot.tolist(),
        "rotation_wxyz": q_wxyz,
        "scale": scale,
    }


def handle_predict(
    image_bytes: bytes,
    ply_bytes: bytes,
    masks_zip_bytes: bytes,
    c2r: np.ndarray,
    intrinsics: dict[str, float | int],
) -> bytes:
    """
    Runs inference and returns the result .zip as bytes.
    Inputs:
      - image_bytes: RGB capture (.png)
      - ply_bytes: point cloud (.ply)
      - masks_zip_bytes: a .zip containing mask PNGs
      - c2r: camera-to-robot transformation matrix
      - intrinsics: pipeline-selected RGB/PLY calibration
    """
    tmpdir = tempfile.mkdtemp()

    image_path = os.path.join(tmpdir, "image.png")
    ply_path = os.path.join(tmpdir, "pointcloud.ply")
    masks_dir = os.path.join(tmpdir, "masks")
    os.makedirs(masks_dir)

    with open(image_path, "wb") as f:
        f.write(image_bytes)
    with open(ply_path, "wb") as f:
        f.write(ply_bytes)

    masks_zip = os.path.join(tmpdir, "masks.zip")
    with open(masks_zip, "wb") as f:
        f.write(masks_zip_bytes)
    with zipfile.ZipFile(masks_zip, "r") as z:
        z.extractall(masks_dir)

    # Build pointmap
    pointmap_np = ply_to_pointmap(ply_path, intrinsics)
    pointmap_tensor = torch.from_numpy(pointmap_np)
    pointmap_tensor[pointmap_tensor.sum(dim=-1) == 0] = float("nan")

    # Load image
    img = load_image(image_path)

    # Process each mask
    mask_files = sorted(os.listdir(masks_dir))
    scene_json = {"objects": []}
    output_dir = os.path.join(tmpdir, "output")
    os.makedirs(output_dir)

    for mask_file in mask_files:
        if not mask_file.lower().endswith(".png"):
            continue
        name = os.path.splitext(mask_file)[0]
        mask_path = os.path.join(masks_dir, mask_file)
        print(f"\n── {name} ──")

        try:
            mask = load_mask(mask_path)
            result = model(img, mask, seed=SEED, pointmap=pointmap_tensor)

            glb = result.get("glb")
            if glb is None:
                print(f"  WARNING: no GLB for {name}")
                continue

            glb_path = os.path.join(output_dir, f"{name}.glb")
            glb.export(glb_path)
            print(f"  saved: {glb_path}")

            tfm = transform_to_robot(result, c2r)
            scene_json["objects"].append({
                "name": name,
                "glb": f"{name}.glb",
                "translation": tfm["translation"],
                "rotation_wxyz": tfm["rotation_wxyz"],
                "scale": tfm["scale"],
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            continue

    # Write transforms.json
    json_path = os.path.join(output_dir, "transforms.json")
    with open(json_path, "w") as f:
        json.dump(scene_json, f, indent=2)

    # Zip everything up
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(output_dir):
            zf.write(os.path.join(output_dir, fname), fname)
    buf.seek(0)
    return buf.getvalue()

def recv_request(sock):
        """
        Wire protocol (multipart REQ→REP):
          frame 0: command ("predict" or "health")
          for "predict":
            frame 1: image bytes
            frame 2: ply bytes
            frame 3: masks zip bytes
            frame 4: c2r bytes
            frame 5: camera intrinsics YAML bytes
        """
        frames = sock.recv_multipart()
        cmd = frames[0].decode("utf-8")
        return cmd, frames[1:]


def main():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(BIND_ADDR)
    print(f"ZMQ REP server listening on {BIND_ADDR}")

    while True:
        try:
            cmd, payload = recv_request(sock)

            if cmd == "health":
                sock.send_multipart([b"ok", json.dumps(
                    {"status": "ok", "model": "loaded"}
                ).encode("utf-8")])
                continue

            if cmd == "predict":
                if len(payload) != 5:
                    raise ValueError(
                        f"predict requires 5 payload frames, received {len(payload)}"
                    )
                image_bytes, ply_bytes, masks_zip_bytes, c2r_bytes, intrinsics_yaml_bytes = payload
                c2r = np.frombuffer(c2r_bytes, dtype=np.float64).reshape(4, 4)
                intrinsics = load_intrinsics(intrinsics_yaml_bytes)
                print(
                    "[SAM3D] Request calibration: "
                    f"fx={intrinsics['fx']:.3f}, fy={intrinsics['fy']:.3f}, "
                    f"cx={intrinsics['cx']:.3f}, cy={intrinsics['cy']:.3f}, "
                    f"{intrinsics['width']}x{intrinsics['height']}",
                    flush=True,
                )
                result_zip = handle_predict(
                    image_bytes, ply_bytes, masks_zip_bytes, c2r, intrinsics
                )
                sock.send_multipart([b"ok", result_zip])
                continue

            # Unknown command
            sock.send_multipart([b"error", f"unknown command: {cmd}".encode("utf-8")])

        except Exception as e:
            traceback.print_exc()
            # REP sockets MUST reply to keep the state machine happy
            try:
                sock.send_multipart([b"error", str(e).encode("utf-8")])
            except zmq.ZMQError:
                pass


if __name__ == "__main__":
    main()
