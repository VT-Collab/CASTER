import os
import sys
import glob
import json
import types
import logging
import hashlib
import sysconfig
from pathlib import Path

import hydra
import trimesh
import numpy as np
from termcolor import colored
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from pydantic import Field

from caster.servers.http_service import InferenceRequest, create_app, run_inference, serve


class GraspRequest(InferenceRequest):
    transforms_path: str
    target: str
    point_yx: list[float] | None = Field(min_length=2, max_length=2)
    table_aware: bool
    image_path: str
    c2r_path: str
    intrinsics_path: str


def create_grasp_app(grasp):
    app = create_app("m2t2")

    @app.post("/grasp")
    def grasp_request(request: GraspRequest) -> dict:
        return run_inference(grasp, **request.model_dump())

    return app


logger = logging.getLogger(__name__)


def _use_uv_cuda() -> Path:
    cuda = Path(sysconfig.get_path("purelib")) / "nvidia/cu13"
    for name in ("PATH", "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH", "LD_LIBRARY_PATH"):
        paths = os.environ.get(name, "").split(os.pathsep)
        os.environ[name] = os.pathsep.join(path for path in paths if "/cuda/" not in path.lower())
    os.environ["CUDA_HOME"] = str(cuda)
    os.environ["CUDA_PATH"] = str(cuda)
    os.environ["CUDA_ROOT"] = str(cuda)
    os.environ["PATH"] = f"{cuda / 'bin'}{os.pathsep}{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}"
    return cuda


def _load_model(root: Path, checkpoint: str | None):
    cuda = _use_uv_cuda()

    import torch
    from torch.utils.cpp_extension import get_default_build_root, load

    source = root / "pointnet2_ops/pointnet2_ops/_ext-src"
    sources = glob.glob(str(source / "src/*.cpp")) + glob.glob(str(source / "src/*.cu"))
    build_root = Path(os.environ.get("TORCH_EXTENSIONS_DIR", get_default_build_root()))
    tag = hashlib.sha256(str(cuda).encode()).hexdigest()[:12]
    link_dir = build_root / f"caster_cuda_{tag}"
    link_dir.mkdir(parents=True, exist_ok=True)
    runtime_link = link_dir / "libcudart.so"
    if not runtime_link.is_symlink():
        runtime_link.symlink_to(cuda / "lib/libcudart.so.13")
    extension = load(
        "_ext", sources, extra_include_paths=[str(source / "include")],
        extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "-Xfatbin", "-compress-all"],
        extra_ldflags=[f"-L{link_dir}", f"-Wl,-rpath,{cuda / 'lib'}"],
        with_cuda=True,
    )
    pointnet2_ops = types.ModuleType("pointnet2_ops")
    pointnet2_ops._ext = extension
    sys.modules["pointnet2_ops"] = pointnet2_ops
    sys.path.insert(0, str(root))

    from m2t2.m2t2 import M2T2

    weights = checkpoint or hf_hub_download("wentao-yuan/m2t2", "m2t2.pth")
    config = OmegaConf.load(root / "config.yaml")
    model = M2T2.from_config(config.m2t2)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    model = model.cuda().eval()
    logger.info("%s %s", colored("M2T2 ready:", "green"), weights)
    return model, config


def _scene_points(transforms_path: Path, target: str):
    objects = json.loads(transforms_path.read_text())["objects"]
    points = []
    target_points = None
    per_object = max(500, 16384 // len(objects))
    correction = Rotation.from_quat([np.sqrt(0.5), 0, 0, np.sqrt(0.5)]).as_matrix()

    for item in objects:
        name = item["name"]
        mesh_path = transforms_path.parent / f"{name}.obj"
        if not mesh_path.exists():
            mesh_path = transforms_path.parent / item["glb"]
        mesh = trimesh.load(mesh_path, force="mesh")
        mesh.apply_scale(float(item["scale"]))
        samples, _ = trimesh.sample.sample_surface(mesh, per_object)
        w, x, y, z = item["rotation_wxyz"]
        rotation = Rotation.from_quat([x, y, z, w]).as_matrix() @ correction
        samples = samples @ rotation.T + np.asarray(item["translation"])
        points.append(samples)
        if name == target:
            target_points = samples

    scene = np.vstack(points)
    x = np.linspace(scene[:, 0].min() - 0.15, scene[:, 0].max() + 0.15, 40)
    y = np.linspace(scene[:, 1].min() - 0.15, scene[:, 1].max() + 0.15, 40)
    grid_x, grid_y = np.meshgrid(x, y)
    table = np.column_stack((grid_x.ravel(), grid_y.ravel(), np.full(grid_x.size, scene[:, 2].min())))
    return np.vstack((scene, table)), target_points


def _candidate_grasps(model, config, points: np.ndarray, target_points: np.ndarray):
    import torch

    count = int(config.data.get("num_points", 4096))
    chosen = np.random.choice(len(points), count, replace=len(points) < count)
    samples = torch.as_tensor(points[chosen], dtype=torch.float32, device="cuda")
    rgb = torch.zeros_like(samples)
    object_count = int(config.data.get("num_object_points", 1024))
    batch = {
        "inputs": torch.cat((samples - samples.mean(0), rgb), dim=-1)[None],
        "points": samples[None],
        "seg": torch.ones((1, count), dtype=torch.long, device="cuda"),
        "object_inputs": torch.cat((samples[:object_count], rgb[:object_count]), dim=-1)[None],
        "object_center": torch.zeros(1, 3, device="cuda"),
        "cam_pose": torch.eye(4, device="cuda"),
        "bottom_center": torch.zeros(1, 3, device="cuda"),
        "ee_pose": torch.eye(4, device="cuda")[None],
        "task": "pick",
        "task_is_place": torch.tensor([0.0], device="cuda"),
    }
    for mask_threshold, object_threshold in ((0.4, 0.4), (0.2, 0.2), (0.05, 0.01)):
        config.eval.mask_thresh = mask_threshold
        config.eval.object_thresh = object_threshold
        with torch.no_grad():
            output = model.infer(batch, config.eval)
        groups = output.get("grasps", [])
        if not groups or not groups[0] or not any(len(group) for group in groups[0]):
            continue
        grasps = torch.cat(groups[0]).cpu().numpy()
        confidence = torch.cat(output["grasp_confidence"][0]).cpu().numpy()
        contacts = torch.cat(output["grasp_contacts"][0]).cpu().numpy()
        on_target = cKDTree(target_points).query(contacts)[0] <= 1e-4
        if on_target.any():
            return grasps[on_target], confidence[on_target], contacts[on_target]
    raise RuntimeError("M2T2 produced no grasps on the target object")


def _project_contacts(contacts: np.ndarray, c2r_path: Path, intrinsics_path: Path, image_size):
    intrinsics = OmegaConf.load(intrinsics_path)
    width, height = image_size
    camera_from_robot = np.linalg.inv(np.load(c2r_path))
    homogeneous = np.column_stack((contacts, np.ones(len(contacts))))
    camera = (camera_from_robot @ homogeneous.T).T[:, :3]
    scale_x = width / float(intrinsics.width)
    scale_y = height / float(intrinsics.height)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = float(intrinsics.fx) * scale_x * camera[:, 0] / camera[:, 2] + float(intrinsics.cx) * scale_x
        v = float(intrinsics.fy) * scale_y * camera[:, 1] / camera[:, 2] + float(intrinsics.cy) * scale_y
    u[camera[:, 2] <= 0] = np.nan
    v[camera[:, 2] <= 0] = np.nan
    return u, v


class GraspServer:
    def __init__(self, root: Path, checkpoint: str | None):
        self.model, self.config = _load_model(root, checkpoint)

    def grasp(self, transforms_path: str, target: str, point_yx: list | None,
              table_aware: bool, image_path: str, c2r_path: str, intrinsics_path: str) -> dict:
        from PIL import Image

        transforms = Path(transforms_path)
        points, target_points = _scene_points(transforms, target)
        grasps, confidence, contacts = _candidate_grasps(self.model, self.config, points, target_points)
        approach_z = grasps[:, :3, 2] @ np.array([0.0, 0.0, 1.0])
        valid = np.flatnonzero(approach_z <= 0.2) if table_aware else np.arange(len(grasps))
        if not len(valid):
            valid = np.array([int(np.argmin(approach_z))])

        chosen = int(valid[np.argmax(confidence[valid])])
        vlm_pixel = None
        chosen_pixel = None
        if point_yx is not None:
            width, height = Image.open(image_path).size
            vlm_pixel = [float(point_yx[1]) * (width - 1) / 1000, float(point_yx[0]) * (height - 1) / 1000]
            u, v = _project_contacts(contacts, Path(c2r_path), Path(intrinsics_path), (width, height))
            distance = np.hypot(u - vlm_pixel[0], v - vlm_pixel[1])
            projected = valid[np.isfinite(distance[valid])]
            if len(projected):
                anchor = int(projected[np.argmin(distance[projected])])
                anchor_distance = np.linalg.norm(contacts - contacts[anchor], axis=1)
                for radius_px, radius_m in ((25, 0.03), (50, 0.05), (100, 0.08)):
                    nearby = projected[(distance[projected] <= radius_px) & (anchor_distance[projected] <= radius_m)]
                    if len(nearby):
                        chosen = int(nearby[np.argmax(confidence[nearby])])
                        break
                else:
                    chosen = anchor
                chosen_pixel = [float(u[chosen]), float(v[chosen])]

        pose = grasps[chosen]
        qx, qy, qz, qw = Rotation.from_matrix(pose[:3, :3]).as_quat()
        return {
            "object": target, "gripper": "franka_panda", "frame": "scene",
            "position": pose[:3, 3].tolist(),
            "orientation_wxyz": [float(qw), float(qx), float(qy), float(qz)],
            "transform_matrix": pose.tolist(),
            "confidence": float(confidence[chosen]),
            "approach_z_scene": float(approach_z[chosen]),
            "num_candidates": len(grasps),
            "chosen_index": chosen,
            "chosen_contact_3d": contacts[chosen].tolist(),
            "chosen_contact_2d": chosen_pixel,
            "vlm_pinpoint_normalized_yx": point_yx,
            "vlm_pinpoint_2d": vlm_pixel,
            "all_candidate_transforms": grasps.tolist(),
            "all_candidate_confidences": confidence.tolist(),
        }


@hydra.main(version_base=None, config_path="../config/servers", config_name="m2t2_server")
def main(config: DictConfig) -> None:
    root = Path(to_absolute_path(config.m2t2_root))
    checkpoint = to_absolute_path(config.checkpoint) if config.checkpoint else None
    grasp_server = GraspServer(root, checkpoint)
    serve(create_grasp_app(grasp_server.grasp), config.host, int(config.port))


if __name__ == "__main__":
    main()
