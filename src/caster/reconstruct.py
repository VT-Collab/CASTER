"""Create scene masks and SAM 3D meshes.

Required Hydra override: scene_num.
set generate_masks=false only to reuse existing masks.
"""

import io
import os
import re
import sys
import json
import time
import signal
import shutil
import logging
import zipfile
import subprocess
from pathlib import Path
from urllib.parse import urlparse
from contextlib import contextmanager

import zmq
import hydra
import trimesh
import numpy as np
import requests
from PIL import Image
from termcolor import colored
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
from hydra.core.global_hydra import GlobalHydra

from caster.llm import GeminiER2
from caster.utils import InferenceError, check_service, request_inference


logger = logging.getLogger(__name__)


@contextmanager
def _m2t2_server(url: str):
    address = urlparse(url)
    host = address.hostname
    port = address.port or 80
    process = None

    try:
        if host in {"localhost", "127.0.0.1"}:
            try:
                check_service(url, "m2t2")
                logger.info("Reusing M2T2 server at %s", url)
            except requests.ConnectionError:
                logger.info("Starting M2T2 server at %s", url)
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "caster.servers.m2t2_server",
                        f"host={host}",
                        f"port={port}",
                    ],
                    start_new_session=True,
                )
                deadline = time.monotonic() + 900
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(
                            f"M2T2 server exited with code {process.returncode}"
                        )
                    try:
                        check_service(url, "m2t2")
                        break
                    except requests.ConnectionError:
                        if time.monotonic() > deadline:
                            raise TimeoutError(
                                "M2T2 server did not start within 15 minutes"
                            )
                        time.sleep(1)

        yield url
    finally:
        if process is not None and process.poll() is None:
            logger.info("Stopping M2T2 server")
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def _detections(image_path: Path, prompt: str, model: str, filter_phrases: list[str]) -> list[dict]:
    annotations = image_path.with_name("object_detections.json")
    if annotations.is_file():
        return json.loads(annotations.read_text())

    with GeminiER2(model) as gemini:
        detections = gemini.points(image_path, prompt, temperature=0.5)
    return [
        item for item in detections
        if not any(phrase in word for word in item["label"].lower().split() for phrase in filter_phrases)
    ]


def _predict_masks(segmenter, image: Image.Image, label: str):
    result = segmenter.gdino.predict(
        [image], [label], box_threshold=0.3, text_threshold=0.25
    )[0]
    boxes = result["boxes"]
    if len(boxes) == 0:
        return []
    boxes = boxes.cpu().numpy() if hasattr(boxes, "cpu") else np.asarray(boxes)
    masks, _, _ = segmenter.sam.predict(np.asarray(image).copy(), xyxy=boxes)
    return masks


def _make_masks(image_path: Path, masks_dir: Path, prompt: str, model: str, filter_phrases: list[str]) -> list[Path]:
    from lang_sam import LangSAM

    loader = GlobalHydra.instance().hydra.config_loader
    search_path = loader.get_search_path()
    if not any(item.path == "pkg://sam2" for item in search_path.config_search_path):
        search_path.prepend("sam2", "pkg://sam2")
        loader.repository.initialize_sources(search_path)

    detections = _detections(image_path, prompt, model, filter_phrases)
    if not detections:
        raise ValueError("Object detection returned no objects")
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    segmenter = LangSAM()
    masks_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for detection in detections:
        label = detection["label"]
        point = np.asarray(detection["point"], dtype=float)
        name = re.sub(r"[^a-zA-Z0-9_-]", "_", label)
        if "box_xyxy" in detection:
            box = np.asarray(detection["box_xyxy"], dtype=float)
            predictor = segmenter.sam.predictor
            predictor.set_image(np.asarray(image).copy())
            masks, _, _ = predictor.predict(
                box=box * [width, height, width, height] / 1000,
                point_coords=np.array([[point[1] * width / 1000, point[0] * height / 1000]]),
                point_labels=np.array([1]),
                multimask_output=False,
            )
        else:
            masks = _predict_masks(segmenter, image, label)
        x = min(round(point[1] * width / 1000), width - 1)
        y = min(round(point[0] * height / 1000), height - 1)
        chosen = next((np.asarray(mask) for mask in masks if np.asarray(mask)[y, x] > 0), None)
        if chosen is None:
            logger.warning("No mask found for %s", label)
            continue
        path = masks_dir / f"{name}.png"
        Image.fromarray((chosen > 0).astype("uint8") * 255).save(path)
        paths.append(path)
    if not paths:
        raise RuntimeError("No object masks were generated")
    return paths


def _reconstruct_meshes(
    image_path: Path,
    ply_path: Path,
    c2r_path: Path,
    intrinsics: Path,
    masks: list[Path],
    meshes_dir: Path,
    server_addr: str,
) -> None:
    mask_archive = io.BytesIO()
    with zipfile.ZipFile(mask_archive, "w", zipfile.ZIP_DEFLATED) as archive:
        for mask in masks:
            archive.write(mask, mask.name)

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, 30 * 60 * 1000)
    try:
        socket.connect(server_addr)
        socket.send_multipart([
            b"predict",
            image_path.read_bytes(),
            ply_path.read_bytes(),
            mask_archive.getvalue(),
            np.asarray(np.load(c2r_path), dtype=np.float64).tobytes(),
            intrinsics.read_bytes(),
        ])
        status, payload = socket.recv_multipart()
    finally:
        socket.close()
        context.term()
    if status != b"ok":
        raise RuntimeError(f"SAM 3D server: {payload.decode('utf-8')}")

    meshes_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(meshes_dir)
    transforms = json.loads((meshes_dir / "transforms.json").read_text())
    for item in transforms["objects"]:
        glb = meshes_dir / item["glb"]
        trimesh.load(glb, force="mesh").export(glb.with_suffix(".obj"))
        logger.info("%s %s", colored("Saved mesh:", "green"), glb)


def _generate_grasps(
    meshes_dir: Path, image_path: Path, c2r_path: Path, intrinsics: Path, config: DictConfig,
) -> Path:
    from caster.utils import render_grasps

    transforms_path = meshes_dir / "transforms.json"
    names = [item["name"] for item in json.loads(transforms_path.read_text())["objects"]]
    results = {}
    errors = {}

    with GeminiER2(config.gemini_model) as gemini:
        with _m2t2_server(config.m2t2_server_url) as server_url:
            for name in names:
                try:
                    point = gemini.points(
                        image_path, config.grasp_prompt.replace("{object_name}", name), temperature=0.2,
                    )[0]["point"]
                    point = [max(0.0, min(1000.0, float(value))) for value in point]
                except Exception as error:
                    logger.warning("Grasp pinpoint failed for %s: %s", name, error)
                    point = None
                try:
                    results[name] = request_inference(
                        server_url,
                        "grasp",
                        {
                            "transforms_path": str(transforms_path),
                            "target": name,
                            "point_yx": point,
                            "table_aware": config.grasp_table_aware,
                            "image_path": str(image_path),
                            "c2r_path": str(c2r_path),
                            "intrinsics_path": str(intrinsics),
                        },
                        config.server_timeout,
                    )
                    logger.info("%s %s", colored("Generated grasp:", "green"), name)
                except InferenceError as error:
                    errors[name] = str(error)
                    logger.warning("M2T2 failed for %s: %s", name, error)

    if not results:
        raise RuntimeError(f"M2T2 failed for all objects: {errors}")
    visualization = render_grasps(transforms_path, image_path, results)
    saved_results = {
        name: {key: value for key, value in result.items()
               if key not in {"all_candidate_transforms", "all_candidate_confidences"}}
        for name, result in results.items()
    }
    output = meshes_dir / "grasp.json"
    output.write_text(json.dumps({
        "server": "m2t2_grasp", "mode": "all_objects", "objects": saved_results,
        "errors": errors, "visualization": str(visualization),
    }, indent=2))
    logger.info("%s %s", colored("Saved grasps:", "green"), output)
    logger.info("%s %s", colored("Saved visualization:", "green"), visualization)
    return output


def reconstruct(config: DictConfig) -> Path:
    scene_dir = Path(to_absolute_path(f"dataset/scene_{config.scene_num}"))
    source_assets = scene_dir / "assets"
    assets = Path(to_absolute_path(config.output_dir)) if config.output_dir else source_assets
    image_path = source_assets / "rgb.png"
    ply_path = source_assets / "pointcloud.ply"
    c2r_path = source_assets / "C2R.npy"
    intrinsics = Path(to_absolute_path(config.camera_intrinsics))
    masks_dir = assets / "masks"
    if config.generate_masks:
        logger.info(colored("Generating object masks", "cyan"))
        masks = _make_masks(
            image_path, masks_dir, config.detection_prompt, config.gemini_model,
            list(config.filter_phrases),
        )
    else:
        source_masks = source_assets / "masks"
        masks = sorted(source_masks.glob("*.png"))
        if not masks:
            raise FileNotFoundError(f"No masks in {source_masks}; set generate_masks=true")
        masks_dir.mkdir(parents=True, exist_ok=True)
        if masks_dir.resolve() != source_masks.resolve():
            for mask in masks:
                shutil.copy2(mask, masks_dir / mask.name)
        masks = [masks_dir / mask.name for mask in masks]
    logger.info("%s %d", colored("Object masks:", "green"), len(masks))

    meshes_dir = assets / "meshes"
    logger.info(colored("Reconstructing meshes with SAM 3D", "cyan"))
    _reconstruct_meshes(
        image_path, ply_path, c2r_path, intrinsics, masks, meshes_dir,
        config.sam3d_server_addr,
    )
    if config.generate_grasps:
        logger.info(colored("Generating grasps with M2T2", "cyan"))
        _generate_grasps(meshes_dir, image_path, c2r_path, intrinsics, config)
    logger.info("%s %s", colored("Scene assets ready:", "green"), assets)
    return assets


@hydra.main(version_base=None, config_path="config", config_name="reconstruct")
def main(config: DictConfig) -> None:
    reconstruct(config)


if __name__ == "__main__":
    main()
