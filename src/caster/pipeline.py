"""Extract object trajectories from existing scene videos.

Required Hydra overrides: scene_num and video_gen_prompt.
Start the DA3, DEVA, and TrackCraft servers before running this command.
"""

import re
import json
import logging
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from termcolor import colored

from caster.llm import infer_task_features
from caster.feature_extract import extract_task_features
from caster.reconstruct import reconstruct
from caster.utils import (
    generate_depth, real_depth_gen, render_depth, render_masks, request_inference,
)


logger = logging.getLogger(__name__)


def _existing_demo_numbers(
    output_root: Path, feature_dir: Path, task: str
) -> tuple[dict[str, int], int]:
    numbers = {}
    highest = 0
    for folder, suffix in ((output_root, f"_{task}"), (feature_dir, "")):
        pattern = re.compile(r"demo_(\d+)_seed_(.+)" + re.escape(suffix))
        for demo in folder.glob("demo_*_seed_*"):
            if not demo.is_dir():
                continue
            index_match = re.match(r"demo_(\d+)_seed_", demo.name)
            if index_match:
                highest = max(highest, int(index_match.group(1)))
            match = pattern.fullmatch(demo.name)
            if match:
                index, label = int(match.group(1)), match.group(2)
                if label in numbers and numbers[label] != index:
                    raise ValueError(f"Conflicting demo numbers for seed {label}: {numbers[label]} and {index}")
                numbers[label] = index
    return numbers, highest + 1


def _task_features(
    config: DictConfig, scene: Path, objects: list[str], task: str
) -> tuple[list[dict], Path]:
    folder = scene / f"cf_{config.cf_index}_{task}"
    file = folder / "task_feature.json"
    if file.is_file():
        return json.loads(file.read_text()), folder

    instruction = config.llm_prompt
    prompt = (
        f"Video Description: {config.video_gen_prompt.prompt}\n\n"
        f"Instruction: {instruction.prompt}\n"
        f"Output Format: {instruction.output_format}\n"
        f"Example: {instruction.example}\n"
        f"Constraints: {instruction.constraints}\n"
        f"Available Objects: {objects}"
    )
    features = infer_task_features(prompt)
    folder.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(features, indent=2))
    logger.info("%s %s", colored("Saved task features:", "green"), file)
    return features, folder


def pipeline(config: DictConfig) -> list[Path]:
    scene = Path(to_absolute_path(f"dataset/scene_{config.scene_num}"))
    output_root = Path(to_absolute_path(config.output_dir))
    assets = scene / "assets"
    task = HydraConfig.get().runtime.choices.video_gen_prompt
    meshes = assets / "meshes"

    # Prepare scene assets
    if not list(meshes.glob("*.obj")):
        logger.info(colored("No object meshes found; reconstructing scene", "cyan"))
        reconstruction = OmegaConf.load(
            Path(__file__).parent / "config/reconstruct.yaml"
        )
        reconstruction.scene_num = config.scene_num
        reconstruct(reconstruction)

    # Get task feature dimensions
    objects = sorted(mesh.stem for mesh in meshes.glob("*.obj"))
    features, feature_dir = _task_features(config, output_root, objects, task)
    tracked_objects = sorted(
        {
            feature[key]
            for feature in features
            for key in ("target_object", "reference_object")
            if key in feature
        }
    )
    if not tracked_objects:
        raise ValueError("Task feature list contains no objects to track")

    intrinsics = Path(to_absolute_path(config.camera_intrinsics))
    video_dir = Path(to_absolute_path(config.video_dir))
    videos = sorted(video_dir.glob("*.mp4"), key=lambda path: path.name)
    if config.valid_threshold is not None:
        results = json.loads((video_dir / "validation_results.json").read_text())
        videos = [video for video in videos if float(results.get(video.stem.split("_")[-1], {}).get("score", 0))
            >= config.valid_threshold]
    if not videos:
        raise FileNotFoundError(f"No input videos found in {video_dir}")

    demo_numbers, next_index = _existing_demo_numbers(output_root, feature_dir, task)
    trajectories = []
    demo_dirs = []
    for video in videos:
        seed = re.search(r"seed_(\d+)", video.stem)
        label = seed.group(1) if seed else video.stem
        if label not in demo_numbers:
            demo_numbers[label] = next_index
            next_index += 1
        index = demo_numbers[label]
        demo_dirs.append(feature_dir / f"demo_{index}_seed_{label}")
        demo = output_root / f"demo_{index}_seed_{label}_{task}"
        depth_dir = demo / "video_gen"
        depth_dir.mkdir(parents=True, exist_ok=True)
        aligned = depth_dir / "aligned_depth.npy"

        # 1. Depth generation ----------------------------------------------------------------
        raw = generate_depth(video, depth_dir, config.da3_server_url)
        # ------------------------------------------------------------------------------------

        # 2. Real depth alignment ------------------------------------------------------------
        real_depth_gen(
            raw,
            assets / "pointcloud.ply",
            intrinsics,
            aligned,
        )
        # ------------------------------------------------------------------------------------

        # 3. Trajectory recovery -------------------------------------------------------------
        mask_paths = []
        object_outputs = []
        for object_name in tracked_objects:
            name = object_name.replace(" ", "_")
            output = feature_dir / f"demo_{index}_seed_{label}" / name
            output.mkdir(parents=True, exist_ok=True)
            mask = output / "masks.npy"
            if not mask.is_file():
                reference = assets / "masks" / f"{name}.png"
                request_inference(
                    config.mask_server_url,
                    "mask",
                    {
                        "video_path": str(video),
                        "object_name": object_name,
                        "output_path": str(mask),
                        "reference_mask_path": str(reference),
                    },
                    config.server_timeout,
                )
            mask_paths.append(mask)
            object_outputs.append((output, mask))

        for output, mask in object_outputs:
            trajectory = output / "trajectory.json"
            if not trajectory.is_file():
                request_inference(
                    config.trackcraft_server_url,
                    "track",
                    {
                        "video_path": str(video),
                        "masks_path": str(mask),
                        "depth_path": str(aligned),
                        "first_frame_ply": str(assets / "pointcloud.ply"),
                        "camera_intrinsics_yaml": str(intrinsics),
                        "c2r_path": str(assets / "C2R.npy"),
                        "output_dir": str(output),
                    },
                    config.server_timeout,
                )
            trajectories.append(trajectory)
            logger.info("%s %s", colored("Saved trajectory:", "green"), trajectory)
        # ------------------------------------------------------------------------------------

        # 5. visualization -------------------------------------------------------------------
        if config.visualize:
            render_masks(
                video,
                mask_paths,
                feature_dir / f"demo_{index}_seed_{label}/mask_viz_all.mp4",
            )
            render_depth(aligned, depth_dir / "depth_video.mp4")
        # ------------------------------------------------------------------------------------
    extract_task_features(feature_dir, demo_dirs, dict(config.weighting))
    return trajectories


@hydra.main(version_base=None, config_path="config", config_name="pipeline")
def main(config: DictConfig) -> None:
    pipeline(config)


if __name__ == "__main__":
    main()
