"""Generate image-conditioned videos with HunyuanVideo 1.5 and Diffusers.

Required Hydra overrides:
    video_gen_prompt: Name of the prompt config.
    scene_num: Scene number under dataset/.

Use seeds=[0,1] for multiple videos; image_path and output_dir can be overridden.
"""

import json
import logging
from pathlib import Path

import hydra
import torch
from PIL import Image
from termcolor import colored
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
from diffusers.utils import export_to_video
from diffusers import HunyuanVideo15ImageToVideoPipeline

from caster.llm.video_validator import VideoValidator, validate_video


logger = logging.getLogger(__name__)


class VideoGenerator:

    def __init__(self, model_id: str) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("HunyuanVideo 1.5 generation requires a CUDA GPU")

        logger.info("%s %s", colored("Loading model:", "cyan"), model_id)
        self.pipeline = HunyuanVideo15ImageToVideoPipeline.from_pretrained(
            model_id, dtype=torch.bfloat16
        )
        self.pipeline.enable_model_cpu_offload()
        self.pipeline.vae.enable_tiling()

    def generate(
        self,
        prompt: str,
        image: Image.Image,
        output_path: str | Path,
        *,
        seed: int,
        num_frames: int,
        num_inference_steps: int,
        fps: int,
    ) -> Path:
        output_path = Path(output_path).resolve()
        logger.info("%s %s", colored("Generating seed:", "cyan"), seed)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        frames = self.pipeline(
            prompt=prompt,
            image=image,
            generator=generator,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
        ).frames[0]
        export_to_video(frames, str(output_path), fps=fps)
        logger.info("%s %s", colored("Saved video:", "green"), output_path)
        return output_path


def video_gen(
    config: DictConfig,
) -> list[Path]:
    """Generate and validate one MP4 per configured seed."""
    prompt = config.video_gen_prompt.prompt
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt config must contain a nonempty 'prompt' string")

    seeds = list(config.seeds)
    if not seeds or any(not isinstance(seed, int) for seed in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be a nonempty list of unique integers")
    output_dir = Path(to_absolute_path(config.output_dir))
    paths = [output_dir / f"output_seed_{seed}.mp4" for seed in seeds]

    with Image.open(to_absolute_path(config.image_path)) as source:
        image = source.convert("RGB")

    generator = VideoGenerator(config.model_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "validation_results.json"
    results = json.loads(results_path.read_text()) if results_path.is_file() else {}
    validator = VideoValidator(config.validation)
    for seed, path in zip(seeds, paths):
        generator.generate(
            prompt,
            image,
            path,
            seed=seed,
            num_frames=config.num_frames,
            num_inference_steps=config.num_inference_steps,
            fps=config.fps,
        )
        result = validate_video(path, prompt, validator)
        result["seed"] = seed
        results[str(seed)] = result
        temporary = results_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(results_path)
        logger.info("Seed %s: valid=%s, score=%.3f", seed, result["valid"], result["score"])
    return [path.resolve() for path in paths]


@hydra.main(version_base=None, config_path="config", config_name="video_gen")
def main(config: DictConfig) -> None:
    video_gen(config)


if __name__ == "__main__":
    main()
