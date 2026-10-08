import ast
import gc
import json
import logging
import re

import imageio.v2 as imageio
import torch
from torchvision.transforms import InterpolationMode, v2


logger = logging.getLogger(__name__)


class VideoValidator:
    def __init__(self, config):
        self.config = config
        model = config.model
        self.model_name = model.name
        self.device = model.input_device
        self.max_new_tokens = model.max_new_tokens
        self.model = None
        self.processor = None

    def _load_model(self):
        if self.model is not None:
            return
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        logger.info("Loading validator: %s", self.model_name)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            self.model_name,
            dtype=getattr(torch, self.config.model.dtype),
            device_map="cpu",
            attn_implementation=self.config.model.attn_implementation,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = model.eval()

    def offload(self):
        if self.model is not None:
            self.model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    def _load_frames(self, video_path):
        from qwen_vl_utils.vision_process import smart_resize

        config = self.config.video
        reader = imageio.get_reader(str(video_path))
        images, timestamps = [], []
        try:
            fps = reader.get_meta_data().get("fps", config.sampling_fps)
            step = max(1, round(fps / config.sampling_fps))
            for index, frame in enumerate(reader):
                if index % step == 0:
                    images.append(torch.from_numpy(frame.copy()).permute(2, 0, 1))
                    timestamps.append(index / fps)
        finally:
            reader.close()
        if not images:
            raise ValueError(f"No frames could be read from {video_path}")

        if len(images) > config.max_frames:
            indices = [
                round(i * (len(images) - 1) / (config.max_frames - 1))
                for i in range(config.max_frames)
            ]
            images = [images[index] for index in indices]
            timestamps = [timestamps[index] for index in indices]

        _, height, width = images[0].shape
        size = smart_resize(
            height // config.image_downsample_factor,
            width // config.image_downsample_factor,
            factor=32,
        )
        resize = v2.Resize(size, interpolation=InterpolationMode.BILINEAR, antialias=True)
        return [resize(image) for image in images], timestamps

    @staticmethod
    def _parse_result(text):
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        decoder = json.JSONDecoder()
        results = []
        for match in re.finditer(r"\{", text):
            try:
                value, _ = decoder.raw_decode(text[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                results.append(value)
        if results:
            return results[-1]

        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                value = ast.literal_eval(text[start:end + 1])
                if isinstance(value, dict):
                    return value
            except (SyntaxError, ValueError):
                pass
        raise ValueError(f"Qwen returned no valid JSON:\n{text}")

    def validate(self, video_path, prompt):
        self._load_model()
        self.model.to(self.device)
        images, timestamps = self._load_frames(video_path)
        content = []
        for timestamp, image in zip(timestamps, images):
            content.extend([
                {"type": "text", "text": f"{timestamp:.2f}s"},
                {"type": "image", "image": image},
            ])
        content.append({
            "type": "text",
            "text": self.config.prompts.user.replace("{generation_prompt}", prompt),
        })
        messages = [
            {"role": "system", "content": self.config.prompts.system},
            {"role": "user", "content": content},
        ]
        initial_messages = list(messages)
        thinking = self.config.model.enable_thinking
        attempts = self.config.evaluation.max_format_attempts
        last_error = None
        for attempt in range(attempts):
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                add_vision_id=False, enable_thinking=thinking,
            )
            inputs = self.processor(
                text=[text], images=images, videos=None,
                padding=True, return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens,
                    do_sample=False, use_cache=True,
                )
            generated = [
                output[len(input_ids):]
                for input_ids, output in zip(inputs.input_ids, generated)
            ]
            output = self.processor.batch_decode(
                generated, skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            truncated = len(generated[0]) >= self.max_new_tokens
            try:
                if truncated:
                    thinking = False
                    raise ValueError(f"Generation hit the max_new_tokens limit of {self.max_new_tokens}")
                data = self._parse_result(output)
                score = float(data["score"])
                return {
                    "valid": bool(data["valid"]) and score >= self.config.evaluation.valid_threshold,
                    "score": score,
                    "reason": data["reason"],
                    "matched_elements": data.get("matched_elements", []),
                    "missing_or_incorrect_elements": data.get("missing_or_incorrect_elements", []),
                    "model": self.model_name,
                }
            except (ValueError, KeyError, TypeError) as error:
                last_error = error
                logger.warning("Validation response attempt %s/%s failed: %s", attempt + 1, attempts, error)
                if attempt + 1 < attempts:
                    if truncated:
                        messages = list(initial_messages)
                    else:
                        messages.extend([
                            {"role": "assistant", "content": [{"type": "text", "text": output}]},
                            {"role": "user", "content": [{"type": "text", "text": (
                                "Return the final answer as one JSON object. "
                                "It must begin with { and end with }."
                            )}]},
                        ])
        raise RuntimeError(f"Qwen failed after {attempts} attempts") from last_error


def validate_video(video_path, prompt, validator):
    """Return a validation record and offload the reusable model afterward."""
    try:
        return validator.validate(video_path, prompt)
    except Exception as error:
        logger.exception("Validation failed for %s", video_path)
        return {
            "valid": False,
            "score": 0.0,
            "reason": f"Validation failed with error: {error}",
            "matched_elements": [],
            "missing_or_incorrect_elements": [],
            "model": validator.model_name,
        }
    finally:
        validator.offload()
