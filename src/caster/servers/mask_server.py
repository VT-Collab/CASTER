"""Track an object's asset mask through a video with DEVA."""

import logging
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from termcolor import colored

from caster.servers.http_service import InferenceRequest, create_app, run_inference, serve as serve_http


class MaskRequest(InferenceRequest):
    video_path: str
    object_name: str
    output_path: str
    reference_mask_path: str


def create_mask_app(track):
    app = create_app("deva")

    @app.post("/mask")
    def mask(request: MaskRequest) -> dict:
        output = run_inference(track, **request.model_dump())
        return {"output_path": output}

    return app


logger = logging.getLogger(__name__)
DEVA_ROOT = Path(__file__).resolve().parents[2] / "DEVA"


def serve(port: int = 29003) -> None:
    sys.path.insert(0, str(DEVA_ROOT))
    from deva.inference.demo_utils import get_input_frame_for_deva
    from deva.inference.inference_core import DEVAInferenceCore
    from deva.model.network import DEVA

    config_path = Path(__file__).resolve().parents[1] / "config/servers/mask_server.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["deva_checkpoint"] = str(Path(config["deva_checkpoint"]).resolve())
    checkpoint = Path(config["deva_checkpoint"])
    if not checkpoint.exists():
        existing = Path(__file__).resolve().parents[3] / "saves/DEVA-propagation.pth"
        if existing.is_file():
            checkpoint = existing
            config["deva_checkpoint"] = str(checkpoint)
        else:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            url = "https://github.com/hkchengrex/Tracking-Anything-with-DEVA/releases/download/v1.0/DEVA-propagation.pth"
            logger.info("Downloading DEVA weights to %s", checkpoint)
            urllib.request.urlretrieve(url, checkpoint)
    model = DEVA(config).cuda().eval()
    model.load_weights(torch.load(config["deva_checkpoint"], map_location="cuda"))
    logger.info(colored("DEVA weights loaded", "green"))

    def track(
        video_path: str, object_name: str, output_path: str, reference_mask_path: str
    ) -> str:
        capture = cv2.VideoCapture(video_path)
        frame_ok, frame = capture.read()
        if not frame_ok:
            raise ValueError(f"Cannot read video: {video_path}")

        height, width = frame.shape[:2]
        reference = cv2.imread(reference_mask_path, cv2.IMREAD_GRAYSCALE)
        if reference is None:
            raise FileNotFoundError(reference_mask_path)
        initial = cv2.resize(
            reference, (width, height), interpolation=cv2.INTER_NEAREST
        )
        initial = torch.as_tensor(initial > 0, dtype=torch.uint8, device="cuda")
        tracker = DEVAInferenceCore(model, config=config)
        tracker.enabled_long_id()
        masks = []

        with torch.inference_mode():
            while frame_ok:
                image = get_input_frame_for_deva(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), config["size"]
                )
                with torch.autocast("cuda", enabled=config["amp"]):
                    if not masks:
                        first_mask = torch.nn.functional.interpolate(
                            initial[None, None].float(),
                            size=image.shape[1:],
                            mode="nearest",
                        )[0, 0].to(torch.uint8)
                        probabilities = tracker.step(image, first_mask, objects=[1])
                    else:
                        probabilities = tracker.step(image)
                probabilities = torch.nn.functional.interpolate(
                    probabilities[:, None],
                    (height, width),
                    mode="bilinear",
                    align_corners=False,
                )[:, 0]
                masks.append(torch.argmax(probabilities, dim=0).byte().cpu().numpy())
                frame_ok, frame = capture.read()

        capture.release()
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, {"masks": np.stack(masks), "target_object": object_name})
        logger.info("%s %s", colored("Saved masks:", "green"), output)
        torch.cuda.empty_cache()
        return str(output)

    serve_http(create_mask_app(track), "localhost", port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    serve()
