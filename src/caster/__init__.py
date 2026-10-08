"""Public-facing video and scene generation tools."""

import os
from pathlib import Path


MODEL_CACHE = Path(__file__).resolve().parents[2] / "models"
os.environ.setdefault("HF_HOME", str(MODEL_CACHE / "huggingface"))
os.environ.setdefault("MODELSCOPE_CACHE", str(MODEL_CACHE / "modelscope"))
os.environ.setdefault("TORCH_HOME", str(MODEL_CACHE / "torch"))
