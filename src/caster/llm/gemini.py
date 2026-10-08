"""Gemini Robotics ER 2 image queries."""

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


class GeminiER2:
    def __init__(self, model: str = "gemini-robotics-er-2-preview") -> None:
        load_dotenv()
        api_key = os.environ.get("GENAI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Set GENAI_API_KEY in the .env file.")
        self.model = model
        self.client = genai.Client(api_key=api_key)

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "GeminiER2":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def points(self, image_path: Path, prompt: str, temperature: float) -> list[dict]:
        mime_type = "image/jpeg" if image_path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
        response = self.client.models.generate_content(
            model=self.model,
            contents=[types.Part.from_bytes(data=image_path.read_bytes(), mime_type=mime_type), prompt],
            config=types.GenerateContentConfig(
                temperature=temperature,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        match = re.search(r"\[\s*\{.*?\}\s*\]", response.text or "", re.DOTALL)
        if not match:
            raise ValueError("Gemini did not return a point list")
        return json.loads(match.group())

    def json_list(self, prompt: str, temperature: float = 0.2) -> list[dict]:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=temperature),
        )
        match = re.search(r"\[.*\]", response.text or "", re.DOTALL)
        if not match:
            raise ValueError("Gemini did not return a JSON list")
        return json.loads(match.group())
