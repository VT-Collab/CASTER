"""Text-only task reasoning through an OpenAI-compatible endpoint."""

import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI


def infer_task_features(prompt: str) -> list[dict]:
    load_dotenv()
    with OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.getenv("OPENAI_API_BASE"),
    ) as client:
        response = client.chat.completions.create(
            model=os.environ["OPENAI_MODEL"],
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )
    content = response.choices[0].message.content or ""
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
    features = json.loads(content)
    if not isinstance(features, list):
        raise ValueError("Expected a JSON list of task features")
    return features
