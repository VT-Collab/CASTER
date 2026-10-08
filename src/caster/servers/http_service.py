"""Shared HTTP setup for Caster's inference services."""

import logging
from threading import Lock

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict


logger = logging.getLogger(__name__)
_inference_lock = Lock()


class InferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


def create_app(name: str) -> FastAPI:
    app = FastAPI(title=f"Caster {name}")

    @app.get("/health")
    def health() -> dict:
        return {"service": name, "status": "ok"}

    return app


def run_inference(function, **arguments):
    # Each service owns one model; serialize requests to its GPU state.
    with _inference_lock:
        try:
            return function(**arguments)
        except Exception as error:
            logger.exception("Inference failed")
            raise HTTPException(
                status_code=500, detail=f"{type(error).__name__}: {error}"
            ) from error


def serve(app: FastAPI, host: str, port: int) -> None:
    uvicorn.run(app, host=host, port=port, workers=1)
