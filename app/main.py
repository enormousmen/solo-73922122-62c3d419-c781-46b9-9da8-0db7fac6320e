"""FastAPI surface for the single-job resumable training service.

Endpoints (interactive docs at /docs):

    GET  /                     service info
    GET  /config               effective default config + runtime info
    POST /train/start          start a fresh run (optional config overrides, force)
    POST /train/pause          stop training; 200 only after a full checkpoint is persisted
    POST /train/resume         continue from the latest checkpoint / paused state
    GET  /train/status         progress and checkpoint health
    GET  /train/losses         per-batch loss history
    POST /train/lr-probe       compare two candidate learning rates on the next batch
    GET  /train/parameters     download model parameters (.pt) from latest checkpoint
    GET  /train/checkpoint     download the full checkpoint (.pt)
"""
from __future__ import annotations

import asyncio
import math
import os
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from .config import load_default_config
from .trainer import (
    CheckpointSaveError,
    ConflictError,
    JobManager,
    NotFoundError,
    PauseTimeoutError,
    ProbeError,
)

manager: JobManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    global manager
    runs_dir = os.environ.get("RUNS_DIR", "runs")
    manager = JobManager(runs_dir=runs_dir, default_config=load_default_config())
    yield


app = FastAPI(
    title="Resumable Training API",
    version="1.0.0",
    description=(
        "Single-job CPU training service (two-layer FC net + Dropout + Adam). "
        "Pause/resume and kill/restart recovery reproduce uninterrupted runs "
        "bit-exactly under the same environment, seed and config."
    ),
    lifespan=lifespan,
)


def _conflict(exc: Exception) -> HTTPException:
    return HTTPException(status_code=409, detail=str(exc))


def _json_safe(value):
    """Replace non-finite floats so validation errors can always be encoded."""
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(request, exc: RequestValidationError):
    """422 with a JSON-safe body (the default handler chokes on inf/NaN input)."""
    return JSONResponse(
        status_code=422,
        content={"detail": _json_safe(jsonable_encoder(exc.errors()))},
    )


class LRProbeRequest(BaseModel):
    """Two distinct finite positive learning rates to compare."""
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    lr_a: float = Field(strict=True, gt=0,
                        description="first candidate learning rate (> 0, finite)")
    lr_b: float = Field(strict=True, gt=0,
                        description="second candidate learning rate (> 0, finite)")


# TEST-ONLY endpoint (enabled via ENABLE_TEST_CRASH_ENDPOINT=1): hard-exit the
# process with os._exit, simulating SIGKILL / preemption / power loss.  Used by
# scripts/verify_repro.py; never enable in production.
if os.environ.get("ENABLE_TEST_CRASH_ENDPOINT") == "1":
    @app.post("/test/crash", include_in_schema=False)
    def _test_crash() -> None:  # pragma: no cover - kills the process
        os._exit(1)


@app.get("/")
def root() -> dict:
    return {
        "service": "resumable-training-api",
        "docs": "/docs",
        "endpoints": [
            "GET /config",
            "POST /train/start",
            "POST /train/pause",
            "POST /train/resume",
            "GET /train/status",
            "GET /train/losses",
            "POST /train/lr-probe",
            "GET /train/parameters",
            "GET /train/checkpoint",
        ],
    }


@app.get("/config")
def get_config() -> dict:
    return {
        "default_config": manager.default_config.to_dict(),
        "runs_dir": str(manager.runs_dir),
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
    }


@app.post("/train/start")
def start(body: dict[str, Any] | None = Body(default=None)) -> dict:
    """Start a fresh run. Body: {"force": bool, "config": {overrides}}."""
    body = body or {}
    force = bool(body.get("force", False))
    overrides = body.get("config") or {}
    try:
        return manager.start(overrides=overrides, force=force)
    except ConflictError as exc:
        raise _conflict(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/train/pause")
async def pause() -> dict:
    """Stop training; responds only after the trainer stopped AND a complete
    checkpoint is persisted.  Runs in a thread so other queries stay live."""
    try:
        return await asyncio.to_thread(manager.pause)
    except PauseTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except CheckpointSaveError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ConflictError as exc:
        raise _conflict(exc) from exc


@app.post("/train/resume")
def resume() -> dict:
    """Resume from the paused state / latest checkpoint. Only one trainer can
    run at a time: concurrent resumes yield exactly one 200 and 409s."""
    try:
        return manager.resume()
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ConflictError as exc:
        raise _conflict(exc) from exc


@app.get("/train/status")
def status() -> dict:
    return manager.status()


@app.get("/train/losses")
def losses(
    since_step: int | None = Query(default=None, description="only losses with step > this"),
    limit: int | None = Query(default=None, description="max number of entries"),
) -> dict:
    return manager.get_losses(since_step=since_step, limit=limit)


@app.post("/train/lr-probe")
async def lr_probe(req: LRProbeRequest) -> dict:
    """Compare two candidate learning rates on the checkpoint's next batch.

    Both candidates start from the same most recent successfully saved
    checkpoint (model + Adam + RNG) and take one update on the same batch;
    the response reports the eval-mode cross-entropy of that batch before and
    after each candidate's update, plus the checkpoint's run_id, epoch and
    batch.  The online job is not modified.  Runs in a thread so status and
    loss queries stay responsive.
    """
    if req.lr_a == req.lr_b:
        raise HTTPException(status_code=422,
                            detail="lr_a and lr_b must be different")
    try:
        return await asyncio.to_thread(manager.lr_probe, req.lr_a, req.lr_b)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ConflictError as exc:
        raise _conflict(exc) from exc
    except ProbeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/train/parameters")
def parameters() -> Response:
    """Download current model parameters (.pt) from the latest checkpoint."""
    try:
        data = manager.read_parameters()
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={"Content-Disposition": 'attachment; filename="model_parameters.pt"'},
    )


@app.get("/train/checkpoint")
def checkpoint() -> FileResponse:
    """Download the full checkpoint (model + optimizer + RNG + history)."""
    try:
        path = manager.checkpoint_file()
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path, filename="checkpoint.pt",
                        media_type="application/octet-stream")
