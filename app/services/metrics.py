"""Evaluation, similarity, and hardware metrics for ProductStudio AI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    import torch


_CLIP_MODEL = None
_CLIP_PROCESSOR = None


@dataclass(frozen=True)
class LatencySummary:
    """Aggregated latency and memory metrics for one scheduler/steps combination."""

    scheduler: str
    steps: int
    count: int
    p50_seconds: float
    p95_seconds: float
    mean_seconds: float
    mean_peak_gpu_memory_mb: float | None


def get_clip_pipeline(model_id: str = "openai/clip-vit-base-patch32", device: str = "cpu"):
    """Lazy load CLIP model and processor for prompt-image alignment scoring."""

    global _CLIP_MODEL, _CLIP_PROCESSOR
    if _CLIP_MODEL is not None and _CLIP_PROCESSOR is not None:
        return _CLIP_MODEL, _CLIP_PROCESSOR

    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as error:
        raise RuntimeError("Transformers must be installed to compute CLIP similarity.") from error

    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)
    model.eval()

    _CLIP_MODEL = model
    _CLIP_PROCESSOR = processor
    return _CLIP_MODEL, _CLIP_PROCESSOR


def compute_clip_similarity(
    image: Image.Image,
    text: str,
    model_id: str = "openai/clip-vit-base-patch32",
    device: str = "cpu",
) -> float:
    """Compute cosine similarity between image and text prompt in CLIP embedding space."""

    import torch
    import torch.nn.functional as F

    model, processor = get_clip_pipeline(model_id=model_id, device=device)

    inputs = processor(
        text=[text],
        images=[image.convert("RGB")],
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
        text_features = model.get_text_features(input_ids=inputs["input_ids"])

        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)

        similarity = (image_features * text_features).sum(dim=-1).item()

    return float(round(similarity, 4))


# Alias for backward/spec compatibility
clip_similarity = compute_clip_similarity


def compute_percentiles(values: list[float]) -> dict[str, float]:
    """Compute summary statistics (p50, p95, mean, min, max) for benchmark metrics."""

    if not values:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}

    arr = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(round(np.percentile(arr, 50), 3)),
        "p95": float(round(np.percentile(arr, 95), 3)),
        "mean": float(round(float(np.mean(arr)), 3)),
        "min": float(round(float(np.min(arr)), 3)),
        "max": float(round(float(np.max(arr)), 3)),
    }


def summarize_latencies(
    rows: list[dict[str, Any]],
    scheduler: str,
    steps: int,
) -> LatencySummary:
    """Aggregate latencies and peak memory across matching successful runs."""

    matched = [
        r
        for r in rows
        if r.get("scheduler") == scheduler
        and r.get("steps") == steps
        and r.get("status") == "ok"
    ]
    latencies = [float(r["latency_seconds"]) for r in matched if r.get("latency_seconds") is not None]
    vram = [float(r["peak_gpu_memory_mb"]) for r in matched if r.get("peak_gpu_memory_mb") is not None]

    if not latencies:
        return LatencySummary(
            scheduler=scheduler,
            steps=steps,
            count=0,
            p50_seconds=0.0,
            p95_seconds=0.0,
            mean_seconds=0.0,
            mean_peak_gpu_memory_mb=None,
        )

    percentiles = compute_percentiles(latencies)
    mean_vram = round(float(sum(vram) / len(vram)), 2) if vram else None

    return LatencySummary(
        scheduler=scheduler,
        steps=steps,
        count=len(latencies),
        p50_seconds=percentiles["p50"],
        p95_seconds=percentiles["p95"],
        mean_seconds=percentiles["mean"],
        mean_peak_gpu_memory_mb=mean_vram,
    )


def get_peak_gpu_memory_mb() -> float | None:
    """Return peak GPU memory allocated in megabytes since last reset, if CUDA is available."""

    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / (1024**2), 2)
    except Exception:
        pass
    return None


def reset_peak_gpu_memory() -> None:
    """Reset peak GPU memory statistics if CUDA is available."""

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
