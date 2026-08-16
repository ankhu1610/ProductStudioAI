"""Generation, compositing, metrics, and tracking services."""

from app.services.compositing import composite_original_foreground, foreground_pixel_difference
from app.services.image_io import InputValidationError, PreparedInputs, load_image, load_mask, prepare_inputs
from app.services.metrics import (
    LatencySummary,
    clip_similarity,
    compute_clip_similarity,
    compute_percentiles,
    get_peak_gpu_memory_mb,
    reset_peak_gpu_memory,
    summarize_latencies,
)
from app.services.pipeline import PipelineResult, ProductStudioPipeline, VariantResult

__all__ = [
    "InputValidationError",
    "LatencySummary",
    "PipelineResult",
    "PreparedInputs",
    "ProductStudioPipeline",
    "VariantResult",
    "clip_similarity",
    "composite_original_foreground",
    "compute_clip_similarity",
    "compute_percentiles",
    "foreground_pixel_difference",
    "get_peak_gpu_memory_mb",
    "load_image",
    "load_mask",
    "prepare_inputs",
    "reset_peak_gpu_memory",
    "summarize_latencies",
]
