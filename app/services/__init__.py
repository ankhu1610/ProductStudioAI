"""Generation, compositing, metrics, and tracking services."""

from app.services.compositing import composite_original_foreground, foreground_pixel_difference
from app.services.image_io import InputValidationError, PreparedInputs, load_image, load_mask, prepare_inputs
from app.services.pipeline import PipelineResult, ProductStudioPipeline, VariantResult

__all__ = [
    "InputValidationError",
    "PipelineResult",
    "PreparedInputs",
    "ProductStudioPipeline",
    "VariantResult",
    "composite_original_foreground",
    "foreground_pixel_difference",
    "load_image",
    "load_mask",
    "prepare_inputs",
]
