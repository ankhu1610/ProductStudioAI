"""Image and product-mask validation and preprocessing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


class InputValidationError(ValueError):
    """Raised when a product image or mask cannot be safely processed."""


@dataclass(frozen=True)
class PreparedInputs:
    """Normalized inputs used by the inpainting pipeline and final composite."""

    image: Image.Image
    protected_mask: Image.Image
    inpaint_mask: Image.Image


def load_image(path: str | Path) -> Image.Image:
    """Load an image, apply EXIF orientation, and convert it to RGB."""

    try:
        with Image.open(path) as source:
            return ImageOps.exif_transpose(source).convert("RGB")
    except (FileNotFoundError, OSError) as error:
        raise InputValidationError(f"Could not load product image: {path}") from error


def load_mask(path: str | Path) -> Image.Image:
    """Load a mask as grayscale; white means protected product."""

    try:
        with Image.open(path) as source:
            return ImageOps.exif_transpose(source).convert("L")
    except (FileNotFoundError, OSError) as error:
        raise InputValidationError(f"Could not load product mask: {path}") from error


def _center_crop_and_resize(image: Image.Image, width: int, height: int, resample: int) -> Image.Image:
    """Resize while preserving aspect ratio, then center crop to target dimensions."""

    source_width, source_height = image.size
    scale = max(width / source_width, height / source_height)
    resized = image.resize((round(source_width * scale), round(source_height * scale)), resample)
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def prepare_inputs(
    image: Image.Image,
    protected_mask: Image.Image,
    width: int = 512,
    height: int = 512,
    threshold: int = 127,
) -> PreparedInputs:
    """Normalize inputs and return both protected and inverted inpainting masks.

    ProductStudio convention: white in the protected mask identifies product pixels.
    Diffusers inpainting convention: white identifies pixels to regenerate. Therefore the
    model's inpaint mask is the inverse of the protected product mask.
    """

    if width < 64 or height < 64 or width % 8 or height % 8:
        raise InputValidationError("target dimensions must be at least 64 and divisible by 8")
    if image.size != protected_mask.size:
        raise InputValidationError(
            f"image and mask dimensions must match before preprocessing: {image.size} != {protected_mask.size}"
        )
    image = _center_crop_and_resize(image.convert("RGB"), width, height, Image.Resampling.LANCZOS)
    mask = _center_crop_and_resize(protected_mask.convert("L"), width, height, Image.Resampling.NEAREST)
    mask_values = np.asarray(mask)
    binary = np.where(mask_values > threshold, 255, 0).astype(np.uint8)
    product_pixels = int(np.count_nonzero(binary))
    if product_pixels == 0:
        raise InputValidationError("mask contains no protected product pixels")
    if product_pixels == binary.size:
        raise InputValidationError("mask protects the entire image; no background remains to generate")
    protected = Image.fromarray(binary, mode="L")
    inpaint = ImageOps.invert(protected)
    return PreparedInputs(image=image, protected_mask=protected, inpaint_mask=inpaint)

