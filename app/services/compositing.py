"""Foreground-preserving final composite and deterministic fidelity metrics."""

from __future__ import annotations

import numpy as np
from PIL import Image


def composite_original_foreground(
    original_image: Image.Image,
    generated_image: Image.Image,
    protected_mask: Image.Image,
) -> Image.Image:
    """Restore source product pixels exactly over a generated background.

    White mask pixels select the original product; black pixels select the generated
    background. This is the final fidelity guarantee for the MVP.
    """

    if original_image.size != generated_image.size or original_image.size != protected_mask.size:
        raise ValueError("original image, generated image, and mask must have identical dimensions")
    return Image.composite(original_image.convert("RGB"), generated_image.convert("RGB"), protected_mask.convert("L"))


def foreground_pixel_difference(
    original_image: Image.Image,
    final_image: Image.Image,
    protected_mask: Image.Image,
) -> float:
    """Return the maximum RGB-channel difference inside the protected product region."""

    if original_image.size != final_image.size or original_image.size != protected_mask.size:
        raise ValueError("original image, final image, and mask must have identical dimensions")
    original = np.asarray(original_image.convert("RGB"), dtype=np.int16)
    final = np.asarray(final_image.convert("RGB"), dtype=np.int16)
    protected = np.asarray(protected_mask.convert("L"), dtype=bool)
    if not np.any(protected):
        raise ValueError("mask contains no protected product pixels")
    return float(np.abs(original[protected] - final[protected]).max())

