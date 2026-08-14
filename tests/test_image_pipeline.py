
from PIL import Image
import pytest

from app.services.compositing import (
    composite_original_foreground,
    foreground_pixel_difference,
)
from app.services.image_io import InputValidationError, prepare_inputs


def test_prepare_inputs_inverts_product_mask_for_inpainting(
    image,
    product_mask,
):
    prepared = prepare_inputs(
        image("red"),
        product_mask(),
        width=64,
        height=64,
    )

    assert prepared.image.size == (64, 64)
    assert prepared.protected_mask.getpixel((32, 32)) == 255
    assert prepared.inpaint_mask.getpixel((32, 32)) == 0
    assert prepared.protected_mask.getpixel((0, 0)) == 0
    assert prepared.inpaint_mask.getpixel((0, 0)) == 255


def test_composite_keeps_original_foreground_exactly(
    image,
    product_mask,
):
    original = image("red", size=(64, 64))
    generated = image("blue", size=(64, 64))
    mask = product_mask(size=(64, 64))

    final = composite_original_foreground(
        original,
        generated,
        mask,
    )

    assert final.getpixel((32, 32)) == (255, 0, 0)
    assert final.getpixel((0, 0)) == (0, 0, 255)
    assert foreground_pixel_difference(
        original,
        final,
        mask,
    ) == 0.0


def test_prepare_inputs_rejects_mismatched_input_sizes(
    image,
    product_mask,
):
    with pytest.raises(
        InputValidationError,
        match="dimensions must match",
    ):
        prepare_inputs(
            image("red", size=(100, 80)),
            product_mask(size=(50, 40)),
            width=64,
            height=64,
        )


def test_prepare_inputs_rejects_empty_and_full_mask(image):
    with pytest.raises(InputValidationError, match="no protected"):
        prepare_inputs(
            image("red"),
            Image.new("L", (100, 80), color=0),
            width=64,
            height=64,
        )

    with pytest.raises(InputValidationError, match="entire image"):
        prepare_inputs(
            image("red"),
            Image.new("L", (100, 80), color=255),
            width=64,
            height=64,
        )