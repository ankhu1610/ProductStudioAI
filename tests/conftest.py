from PIL import Image, ImageDraw
import pytest


@pytest.fixture
def image():
    def _image(color, size=(100, 80)):
        return Image.new("RGB", size, color=color)

    return _image


@pytest.fixture
def product_mask():
    def _product_mask(size=(100, 80)):
        mask = Image.new("L", size, color=0)
        draw = ImageDraw.Draw(mask)

        width, height = size
        box = (
            int(width * 0.25),
            int(height * 0.25),
            int(width * 0.75),
            int(height * 0.75),
        )

        draw.rectangle(box, fill=255)
        return mask

    return _product_mask