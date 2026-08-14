"""Create license-safe synthetic inputs for interface and pipeline testing.

These are simple geometric stand-ins, not generated commercial product photos.
They let reviewers exercise validation, mask handling, and compositing without
private assets. Replace only with authorized images in real demonstrations.
"""

from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "demo_inputs"


def save_case(name: str, shape: str, color: str) -> None:
    image = Image.new("RGB", (512, 512), "#e5e7eb")
    mask = Image.new("L", (512, 512), 0)
    drawing = ImageDraw.Draw(image)
    mask_drawing = ImageDraw.Draw(mask)
    box = (140, 150, 372, 362)
    if shape == "ellipse":
        drawing.ellipse(box, fill=color, outline="#111827", width=5)
        mask_drawing.ellipse(box, fill=255)
    elif shape == "rounded":
        drawing.rounded_rectangle(box, radius=34, fill=color, outline="#111827", width=5)
        mask_drawing.rounded_rectangle(box, radius=34, fill=255)
    else:
        drawing.polygon([(165, 345), (230, 160), (345, 235), (310, 360)], fill=color, outline="#111827", width=5)
        mask_drawing.polygon([(165, 345), (230, 160), (345, 235), (310, 360)], fill=255)
    image.save(OUTPUT / f"{name}.png")
    mask.save(OUTPUT / f"{name}_mask.png")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = (
        ("sneaker", "polygon", "#2563eb"),
        ("watch", "ellipse", "#ca8a04"),
        ("bottle", "rounded", "#16a34a"),
        ("headphones", "ellipse", "#7c3aed"),
        ("bag", "rounded", "#be123c"),
        ("skincare", "rounded", "#0891b2"),
        ("chair", "rounded", "#b45309"),
        ("coffee-maker", "rounded", "#334155"),
        ("lamp", "polygon", "#f59e0b"),
        ("jacket", "polygon", "#15803d"),
    )
    for name, shape, color in cases:
        save_case(name, shape, color)
    print(f"Created synthetic demo assets in {OUTPUT}")


if __name__ == "__main__":
    main()
