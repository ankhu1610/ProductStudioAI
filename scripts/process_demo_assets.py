"""Process generated photorealistic product photos into data/demo_inputs with accurate masks."""

import glob
import os
from pathlib import Path
from PIL import Image, ImageFilter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "demo_inputs"
BRAIN_DIR = Path(r"C:\Users\Ankit\.gemini\antigravity-ide\brain\d0ec6aab-cb21-4b89-b45e-6a4721cc1648")

MAPPING = {
    "sneaker": "demo_sneaker_*.jpg",
    "watch": "demo_watch_*.jpg",
    "bottle": "demo_bottle_*.jpg",
    "headphones": "demo_headphones_*.jpg",
    "bag": "demo_bag_*.jpg",
    "skincare": "demo_skincare_*.jpg",
    "chair": "demo_chair_*.jpg",
    "coffee-maker": "demo_coffee_*.jpg",
    "lamp": "demo_lamp_*.jpg",
}


def process_assets() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, pattern in MAPPING.items():
        matches = list(BRAIN_DIR.glob(pattern))
        if not matches:
            print(f"Skipping {name}, no match found in {BRAIN_DIR}")
            continue

        img_path = matches[0]
        img = Image.open(img_path).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
        arr = np.array(img, dtype=np.float32)

        # Estimate white studio background from borders
        top = arr[:15, :, :].reshape(-1, 3)
        bottom = arr[-15:, :, :].reshape(-1, 3)
        left = arr[:, :15, :].reshape(-1, 3)
        right = arr[:, -15:, :].reshape(-1, 3)
        border = np.concatenate([top, bottom, left, right], axis=0)
        bg_color = np.median(border, axis=0)

        # Distance from background color
        dist = np.linalg.norm(arr - bg_color, axis=-1)

        # Thresholding
        fg_mask = (dist > 18.0).astype(np.uint8) * 255
        mask_pil = Image.fromarray(fg_mask, mode="L")

        # Morphological closing to fill small holes and smooth edges
        dilated = mask_pil.filter(ImageFilter.MaxFilter(7))
        closed = dilated.filter(ImageFilter.MinFilter(7))

        out_img_path = OUT_DIR / f"{name}.png"
        out_mask_path = OUT_DIR / f"{name}_mask.png"

        img.save(out_img_path)
        closed.save(out_mask_path)
        print(f"Generated realistic asset: {out_img_path.name} and {out_mask_path.name}")


if __name__ == "__main__":
    process_assets()
