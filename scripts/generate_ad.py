"""CLI runner for ProductStudio AI Phase 1 background generation & compositing."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from app.core.config import get_settings
from app.schemas.generation import GenerationRequest
from app.services.pipeline import ProductStudioPipeline


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate prompt-controlled e-commerce ad backgrounds while preserving product foreground pixels."
    )
    parser.add_argument("--image", required=True, type=Path, help="Path to input product RGB image")
    parser.add_argument("--mask", required=True, type=Path, help="Path to input product mask (white=product)")
    parser.add_argument("--prompt", required=True, type=str, help="Background generation text prompt")
    parser.add_argument("--negative-prompt", type=str, default="", help="Negative text prompt")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--steps", type=int, default=15, help="Number of inference steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--width", type=int, default=512, help="Output image width (divisible by 8)")
    parser.add_argument("--height", type=int, default=512, help="Output image height (divisible by 8)")
    parser.add_argument("--variants", type=int, default=1, help="Number of variant images to generate")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to save generated outputs")

    args = parser.parse_args()

    request = GenerationRequest(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        width=args.width,
        height=args.height,
        num_variants=args.variants,
    )

    settings = get_settings()
    pipeline = ProductStudioPipeline(settings=settings)

    print("ProductStudio AI - Phase 1 Background Replacement")
    print("=" * 60)
    print(f"Product Image  : {args.image}")
    print(f"Product Mask   : {args.mask}")
    print(f"Prompt         : {args.prompt}")
    print(f"Resolution     : {args.width}x{args.height}")
    print(f"Steps          : {args.steps} | Seed: {args.seed} | CFG: {args.guidance_scale}")
    print(f"Variants       : {args.variants}")
    print("=" * 60)

    try:
        result = pipeline.run(
            image_input=args.image,
            mask_input=args.mask,
            request=request,
            output_dir=args.output_dir,
        )
    except Exception as err:
        print(f"\n[ERROR] Generation failed: {err}", file=sys.stderr)
        return 1

    print("\nGeneration Complete!")
    print(f"Job ID     : {result.request_id}")
    print(f"Output Dir : {result.output_dir.resolve()}")
    print("-" * 60)
    for idx, variant in enumerate(result.variants, 1):
        print(f"Variant #{idx}:")
        print(f"  Final Image      : {variant.final_path.name}")
        print(f"  Generated BG     : {variant.generated_path.name}")
        print(f"  Metadata JSON    : {variant.metadata_path.name}")
        print(f"  Latency          : {variant.metadata.latency_seconds:.2f}s")
        print(f"  Foreground Pixel Diff : {variant.metadata.foreground_pixel_difference:.4f}")
        print("-" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
