"""Gradio interactive web interface for ProductStudio AI."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import gradio as gr
from PIL import Image

logger = logging.getLogger(__name__)

from app.core.config import Settings, get_settings
from app.schemas.generation import GenerationRequest
from app.services.database import DatabaseService, JobRecord
from app.services.pipeline import PipelineResult, ProductStudioPipeline
from app.services.tracking import MLflowTracker

ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = ROOT / "data" / "demo_inputs"

PRESETS = {
    "Sneaker on Wet Neon Asphalt": {
        "image": DEMO_DIR / "sneaker.png",
        "mask": DEMO_DIR / "sneaker_mask.png",
        "prompt": "premium sneaker campaign on wet neon asphalt at night, cinematic reflections, product photography",
        "seed": 101,
        "steps": 15,
        "scheduler": "ddim",
    },
    "Luxury Watch on White Marble": {
        "image": DEMO_DIR / "watch.png",
        "mask": DEMO_DIR / "watch_mask.png",
        "prompt": "luxury watch on a white marble surface, soft window light, high-end studio product photography",
        "seed": 102,
        "steps": 15,
        "scheduler": "ddim",
    },
    "Beverage Bottle on Sunlit Beach": {
        "image": DEMO_DIR / "bottle.png",
        "mask": DEMO_DIR / "bottle_mask.png",
        "prompt": "refreshing beverage campaign on a sunlit tropical beach, condensation, premium commercial photography",
        "seed": 103,
        "steps": 15,
        "scheduler": "ddim",
    },
    "Wireless Headphones on Walnut Desk": {
        "image": DEMO_DIR / "headphones.png",
        "mask": DEMO_DIR / "headphones_mask.png",
        "prompt": "wireless headphones on a minimalist walnut desk, warm morning light, editorial product photograph",
        "seed": 104,
        "steps": 15,
        "scheduler": "ddim",
    },
    "Designer Handbag in Parisian Cafe": {
        "image": DEMO_DIR / "bag.png",
        "mask": DEMO_DIR / "bag_mask.png",
        "prompt": "fashion handbag in an elegant cafe setting, shallow depth of field, luxury campaign photography",
        "seed": 105,
        "steps": 15,
        "scheduler": "ddim",
    },
    "Skincare Bottle in Spa Bathroom": {
        "image": DEMO_DIR / "skincare.png",
        "mask": DEMO_DIR / "skincare_mask.png",
        "prompt": "skincare product on a clean spa bathroom counter, botanical shadows, premium beauty advertisement",
        "seed": 106,
        "steps": 15,
        "scheduler": "ddim",
    },
    "Modern Lounge Chair in Living Room": {
        "image": DEMO_DIR / "chair.png",
        "mask": DEMO_DIR / "chair_mask.png",
        "prompt": "modern lounge chair in a sunlit Scandinavian living room, interior design editorial",
        "seed": 107,
        "steps": 15,
        "scheduler": "ddim",
    },
    "Espresso Machine in Stylish Kitchen": {
        "image": DEMO_DIR / "coffee-maker.png",
        "mask": DEMO_DIR / "coffee-maker_mask.png",
        "prompt": "espresso machine in a stylish home kitchen, warm morning light, premium appliance advertisement",
        "seed": 108,
        "steps": 15,
        "scheduler": "ddim",
    },
    "Table Lamp in Modern Bedroom": {
        "image": DEMO_DIR / "lamp.png",
        "mask": DEMO_DIR / "lamp_mask.png",
        "prompt": "designer table lamp in a calm modern bedroom, evening ambience, interior photography",
        "seed": 109,
        "steps": 15,
        "scheduler": "ddim",
    },
}


def build_gradio_app(
    settings: Settings | None = None,
    pipeline: ProductStudioPipeline | None = None,
    db: DatabaseService | None = None,
    tracker: MLflowTracker | None = None,
) -> gr.Blocks:
    """Create the interactive Gradio Blocks application."""
    settings = settings or get_settings()
    pipeline = pipeline or ProductStudioPipeline(settings=settings)
    db = db or DatabaseService(settings=settings)
    tracker = tracker or MLflowTracker(settings=settings)

    def on_generate(
        image_input: Image.Image | None,
        mask_input: Image.Image | None,
        prompt: str,
        negative_prompt: str,
        scheduler: str,
        steps: int,
        guidance_scale: float,
        seed: int,
        num_variants: int,
    ) -> tuple[list[Image.Image], list[Image.Image], str, str]:
        if image_input is None:
            return [], [], "Error: Please upload a product image.", "{}"
        if mask_input is None:
            return [], [], "Error: Please upload a product mask (white=product).", "{}"
        if not prompt or len(prompt.strip()) < 3:
            return [], [], "Error: Prompt must be at least 3 characters long.", "{}"

        req = GenerationRequest(
            prompt=prompt.strip(),
            negative_prompt=negative_prompt.strip(),
            scheduler=scheduler,  # type: ignore[arg-type]
            steps=int(steps),
            guidance_scale=float(guidance_scale),
            seed=int(seed),
            num_variants=int(num_variants),
            width=settings.default_width,
            height=settings.default_height,
        )

        try:
            result = pipeline.run(
                image_input=image_input,
                mask_input=mask_input,
                request=req,
            )

            first_variant = result.variants[0]
            record = JobRecord(
                job_id=result.request_id,
                created_at=first_variant.metadata.created_at,
                status="completed",
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                scheduler=req.scheduler,
                steps=req.steps,
                guidance_scale=req.guidance_scale,
                seed=req.seed,
                resolution=f"{req.width}x{req.height}",
                num_variants=req.num_variants,
                latency_seconds=first_variant.metadata.latency_seconds,
                peak_gpu_memory_mb=first_variant.metadata.peak_gpu_memory_mb,
                foreground_pixel_difference=first_variant.metadata.foreground_pixel_difference,
                output_dir=str(result.output_dir),
                final_images=[v.final_path.name for v in result.variants],
                metadata_files=[v.metadata_path.name for v in result.variants],
            )
            db.save_job(record)
            tracker.log_generation(result)

            final_imgs = [v.final_image for v in result.variants]
            bg_imgs = [v.generated_background for v in result.variants]

            vram_info = f" | VRAM: {first_variant.metadata.peak_gpu_memory_mb} MB" if first_variant.metadata.peak_gpu_memory_mb else ""
            status_text = (
                f"Generated {len(result.variants)} variant(s) in {first_variant.metadata.latency_seconds:.2f}s "
                f"| Pixel Diff: {first_variant.metadata.foreground_pixel_difference:.4f}{vram_info} "
                f"| Job ID: {result.request_id}"
            )
            meta_json = first_variant.metadata.model_dump_json(indent=2)
            return final_imgs, bg_imgs, status_text, meta_json

        except Exception as err:
            logger.error("Generation failed: %s", err, exc_info=True)
            return [], [], f"Generation failed: {err}", "{}"

    def on_select_preset(preset_name: str) -> tuple[Image.Image | None, Image.Image | None, str, int, int, str]:
        if preset_name not in PRESETS:
            return None, None, "", 42, 15, "ddim"
        p = PRESETS[preset_name]
        img = Image.open(p["image"]) if p["image"].exists() else None
        mask = Image.open(p["mask"]) if p["mask"].exists() else None
        return img, mask, p["prompt"], p["seed"], p["steps"], p["scheduler"]

    with gr.Blocks(title="ProductStudio AI - E-Commerce Ad Studio") as demo:
        gr.Markdown(
            """
            # ProductStudio AI
            ### Mask-Aware Latent Diffusion Platform for E-Commerce Ad Creative Generation
            Upload a product photo and mask, describe your desired environment, and generate professional ad creatives that **100% preserve your product pixels**.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Product & Mask Inputs")
                preset_dropdown = gr.Dropdown(
                    label="Load Example Product Preset",
                    choices=list(PRESETS.keys()),
                    value=None,
                )

                with gr.Row():
                    img_input = gr.Image(label="Product Image (RGB)", type="pil")
                    mask_input = gr.Image(label="Product Mask (White = Product)", type="pil", image_mode="L")

                gr.Markdown("### 2. Creative Direction")
                prompt_input = gr.Textbox(
                    label="Background Prompt",
                    placeholder="e.g. premium sneaker campaign on wet neon asphalt at night, cinematic reflections",
                    lines=2,
                )
                neg_prompt_input = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="text, watermark, blurry, deformed background, duplicate product",
                    lines=1,
                )

                with gr.Accordion("Advanced Generation Settings", open=False):
                    with gr.Row():
                        scheduler_input = gr.Dropdown(
                            label="Scheduler",
                            choices=["ddim", "pndm", "dpm_solver", "euler_a"],
                            value="ddim",
                        )
                        steps_input = gr.Slider(label="Inference Steps", minimum=10, maximum=50, step=5, value=15)

                    with gr.Row():
                        cfg_input = gr.Slider(label="Guidance Scale (CFG)", minimum=1.0, maximum=15.0, step=0.5, value=7.5)
                        variants_input = gr.Slider(label="Number of Variants", minimum=1, maximum=4, step=1, value=1)

                    seed_input = gr.Number(label="Random Seed", value=42, precision=0)

                generate_btn = gr.Button("Generate Ad Creative", variant="primary", size="lg")

            with gr.Column(scale=1):
                gr.Markdown("### 3. Generated Ad Creatives")
                status_box = gr.Textbox(label="Status & Performance", interactive=False)

                with gr.Tab("Final Composites (Preserved Product)"):
                    gallery_output = gr.Gallery(label="Final Ads", columns=2, height=400, preview=True)

                with gr.Tab("Raw Generated Backgrounds"):
                    bg_gallery_output = gr.Gallery(label="Generated Environments", columns=2, height=400, preview=True)

                with gr.Tab("Reproducibility Metadata"):
                    metadata_box = gr.Code(label="JSON Metadata", language="json", interactive=False)

        preset_dropdown.change(
            fn=on_select_preset,
            inputs=[preset_dropdown],
            outputs=[img_input, mask_input, prompt_input, seed_input, steps_input, scheduler_input],
        )

        generate_btn.click(
            fn=on_generate,
            inputs=[
                img_input,
                mask_input,
                prompt_input,
                neg_prompt_input,
                scheduler_input,
                steps_input,
                cfg_input,
                seed_input,
                variants_input,
            ],
            outputs=[gallery_output, bg_gallery_output, status_box, metadata_box],
        )

    return demo
