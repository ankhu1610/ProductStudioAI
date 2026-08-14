"""Lazy SD inpainting service used by Phase 1 and later API/UI layers."""

from __future__ import annotations

import time

from PIL import Image

from app.core.config import Settings
from app.schemas.generation import GenerationRequest


class GenerationService:
    """Loads the pretrained inpainting pipeline only when generation is requested."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            import torch
            from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline
        except ImportError as error:
            raise RuntimeError(
                "Generation dependencies are not installed. Install CUDA PyTorch, then run "
                'pip install -e ".[dev,tracking]".'
            ) from error
        if self.settings.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no CUDA-enabled PyTorch GPU is available")
        dtype = torch.float16 if (self.settings.device == "cuda" and self.settings.precision == "fp16") else torch.float32
        model_source = str(self.settings.model_path or self.settings.model_id)
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(model_source, torch_dtype=dtype)
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        self._pipeline = pipeline.to(self.settings.device)
        return self._pipeline

    def generate_background(self, image: Image.Image, inpaint_mask: Image.Image, request: GenerationRequest) -> tuple[Image.Image, float]:
        """Generate one prompt-controlled background; white mask pixels are regenerated."""

        pipeline = self._get_pipeline()
        import torch

        generator = torch.Generator(device=self.settings.device).manual_seed(request.seed)
        started_at = time.perf_counter()
        result = pipeline(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or None,
            image=image,
            mask_image=inpaint_mask,
            num_inference_steps=request.steps,
            guidance_scale=request.guidance_scale,
            generator=generator,
            width=request.width,
            height=request.height,
        ).images[0]
        return result, time.perf_counter() - started_at
