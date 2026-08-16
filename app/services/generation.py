"""Lazy SD inpainting service used by Phase 1 and later API/UI layers."""

from __future__ import annotations

import time
from typing import Literal

from PIL import Image

from app.core.config import Settings
from app.schemas.generation import GenerationRequest

SchedulerName = Literal["ddim", "pndm", "dpm_solver", "euler_a"]


class GenerationService:
    """Loads the pretrained inpainting pipeline only when generation is requested."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._pipeline = None
        self._active_scheduler: SchedulerName | None = None

    def _get_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            import torch
            from diffusers import StableDiffusionInpaintPipeline
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
        self._pipeline = pipeline.to(self.settings.device)
        return self._pipeline

    def _apply_scheduler(self, scheduler: SchedulerName) -> None:
        if self._active_scheduler == scheduler and self._pipeline is not None:
            return
        try:
            from diffusers import (
                DDIMScheduler,
                DPMSolverMultistepScheduler,
                EulerAncestralDiscreteScheduler,
                PNDMScheduler,
            )
        except ImportError as error:
            raise RuntimeError(
                "Generation dependencies are not installed. Install CUDA PyTorch, then run "
                'pip install -e ".[dev,tracking]".'
            ) from error
        pipeline = self._get_pipeline()
        config = pipeline.scheduler.config
        if scheduler == "ddim":
            pipeline.scheduler = DDIMScheduler.from_config(config)
        elif scheduler == "pndm":
            pipeline.scheduler = PNDMScheduler.from_config(config)
        elif scheduler == "dpm_solver":
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(config)
        elif scheduler == "euler_a":
            pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(config)
        else:
            raise ValueError(f"unsupported scheduler: {scheduler}")
        self._active_scheduler = scheduler

    def generate_background(
        self,
        image: Image.Image,
        inpaint_mask: Image.Image,
        request: GenerationRequest,
    ) -> tuple[Image.Image, float, float | None]:
        """Generate one prompt-controlled background; white mask pixels are regenerated."""

        import torch

        pipeline = self._get_pipeline()
        self._apply_scheduler(request.scheduler)

        peak_gpu_memory_mb: float | None = None
        if self.settings.device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

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
        latency = time.perf_counter() - started_at

        if self.settings.device == "cuda" and torch.cuda.is_available():
            peak_gpu_memory_mb = round(torch.cuda.max_memory_allocated() / (1024**2), 2)

        return result, latency, peak_gpu_memory_mb
