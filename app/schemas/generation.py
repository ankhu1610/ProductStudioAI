"""Typed request and result models for product-background generation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class GenerationRequest(BaseModel):
    """Validated settings for one product-background generation request."""

    prompt: str = Field(min_length=3, max_length=500)
    negative_prompt: str = ""
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    steps: int = Field(default=15, ge=1, le=100)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    width: int = Field(default=512, ge=64, le=1024)
    height: int = Field(default=512, ge=64, le=1024)
    num_variants: int = Field(default=1, ge=1, le=4)
    scheduler: Literal["ddim", "pndm", "dpm_solver", "euler_a"] = "ddim"

    @field_validator("width", "height")
    @classmethod
    def dimensions_must_be_divisible_by_eight(cls, value: int) -> int:
        if value % 8 != 0:
            raise ValueError("image dimensions must be divisible by 8")
        return value


class GenerationMetadata(BaseModel):
    """Fields needed to reproduce and audit an output image."""

    request_id: str
    created_at: str
    base_model: str
    scheduler: str
    seed: int
    steps: int
    guidance_scale: float
    prompt: str
    negative_prompt: str
    resolution: str
    latency_seconds: float
    peak_gpu_memory_mb: float | None = None
    device: str
    foreground_pixel_difference: float
    clip_similarity: float | None = None
    input_image: str
    input_mask: str
    generated_image: str
    final_image: str
