"""End-to-end Phase 1 product background generation and compositing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

from PIL import Image

from app.core.config import Settings, get_settings
from app.schemas.generation import GenerationMetadata, GenerationRequest
from app.services.compositing import composite_original_foreground, foreground_pixel_difference
from app.services.generation import GenerationService
from app.services.image_io import PreparedInputs, load_image, load_mask, prepare_inputs


@dataclass(frozen=True)
class VariantResult:
    """Artifacts and metadata produced for one generation variant."""

    final_image: Image.Image
    generated_background: Image.Image
    metadata: GenerationMetadata
    final_path: Path
    generated_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class PipelineResult:
    """Complete collection of output artifacts for a multi-variant request."""

    request_id: str
    output_dir: Path
    prepared_inputs: PreparedInputs
    variants: list[VariantResult]


class ProductStudioPipeline:
    """Orchestrates image preprocessing, background generation, compositing, and metadata saving."""

    def __init__(
        self,
        settings: Settings | None = None,
        generation_service: GenerationService | None = None,
    ):
        self.settings = settings or get_settings()
        self.generation_service = generation_service or GenerationService(self.settings)

    def run(
        self,
        image_input: str | Path | Image.Image,
        mask_input: str | Path | Image.Image,
        request: GenerationRequest,
        output_dir: str | Path | None = None,
    ) -> PipelineResult:
        """Run the end-to-end Phase 1 workflow and persist images and JSON metadata."""

        # 1. Load inputs if filepaths were provided
        image = image_input if isinstance(image_input, Image.Image) else load_image(image_input)
        mask = mask_input if isinstance(mask_input, Image.Image) else load_mask(mask_input)

        # 2. Preprocess & validate inputs
        prepared = prepare_inputs(image, mask, width=request.width, height=request.height)

        # 3. Setup output directory structure
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        req_hash = uuid4().hex[:6]
        request_id = f"{timestamp}_{req_hash}"

        base_output_dir = Path(output_dir or self.settings.output_dir)
        job_dir = base_output_dir / request_id
        job_dir.mkdir(parents=True, exist_ok=True)

        input_img_path = job_dir / "input_image.png"
        input_mask_path = job_dir / "input_mask.png"
        prepared.image.save(input_img_path)
        prepared.protected_mask.save(input_mask_path)

        # 4. Generate variants sequentially
        variants: list[VariantResult] = []
        for v_idx in range(request.num_variants):
            variant_seed = request.seed + v_idx
            variant_request = request.model_copy(update={"seed": variant_seed})

            # Generate background via diffusers pipeline
            gen_bg, latency, peak_gpu_memory_mb = self.generation_service.generate_background(
                prepared.image,
                prepared.inpaint_mask,
                variant_request,
            )

            # Composite original foreground over generated background
            final_composite = composite_original_foreground(
                prepared.image,
                gen_bg,
                prepared.protected_mask,
            )

            # Verify foreground fidelity score (must be 0.0)
            pixel_diff = foreground_pixel_difference(
                prepared.image,
                final_composite,
                prepared.protected_mask,
            )

            # Save variant output files
            final_path = job_dir / f"final_variant_{v_idx + 1}.png"
            gen_path = job_dir / f"generated_bg_variant_{v_idx + 1}.png"
            meta_path = job_dir / f"metadata_variant_{v_idx + 1}.json"

            final_composite.save(final_path)
            gen_bg.save(gen_path)

            # Build reproducibility metadata
            metadata = GenerationMetadata(
                request_id=request_id,
                created_at=datetime.now(timezone.utc).isoformat(),
                base_model=str(self.settings.model_path or self.settings.model_id),
                scheduler=request.scheduler,
                seed=variant_seed,
                steps=request.steps,
                guidance_scale=request.guidance_scale,
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                resolution=f"{request.width}x{request.height}",
                latency_seconds=round(latency, 3),
                peak_gpu_memory_mb=peak_gpu_memory_mb,
                device=self.settings.device,
                foreground_pixel_difference=pixel_diff,
                input_image=str(input_img_path),
                input_mask=str(input_mask_path),
                generated_image=str(gen_path),
                final_image=str(final_path),
            )

            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata.model_dump(), f, indent=2)

            variants.append(
                VariantResult(
                    final_image=final_composite,
                    generated_background=gen_bg,
                    metadata=metadata,
                    final_path=final_path,
                    generated_path=gen_path,
                    metadata_path=meta_path,
                )
            )

        return PipelineResult(
            request_id=request_id,
            output_dir=job_dir,
            prepared_inputs=prepared,
            variants=variants,
        )
