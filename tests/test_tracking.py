from unittest.mock import MagicMock
from PIL import Image
import pytest

from app.core.config import Settings
from app.schemas.generation import GenerationMetadata
from app.services.pipeline import PipelineResult, VariantResult
from app.services.tracking import MLflowTracker


def test_mlflow_tracker_mock(tmp_path):
    settings = Settings(device="cpu", mlflow_tracking_uri=str(tmp_path / "mlruns"))
    tracker = MLflowTracker(tracking_uri=str(tmp_path / "mlruns"), settings=settings)

    # Build mock pipeline result
    job_dir = tmp_path / "job_track_123"
    job_dir.mkdir(parents=True, exist_ok=True)
    final_p = job_dir / "final_variant_1.png"
    gen_p = job_dir / "generated_bg_variant_1.png"
    meta_p = job_dir / "metadata_variant_1.json"

    Image.new("RGB", (64, 64), color="red").save(final_p)
    Image.new("RGB", (64, 64), color="blue").save(gen_p)
    meta_p.write_text('{"test": true}')

    meta = GenerationMetadata(
        request_id="job_track_123",
        created_at="2026-08-16T12:00:00+00:00",
        base_model="test-model",
        scheduler="ddim",
        seed=42,
        steps=15,
        guidance_scale=7.5,
        prompt="luxury watch on marble",
        negative_prompt="",
        resolution="512x512",
        latency_seconds=1.5,
        peak_gpu_memory_mb=2048.0,
        device="cpu",
        foreground_pixel_difference=0.0,
        clip_similarity=0.32,
        input_image="input.png",
        input_mask="mask.png",
        generated_image=str(gen_p),
        final_image=str(final_p),
    )
    variant = VariantResult(
        final_image=Image.new("RGB", (64, 64)),
        generated_background=Image.new("RGB", (64, 64)),
        metadata=meta,
        final_path=final_p,
        generated_path=gen_p,
        metadata_path=meta_p,
    )
    result = PipelineResult(
        request_id="job_track_123",
        output_dir=job_dir,
        prepared_inputs=MagicMock(),
        variants=[variant],
    )

    run_id = tracker.log_generation(result)
    assert run_id is not None or not tracker.is_available()
