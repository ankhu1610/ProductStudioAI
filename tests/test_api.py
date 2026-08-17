import io
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from PIL import Image
import pytest

from app.api.app import create_app
from app.core.config import Settings
from app.schemas.generation import GenerationMetadata
from app.services.database import DatabaseService
from app.services.pipeline import PipelineResult, VariantResult


def _create_test_image_bytes(color="red", size=(64, 64), mode="RGB") -> bytes:
    img = Image.new(mode, size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def mock_app(tmp_path):
    settings = Settings(device="cpu", output_dir=tmp_path)
    db = DatabaseService(db_path=tmp_path / "test.db")

    # Mock pipeline
    mock_pipeline = MagicMock()

    def fake_run(image_input, mask_input, request, output_dir=None):
        job_dir = tmp_path / "job_mock"
        job_dir.mkdir(parents=True, exist_ok=True)
        final_p = job_dir / "final_variant_1.png"
        gen_p = job_dir / "generated_bg_variant_1.png"
        meta_p = job_dir / "metadata_variant_1.json"

        Image.new("RGB", (64, 64), color="blue").save(final_p)
        Image.new("RGB", (64, 64), color="green").save(gen_p)
        meta_p.write_text('{"mock": true}')

        meta = GenerationMetadata(
            request_id="job_mock",
            created_at="2026-08-16T12:00:00+00:00",
            base_model="test-model",
            scheduler=request.scheduler,
            seed=request.seed,
            steps=request.steps,
            guidance_scale=request.guidance_scale,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            resolution=f"{request.width}x{request.height}",
            latency_seconds=1.25,
            peak_gpu_memory_mb=None,
            device="cpu",
            foreground_pixel_difference=0.0,
            input_image=str(job_dir / "input.png"),
            input_mask=str(job_dir / "mask.png"),
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
        return PipelineResult(
            request_id="job_mock",
            output_dir=job_dir,
            prepared_inputs=MagicMock(),
            variants=[variant],
        )

    mock_pipeline.run.side_effect = fake_run

    app = create_app(settings=settings, pipeline=mock_pipeline, db=db)
    return app, tmp_path


def test_api_health(mock_app):
    app, _ = mock_app
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "model_id" in data


def test_api_generate_and_get_job(mock_app):
    app, _ = mock_app
    client = TestClient(app)

    img_bytes = _create_test_image_bytes(color="red", size=(64, 64), mode="RGB")
    mask_bytes = _create_test_image_bytes(color=255, size=(64, 64), mode="L")

    resp = client.post(
        "/api/v1/generate",
        files={
            "image": ("product.png", img_bytes, "image/png"),
            "mask": ("mask.png", mask_bytes, "image/png"),
        },
        data={
            "prompt": "luxury sneaker campaign",
            "seed": 100,
            "steps": 15,
            "scheduler": "ddim",
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["job_id"] == "job_mock"
    assert data["status"] == "completed"
    assert len(data["variants"]) == 1

    # Fetch job via GET /api/v1/jobs/{job_id}
    job_resp = client.get("/api/v1/jobs/job_mock")
    assert job_resp.status_code == 200
    job_data = job_resp.json()
    assert job_data["job_id"] == "job_mock"
    assert job_data["prompt"] == "luxury sneaker campaign"

    # Fetch output file via GET /api/v1/outputs/{job_id}/{filename}
    file_resp = client.get("/api/v1/outputs/job_mock/final_variant_1.png")
    assert file_resp.status_code == 200
    assert file_resp.headers["content-type"] == "image/png"


def test_api_job_not_found(mock_app):
    app, _ = mock_app
    client = TestClient(app)
    resp = client.get("/api/v1/jobs/non_existent_job")
    assert resp.status_code == 404
