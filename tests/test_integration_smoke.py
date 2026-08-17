import io
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
import pytest

from app.api.app import create_app
from app.core.config import Settings
from app.services.database import DatabaseService
from app.services.pipeline import ProductStudioPipeline
from app.services.tracking import MLflowTracker


def _create_synthetic_test_inputs():
    img = Image.new("RGB", (64, 64), color="red")
    mask = Image.new("L", (64, 64), color=0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle((16, 16, 48, 48), fill=255)

    img_buf = io.BytesIO()
    mask_buf = io.BytesIO()
    img.save(img_buf, format="PNG")
    mask.save(mask_buf, format="PNG")
    return img_buf.getvalue(), mask_buf.getvalue()


def test_full_integration_smoke(tmp_path):
    # 1. Setup isolated settings and services
    settings = Settings(
        device="cpu",
        output_dir=tmp_path / "outputs",
        database_url=f"sqlite:///{tmp_path / 'smoke.db'}",
        mlflow_tracking_uri=str(tmp_path / "mlruns"),
    )
    db = DatabaseService(settings=settings)
    tracker = MLflowTracker(settings=settings)

    # Mock inpainting generator to return synthetic generated background
    mock_gen = MagicMock()
    mock_gen.generate_background.return_value = (
        Image.new("RGB", (64, 64), color="blue"),
        0.5,
        1024.0,
    )
    pipeline = ProductStudioPipeline(settings=settings, generation_service=mock_gen)

    # 2. Build FastAPI application
    app = create_app(settings=settings, pipeline=pipeline, db=db, tracker=tracker)
    client = TestClient(app)

    # 3. Health check
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["status"] == "healthy"

    # 4. Execute Generation Request via POST /api/v1/generate
    img_bytes, mask_bytes = _create_synthetic_test_inputs()
    gen_resp = client.post(
        "/api/v1/generate",
        files={
            "image": ("test_product.png", img_bytes, "image/png"),
            "mask": ("test_mask.png", mask_bytes, "image/png"),
        },
        data={
            "prompt": "luxury product in sunlit studio",
            "negative_prompt": "blurry",
            "scheduler": "ddim",
            "steps": 10,
            "guidance_scale": 7.5,
            "seed": 42,
            "width": 64,
            "height": 64,
            "num_variants": 1,
        },
    )
    assert gen_resp.status_code == 201
    gen_data = gen_resp.json()
    job_id = gen_data["job_id"]
    assert gen_data["status"] == "completed"
    assert len(gen_data["variants"]) == 1

    # 5. Verify SQLite Database Record via GET /api/v1/jobs/{job_id}
    job_resp = client.get(f"/api/v1/jobs/{job_id}")
    assert job_resp.status_code == 200
    job_data = job_resp.json()
    assert job_data["job_id"] == job_id
    assert job_data["prompt"] == "luxury product in sunlit studio"
    assert job_data["foreground_pixel_difference"] == 0.0

    # 6. Verify Output Artifacts via GET /api/v1/outputs/{job_id}/{filename}
    final_img_resp = client.get(f"/api/v1/outputs/{job_id}/final_variant_1.png")
    assert final_img_resp.status_code == 200
    assert final_img_resp.headers["content-type"] == "image/png"

    meta_file_resp = client.get(f"/api/v1/outputs/{job_id}/metadata_variant_1.json")
    assert meta_file_resp.status_code == 200
    assert meta_file_resp.headers["content-type"] == "application/json"
