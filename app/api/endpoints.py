"""FastAPI endpoints for ProductStudio AI generation, health, jobs, and output serving."""

from __future__ import annotations

from datetime import datetime, timezone
import io
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse
from PIL import Image

from app.core.config import Settings, get_settings
from app.schemas.generation import GenerationRequest
from app.services.database import DatabaseService, JobRecord
from app.services.image_io import InputValidationError
from app.services.pipeline import PipelineResult, ProductStudioPipeline
from app.services.tracking import MLflowTracker

router = APIRouter()


def get_app_settings(request: Request) -> Settings:
    return getattr(request.app.state, "settings", get_settings())


def get_db_service(request: Request) -> DatabaseService:
    return getattr(request.app.state, "db_service", DatabaseService(settings=get_app_settings(request)))


def get_pipeline(request: Request) -> ProductStudioPipeline:
    return getattr(request.app.state, "pipeline", ProductStudioPipeline(settings=get_app_settings(request)))


def get_tracker(request: Request) -> MLflowTracker:
    return getattr(request.app.state, "tracker", MLflowTracker(settings=get_app_settings(request)))


@router.get("/health")
def health_check(settings: Settings = Depends(get_app_settings)):
    """System health check, GPU diagnostics, and model status."""
    cuda_available = False
    gpu_name = None
    gpu_vram = None

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_vram = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB"
    except Exception:
        pass

    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": settings.model_id,
        "device": settings.device,
        "precision": settings.precision,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "gpu_vram": gpu_vram,
    }


@router.post("/api/v1/generate", status_code=status.HTTP_201_CREATED)
async def generate_ad_creative(
    image: UploadFile = File(..., description="Product RGB image file"),
    mask: UploadFile = File(..., description="Product mask image file (white=product)"),
    prompt: str = Form(..., min_length=3, max_length=500),
    negative_prompt: str = Form(""),
    seed: int = Form(42),
    steps: int = Form(15),
    guidance_scale: float = Form(7.5),
    width: int = Form(512),
    height: int = Form(512),
    num_variants: int = Form(1),
    scheduler: str = Form("ddim"),
    pipeline: ProductStudioPipeline = Depends(get_pipeline),
    db: DatabaseService = Depends(get_db_service),
    tracker: MLflowTracker = Depends(get_tracker),
):
    """Generate prompt-controlled e-commerce ad backgrounds while preserving product foreground."""
    # 1. Parse and validate image inputs
    try:
        image_bytes = await image.read()
        mask_bytes = await mask.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        pil_mask = Image.open(io.BytesIO(mask_bytes)).convert("L")
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image or mask file: {err}",
        ) from err

    # 2. Validate request parameters
    try:
        req = GenerationRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            num_variants=num_variants,
            scheduler=scheduler,  # type: ignore[arg-type]
        )
    except Exception as err:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(err)) from err

    # 3. Execute pipeline
    try:
        result: PipelineResult = pipeline.run(
            image_input=pil_image,
            mask_input=pil_mask,
            request=req,
        )
    except InputValidationError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err)) from err
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error during generation: {err}",
        ) from err

    # 4. Save to database
    final_paths = [v.final_path.name for v in result.variants]
    meta_paths = [v.metadata_path.name for v in result.variants]
    first_meta = result.variants[0].metadata

    record = JobRecord(
        job_id=result.request_id,
        created_at=first_meta.created_at,
        status="completed",
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        scheduler=req.scheduler,
        steps=req.steps,
        guidance_scale=req.guidance_scale,
        seed=req.seed,
        resolution=f"{req.width}x{req.height}",
        num_variants=req.num_variants,
        latency_seconds=first_meta.latency_seconds,
        peak_gpu_memory_mb=first_meta.peak_gpu_memory_mb,
        foreground_pixel_difference=first_meta.foreground_pixel_difference,
        output_dir=str(result.output_dir),
        final_images=final_paths,
        metadata_files=meta_paths,
    )
    db.save_job(record)

    # 5. Log to MLflow if available
    try:
        tracker.log_generation(result)
    except Exception:
        pass

    # 6. Build response
    variants_resp = []
    for v in result.variants:
        variants_resp.append(
            {
                "final_image_url": f"/api/v1/outputs/{result.request_id}/{v.final_path.name}",
                "generated_bg_url": f"/api/v1/outputs/{result.request_id}/{v.generated_path.name}",
                "metadata_url": f"/api/v1/outputs/{result.request_id}/{v.metadata_path.name}",
                "metadata": v.metadata.model_dump(),
            }
        )

    return {
        "job_id": result.request_id,
        "status": "completed",
        "created_at": first_meta.created_at,
        "prompt": req.prompt,
        "scheduler": req.scheduler,
        "steps": req.steps,
        "resolution": f"{req.width}x{req.height}",
        "variants": variants_resp,
    }


@router.get("/api/v1/jobs/{job_id}")
def get_job_by_id(job_id: str, db: DatabaseService = Depends(get_db_service)):
    """Retrieve metadata and results for a completed generation job."""
    record = db.get_job(job_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found")

    variant_urls = [f"/api/v1/outputs/{job_id}/{img}" for img in (record.final_images or [])]
    return {
        "job_id": record.job_id,
        "created_at": record.created_at,
        "status": record.status,
        "prompt": record.prompt,
        "scheduler": record.scheduler,
        "steps": record.steps,
        "seed": record.seed,
        "resolution": record.resolution,
        "num_variants": record.num_variants,
        "latency_seconds": record.latency_seconds,
        "peak_gpu_memory_mb": record.peak_gpu_memory_mb,
        "foreground_pixel_difference": record.foreground_pixel_difference,
        "final_images": variant_urls,
        "error": record.error,
    }


@router.get("/api/v1/jobs")
def list_all_jobs(limit: int = 50, offset: int = 0, db: DatabaseService = Depends(get_db_service)):
    """List recent generation jobs."""
    records = db.list_jobs(limit=limit, offset=offset)
    return [
        {
            "job_id": r.job_id,
            "created_at": r.created_at,
            "status": r.status,
            "prompt": r.prompt,
            "scheduler": r.scheduler,
            "steps": r.steps,
            "latency_seconds": r.latency_seconds,
            "final_images": [f"/api/v1/outputs/{r.job_id}/{img}" for img in (r.final_images or [])],
        }
        for r in records
    ]


@router.get("/api/v1/outputs/{job_id}/{filename}")
def get_output_file(
    job_id: str,
    filename: str,
    settings: Settings = Depends(get_app_settings),
):
    """Safely serve generated image or metadata files."""
    # Prevent path traversal attacks
    if ".." in job_id or ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid path parameters")

    file_path = settings.output_dir / job_id / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"File {filename} not found for job {job_id}")

    media_type = "application/json" if filename.endswith(".json") else "image/png"
    return FileResponse(path=file_path, media_type=media_type)
