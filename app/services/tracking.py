"""MLflow experiment and artifact tracking service for ProductStudio AI."""

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.services.pipeline import PipelineResult

logger = get_logger("tracking")


class MLflowTracker:
    """Tracks generation experiments, hyperparameters, performance metrics, and image artifacts."""

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str = "productstudio-generations",
        settings: Settings | None = None,
    ):
        self.settings = settings or get_settings()
        self.tracking_uri = tracking_uri or self.settings.mlflow_tracking_uri
        self.experiment_name = experiment_name

    def is_available(self) -> bool:
        """Check if MLflow is installed and accessible."""
        try:
            import mlflow

            return True
        except ImportError:
            return False

    def log_generation(self, result: PipelineResult) -> str | None:
        """Log a completed PipelineResult to MLflow and return the run ID."""
        if not self.is_available():
            logger.debug("MLflow is not installed; skipping experiment tracking.")
            return None

        try:
            import mlflow

            uri = str(self.tracking_uri)
            if "://" not in uri:
                uri = Path(uri).resolve().as_uri()

            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(self.experiment_name)

            first_variant = result.variants[0]
            meta = first_variant.metadata

            run_name = f"gen_{result.request_id}"
            with mlflow.start_run(run_name=run_name) as run:
                # 1. Log parameters
                mlflow.log_param("request_id", result.request_id)
                mlflow.log_param("base_model", meta.base_model)
                mlflow.log_param("scheduler", meta.scheduler)
                mlflow.log_param("prompt", meta.prompt)
                mlflow.log_param("negative_prompt", meta.negative_prompt)
                mlflow.log_param("steps", meta.steps)
                mlflow.log_param("guidance_scale", meta.guidance_scale)
                mlflow.log_param("seed", meta.seed)
                mlflow.log_param("resolution", meta.resolution)
                mlflow.log_param("num_variants", len(result.variants))
                mlflow.log_param("device", meta.device)

                # 2. Log metrics
                mlflow.log_metric("latency_seconds", meta.latency_seconds)
                mlflow.log_metric("foreground_pixel_difference", meta.foreground_pixel_difference)
                if meta.peak_gpu_memory_mb is not None:
                    mlflow.log_metric("peak_gpu_memory_mb", meta.peak_gpu_memory_mb)
                if meta.clip_similarity is not None:
                    mlflow.log_metric("clip_similarity", meta.clip_similarity)

                # 3. Log artifacts from output directory
                job_dir = result.output_dir
                if job_dir.exists():
                    for file_path in job_dir.glob("*"):
                        if file_path.is_file():
                            mlflow.log_artifact(str(file_path))

                logger.info(f"Successfully logged generation {result.request_id} to MLflow run {run.info.run_id}")
                return run.info.run_id

        except Exception as err:
            logger.warning(f"Failed to log to MLflow: {err}")
            return None
