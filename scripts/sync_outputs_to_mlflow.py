"""Sync all past generated output jobs into MLflow tracking database."""

import json
from pathlib import Path
import os
import mlflow

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

def sync():
    tracking_uri = "sqlite:///outputs/mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = "productstudio-generations"
    mlflow.set_experiment(experiment_name)
    
    outputs_dir = Path("outputs")
    meta_files = list(outputs_dir.glob("*/metadata_variant_*.json"))
    print(f"Found {len(meta_files)} past variant metadata files in outputs/")
    
    # Get existing logged request_ids in MLflow
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    existing_runs = client.search_runs(experiment_ids=[exp.experiment_id])
    existing_req_ids = {r.data.params.get("request_id") for r in existing_runs if r.data.params.get("request_id")}
    
    count = 0
    for meta_file in sorted(meta_files):
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            
            req_id = meta.get("request_id")
            if not req_id or req_id in existing_req_ids:
                continue
            
            run_name = f"gen_{req_id}"
            job_dir = meta_file.parent
            
            with mlflow.start_run(run_name=run_name) as run:
                # 1. Log parameters
                mlflow.log_param("request_id", req_id)
                mlflow.log_param("base_model", meta.get("base_model", "stable-diffusion-inpainting"))
                mlflow.log_param("scheduler", meta.get("scheduler", "ddim"))
                mlflow.log_param("prompt", meta.get("prompt", ""))
                mlflow.log_param("negative_prompt", meta.get("negative_prompt", ""))
                mlflow.log_param("steps", meta.get("steps", 15))
                mlflow.log_param("guidance_scale", meta.get("guidance_scale", 7.5))
                mlflow.log_param("seed", meta.get("seed", 42))
                mlflow.log_param("resolution", meta.get("resolution", "512x512"))
                mlflow.log_param("device", meta.get("device", "cuda"))
                
                # 2. Log metrics
                if "latency_seconds" in meta and meta["latency_seconds"] is not None:
                    mlflow.log_metric("latency_seconds", float(meta["latency_seconds"]))
                if "foreground_pixel_difference" in meta and meta["foreground_pixel_difference"] is not None:
                    mlflow.log_metric("foreground_pixel_difference", float(meta["foreground_pixel_difference"]))
                if "peak_gpu_memory_mb" in meta and meta["peak_gpu_memory_mb"] is not None:
                    mlflow.log_metric("peak_gpu_memory_mb", float(meta["peak_gpu_memory_mb"]))
                if "clip_similarity" in meta and meta["clip_similarity"] is not None:
                    mlflow.log_metric("clip_similarity", float(meta["clip_similarity"]))
                
                # 3. Log artifacts
                for img_file in job_dir.glob("*"):
                    if img_file.is_file():
                        mlflow.log_artifact(str(img_file))
                
                existing_req_ids.add(req_id)
                count += 1
                print(f"Logged {req_id} -> MLflow run {run.info.run_id}")
        except Exception as e:
            print(f"Failed to sync {meta_file}: {e}")
            
    print(f"\nSuccessfully synced {count} past experiments into MLflow!")

if __name__ == "__main__":
    sync()
