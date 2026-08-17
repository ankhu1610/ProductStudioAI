"""SQLite persistence service for ProductStudio AI generation jobs and reproducibility records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any

from app.core.config import Settings, get_settings


@dataclass
class JobRecord:
    """Persistent database record for one generation request."""

    job_id: str
    created_at: str
    status: str
    prompt: str
    negative_prompt: str
    scheduler: str
    steps: int
    guidance_scale: float
    seed: int
    resolution: str
    num_variants: int
    latency_seconds: float | None = None
    peak_gpu_memory_mb: float | None = None
    foreground_pixel_difference: float | None = None
    clip_similarity: float | None = None
    output_dir: str | None = None
    final_images: list[str] | None = None
    metadata_files: list[str] | None = None
    error: str | None = None


class DatabaseService:
    """Manages SQLite storage for generation jobs and reproducibility metadata."""

    def __init__(self, db_path: str | Path | None = None, settings: Settings | None = None):
        if db_path is not None:
            self.db_path = Path(db_path)
        else:
            settings = settings or get_settings()
            # Handle sqlite:/// prefix if present
            url = settings.database_url
            if url.startswith("sqlite:///"):
                self.db_path = Path(url[len("sqlite:///") :])
            else:
                self.db_path = Path(url)

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS generation_jobs (
                    job_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    negative_prompt TEXT NOT NULL,
                    scheduler TEXT NOT NULL,
                    steps INTEGER NOT NULL,
                    guidance_scale REAL NOT NULL,
                    seed INTEGER NOT NULL,
                    resolution TEXT NOT NULL,
                    num_variants INTEGER NOT NULL,
                    latency_seconds REAL,
                    peak_gpu_memory_mb REAL,
                    foreground_pixel_difference REAL,
                    clip_similarity REAL,
                    output_dir TEXT,
                    final_images_json TEXT,
                    metadata_files_json TEXT,
                    error TEXT
                )
                """
            )
            conn.commit()

    def save_job(self, record: JobRecord) -> None:
        """Insert or update a job record."""
        final_images_json = json.dumps(record.final_images or [])
        metadata_files_json = json.dumps(record.metadata_files or [])

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO generation_jobs (
                    job_id, created_at, status, prompt, negative_prompt, scheduler,
                    steps, guidance_scale, seed, resolution, num_variants,
                    latency_seconds, peak_gpu_memory_mb, foreground_pixel_difference,
                    clip_similarity, output_dir, final_images_json, metadata_files_json, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.job_id,
                    record.created_at,
                    record.status,
                    record.prompt,
                    record.negative_prompt,
                    record.scheduler,
                    record.steps,
                    record.guidance_scale,
                    record.seed,
                    record.resolution,
                    record.num_variants,
                    record.latency_seconds,
                    record.peak_gpu_memory_mb,
                    record.foreground_pixel_difference,
                    record.clip_similarity,
                    record.output_dir,
                    final_images_json,
                    metadata_files_json,
                    record.error,
                ),
            )
            conn.commit()

    def get_job(self, job_id: str) -> JobRecord | None:
        """Fetch a job record by job_id."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM generation_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()

            if row is None:
                return None

            return self._row_to_record(row)

    def list_jobs(self, limit: int = 50, offset: int = 0) -> list[JobRecord]:
        """Fetch a paginated list of recent generation jobs."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM generation_jobs
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()

            return [self._row_to_record(r) for r in rows]

    def _row_to_record(self, row: sqlite3.Row) -> JobRecord:
        final_images = json.loads(row["final_images_json"]) if row["final_images_json"] else []
        metadata_files = json.loads(row["metadata_files_json"]) if row["metadata_files_json"] else []

        return JobRecord(
            job_id=row["job_id"],
            created_at=row["created_at"],
            status=row["status"],
            prompt=row["prompt"],
            negative_prompt=row["negative_prompt"],
            scheduler=row["scheduler"],
            steps=row["steps"],
            guidance_scale=row["guidance_scale"],
            seed=row["seed"],
            resolution=row["resolution"],
            num_variants=row["num_variants"],
            latency_seconds=row["latency_seconds"],
            peak_gpu_memory_mb=row["peak_gpu_memory_mb"],
            foreground_pixel_difference=row["foreground_pixel_difference"],
            clip_similarity=row["clip_similarity"],
            output_dir=row["output_dir"],
            final_images=final_images,
            metadata_files=metadata_files,
            error=row["error"],
        )
