"""Phase 2 benchmark runner for scheduler and step-count comparisons."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import Settings, get_settings
from app.schemas.generation import GenerationRequest
from app.services.metrics import clip_similarity, summarize_latencies
from app.services.pipeline import ProductStudioPipeline

DEFAULT_STEP_COUNTS = (10, 15, 25, 50)
DEFAULT_SCHEDULERS = ("ddim", "pndm")
BENCHMARK_SPEC_PATH = Path(__file__).resolve().parent / "benchmark_prompts.json"


@dataclass(frozen=True)
class BenchmarkCase:
    """One versioned product/prompt pair from benchmark_prompts.json."""

    id: str
    image: Path
    mask: Path
    prompt: str
    seed: int


@dataclass(frozen=True)
class BenchmarkRunRow:
    """One measured benchmark execution."""

    case_id: str
    scheduler: str
    steps: int
    seed: int
    status: str
    latency_seconds: float | None
    peak_gpu_memory_mb: float | None
    foreground_pixel_difference: float | None
    clip_similarity: float | None
    output_dir: str | None
    final_image: str | None
    error: str | None = None


def load_benchmark_spec(path: Path = BENCHMARK_SPEC_PATH) -> tuple[dict[str, Any], list[BenchmarkCase]]:
    """Load the canonical benchmark cases and shared defaults."""

    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)

    root = path.resolve().parents[1]
    defaults = payload.get("defaults", {})
    cases: list[BenchmarkCase] = []
    for entry in payload["cases"]:
        cases.append(
            BenchmarkCase(
                id=entry["id"],
                image=(root / entry["image"]).resolve(),
                mask=(root / entry["mask"]).resolve(),
                prompt=entry["prompt"],
                seed=int(entry["seed"]),
            )
        )
    return defaults, cases


def _write_csv(path: Path, rows: list[BenchmarkRunRow]) -> None:
    fieldnames = list(BenchmarkRunRow.__annotations__.keys())
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _write_human_rating_template(path: Path, rows: list[BenchmarkRunRow]) -> None:
    fieldnames = [
        "case_id",
        "scheduler",
        "steps",
        "clip_similarity",
        "latency_seconds",
        "quality_rating_1_5",
        "prompt_alignment_1_5",
        "boundary_quality_1_5",
        "notes",
        "final_image",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if row.status != "ok":
                continue
            writer.writerow(
                {
                    "case_id": row.case_id,
                    "scheduler": row.scheduler,
                    "steps": row.steps,
                    "clip_similarity": row.clip_similarity,
                    "latency_seconds": row.latency_seconds,
                    "quality_rating_1_5": "",
                    "prompt_alignment_1_5": "",
                    "boundary_quality_1_5": "",
                    "notes": "",
                    "final_image": row.final_image,
                }
            )


def _pick_recommendation(summaries: list[dict[str, object]]) -> dict[str, object]:
    """Prefer the fastest configuration that keeps the highest mean CLIP score."""

    ok_rows = [row for row in summaries if row["count"] > 0]
    if not ok_rows:
        return {"scheduler": None, "steps": None, "reason": "no successful benchmark runs"}

    max_clip = max(float(row["mean_clip_similarity"]) for row in ok_rows)
    candidates = [row for row in ok_rows if float(row["mean_clip_similarity"]) >= max_clip - 0.01]
    best = min(candidates, key=lambda row: (float(row["p95_seconds"]), int(row["steps"])))
    return {
        "scheduler": best["scheduler"],
        "steps": best["steps"],
        "p50_seconds": best["p50_seconds"],
        "p95_seconds": best["p95_seconds"],
        "mean_clip_similarity": best["mean_clip_similarity"],
        "reason": "highest mean CLIP among fastest p95 latency candidates",
    }


def run_benchmark_suite(
    *,
    settings: Settings | None = None,
    output_dir: Path | None = None,
    step_counts: tuple[int, ...] = DEFAULT_STEP_COUNTS,
    schedulers: tuple[str, ...] = DEFAULT_SCHEDULERS,
    spec_path: Path = BENCHMARK_SPEC_PATH,
    compute_clip: bool = True,
    log_mlflow: bool = True,
) -> Path:
    """Execute all benchmark cases and write CSV, summary JSON, and optional MLflow artifacts."""

    defaults, cases = load_benchmark_spec(spec_path)
    settings = settings or get_settings()
    pipeline = ProductStudioPipeline(settings=settings)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    benchmark_dir = Path(output_dir or settings.output_dir / "benchmarks" / timestamp)
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    rows: list[BenchmarkRunRow] = []
    for case in cases:
        for scheduler in schedulers:
            for steps in step_counts:
                request = GenerationRequest(
                    prompt=case.prompt,
                    negative_prompt=str(defaults.get("negative_prompt", "")),
                    seed=case.seed,
                    steps=steps,
                    guidance_scale=float(defaults.get("guidance_scale", 7.5)),
                    width=int(defaults.get("width", 512)),
                    height=int(defaults.get("height", 512)),
                    num_variants=1,
                    scheduler=scheduler,  # type: ignore[arg-type]
                )
                run_dir = benchmark_dir / f"{case.id}_{scheduler}_{steps}"
                try:
                    result = pipeline.run(
                        image_input=case.image,
                        mask_input=case.mask,
                        request=request,
                        output_dir=run_dir,
                    )
                    variant = result.variants[0]
                    clip_score = clip_similarity(variant.final_image, case.prompt) if compute_clip else None
                    rows.append(
                        BenchmarkRunRow(
                            case_id=case.id,
                            scheduler=scheduler,
                            steps=steps,
                            seed=case.seed,
                            status="ok",
                            latency_seconds=variant.metadata.latency_seconds,
                            peak_gpu_memory_mb=variant.metadata.peak_gpu_memory_mb,
                            foreground_pixel_difference=variant.metadata.foreground_pixel_difference,
                            clip_similarity=clip_score,
                            output_dir=str(result.output_dir),
                            final_image=str(variant.final_path),
                        )
                    )
                except Exception as error:  # noqa: BLE001 - benchmark should continue after one failure
                    rows.append(
                        BenchmarkRunRow(
                            case_id=case.id,
                            scheduler=scheduler,
                            steps=steps,
                            seed=case.seed,
                            status="failed",
                            latency_seconds=None,
                            peak_gpu_memory_mb=None,
                            foreground_pixel_difference=None,
                            clip_similarity=None,
                            output_dir=str(run_dir),
                            final_image=None,
                            error=str(error),
                        )
                    )

    results_csv = benchmark_dir / "benchmark_results.csv"
    _write_csv(results_csv, rows)
    _write_human_rating_template(benchmark_dir / "human_rating_sheet.csv", rows)

    row_dicts = [asdict(row) for row in rows]
    summaries: list[dict[str, object]] = []
    for scheduler in schedulers:
        for steps in step_counts:
            summary = summarize_latencies(row_dicts, scheduler=scheduler, steps=steps)
            clip_values = [
                float(row["clip_similarity"])
                for row in row_dicts
                if row["scheduler"] == scheduler
                and row["steps"] == steps
                and row["status"] == "ok"
                and row["clip_similarity"] is not None
            ]
            summaries.append(
                {
                    **asdict(summary),
                    "mean_clip_similarity": round(float(sum(clip_values) / len(clip_values)), 4)
                    if clip_values
                    else None,
                }
            )

    summary_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "spec_path": str(spec_path.resolve()),
        "step_counts": list(step_counts),
        "schedulers": list(schedulers),
        "successful_runs": sum(1 for row in rows if row.status == "ok"),
        "failed_runs": sum(1 for row in rows if row.status == "failed"),
        "summaries": summaries,
        "recommendation": _pick_recommendation(summaries),
    }
    summary_json = benchmark_dir / "benchmark_summary.json"
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    if log_mlflow:
        _log_to_mlflow(settings, benchmark_dir, summary_payload, results_csv)

    return benchmark_dir


def _log_to_mlflow(
    settings: Settings,
    benchmark_dir: Path,
    summary_payload: dict[str, object],
    results_csv: Path,
) -> None:
    try:
        import mlflow
    except ImportError:
        return

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("productstudio-benchmarks")
    recommendation = summary_payload["recommendation"]
    assert isinstance(recommendation, dict)

    with mlflow.start_run(run_name=benchmark_dir.name):
        mlflow.log_param("step_counts", ",".join(str(value) for value in summary_payload["step_counts"]))
        mlflow.log_param("schedulers", ",".join(str(value) for value in summary_payload["schedulers"]))
        mlflow.log_metric("successful_runs", float(summary_payload["successful_runs"]))
        mlflow.log_metric("failed_runs", float(summary_payload["failed_runs"]))

        summaries = summary_payload["summaries"]
        assert isinstance(summaries, list)
        for entry in summaries:
            assert isinstance(entry, dict)
            tag = f"{entry['scheduler']}_{entry['steps']}"
            mlflow.log_metric(f"p50_seconds_{tag}", float(entry["p50_seconds"]))
            mlflow.log_metric(f"p95_seconds_{tag}", float(entry["p95_seconds"]))
            if entry.get("mean_clip_similarity") is not None:
                mlflow.log_metric(f"mean_clip_{tag}", float(entry["mean_clip_similarity"]))
            if entry.get("mean_peak_gpu_memory_mb") is not None:
                mlflow.log_metric(f"mean_vram_mb_{tag}", float(entry["mean_peak_gpu_memory_mb"]))

        if recommendation.get("scheduler") is not None:
            mlflow.log_param("recommended_scheduler", str(recommendation["scheduler"]))
            mlflow.log_param("recommended_steps", int(recommendation["steps"]))
            if recommendation.get("p95_seconds") is not None:
                mlflow.log_metric("recommended_p95_seconds", float(recommendation["p95_seconds"]))
            if recommendation.get("mean_clip_similarity") is not None:
                mlflow.log_metric("recommended_mean_clip", float(recommendation["mean_clip_similarity"]))

        mlflow.log_artifact(str(results_csv))
        mlflow.log_artifact(str(benchmark_dir / "benchmark_summary.json"))
        mlflow.log_artifact(str(benchmark_dir / "human_rating_sheet.csv"))
