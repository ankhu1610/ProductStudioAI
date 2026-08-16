"""Benchmark execution runner for measuring sampling quality, latency, and hardware metrics."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from PIL import Image

from app.core.config import Settings, get_settings
from app.schemas.generation import GenerationRequest
from app.services.compositing import composite_original_foreground, foreground_pixel_difference
from app.services.generation import GenerationService
from app.services.image_io import load_image, load_mask, prepare_inputs
from app.services.metrics import (
    compute_clip_similarity,
    compute_percentiles,
    get_peak_gpu_memory_mb,
    reset_peak_gpu_memory,
)


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    image_path: Path
    mask_path: Path
    prompt: str
    seed: int


@dataclass
class BenchmarkRunResult:
    case_id: str
    scheduler: str
    steps: int
    seed: int
    prompt: str
    resolution: str
    latency_seconds: float
    peak_gpu_memory_mb: float | None
    foreground_pixel_diff: float
    clip_similarity: float | None
    final_image_path: str
    generated_image_path: str


class BenchmarkRunner:
    """Runs systematic benchmarks across schedulers and step counts."""

    def __init__(
        self,
        settings: Settings | None = None,
        generation_service: GenerationService | None = None,
        spec_path: str | Path = "eval/benchmark_prompts.json",
    ):
        self.settings = settings or get_settings()
        self.generation_service = generation_service or GenerationService(self.settings)
        self.spec_path = Path(spec_path)
        self._load_spec()

    def _load_spec(self) -> None:
        with open(self.spec_path, encoding="utf-8") as f:
            data = json.load(f)
        self.defaults = data.get("defaults", {})
        self.cases: list[BenchmarkCase] = [
            BenchmarkCase(
                id=case["id"],
                image_path=Path(case["image"]),
                mask_path=Path(case["mask"]),
                prompt=case["prompt"],
                seed=int(case["seed"]),
            )
            for case in data.get("cases", [])
        ]

    def run_benchmark(
        self,
        steps: list[int] | None = None,
        schedulers: list[str] | None = None,
        case_ids: list[str] | None = None,
        skip_clip: bool = False,
        output_dir: str | Path | None = None,
        log_mlflow: bool = True,
    ) -> dict[str, Any]:
        """Execute benchmark matrix and produce CSV results, summary JSON, and rating sheet."""

        target_steps = steps or [10, 15, 25, 50]
        target_schedulers = schedulers or ["ddim", "pndm"]
        filtered_cases = (
            [c for c in self.cases if c.id in case_ids] if case_ids else self.cases
        )

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        bench_dir = Path(output_dir or (self.settings.output_dir / "benchmarks" / timestamp))
        images_dir = bench_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        results: list[BenchmarkRunResult] = []

        print(f"Starting ProductStudio AI Benchmark ({len(filtered_cases)} cases x {len(target_schedulers)} schedulers x {len(target_steps)} step counts)")
        print("=" * 70)

        for case in filtered_cases:
            orig_img = load_image(case.image_path)
            orig_mask = load_mask(case.mask_path)
            prepared = prepare_inputs(
                orig_img,
                orig_mask,
                width=self.defaults.get("width", 512),
                height=self.defaults.get("height", 512),
            )

            for sched in target_schedulers:
                for step_count in target_steps:
                    req = GenerationRequest(
                        prompt=case.prompt,
                        negative_prompt=self.defaults.get("negative_prompt", ""),
                        seed=case.seed,
                        steps=step_count,
                        guidance_scale=self.defaults.get("guidance_scale", 7.5),
                        width=self.defaults.get("width", 512),
                        height=self.defaults.get("height", 512),
                        num_variants=1,
                        scheduler=sched,  # type: ignore[arg-type]
                    )

                    reset_peak_gpu_memory()

                    gen_bg, latency, peak_vram = self.generation_service.generate_background(
                        prepared.image,
                        prepared.inpaint_mask,
                        req,
                    )

                    if peak_vram is None:
                        peak_vram = get_peak_gpu_memory_mb()

                    final_comp = composite_original_foreground(
                        prepared.image,
                        gen_bg,
                        prepared.protected_mask,
                    )

                    pixel_diff = foreground_pixel_difference(
                        prepared.image,
                        final_comp,
                        prepared.protected_mask,
                    )

                    clip_score: float | None = None
                    if not skip_clip:
                        try:
                            clip_score = compute_clip_similarity(
                                final_comp,
                                case.prompt,
                                device=self.settings.device if self.settings.device == "cuda" else "cpu",
                            )
                        except Exception as err:
                            print(f"[Warning] CLIP scoring skipped for {case.id}: {err}")

                    final_filename = f"{case.id}_{sched}_{step_count}s_final.png"
                    gen_filename = f"{case.id}_{sched}_{step_count}s_bg.png"
                    final_path = images_dir / final_filename
                    gen_path = images_dir / gen_filename

                    final_comp.save(final_path)
                    gen_bg.save(gen_path)

                    run_result = BenchmarkRunResult(
                        case_id=case.id,
                        scheduler=sched,
                        steps=step_count,
                        seed=case.seed,
                        prompt=case.prompt,
                        resolution=f"{req.width}x{req.height}",
                        latency_seconds=round(latency, 3),
                        peak_gpu_memory_mb=peak_vram,
                        foreground_pixel_diff=pixel_diff,
                        clip_similarity=clip_score,
                        final_image_path=str(final_path),
                        generated_image_path=str(gen_path),
                    )
                    results.append(run_result)

                    print(
                        f"[{case.id:18}] {sched.upper():5} | Steps: {step_count:2} | "
                        f"Latency: {latency:5.2f}s | Diff: {pixel_diff:.1f} | "
                        f"CLIP: {clip_score if clip_score is not None else 0.0:.3f}"
                    )

        # 1. Export CSV
        csv_path = bench_dir / "benchmark_results.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "case_id",
                "scheduler",
                "steps",
                "seed",
                "prompt",
                "resolution",
                "latency_seconds",
                "peak_gpu_memory_mb",
                "foreground_pixel_diff",
                "clip_similarity",
                "final_image_path",
                "generated_image_path",
            ])
            for r in results:
                writer.writerow([
                    r.case_id,
                    r.scheduler,
                    r.steps,
                    r.seed,
                    r.prompt,
                    r.resolution,
                    r.latency_seconds,
                    r.peak_gpu_memory_mb if r.peak_gpu_memory_mb is not None else "",
                    r.foreground_pixel_diff,
                    r.clip_similarity if r.clip_similarity is not None else "",
                    r.final_image_path,
                    r.generated_image_path,
                ])

        # 2. Export Human Rating Sheet Template
        rating_sheet_path = bench_dir / "human_rating_sheet.csv"
        with open(rating_sheet_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "case_id",
                "scheduler",
                "steps",
                "prompt",
                "final_image_path",
                "product_fidelity_1to5",
                "prompt_alignment_1to5",
                "boundary_quality_1to5",
                "overall_score_1to5",
                "notes",
            ])
            for r in results:
                writer.writerow([
                    r.case_id,
                    r.scheduler,
                    r.steps,
                    r.prompt,
                    r.final_image_path,
                    "",
                    "",
                    "",
                    "",
                    "",
                ])

        # 3. Compute Aggregated Summary
        summary = self._compute_summary(results, target_schedulers, target_steps)
        summary_path = bench_dir / "benchmark_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # 4. Optional MLflow Logging
        if log_mlflow:
            self._log_to_mlflow(summary, csv_path, summary_path, rating_sheet_path)

        print("=" * 70)
        print(f"Benchmark finished! Outputs saved to: {bench_dir}")
        print(f"Summary: {summary_path.name} | Results: {csv_path.name} | Rating Sheet: {rating_sheet_path.name}")
        if "recommended_configuration" in summary:
            rec = summary["recommended_configuration"]
            print(f"Recommended Setting: {rec.get('scheduler', '').upper()} at {rec.get('steps', '')} steps (p50: {rec.get('p50_latency', '')}s, CLIP: {rec.get('mean_clip', '')})")

        return {
            "output_dir": str(bench_dir),
            "results_csv": str(csv_path),
            "summary_json": str(summary_path),
            "rating_sheet_csv": str(rating_sheet_path),
            "summary": summary,
        }

    def _compute_summary(
        self,
        results: list[BenchmarkRunResult],
        schedulers: list[str],
        steps: list[int],
    ) -> dict[str, Any]:
        grouped: dict[str, Any] = {}

        best_score = -1.0
        recommended: dict[str, Any] = {}

        for sched in schedulers:
            for s in steps:
                sub = [r for r in results if r.scheduler == sched and r.steps == s]
                if not sub:
                    continue
                latencies = [r.latency_seconds for r in sub]
                vram_vals = [r.peak_gpu_memory_mb for r in sub if r.peak_gpu_memory_mb is not None]
                clip_vals = [r.clip_similarity for r in sub if r.clip_similarity is not None]

                lat_stats = compute_percentiles(latencies)
                mean_vram = round(float(sum(vram_vals) / len(vram_vals)), 2) if vram_vals else None
                mean_clip = round(float(sum(clip_vals) / len(clip_vals)), 4) if clip_vals else None

                key = f"{sched}_{s}steps"
                grouped[key] = {
                    "scheduler": sched,
                    "steps": s,
                    "total_runs": len(sub),
                    "latency": lat_stats,
                    "mean_peak_gpu_memory_mb": mean_vram,
                    "mean_clip_similarity": mean_clip,
                    "foreground_pixel_diff_max": max(r.foreground_pixel_diff for r in sub),
                }

                # Recommendation heuristic: balance good quality (s >= 15) with low latency
                clip_weight = mean_clip if mean_clip is not None else 0.5
                latency_pen = lat_stats["p50"] if lat_stats["p50"] > 0 else 1.0
                score = (clip_weight * 100) / (latency_pen ** 0.5) if s >= 15 else -1.0

                if score > best_score:
                    best_score = score
                    recommended = {
                        "scheduler": sched,
                        "steps": s,
                        "p50_latency": lat_stats["p50"],
                        "p95_latency": lat_stats["p95"],
                        "mean_clip": mean_clip,
                    }

        if not recommended and grouped:
            first_key = next(iter(grouped))
            first_entry = grouped[first_key]
            recommended = {
                "scheduler": first_entry["scheduler"],
                "steps": first_entry["steps"],
                "p50_latency": first_entry["latency"]["p50"],
                "p95_latency": first_entry["latency"]["p95"],
                "mean_clip": first_entry["mean_clip_similarity"],
            }

        return {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "configurations": grouped,
            "recommended_configuration": recommended,
        }

    def _log_to_mlflow(
        self,
        summary: dict[str, Any],
        csv_path: Path,
        summary_path: Path,
        rating_sheet_path: Path,
    ) -> None:
        try:
            import mlflow

            mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
            mlflow.set_experiment("productstudio-benchmarks")

            with mlflow.start_run(run_name=f"benchmark_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"):
                rec = summary.get("recommended_configuration", {})
                if rec:
                    mlflow.log_param("recommended_scheduler", rec.get("scheduler"))
                    mlflow.log_param("recommended_steps", rec.get("steps"))
                    if rec.get("p50_latency") is not None:
                        mlflow.log_metric("rec_p50_latency", rec["p50_latency"])
                    if rec.get("p95_latency") is not None:
                        mlflow.log_metric("rec_p95_latency", rec["p95_latency"])
                    if rec.get("mean_clip") is not None:
                        mlflow.log_metric("rec_mean_clip", rec["mean_clip"])

                for conf_name, conf_data in summary.get("configurations", {}).items():
                    lat = conf_data.get("latency", {})
                    mlflow.log_metric(f"{conf_name}_p50_latency", lat.get("p50", 0.0))
                    mlflow.log_metric(f"{conf_name}_p95_latency", lat.get("p95", 0.0))
                    if conf_data.get("mean_clip_similarity") is not None:
                        mlflow.log_metric(f"{conf_name}_mean_clip", conf_data["mean_clip_similarity"])
                    if conf_data.get("mean_peak_gpu_memory_mb") is not None:
                        mlflow.log_metric(f"{conf_name}_peak_vram_mb", conf_data["mean_peak_gpu_memory_mb"])

                mlflow.log_artifact(str(csv_path))
                mlflow.log_artifact(str(summary_path))
                mlflow.log_artifact(str(rating_sheet_path))
        except Exception as err:
            print(f"[MLflow] Optional experiment tracking skipped: {err}")
