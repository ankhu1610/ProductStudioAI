"""CLI entrypoint for running ProductStudio AI Phase 2 benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from app.core.config import get_settings
from eval.runner import BenchmarkRunner


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run reproducible sampling and latency benchmarks across schedulers and step counts."
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[10, 15, 25, 50],
        help="List of inference step counts to benchmark (e.g. --steps 10 15 25 50)",
    )
    parser.add_argument(
        "--schedulers",
        type=str,
        nargs="+",
        default=["ddim", "pndm"],
        help="List of schedulers to benchmark (e.g. --schedulers ddim pndm dpm_solver)",
    )
    parser.add_argument(
        "--cases",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of case IDs to run (e.g. --cases sneaker-neon watch-marble)",
    )
    parser.add_argument(
        "--skip-clip",
        action="store_true",
        help="Skip CLIP text-image similarity evaluation (faster on low-resource hardware)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Custom directory for benchmark results",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow experiment artifact and metrics logging",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("eval/benchmark_prompts.json"),
        help="Path to benchmark specification JSON",
    )

    args = parser.parse_args()

    settings = get_settings()
    try:
        runner = BenchmarkRunner(settings=settings, spec_path=args.spec)
        runner.run_benchmark(
            steps=args.steps,
            schedulers=args.schedulers,
            case_ids=args.cases,
            skip_clip=args.skip_clip,
            output_dir=args.output_dir,
            log_mlflow=not args.no_mlflow,
        )
    except Exception as err:
        print(f"\n[ERROR] Benchmark execution failed: {err}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
