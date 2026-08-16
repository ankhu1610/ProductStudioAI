# Evaluation Assets

`benchmark_prompts.json` is the canonical, version-controlled Phase 2 benchmark specification. It fixes prompts, seeds, resolution, and image/mask paths so scheduler and code changes can be compared fairly.

The referenced demo inputs are license-safe synthetic assets under `data/demo_inputs/`; do not replace them with private merchant images.

## Phase 2 benchmark runner

Run the full suite on CUDA:

```powershell
python -m scripts.benchmark
```

This executes all 10 cases at step counts 10, 15, 25, and 50 for both DDIM and PNDM schedulers. Each run records:

- end-to-end latency
- peak GPU memory
- foreground pixel difference (must remain 0.0)
- CLIP text-image similarity on the final composite

Outputs are written under `outputs/benchmarks/<timestamp>/`:

- `benchmark_results.csv` — one row per case/scheduler/step run
- `benchmark_summary.json` — p50/p95 latency, mean CLIP, and a recommended scheduler/step pair
- `human_rating_sheet.csv` — template for manual quality, prompt alignment, and boundary ratings

Optional flags:

```powershell
python -m scripts.benchmark --steps 10 15 --schedulers ddim --skip-clip
```

Results are also logged to the local MLflow experiment `productstudio-benchmarks` when MLflow is installed.
