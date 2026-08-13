# Evaluation Assets

`benchmark_prompts.json` is the canonical, version-controlled Phase 2 benchmark specification. It fixes prompts, seeds, resolution, and image/mask paths so scheduler and code changes can be compared fairly.

The referenced demo inputs will be added in Phase 1. They must be license-safe assets committed under `data/demo_inputs/`; do not replace them with private merchant images. Phase 2 will add `eval_suite.py`, which executes these exact cases at 10, 15, 25, and 50 sampling steps and writes the benchmark results.

