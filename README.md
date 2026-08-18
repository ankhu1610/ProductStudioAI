# ProductStudio AI

ProductStudio AI is a planned mask-aware latent-diffusion service for e-commerce ad creative generation. A merchant uploads a product image, a product mask, and a background prompt; the service generates a new environment while restoring the original product pixels in the final output.


## Business problem

E-commerce merchants need varied, professional product visuals without repeated photoshoots or manual compositing. Generic text-to-image generation can distort product logos, materials, and geometry. ProductStudio AI targets this failure mode with mask-guided background generation followed by final pixel-space compositing of the source foreground.


## Reproducible benchmarks

The fixed benchmark cases live in [`eval/benchmark_prompts.json`](eval/benchmark_prompts.json).

```powershell
python -m scripts.benchmark
```

See [`eval/README.md`](eval/README.md) for output files, metrics, and optional flags.

## Repository layout

```text
app/        ProductStudio application package (being built)
data/       Setup instructions; local model/data directories are ignored
eval/       Versioned benchmark specification and future evaluation runner
scripts/    Environment and benchmark utilities
sd/         Original from-scratch Stable Diffusion learning implementation
tests/      Automated tests (added in Phase 4)
```

## Original implementation credits

The from-scratch learning implementation is based on the Stable Diffusion architecture and references work from CompVis, Hugging Face Diffusers, and related educational implementations. See the original module comments and diagram PDF for the architecture walkthrough.

