# ProductStudio AI

ProductStudio AI is a planned mask-aware latent-diffusion service for e-commerce ad creative generation. A merchant uploads a product image, a product mask, and a background prompt; the service generates a new environment while restoring the original product pixels in the final output.

## Status

Phase 0 (foundation) is in progress. The current `sd/` directory remains a from-scratch Stable Diffusion v1.5 learning implementation. The production service will be built in `app/` in later phases.

## Business problem

E-commerce merchants need varied, professional product visuals without repeated photoshoots or manual compositing. Generic text-to-image generation can distort product logos, materials, and geometry. ProductStudio AI targets this failure mode with mask-guided background generation followed by final pixel-space compositing of the source foreground.

## Planned architecture

```text
Gradio UI / REST API
       -> input validation
       -> SD 1.5 inpainting + DDIM
       -> foreground pixel-space compositing
       -> image + reproducibility metadata
       -> local storage, SQLite, and MLflow tracking
```

## Quick start: Phase 0 environment

1. Install Python 3.10-3.12 and create a virtual environment.
2. Install a CUDA-enabled PyTorch build appropriate for your NVIDIA driver/GPU using the [official PyTorch selector](https://pytorch.org/get-started/locally/). For the RTX 5070 Ti, use a current CUDA build rather than the legacy `requirements.txt` pins.
3. Install the application dependencies:

   ```powershell
   pip install -e ".[dev,tracking]"
   ```

4. Copy `.env.example` to `.env`.
5. Verify the runtime:

   ```powershell
   python -m scripts.verify_environment
   ```

For the target RTX 5070 Ti Laptop GPU (12 GB VRAM), begin at 512x512, FP16, batch size 1, and one inference worker.

## Model and data setup

Read [`data/README_SETUP.md`](data/README_SETUP.md) before downloading any model assets. Model weights, datasets, private product images, masks, generated outputs, and credentials are intentionally excluded from Git.

## Reproducible benchmarks

The fixed benchmark cases live in [`eval/benchmark_prompts.json`](eval/benchmark_prompts.json). Phase 2 will add the evaluation runner; it will use these committed prompts, masks, seeds, and step counts to reproduce benchmark numbers.

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

