# ProductStudio AI - System Architecture & Technical Design

## 1. Overview

**ProductStudio AI** is a production MLOps platform engineered to generate high-converting, prompt-controlled e-commerce ad backgrounds while **guaranteeing 100% pixel fidelity** for the merchant's product foreground.

```text
[ Merchant / User ]
        │ (RGB Image + Binary Mask + Creative Prompt)
        ▼
[ Gradio Web UI / FastAPI REST Endpoints ]
        │
        ▼
[ Preprocessing & Input Validation ] ───▶ (Aspect Crop, 512x512, Mask Invert)
        │
        ▼
[ Latent Diffusion Inpainting Engine ] ──▶ (SD 1.5 Inpainting + Multi-Scheduler)
        │                                  (DDIM / PNDM / DPM-Solver / Euler-A)
        ▼
[ Pixel-Space Foreground Compositor ] ──▶ Final = Mask x Original + (1-Mask) x Generated
        │                                  Fidelity Verification: Pixel Diff == 0.0
        ▼
[ Storage & MLOps Tracking Layer ]
   ├── Local Filesystem:  outputs/<request_id>/ (Images + JSON Metadata)
   ├── SQLite Database:   productstudio.db (Job records & indices)
   └── MLflow Tracking:   mlruns/ (Hyperparameters, Metrics, Artifacts)
```

---

## 2. Core Subsystems

### A. Preprocessing & Mask Conventions
- **ProductStudio Protected Mask**: White ($255$) identifies the product region to preserve; Black ($0$) identifies background.
- **Diffusers Inpainting Convention**: White ($255$) identifies the region to regenerate.
- `prepare_inputs` automatically inverts the mask for the diffusion UNet, resizes with aspect ratio preservation (LANCZOS for image, NEAREST for mask), and enforces 8-pixel dimension divisibility.

### B. Inpainting & Multi-Scheduler Sampling
- Base Checkpoint: `stable-diffusion-v1-5/stable-diffusion-inpainting`.
- Supported Schedulers:
  - **DDIM** (`DDIMScheduler`): Fast deterministic sampling (10-25 steps).
  - **PNDM** (`PNDMScheduler`): Stable baseline multi-step scheduler.
  - **DPM-Solver** (`DPMSolverMultistepScheduler`): Ultra-fast high-order ODE solver.
  - **Euler-A** (`EulerAncestralDiscreteScheduler`): Fast ancestral sampler.

### C. Deterministic Foreground Compositing
- Formula: $\text{Final} = \text{Mask} \times \text{Original} + (1 - \text{Mask}) \times \text{Generated}$.
- Metric: $\text{Foreground Pixel Difference} = \max | \text{Original}_{\text{product}} - \text{Final}_{\text{product}} | \equiv 0.0$.

### D. Persistence & Observability
- **SQLite**: Local structured index for job statuses, prompt history, and execution metrics.
- **MLflow**: Centralized experiment logging for benchmark comparisons and production request auditing.
- **JSON Metadata**: Self-contained reproducibility file (`metadata_variant_*.json`) stored alongside each image asset.
