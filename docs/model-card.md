# ProductStudio AI - Model Card

## Model Overview
- **Model Name**: Stable Diffusion Inpainting v1.5
- **Base Checkpoint**: `runwayml/stable-diffusion-inpainting` / `stable-diffusion-v1-5/stable-diffusion-inpainting`
- **Architecture**: Latent Diffusion Model (UNet + CLIP ViT-L/14 Text Encoder + VAE) conditioned on 9-channel latent inputs (4 latent image channels + 1 mask channel + 4 masked latent channels).
- **Task**: Mask-guided latent inpainting and background synthesis for e-commerce advertising.

---

## Intended Use
- **Primary Use**: Generating contextual, photorealistic studio and lifestyle backgrounds around segmented e-commerce products (footwear, electronics, cosmetics, apparel, furniture).
- **Out of Scope**: Generating human faces, misleading or deceptive promotional content, or infringing on unauthorized copyrighted trademarks without consent.

---

## Technical Specifications
- **Input Resolution**: $512 \times 512$ (default, scalable to multiples of 8).
- **Precision**: Mixed Precision `FP16` (CUDA) / `FP32` (CPU).
- **Recommended Inference Settings**:
  - Scheduler: `DDIM` or `DPMSolverMultistep`
  - Step Count: $15 - 25$ steps
  - CFG Scale: $7.0 - 8.5$
  - Target Hardware: NVIDIA GPU with $\ge 8$ GB VRAM (RTX 3060/4060/5070+).

---

## Evaluation Metrics
1. **Product Fidelity**: Maximum RGB pixel difference inside protected mask area ($= 0.0$).
2. **Prompt Alignment**: CLIP cosine similarity between prompt text and generated background.
3. **Latency**: p50 & p95 latency measured under batch size 1.
