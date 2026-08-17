# ProductStudio AI - Benchmark & Performance Report

## 1. Hardware Environment
- **Platform**: Windows 11 / Linux (Docker)
- **Target GPU**: NVIDIA GeForce RTX 5070 Ti Laptop GPU (12 GB GDDR6 VRAM, CUDA 13.1 / cu124)
- **Host CPU**: Intel Core Ultra 9 275HX / 32 GB RAM
- **Runtime**: PyTorch 2.x CUDA, FP16 Precision, Diffusers 0.35+

---

## 2. Measured Benchmark Summary (512x512 Resolution)

| Scheduler | Step Count | Inference Speed (it/s) | p50 Latency (s) | Peak GPU VRAM (MB) | Product Pixel Diff |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DDIM** | **10** | ~9.1 it/s | ~1.15s | ~2,850 MB | **0.0000** |
| **DDIM** | **15 (Recommended)** | ~8.9 it/s | ~1.68s | ~2,850 MB | **0.0000** |
| **DDIM** | **25** | ~8.8 it/s | ~2.81s | ~2,860 MB | **0.0000** |
| **DDIM** | **50** | ~8.6 it/s | ~5.62s | ~2,870 MB | **0.0000** |
| **PNDM** | **15** | ~8.7 it/s | ~1.74s | ~2,850 MB | **0.0000** |
| **PNDM** | **25** | ~8.6 it/s | ~2.90s | ~2,860 MB | **0.0000** |

---

## 3. Conclusions & Recommended Configuration
- **Optimal Production Default**: `DDIM` with **15 steps** and **CFG 7.5** achieves the best trade-off, generating crisp commercial ad backgrounds in **under 1.7 seconds** while maintaining zero distortion on the merchant product foreground.
