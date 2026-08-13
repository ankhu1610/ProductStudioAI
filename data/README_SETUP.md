# Model and Asset Setup

This repository does not include model weights, tokenizer assets, private product images, masks, or generated outputs. Do not commit any of them.

## ProductStudio application model

The ProductStudio application (Phase 1 onward) is designed for an SD 1.5 **inpainting** checkpoint. Before downloading or using a model, review and accept its license and terms on the provider's model page. Record the exact model ID or local path in `.env`.

Default configuration:

```text
PRODUCTSTUDIO_MODEL_ID=stable-diffusion-v1-5/stable-diffusion-inpainting
```

If using a local model copy, put it under `data/models/` and set:

```text
PRODUCTSTUDIO_MODEL_PATH=data/models/<model-directory>
```

`data/models/` is intentionally ignored by Git.

## Original from-scratch SD learning implementation

The existing `sd/` notebooks use the original Stable Diffusion v1.5 checkpoint format. To run them, create the following sibling directories:

```text
LLMOPS/
  data/
    vocab.json
    merges.txt
    v1-5-pruned-emaonly.ckpt
  images/
    dog.jpg                 # optional demo input
```

Download `vocab.json`, `merges.txt`, and `v1-5-pruned-emaonly.ckpt` from the Stable Diffusion v1.5 model repository after accepting its access terms. Keep these files local; the checkpoint is large and must never be committed.

Expected notebook references:

```text
sd/demo.ipynb -> ../data/vocab.json
sd/demo.ipynb -> ../data/merges.txt
sd/demo.ipynb -> ../data/v1-5-pruned-emaonly.ckpt
```

## Verify before running

1. Copy `.env.example` to `.env` and adjust the model path/ID if required.
2. Create a virtual environment and install the project dependencies.
3. Run `python -m scripts.verify_environment`.
4. Confirm the output shows CUDA available and approximately 12 GB of GPU VRAM.

For the RTX 5070 Ti Laptop GPU, start with 512x512 images, FP16 precision, batch size 1, and one GPU worker.

