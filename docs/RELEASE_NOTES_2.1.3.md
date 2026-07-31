# StarNodes 2.1.3 Release Notes

## 🎉 New Node: ⭐ Star Flux2/Qwen-Image-Edit Inpainter

All-in-one inpainting node built for **Flux2** models (Flux2 Dev and Flux2 Klein) and **Qwen-Image-Edit** models.

### What it does
Replicates a complete Flux2 inpainting pipeline in a single node:
- Smart crop-and-stitch around the masked area (settings hardcoded to sensible defaults)
- Flux2 text conditioning with real CFG (empty negative prompt encoded automatically)
- Up to 4 optional reference images, resized to ~1MP and injected as `reference_latents`
- Optional automatic use of the inpaint area itself as a reference for consistent style/content
- InpaintModelConditioning with noise mask
- Differential Diffusion for soft mask boundaries
- Sampling with full sampler settings (seed, steps, cfg, sampler, scheduler, denoise)

### Defaults tuned for Flux2
- steps: 20, cfg: 5.0, sampler: euler, scheduler: simple, denoise: 1.0

### Usage
Connect a Flux2 model (with Mistral 3 Small CLIP, type `flux2`, and Flux2 VAE) or a Qwen-Image-Edit model (with Qwen2.5-VL encoder and Qwen-Image VAE). Apply LoRAs to the model before connecting it. Provide image, mask and prompt — done.

### Qwen-Image-Edit compatibility
Video-style VAE decode output (5D) is flattened automatically, so the same node works with both model families.

## Changed Files
- `misc/StarFlux2Inpainter.py` (new)
- `web/docs/StarFlux2Inpainter.md` (new)
- `__init__.py` (node registration)
- `README.md` (version + node list)
- `pyproject.toml` (version 2.1.3)
