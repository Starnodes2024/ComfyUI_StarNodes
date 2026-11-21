# StarNodes 3.6.4 — Discord Release Post

**⭐ StarNodes 3.6.4 is out!** ⭐

**New nodes**
- **⭐ Star Model Loader (+ Advanced)** — checkpoint/model + text encoders + VAEs + sampler settings in one node, with presets. Fixed sockets that never drop your wires again — filled = selected, hollow = unused
- **⭐ Star Universal Conditioner** — text + up to 10 reference images, with megapixel resize for vision refs (Flux2, Qwen-Edit, MOSS…)
- **⭐ Star LoRA Converter** — shrink LoRA rank with fast SVD + save as FP16/FP8, right into models/loras
- **⭐ Star Regional Spatial LoRA Canvas + Regional Sampler** — different prompts & LoRAs in different image regions
- **⭐ Star Face Detailer** — WeVisOne-VL segmentation, per-face prompts & LoRAs, seam-free paste-back
- **⭐ Star Minimax Remake Option** — Viggle-style video remake without the Viggle pack
- **⭐ Star Model Merge / AIO Merge** and **⭐ Star GPU Cooldown**

**Improved**
- ⭐ Star Video Loader: crop dialog with live preview, resize presets, seconds output
- ⭐ Star Minimax AIO: remake mode, 17n+5 frame fix, reference MP caps
- ⭐ Star Preview: Flux2 Tiny VAE previews
- ⭐ Star Custom Sigmas: "Viggle Reference 6 Steps" preset
- CUDA stability fixes, Qwen 2.1 latents & conditioning fixes, 2 new dataset wildcard packs

**Update:** ComfyUI Manager → Update, or `git pull` in
`ComfyUI/custom_nodes/ComfyUI_StarNodes`
Fresh install: `git clone https://github.com/Starnodes2024/ComfyUI_StarNodes`
Then **restart ComfyUI + Ctrl+F5**

https://github.com/Starnodes2024/ComfyUI_StarNodes
