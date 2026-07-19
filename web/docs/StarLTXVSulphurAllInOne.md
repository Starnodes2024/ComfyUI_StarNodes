# ⭐ Star LTXV All-in-One (2-Pass) — Help

A single ComfyUI node that replaces the two subgraphs (and their group-bypass
plumbing) of the **"LTXV Sulphur All in 1"** workflow.

It performs the exact same pipeline:

1. **Pass 1** at half resolution (`euler_ancestral_cfg_pp`, 8/12/16-step sigma preset)
2. Split A/V latents → **2x latent upscale** (spatial upscaler model)
3. Optional guide-image re-injection → **pass 2** at full resolution (`euler_cfg_pp`, fixed 5-step sigmas)
4. Decode video (VAE) **and** audio (audio VAE) — the model generates audio even in T2V/I2V mode

A **fancy animated DOM progress bar** is shown on the node itself during both
passes, with the standard ComfyUI progress bar as fallback.

---

## What you still need on the canvas (same as before)

- `DualCLIPLoader` (gemma-3 + text projection, type `ltxv`) → two `CLIPTextEncode` → `positive` / `negative`
- `VAELoader` (LTX23 **video** VAE) → `vae`
- `VAELoader` (LTX23 **audio** VAE) → `audio_vae`
- `LatentUpscaleModelLoader` (ltx-2.3 spatial upscaler x2) → `upscale_model`
- Optional: `LoadImage` → `image`, `LoadAudio` → `audio`
- Optional: your Ollama prompt helper, save/compressor nodes, RTX upscale — all downstream as before
- Note on SageAttention: the base model now loads *inside* this node, so an external
  `PatchSageAttentionKJ` can't be wired in front of it. Your workflow shipped with that node
  bypassed anyway — if you ever need it, that's a reason to switch to the loader+engine split.

---

## Node reference

**Mode**
- `text_to_video` — no inputs needed besides conditioning
- `image_to_video` — needs `image`
- `image_audio_to_video` — needs `image` **and** `audio` (audio is trimmed to the video length and preserved, not re-generated)

**Model**
- `base_model` — dropdown of `models/diffusion_models`
- `lora_1..3` + strengths — dropdowns of `models/loras`, `None` to skip
- The base model + LoRA stack is **cached**: changing prompts, seed, size, sigmas, mode etc. does **not**
  reload the model. Changing the model/LoRA selection reloads once. One configuration is kept in memory.

**Sigmas**
- `sigma_preset`: `12 steps` (workflow default), `8 steps` (fast), `16 steps` (quality) — the three
  schedules from the workflow's note node — or `custom` (uses `custom_sigmas_pass1`)
- `sigmas_pass2`: the second-pass schedule, default `0.85, 0.725, 0.6, 0.4219, 0.0`

**Video**
- `width` × `height` (snapped to /32), `frame_rate`, `duration_seconds` → frames are snapped to 8n+1
  (4.0 s @ 25 fps → 97 frames, like the original)
- `seed` is shared by both passes (as in the workflow)

**Defaults carried over from the workflow**
- `cfg` 1.0 (distilled), guide strengths 0.7 (pass 1) / 1.0 (pass 2), guide resize 1536, img compression 18
- Note: the original workflow ran its distilled LoRA at strength **0.6** — set `lora_1_strength` accordingly.

**Outputs** — `images` (IMAGE), `audio` (AUDIO), `frame_rate` (FLOAT) — wire straight into your
`StarVideoCompressor` / save nodes / RTX upscale chain as before.

---

## Small intentional differences from the workflow

- Pass 2 uses the same seed as pass 1 (the Image+Audio subgraph used a fixed internal seed 0 — almost
  certainly an oversight).
- The encoded-audio noise mask is built to match the audio latent shape (the SolidMask in the subgraph
  had leftover 1024×1024 dims); effect is identical: input audio is preserved through both passes.
- Decode uses plain `VAEDecode` (the active path in the workflow; the tiled decode was bypassed).
