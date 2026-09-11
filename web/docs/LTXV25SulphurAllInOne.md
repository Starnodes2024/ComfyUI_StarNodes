# ⭐ Star LTXV 2.5 All-in-One — Help

A single ComfyUI node that replaces the official **LTX-2.5** workflow templates
(text-to-video, image-to-video and first/last-frame-to-video), plus a dedicated
**audio-only** mode.

LTXV 2.5 uses a **single** text encoder (type `ltxv`, e.g. the gemma-4 12B with
projection checkpoint) — no DualCLIPLoader needed. Models, LoRAs, CLIP and VAEs
are loaded inside the node and **cached**: they are only re-loaded when the
selection actually changes.

A fancy animated DOM progress bar is shown on the node itself during every
sampling pass, with the standard ComfyUI progress bar as fallback.

---

## Modes

| Mode | Needs | Pipeline |
|---|---|---|
| `text_to_video` | prompt only | two passes: half res → 2x latent upscale → full res |
| `image_to_video` | `first_frame` | two passes, guide image injected at 0.7 (pass 1) / 1.0 (pass 2) |
| `image_audio_to_video` | `first_frame` + `audio` | two passes, your audio is encoded and preserved (not re-generated) |
| `first_last_frame_to_video` | `first_frame` + `last_frame` | **single** full-res pass; both frames are added as keyframe guides (strength 0.7, like the official flf2v template) and cropped back out after sampling |
| `audio_only` | prompt only | one plain **30-step** pass at 64×64 (normal scheduler, no custom sigmas). Only the `audio` output matters; `images` still returns the tiny 64×64 video as a visual reference |

---

## Audio output — what you get and why

**Generated audio is always decoded from the first sampling pass** (the one with
the most steps), never from the upscale refine pass — the refine pass adds
nothing to sound quality.

Priority order for the `audio` output:

1. `image_audio_to_video` mode → the connected `audio` input is always passed through unchanged.
2. otherwise → the model-generated audio from the first pass.

(The `audio` input is ignored in all modes except `image_audio_to_video`.)

---

## Steps / sigmas

- `sigma_preset`:
  - `12 steps` (default), `8 steps` (faster), `16 steps` (finer) — the baked
    schedules from the original workflow's note node.
  - `20 / 30 / 40 / 50 steps` — plain sampler steps with the **normal**
    scheduler, no custom sigmas.
  - `custom` — uses the `custom_sigmas_pass1` text field.
- `sigmas_pass2` — the second-pass (refine) schedule for the two-pass modes,
  default `0.85, 0.7250, 0.4219, 0.0`.
- The `audio_only` pass always runs **30 steps, normal scheduler**,
  independent of `sigma_preset`.

---

## Models

- `base_model` — LTXV 2.5 A/V checkpoint from `models/diffusion_models`.
  NVFP4 / INT8 convrot checkpoints (e.g.
  `ltx-2.5-22b-distilled-transformer-nvfp4.safetensors`) are supported — keep
  `weight_dtype` on `default` for those so the native quantization is used.
- `model_override` (optional MODEL input) — replaces the `base_model` dropdown
  (e.g. a model pre-patched with flash/sage attention). The LoRA stack is still
  applied on top of it.
- `clip_1` — single LTXV 2.5 text encoder from `models/text_encoders` (type ltxv).
- `vae` / `audio_vae` — LTXV 2.5 video and audio VAE from `models/vae`.
- `upscale_model` — LTXV 2.5 latent spatial upscaler x2 from
  `models/latent_upscale_models` (only used by the two-pass modes).
- `lora_1..3` + strengths — optional LoRA stack, applied in order.

Model links (HuggingFace, access required): <https://huggingface.co/Lightricks/LTX-2.5>

---

## Video size & length

- `video_size`: `HD` (~1280 px), `FHD` (~1920 px) — same ratio tables as the
  Star LTX Video Settings node — or `Custom` (`custom_width` × `custom_height`).
- `ratio_from_image` picks the closest preset ratio to the connected
  `first_frame`.
- Width/height are snapped to *(multiple of 32) + 1*; frames are snapped to
  *8n + 1* (4 s @ 25 fps → 97 frames).
- `seed` is shared by all passes.

## Outputs

- `images` (IMAGE) — the decoded video frames.
- `audio` (AUDIO) — see the priority list above.
- `frame_rate` (FLOAT) — pass straight into your save/compressor nodes.
- `latent` (LATENT) — the combined sampled A/V latent, before decoding.
- `model` (MODEL) — the model used for sampling (with the LoRA stack applied),
  ready for downstream reuse.
- `clip` (CLIP) — the loaded text encoder.
- `vae` / `audio_vae` (VAE) — the loaded video and audio VAEs.

## Sound processing (`sound_settings` input)

Connect a **⭐ Star Video Sound Enricher Option** node here and the audio
output is cleaned up and enriched internally (de-harsh, bass/warmth boost,
high-fizz taming, at least 44.1 kHz — never downsampled) — identical
processing to the standalone ⭐ Star Video Sound Enricher node, just without
extra audio wiring. Applies to generated audio and passed-through audio alike.
Not connected → audio output stays untouched.

---

## Small intentional differences from the templates

- All passes share one seed (the official t2v/i2v templates refine pass 2 with
  a fixed internal seed 42 — almost certainly an oversight).
- Audio is decoded from the first pass instead of the refine pass (better sound).
- `first_last_frame_to_video` decodes with tile overlap 64 like the flf2v
  template; the two-pass modes use overlap 32.
