# ⭐ Star Qwen Edit Encoder

A faster, feature-equivalent copy of ComfyUI's `TextEncodeQwenImageEdit`. Produces conditioning from a text prompt for Qwen image edit, optionally attaching a reference latent derived from an image+VAE.

Category: `⭐StarNodes/Conditioning`

## Inputs
- **clip (CLIP)**: CLIP model used to tokenize and encode the prompt (+images).
- **prompt (STRING)**: Text prompt. Supports multiline and dynamic prompts.

Optional:
- **vae (VAE)**: If provided with `image`, the node encodes a reference latent for Qwen edit guidance.
- **image (IMAGE)**: Optional image used for reference latent and Qwen image-aware tokenization.
- **reference_latent (LATENT)**: If provided, the node skips VAE encoding and attaches this latent directly.
- **resize_mode (lanczos | bicubic)**: Filter used by the internal resize. Default: `lanczos`.
- **skip_upscale_if_match (BOOLEAN)**: If true, skip resampling when the input already matches the chosen target resolution. Default: `True`.
- **ar_skip_epsilon (FLOAT)**: Additional AR-based skip. If the image's aspect ratio is within this epsilon of the target AR, resampling is skipped. Default: `0.002`.
  Example: target AR 1.50, image AR 1.498 (diff 0.002) → skip; image AR 1.495 (diff 0.005) → resample.
- **cache_tokens (BOOLEAN)**: Cache prompt-only encodings (no image/reference_latent). Keyed by `(clip-id, prompt, cache_bust)`. Default: `False`.
- **cache_bust (STRING)**: A small version tag for the cache when `cache_tokens` is enabled. Change it (e.g., to "v2") to force a fresh compute even if `prompt` and `clip` are unchanged. Default: empty.

## Outputs
- **CONDITIONING**: Conditioning for Qwen, with `reference_latents` attached if available.

## Behavior & Notes
- When `image` and `vae` are both provided, the node chooses a preferred Qwen resolution closest to the image's aspect ratio and resamples the image accordingly (unless skipping is triggered).
- Skipping criteria:
  - Exact size match to the target resolution; or
  - Aspect ratio within `ar_skip_epsilon` of the target AR (and `skip_upscale_if_match` is true).
- `torch.inference_mode()` is used to eliminate autograd overhead.
- For repeat runs with the same prompt and no image/latent, enable `cache_tokens` to avoid redundant work.

## Tips
- Keep `resize_mode = lanczos` for maximum fidelity; switch to `bicubic` if you confirm speed benefits with negligible visual impact.
- If you already computed an upstream image latent, pass it via `reference_latent` and leave `image` unset.

## Changelog
- v1.1: Added AR epsilon skip, resize mode, prompt-only cache, direct latent input, and inference-mode execution.
- v1.0: Initial port from `TextEncodeQwenImageEdit`.
