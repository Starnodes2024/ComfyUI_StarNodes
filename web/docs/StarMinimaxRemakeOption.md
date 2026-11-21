# ⭐ Star Minimax Remake Option

Viggle-Remake mode for the **⭐ Star Minimax All In One** node — without the
Viggle node pack. Connect it to the AIO's `remake_settings` input and the
run switches to the Viggle-Remake pipeline: the **swap** or **edit** LoRA is
patched onto the MiniMax H3 diffusion model, the source clip conditions the
generation as `<Video 1>`, your painted reference frames as `<Picture 1>` …
`<Picture 4>`, and the source soundtrack is kept in the output.

Everything runs locally inside ComfyUI — no external API, no extra nodes.

## What it does inside the AIO

- **LoRA** — patched onto the model bypass-style (patches, not merged — the
  same quality path as *LoraLoaderBypassModelOnly*, which the Viggle docs
  recommend over merging into the bf16 weights).
- **Source clip** — cover-cropped onto the AIO's `aspect_ratio` × `megapixels`
  canvas, cut to the `duration` length, encoded and conditioned as
  `<Video 1>` with the reference script's 2 fps text labels (text-only
  presentation — the references condition through the DiT latent path, not
  the text encoder's vision tower).
- **References** — each encoded at the canvas short edge (rounded to 32) and
  conditioned as `<Picture N>`, in slot order.
- **Pins** — each pinned reference additionally becomes a keyframe at that
  output frame for the first part of the denoise, then drops out below the
  `keyframe_until` sigma (the Viggle *RemakeKeyframeUntil* behavior).
- **Soundtrack** — the source audio is resampled to 32 kHz stereo, encoded
  and held clean for the whole denoise (noise mask 0), so the original
  soundtrack passes through to the AUDIO output unchanged. Silence when not
  connected.
- **Noise** — the starting noise is drawn in the Viggle reference script's
  order (one draw per reference/keyframe latent first), so a seed reproduces
  the same video as `inference/sample.py` / the Viggle nodes.

## Inputs

| Input | Type | Notes |
|---|---|---|
| `lora` | dropdown | the Viggle-Remake LoRA file from `models/loras` — `swap` (replace people/characters) or `edit` (change objects/setting/weather/light), or a `lora_lowrank` variant |
| `lora_strength` | FLOAT | LoRA strength, 1.0 = full |
| `source_video` | IMAGE | the source clip's frames at **24 fps** — supplies motion, camera and framing. Use the `images` output of ⭐ Star Video Loader with `force_rate` 24 |
| `ref_image_0…3` | IMAGE | 1–4 painted reference frames / character references, in order — `<Picture 1>` … `<Picture 4>` |
| `ref_image_size` | combo | cap on the encoded reference size — `none` (Viggle sizing: canvas short edge), `0.5` / `1.0` / `1.5 MP` bound the reference area (aspect preserved, default `1.0 MP`) — saves VRAM and ref tokens on large canvases |
| `pins` | STRING | optional: one output frame per reference, e.g. `12, 190` — give one pin per reference or none |
| `keyframe_until` | FLOAT | sigma cutoff for pinned keyframes (Viggle default 0.9) |
| `source_audio` | AUDIO | optional — the soundtrack kept in the output, e.g. ⭐ Star Video Loader's `audio` output |

## Output

| Output | Type | Notes |
|---|---|---|
| `remake_settings` | STAR_MINIMAX_REMAKE | connects to the `remake_settings` input of ⭐ Star Minimax All In One |

## Setup

1. Download the LoRA(s) from `Viggle/Viggle-Remake` (`lora/swap` and/or
   `lora/edit`, or the smaller `lora_lowrank` rank-128/64 variants) and place
   the `.safetensors` file(s) in `models/loras` like any other LoRA.
2. Load the source clip with ⭐ Star Video Loader (`force_rate` 24) — wire
   `images` → `source_video` and `audio` → `source_audio`.
3. Paint your reference frame(s) — typically a frame of the source edited in
   an image editor (the person/object change you want) — and connect up to 4.
4. Write the prompt in MiniMax H3's six-section format and refer to the
   references as `<Video 1>` / `<Picture 1>` … `<Picture 4>`, exactly like in
   the Viggle workflows.
5. Set the AIO's `duration` — a cap on the output length; the actual output
   is the smaller of `duration` and the source clip, trimmed to the exact
   frame count (wire the loader's `seconds` output into `duration` to always
   get the full clip). `aspect_ratio` × `megapixels` set the output canvas
   (the source is cover-cropped to it).

## Notes

- Video mode only — the AIO raises an error if `remake_settings` is connected
  in `image` mode.
- The AIO's own `ref_image` / `ref_video` / `ref_audio` inputs are ignored
  while a remake bundle is connected (a warning is printed). RefMod blocks
  and multiref guides still apply on top, and a latent-upscale refine pass
  keeps the held soundtrack clean too.
- Pins are output frames in `[0, frames)` — frame 12 means the pinned
  reference is treated as the painted version of output frame 12.
- Recommended output lengths follow the Viggle training range (~90–345
  frames ≈ 3.75–14.4 s at 24 fps), but any `17n+5` length works.
- **Exact-length output**: internally the render is padded up to the `17n+5`
  grid by repeating the source's last frames, then the decoded video *and*
  the held soundtrack are trimmed back to the requested frame count — parts
  cut from a longer clip join seamlessly, with no dropped frames and no gap
  in the audio. Enable `minimax_snap` on ⭐ Star Video Loader to cut on the
  grid directly (no padding needed at all).
