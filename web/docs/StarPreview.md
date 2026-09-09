# ⭐ Star Preview — Help

Live **animated sampling preview** for the ⭐StarNodes video all-in-one
nodes and the ⭐ StarSampler (Unified). Connect it once and watch the video
(or image) take shape step by step while the node is still sampling — no
extra model inputs, no VAE hookup needed.

```
[⭐ Star Preview] ──star_preview──> preview input of the All-In-One node
                                   or the ⭐ StarSampler (Unified)
                                        │
                                        └─ while sampling runs, an animated
                                           preview appears on the Star
                                           Preview node (updated every step)
```

Works like the KJNodes "Model Preview Override", but fully self-contained in
the StarNodes pack (KJNodes is **not** required) and reduced to the video
preview only — no charts, no sigma graphs.

## Compatible Nodes

Connect `star_preview` to the **`preview`** input of:

- ⭐ **Star LTXV All-in-One (2-Pass)** (`LTXVSulphurAllInOne`)
- ⭐ **Star LTXV 2.5 All-in-One (BETA)** (`LTXV25SulphurAllInOne`)
- ⭐ **Star Minimax All In One** (`StarMinimaxAllInOne`)
- ⭐ **StarSampler (Unified)** (`StarSampler`) — image models (SD/SDXL/Flux/
  ZIT) and video latents alike; for images the preview is a single JPEG that
  sharpens step by step instead of an animated WebP

## Widgets

- **preview_vae** — optional tiny preview VAE from `models/vae_approx`
  (e.g. a `taehv` / `taeltx` / `taeh3` decoder). With a decoder selected,
  the live preview is a real tiny-VAE decode with truer colors than
  Latent2RGB. Video vs image decoders are detected automatically from the
  file itself, so renamed files and subfolders work fine. Default `none`
  keeps the fast Latent2RGB preview — no file needed.

> Note: this dropdown is **only** for the live preview. The video/audio
> VAEs used for the final output are still chosen on the all-in-one node
> itself; the audio VAE is never needed for the preview.
>
> For image models on the ⭐ StarSampler (Unified), pick the matching image
> TAE (e.g. `taesd` for SD 1.5, `taef1` for Flux) — or leave it at `none`.

## Fixed Settings

Everything else is hardcoded on purpose:

| Setting | Value |
|---|---|
| Preview size | 512 px (longest side, small latents are upscaled) |
| Image quality | 80 (JPEG / WebP) |
| Playback | 8 fps |
| Frames per step | up to 16, sampled evenly across the video |

## How It Works

1. The Star Preview node passes a small options bundle to the all-in-one
   node.
2. The all-in-one node clones its internal diffusion model and attaches a
   lightweight sampler wrapper (the same mechanism KJNodes uses).
3. After every sampling step the current video latent is decoded — either
   with the selected tiny preview VAE or, by default, with the model's
   built-in latent-to-RGB factors (Latent2RGB) — encoded as an animated
   WebP and streamed to this node over the websocket.
4. Encoding runs on a background thread that drops frames when busy, so
   **sampling is never slowed down or blocked** by the preview.

While the preview is active, ComfyUI's built-in latent preview is silenced
for that sampling run (it would decode the same latents a second time).

## Notes

- Both passes of the LTXV pipelines (half-res pass + refine pass) and the
  Minimax refine pass all send previews — the animation restarts with each
  pass.
- A/V latents are handled automatically: only the video member is previewed,
  and LTXV first/last-frame keyframe padding is trimmed from the preview.
- In `audio only` / `audio_only` modes the tiny reference video latent is
  previewed.
- The default preview is a fast Latent2RGB approximation of the latent —
  colors and sharpness of the final output will differ. Select a tiny
  preview VAE in the dropdown for a closer match.
- The node itself has no outputs to wire downstream; it is display-only.
