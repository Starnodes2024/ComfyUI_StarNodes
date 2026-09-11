# ⭐ Star Minimax Multiref Option — Help

Timed keyframe references ("guides") for the **⭐ Star Minimax All In One**
node. This is the multi-frame / keyframe path of the stock MiniMax H3
template — the chain of *Add Guide for MiniMax H3* nodes with their
seconds → frames *Math Expression* nodes — collapsed into a single,
easy-to-use option node.

```
[LoadImage] ╮                [Star Minimax Multiref Option]
[LoadImage] ┼─ guide_image ─>   start @ guide_image_N (s)
[ Video   ] ╯                          │ multiref_settings
                                       v
                        [⭐ Star Minimax All In One]
```

## How it works

1. **Connect reference files** — drag image (or short video) sources onto the
   `guide_image_0…8` slots. Only the first slot is mandatory — the other
   eight are optional; simply leave the unused ones unconnected.
2. **Set the start time** — every slot has its own
   **`start @ guide_image_N (s)`** widget right on the node. Enter the time
   in **seconds** on the output timeline where the reference should be
   pinned (the widget is only used while its slot is connected). That is it
   — no PrimitiveFloat → Math Expression → Add Guide chains anymore.

Inside the AIO node each guide is applied in-process through the **exact code
path of the core `MiniMaxH3AddGuide` node**:

- The start time is converted with `frame_idx = round(seconds × 24)` — the
  same math as the template's `round (a * 24)` Math Expression nodes
  (24 fps). `1.5 s` anchors at frame 36.
- A **single image** anchors a still keyframe at that frame.
- A **batch of 5+ frames** (a loaded video) anchors as a clip starting at
  that frame, cropped down to the model's valid clip lengths
  (5, 22, 39 … `17k + 5` frames). Batches shorter than 5 frames use only the
  first image.
- **Negative start times** count back from the end of the video, exactly like
  negative frame indices on the core node (e.g. `−0.5 s` on a 5 s clip pins
  the guide near the last frames).
- Guides that land outside the timeline fail with a clear error message,
  just like the core node.

## Connectors

| Connector | Type | Notes |
|---|---|---|
| `guide_image_0…8` | IMAGE | up to 9 reference images/clips; only `guide_image_0` is required |
| `start @ guide_image_N (s)` | FLOAT widget | one per slot — start time in seconds on the output timeline (−150 … 150, step 0.1); used only while its slot is connected |
| **multiref_settings** out | MULTIREF_SETTINGS | connect to the `multiref_settings` input of ⭐ Star Minimax All In One |

## Usage

1. Add the node (**⭐StarNodes/Video → ⭐ Star Minimax Multiref Option**) and
   connect its `multiref_settings` output to the ⭐ Star Minimax All In One.
2. Drop a reference onto `guide_image_0` and set where on the output
   timeline the guide starts with the `start @ guide_image_0 (s)` widget.
3. Connect more references to `guide_image_1`, `guide_image_2`, … — each
   slot uses its own `start @ guide_image_N (s)` widget.
4. Write the AIO prompt so it describes the shot changes at those timestamps
   (e.g. *"... at 2.0 seconds cut to the scene of the second shot ..."*) —
   the guides anchor the frames, the prompt tells the model what to do there.

## Notes

- Guides are **timeline anchors**, not semantic references: they do **not**
  get `<Picture i>` tags and do not shift the ordering of the AIO's
  `ref_image_…` / `ref_video_…` slots. A guide pins actual output frames; a
  reference describes content the model should weave in anywhere. Both can be
  used together.
- Works in the AIO's `video` mode and `image` mode (9-frame timeline; a start
  time beyond ~0.33 s lands past the last frame and raises the core node's
  clear out-of-range error).
- When the ⭐ Star Minimax Latent Upscaler Option is also connected, the
  guide latents are resolution-matched and ride into the refine pass
  automatically, so the anchors survive the second sampling pass.
- Requires a ComfyUI version whose core includes the `MiniMaxH3AddGuide`
  node (the MiniMax H3 multiframe template ships it).
