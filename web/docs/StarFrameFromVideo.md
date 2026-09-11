# ⭐ Star Frame From Video

Selects a single frame from a batch of images (e.g., frames produced by a video loader).

- __Category__: `⭐StarNodes/Video`
- __Class__: `StarFrameFromVideo`
- __File__: `star_frame_from_video.py`

## Inputs
- __images__ (IMAGE, required): Batch of frames `[N,H,W,C]`.
- __frame_select_mode__ (CHOICE, required, default: "First Frame"): `First Frame`, `Last Frame`, `Frame Number`.
- __frame_number__ (INT, optional, default: 0): Index used when `Frame Number` mode is selected.

## Outputs
- __frame__ (IMAGE): Single-frame batch `[1,H,W,C]`.

## Behavior
- Works with torch tensors, numpy arrays, or Python lists.
- Safe-guards index and empty batch.

## Usage Tips
- For the first/last frame, no need to set `frame_number`.
- Combine with downstream nodes expecting a single image.

## Version
- Introduced in StarNodes 1.6+
