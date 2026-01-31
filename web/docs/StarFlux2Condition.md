# ⭐ Star Flux2 Condition

A conditioning node for FLUX-style models that:

- Encodes your text prompt with CLIP
- Attaches a **Flux guidance** value to the conditioning
- Optionally encodes up to **5 reference images** into VAE latents and injects them as `reference_latents` (ComfyUI `ReferenceLatent` compatible)

Category: `⭐StarNodes/Conditioning`

## Inputs

Required:
- **clip (CLIP)**: The CLIP model used for tokenization/encoding.
- **vae (VAE)**: VAE used to encode reference images into latent space.
- **text (STRING)**: Prompt text.
- **guidance (FLOAT)**: Flux guidance value to attach into the conditioning dict.

Optional:
- **image_1 (IMAGE)**: Optional reference image 1.
- **image_2 (IMAGE)**: Optional reference image 2.
- **image_3 (IMAGE)**: Optional reference image 3.
- **image_4 (IMAGE)**: Optional reference image 4.
- **image_5 (IMAGE)**: Optional reference image 5.

## Outputs
- **CONDITIONING**: Conditioning compatible with Flux workflows.

## Behavior & Notes
- If one or more images are connected, each image is resized to a maximum of ~1 megapixel (if needed) and then encoded with the provided VAE.
- The resulting latent samples are appended to the conditioning via `reference_latents`, matching ComfyUI's `ReferenceLatent` node behavior.
- If multiple images are provided, multiple reference latents are appended (some models support using multiple reference images).

## Changelog
- v1.0: Initial release.
