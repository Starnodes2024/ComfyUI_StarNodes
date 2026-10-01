# Star Qwen2 Outpainter

## Description
The Star Qwen2 Outpainter is an all-in-one outpainting node built for **Qwen-Image-Edit** models (Qwen-Image-Edit, Edit 2509 and Qwen-Image 2.1). It places your input image centered on a new canvas in the chosen aspect ratio, fills the empty surround with solid red and lets the edit model expand the image into the red areas. The generated canvas is the only reference image — no mask, no crop-and-stitch, just prompt and sampler settings.

## Inputs

### Required
- **model**: The Qwen-Image-Edit diffusion model. Apply LoRAs before connecting the model.
- **clip**: The matching text encoder (Qwen2.5-VL for Qwen-Image-Edit, Qwen3-VL for Qwen-Image 2.1)
- **vae**: The matching VAE (Qwen-Image VAE or the RGBA Qwen-Image 2.1 VAE)
- **image**: The source image to be outpainted — picked right inside the node from the input folder, the output folder (entries ending with `[output]`), or via upload / clipboard paste. The in-node preview shows the real image at its true aspect ratio
- **aspect_ratio**: Target canvas aspect ratio (same list as the Aspect Ratio Advanced node, default: 16:9 landscape)
- **megapixel**: Target canvas size in megapixels (default: 2.0)
- **image_placement**: Where the input image sits on the canvas — any of the 9 positions (`top left` … `bottom right`) or `custom`. With `custom` you can drag the image box inside the in-node preview to define which area gets outpainted, and drag the corner grip to freely resize the source image on the canvas (default: center)
- **prompt**: What the model should do with the red areas (default: `expand the red areas with background and scene that fits the source <image 1>. add a fluffy purple otter with a golden star and the word "STARNODES" to the scene.`)
- **seed**: Random seed for reproducible results
- **steps**: Number of sampling steps (default: 30)
- **cfg**: Classifier-free guidance scale (default: 1.0 — an empty negative prompt is encoded automatically)
- **sampler_name**: Sampling algorithm (default: euler)
- **scheduler**: Noise schedule (default: simple)
- **denoise**: Denoising strength — keep at 1.0 for outpainting (default: 1.0)
- **qwen_image_2_1**: Enable for Qwen-Image 2.1 models — 64-channel /16x latents, image_slots conditioning and RGBA VAE output (default: No)

### Optional
- **reference_image**: Optional second reference image (IMAGE connector) — scaled to ~1 MP and added to the vision tokens and reference latents next to the red canvas, e.g. a style or content reference for the expansion
- **preview**: Optional ⭐ Star Preview connector — shows a live sampling preview on the connected ⭐ Star Preview node

## Outputs
- **input_image**: The loaded source image (EXIF-corrected, RGB) — like the output of a Load Image node, handy for previewing or reusing elsewhere
- **image**: The outpainted result in the new canvas size
- **reference**: The red canvas with your image placed — exactly what the model saw, useful for checking the setup

## Usage
1. Connect a Qwen-Image-Edit model with its matching CLIP and VAE
2. Pick the image you want to expand in the node's image dropdown (or upload / paste one)
3. Pick the target aspect ratio and megapixel size — the in-node preview shows the real image on the canvas in its exact position and size
4. Keep or adapt the prompt (the model looks for the red areas to expand)
5. Run the node — done

## How it works
- The input image is first capped at the selected megapixel size (downscaled only if larger)
- The canvas is built at the chosen ratio and megapixel size and grown just enough so the input image fits centered — the remaining surround is filled with pure red
- The canvas is fed to the text encoder as a vision reference and VAE-encoded into `reference_latents`, the way Qwen-Image-Edit models expect
- The model samples an empty latent at the full canvas size and fills the red areas
- Video-style and RGBA VAE outputs (Qwen Image 2.1) are handled automatically

## Notes
- If the input image already matches the chosen ratio, there is no red surround — the result is just a re-rendered image
- A small red margin produces subtle extension; a big red area gives the model more freedom
- LoRAs: apply them to the model before connecting it to this node
