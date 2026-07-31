# Star Flux2/Qwen-Image-Edit Inpainter

## Description
The Star Flux2/Qwen-Image-Edit Inpainter is an all-in-one inpainting node built for **Flux2** models (Flux2 Dev and Flux2 Klein) and **Qwen-Image-Edit** models. It replicates a complete Flux2 inpainting pipeline in a single node: smart crop-and-stitch around the masked area, Flux2 text conditioning, reference image latents, InpaintModelConditioning, Differential Diffusion and sampling. Crop and stitch settings are hardcoded to sensible defaults so you only deal with the things that matter: prompt, sampler settings and optional reference images.

## Inputs

### Required
- **model**: The diffusion model (Flux2 Dev, Flux2 Klein or Qwen-Image-Edit). Apply LoRAs before connecting the model.
- **clip**: The matching text encoder (e.g. Mistral 3 Small loaded with type `flux2`, or the Qwen2.5-VL encoder for Qwen-Image-Edit)
- **vae**: The matching VAE (Flux2 VAE or Qwen-Image VAE)
- **image**: The source image to be inpainted
- **mask**: The mask indicating areas to be inpainted (white areas will be inpainted)
- **text**: Prompt describing what should be inpainted in the masked area
- **seed**: Random seed for reproducible results
- **steps**: Number of sampling steps (default: 20)
- **cfg**: Classifier-free guidance scale (default: 5.0 — both model families use real CFG with a negative prompt)
- **sampler_name**: Sampling algorithm (default: euler)
- **scheduler**: Noise schedule (default: simple)
- **denoise**: Denoising strength (1.0 for full inpainting, lower for subtle changes)
- **use_inpaint_area_as_reference**: Feed the cropped inpaint area to the model as a reference image. Helps the model keep style and content consistent with the surrounding image (default: Yes)

### Optional
- **reference_image_1 - reference_image_4**: Up to 4 reference images. Each is resized to ~1MP and encoded as a reference latent that guides the generation (e.g. a character, object or style you want to appear in the masked area).

## Outputs
- **image**: The inpainted result image
- **latent**: Latent representation of the inpainted image
- **mask**: The original mask (passed through)
- **seed**: The seed used for generation

## Usage
1. Connect a Flux2 model (Dev or Klein) or a Qwen-Image-Edit model with its matching CLIP and VAE
2. Provide an image and a mask (white = inpaint)
3. Enter a text prompt describing what should appear in the masked area
4. Optionally connect reference images of things you want in the result
5. Run the node — done

## Features

### Built for Flux2 and Qwen-Image-Edit
- Real CFG with an automatically encoded empty negative prompt
- Reference image latents injected into the conditioning (`reference_latents`), the way both model families expect
- The cropped inpaint context is used as an automatic reference so the result blends with the surrounding image
- Video-style VAE outputs (Qwen) are handled automatically

### Smart Crop and Stitch (hardcoded)
- Automatically detects and crops only the masked region with padding
- Mask holes are filled, mask is grown and blurred for smooth transitions
- The inpainted region is seamlessly blended back into the original image
- No settings to tweak — it just works

### Differential Diffusion
- Always applied for soft, artifact-free mask boundaries

## Notes
- Works with Flux2 Dev, Flux2 Klein and Qwen-Image-Edit
- LoRAs: apply them to the model before connecting it to this node
- For subtle edits lower the denoise value; for full replacement keep it at 1.0
- Reference images are great for inserting a specific character or object into the masked area
