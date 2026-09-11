# ⭐ Star QwenEdit+ Conditioner

An advanced conditioning node for Qwen image editing that supports multiple reference images with integrated vision-language processing.

Category: `⭐StarNodes/Conditioning`

## Inputs
- **clip (CLIP)**: CLIP model used to tokenize and encode the prompt with images.
- **prompt (STRING)**: Text prompt describing the desired image modification. Supports multiline and dynamic prompts.

Optional:
- **vae (VAE)**: If provided with images, encodes reference latents for Qwen edit guidance.
- **image1 (IMAGE)**: First optional reference image.
- **image2 (IMAGE)**: Second optional reference image.
- **image3 (IMAGE)**: Third optional reference image.

## Outputs
- **CONDITIONING**: Conditioning for Qwen, with `reference_latents` attached if VAE and images are provided.

## Behavior & Notes
- Supports up to 3 reference images, which are resized and processed for both vision tokenization and latent encoding.
- Uses a specialized Llama template for vision-language integration, prompting the model to describe image features and generate modifications.
- Images are upscaled to 384x384 for vision processing and to approximately 1024x1024 (multiples of 8) for latent encoding.
- Each provided image contributes a vision token in the format "Picture X: <|vision_start|><|image_pad|><|vision_end|>" to the prompt.
- Reference latents are attached to the conditioning if VAE is provided and images are present.

## Tips
- Use this node when you need to condition on multiple reference images for complex image editing tasks.
- Ensure images are provided in order (image1, image2, image3) as they are labeled sequentially in the prompt.
- The node's vision processing helps maintain consistency with original inputs while applying text-based modifications.

## Changelog
- v1.0: Initial implementation supporting multiple reference images with integrated vision processing.
