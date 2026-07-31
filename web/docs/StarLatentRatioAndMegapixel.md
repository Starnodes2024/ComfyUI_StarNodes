# Star Advanves Ratio/Latent

## Description
The **Star Advanved Ratio/Latent** node creates a latent tensor based on a chosen aspect ratio and target resolution. It is useful for quickly setting up latent resolutions that match a desired ratio and overall pixel count, with dimensions always divisible by 16.

## Inputs

### Required
- **ratio**: Aspect ratio preset for the output. Options:
  - **custom**: Uses a neutral 1:1 ratio (or image ratio when *Ratio From Image* is enabled)
  - **1:1**
  - **1:2**
  - **3:4**
  - **2:3**
  - **5:7**
  - **9:16**
  - **9:21**
  - **10:16**
  - **4:3**
  - **16:10**
  - **3:2**
  - **2:1**
  - **7:5**
  - **16:9** *(default)*
  - **21:9**
- **custom_ratio**: Text input for custom aspect ratios when **ratio** is set to `custom`.
  - Format examples: `21:9`, `4:3`, `8:6`
  - Both `:` and `x` separators are accepted (e.g. `21x9`)
  - If parsing fails or values are invalid, the node falls back to `1:1`
- **resolution**: Resolution scale selector, **relative to a diffusion base resolution** plus a few model-friendly presets. Options include:
  - **custom**: Directly uses **custom_width** and **custom_height** as the target resolution (other ratio and resolution settings are ignored).
  - **SD**: A base resolution similar to **512×512** (classic SD-style). Other ratios are computed around this pixel area.
  - **SDXL**: Special label corresponding to scale `1.0`, i.e. an area similar to **1024×1024** (SDXL-style base).
  - Numeric scales labeled as megapixels, e.g. `2 Megapixel`, `3 Megapixel`, `4 Megapixel`, ..., `20 Megapixel`, `50 Megapixel`, `100 Megapixel` *(default: `4 Megapixel`)*
    - Internally, `2 Megapixel` corresponds to ≈ 2× the SDXL base area, `4 Megapixel` ≈ 4×, and higher values scale proportionally.
  - **Qwen Image**: Uses an area based on **1328×1328** (Qwen 1:1 style). Other ratios are computed around this pixel area.
  - **WAN HD**: Uses an area based on **1280×720** (WAN HD-style). Other ratios are computed around this pixel area.
  - **WAN FullHD**: Uses an area based on **1920×1080** (WAN FullHD-style). Other ratios are computed around this pixel area.
- **custom_width**: Target width in pixels when **resolution** is `custom`. Must be divisible by 16 (the node will snap to the nearest multiple of 16).
- **custom_height**: Target height in pixels when **resolution** is `custom`. Must be divisible by 16 (the node will snap to the nearest multiple of 16).
- **latent_channels**: Channel count of the latent tensor. Options:
  - `4 (SD/SDXL)` - For standard SD 1.5/SDXL models (uses 8x downsampling)
  - `16 (FLUX/QWEN/ZIT)` *(default)* - For Flux 1.x, SD3.5, Qwen, and ZIT models (uses 8x downsampling)
  - `128 (FLUX2)` - For Flux 2 models (uses 16x downsampling)
- **ratio_from_image**: Boolean toggle *(Ratio From Image)*. When enabled and an image is connected, the node will approximate the nearest preset ratio from the input image.
 - **batch_size**: Number of latent samples to create in one go. Works like the core latent nodes:
   - `1` *(default)*
   - higher values create a batch of independent zero-initialized latents with the same spatial size.

### Optional
- **image**: Optional image input used when **ratio_from_image** is enabled. Its aspect ratio is used to select the nearest preset ratio.

## Outputs
- **LATENT**: A latent tensor of shape `[batch_size, latent_channels, H_latent, W_latent]` derived from the computed image-space resolution.
- **WIDTH**: Final image-space width (int), always divisible by 16.
- **HEIGHT**: Final image-space height (int), always divisible by 16.

## Behavior
1. The node determines an aspect ratio:
   - If **ratio_from_image** is enabled and an image is provided, the node reads the image aspect ratio and selects the closest preset ratio (1:1, 1:2, 3:4, 2:3, 5:7, 9:16, 9:21, 10:16, 4:3, 16:10, 3:2, 2:1, 7:5, 16:9, 21:9).
   - Otherwise, it uses the chosen **ratio**. For **custom**, a neutral 1:1 ratio is used.
2. The node computes a target pixel area from **megapixels**, scaled from a **1024×1024** base area (not strict camera megapixels).
3. Using the target area and chosen aspect ratio, it computes ideal floating-point width and height.
4. Both width and height are snapped to the nearest multiple of **16**, with a minimum of 16.
5. The latent tensor resolution is derived from these dimensions:
   - For **4 or 16 channels**: Uses 8× downsampling (width/8, height/8)
   - For **128 channels** (Flux 2): Uses 16× downsampling (width/16, height/16)

## Notes
- All output dimensions are forced to be **divisible by 16** to stay friendly with most diffusion model pipelines.
- The **megapixels** setting controls the overall scale of the resolution independent of the aspect ratio.
- Use **ratio_from_image** to quickly match your latent resolution to the shape of an input reference image while still controlling megapixels and latent channels.
