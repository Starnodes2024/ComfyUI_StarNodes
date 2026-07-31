# ⭐ Star Realistic Film Grain

## Overview
The **Star Realistic Film Grain** node generates highly realistic, analog film grain based on authentic film stock profiles. It uses advanced techniques including luminance masking, organic variance (clumping), and soft-light blending to recreate the natural characteristics of silver halide film grain.

## Key Features
- **8 Authentic Film Profiles**: Includes classic stocks like Kodak Tri-X 400, Ilford HP5 Plus, Kodak Portra 400, and more
- **Luminance Masking**: Grain appears primarily in midtones, just like real film (shadows and highlights show less grain)
- **Organic Variance**: Perlin-like clumping creates natural grain clustering instead of uniform noise
- **Auto-Scaling**: Automatically adjusts grain size based on image resolution
- **Color & B/W Grain**: Supports both monochrome and chromatic grain patterns
- **Soft-Light Blending**: Preserves image contrast and color fidelity

## Inputs

### Required
- **image** (IMAGE): Input image to apply film grain to
- **demo_mode** (BOOLEAN): Enable demo grid showing all film profiles (default: False)
- **film_profile** (DROPDOWN): Select from authentic film stock profiles or "Custom" (ignored in demo mode)
- **blend_strength** (FLOAT): Overall grain intensity (0.0-1.0, default: 0.5)
- **auto_scale_grain** (BOOLEAN): Automatically scale grain size based on resolution (default: True)
- **manual_grain_size** (FLOAT): Manual grain size when auto-scale is off (0.1-8.0, default: 1.0)
- **organic_variance** (FLOAT): Amount of natural clumping/clustering (0.0-1.0, default: 0.5)
- **chroma_grain** (BOOLEAN): Enable color grain for Custom profile (default: False)

## Film Profiles

### Black & White Films
- **Kodak Tri-X 400**: Classic, medium grain with good organic variance (size: 1.2)
- **Ilford HP5 Plus 400**: Balanced, moderate grain structure (size: 1.0)
- **Ilford Delta 3200**: Very coarse, high-speed grain with strong clumping (size: 2.6)
- **Kodak T-Max 100**: Very fine, clean grain for sharp images (size: 0.4)

### Color Films
- **Kodak Portra 400**: Popular portrait film with soft, subtle color grain (size: 0.8)
- **Fujifilm Superia 400**: Prominent, structured color grain (size: 1.4)
- **Cinestill 800T**: Motion picture grain with cinematic characteristics (size: 1.6)

### Custom Profile
- **Custom**: Full manual control over grain size, variance, and chroma settings

## Technical Details

### Luminance Masking
The node uses ITU-R BT.709 luminance calculation and applies a bell curve mask:
- **Formula**: `grain_mask = 4.0 * luminance * (1.0 - luminance)`
- **Effect**: Maximum grain at 50% gray, minimal grain in pure blacks and whites
- This mimics how real film grain behaves in different exposure zones

### Organic Variance
Creates natural grain clumping through low-frequency modulation:
- Generates a coarse "cloud" pattern at 1/8th the noise resolution
- Modulates the primary grain density to create realistic clustering
- Higher variance values produce more pronounced clumping

### Auto-Scaling
When enabled, grain size scales relative to a 1024px reference:
- **Formula**: `actual_size = grain_size * (max(height, width) / 1024.0)`
- Ensures consistent grain appearance across different resolutions

### Soft-Light Blending
Uses the Photoshop-style Soft Light blend mode:
- Preserves image contrast and color saturation
- Creates natural integration between grain and image
- Avoids the harsh appearance of simple additive blending

## Demo Mode

When **demo_mode** is enabled, the node creates a comparison grid showing all film profiles:
- Each profile is applied to a 500x500px version of your input image for better visibility
- Grain strength is automatically boosted to minimum 0.8 (or higher if you set it) to make differences more visible
- Profiles are arranged in an automatic grid layout (currently 3x3 for 7 profiles)
- Each tile includes a scaled black label bar at the bottom with the profile name in large text
- Perfect for testing and comparing all available film stocks at once
- The grid automatically adapts if more profiles are added in the future

**Note**: In demo mode, the `film_profile` selection is ignored, and all profiles (except "Custom") are shown.

## Usage Tips

1. **For Vintage Look**: Use Kodak Tri-X 400 or Ilford HP5 Plus with blend_strength 0.5-0.7
2. **For Cinematic Feel**: Try Cinestill 800T with blend_strength 0.6-0.8
3. **For Subtle Enhancement**: Use Kodak T-Max 100 with blend_strength 0.2-0.4
4. **For Heavy Grain**: Ilford Delta 3200 with blend_strength 0.7-1.0
5. **Custom Grain**: Use "Custom" profile and adjust manual_grain_size and organic_variance to taste
6. **Test All Profiles**: Enable demo_mode to see all film stocks side-by-side on your image

## Output
- **image** (IMAGE): Image with realistic film grain applied

## Category
⭐StarNodes/Image & Latent Selection
