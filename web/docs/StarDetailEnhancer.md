# Star Adaptive Detail Enhancer

The **Star Adaptive Detail Enhancer** node adaptively sharpens, enhances, and denoises images based on edge, face, and texture analysis. It is especially useful for improving the perceived detail of generated images, portraits, and textures, while minimizing noise and artifacts.

## Node Class
`AdaptiveDetailEnhancement`

## Category
⭐StarNodes/Image And Latent

## Inputs
| Name                | Type    | Description                                                                 |
|---------------------|---------|-----------------------------------------------------------------------------|
| `image`             | IMAGE   | The input image (Tensor or numpy array).                                    |
| `enhancement_strength` | FLOAT | Overall strength of detail enhancement (0.0–3.0, default: 1.0).             |
| `edge_threshold`    | FLOAT   | Edge detection threshold (0.1–1.0, default: 0.3).                           |
| `face_boost`        | FLOAT   | Additional sharpening for detected faces (0.5–3.0, default: 1.5).           |
| `texture_boost`     | FLOAT   | Additional sharpening for texture regions (0.5–2.5, default: 1.2).          |
| `noise_suppression` | FLOAT   | Amount of denoising (0.0–1.0, default: 0.5).                                |

## Output
| Name    | Type  | Description            |
|---------|-------|------------------------|
| image   | IMAGE | The enhanced image.    |

## How it works
- **Edge Detection:** Finds strong edges using Sobel and Canny filters.
- **Face Detection:** Detects face regions for selective enhancement.
- **Texture Detection:** Identifies high-variance texture areas.
- **Selective Sharpening:** Applies unsharp masking with adaptive masks for edges, faces, and textures.
- **Denoising:** Optionally denoises the image using a bilateral filter.

## Example Usage
- Use for upscaling or final enhancement of portraits, landscapes, and generated art.
- Adjust `enhancement_strength` for subtle or strong effects.
- Increase `face_boost` for portraits. Increase `texture_boost` for landscapes or highly detailed images.
- Use higher `noise_suppression` for noisy images.

---

**Author:** StarNodes Team

For more information, see the [README](../../README.md).
