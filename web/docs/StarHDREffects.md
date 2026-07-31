# ⭐ Star HDR Effects

## Description
Applies an HDR-style local contrast and tone-mapping effect similar to "fake HDR" filters. It boosts local detail and micro-contrast while compressing highlights and optionally adjusting saturation.

## Inputs
- **image** (IMAGE, required): Source image.
- **preset** (CHOICE, required): Named HDR look. Selecting a preset updates the HDR sliders to a good starting point; you can then tweak them freely. Default: `Natural HDR`.
- **strength** (FLOAT, required): Overall blend between original and HDR-processed result. 0.0 = original, 1.0 = full effect, values > 1.0 can exaggerate the look. Default: 0.7.
- **radius** (INT, required): Radius of the local blur used to separate base and detail layers. Larger values create more pronounced, wide-radius HDR halos. Default: 8.
- **local_contrast** (FLOAT, required): Multiplier for the local detail component. Higher values increase micro-contrast and "grungy" HDR detail. Default: 1.0.
- **global_contrast** (FLOAT, required): Multiplier for the base tone curve. Values > 1.0 increase global contrast, values < 1.0 flatten it. Default: 1.0.
- **saturation** (FLOAT, required): Color saturation applied after tone mapping. 1.0 = unchanged, < 1.0 desaturates, > 1.0 boosts color. Default: 1.0.
- **highlight_protection** (FLOAT, required): Protects bright areas from clipping by rolling off highlights. 0.0 = no protection, 1.0 = strong highlight compression. Default: 0.5.

## Outputs
- **image** (IMAGE): HDR-processed image.

## Notes
- For a natural HDR look, try:
  - **strength** ≈ 0.4–0.8
  - **radius** ≈ 6–12
  - **local_contrast** ≈ 0.8–1.2
  - **global_contrast** ≈ 0.9–1.1
  - **saturation** ≈ 0.9–1.1
- For a more stylized, "crunchy" HDR effect, increase **local_contrast** and **strength**, and optionally radius.
- Highlight protection helps avoid blown-out skies and bright areas when pushing contrast.
- The effect is implemented with a log-space base/detail split and runs fully on GPU with PyTorch.

### Presets
- **Natural HDR**: Balanced local contrast and highlight protection for a subtle HDR feel.
- **Strong HDR**: Higher local contrast and strength for a more dramatic, crunchy HDR effect.
- **Soft HDR Matte**: Softer contrast and slightly lowered saturation for a matte, cinematic look.
- **Detail Boost**: Smaller radius and strong local contrast to emphasize fine details.
- **Sky Protect Highlights**: Emphasis on highlight protection and moderate contrast for skies and bright scenes.
