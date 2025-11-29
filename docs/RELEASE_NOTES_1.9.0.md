# 🌟 StarNodes v1.9.0 Release Notes

**Release Date:** November 29, 2025

---

## 🎉 What's New in 1.9.0

### Image Filters & Effects
- **Star HighPass Filter** – High-pass based sharpening filter to enhance fine details and edge contrast.
- **Star Black And White** – Flexible black-and-white conversion with tonal control for cinematic monochrome looks.
- **Star Radial Blur** – Radial blur effect for focus/zoom style motion and creative depth effects.
- **Star Simple Filters** – Expanded image adjustment suite (sharpen, blur, saturation, contrast, brightness, temperature, color matching).

### Workflow & Ratio Utilities
- **Star PSD Saver Adv. Layers** – Advanced PSD exporter with improved layer handling for complex Photoshop workflows.
- **Star Advanced Ratio/Latent** – Combined advanced aspect ratio and latent megapixel helper for precise, resolution-safe sizing.

### LoRA Utilities
- **Star Dynamic LoRA** – Dynamic LoRA loader that lets you configure multiple LoRAs with flexible weights and options in a single node.
- **Star Dynamic LoRA (Model Only)** – Model-only variant of Star Dynamic LoRA that keeps CLIP conditioning unchanged.

### Sampling Utilities
- **Star FlowMatch Option** – Adds FlowMatch-related options to compatible samplers for more control over sampling behavior.

---

## 🔧 Internal & UX Improvements
- Reorganized the `comfyui_starnodes` codebase into logical subfolders (`samplers`, `image_tools`, `qwen`, `infiniteyou`, `text_io`, `external`, `grid`, `misc`).
- Moved documentation markdown files into `docs/` (keeping `README.md` at the root).
- Adjusted JSON resource paths (ratio files) to match the new folder layout.
- Removed unused custom type registration that previously emitted warnings.

These changes are internal and should be fully backward compatible with existing workflows.

---

## 📖 Documentation Updates
- Updated `README.md` with a **New in 1.9.0** section highlighting the new and featured nodes.
- Updated version references in:
  - `__init__.py`
  - `pyproject.toml`
  - `docs/EXAMPLE_WORKFLOWS.md`
  - `web/docs/StarSDUpscaleRefiner.md`

---

## 🔄 Migration from 1.8.x

There are **no breaking changes** expected when upgrading from 1.8.x to 1.9.0.

Recommended actions:
1. Update the extension to 1.9.0.
2. Refresh ComfyUI (and clear browser cache if UI behaves oddly).
3. Try the new filter and ratio/latent helper nodes in your existing workflows.

---

## 📞 Support

- **Issues:** Use GitHub issues to report bugs or request features.
- **Docs:** See `web/docs/` and `docs/` for detailed node documentation.
- **Workflows:** See `docs/EXAMPLE_WORKFLOWS.md` for example pipelines.
