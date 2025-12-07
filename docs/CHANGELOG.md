# Changelog - ComfyUI StarNodes

## Version 1.9.2 (2025-12-07)

### ✨ New Nodes

- **Star Stop And Go** – Interactive workflow control node that lets you pause, review, and decide whether to continue or stop your ComfyUI workflow execution. Works with any data type and supports user-select, pause, and bypass modes.
- **Star Model Packer** – Combines split safetensors model files into a single file and converts them to the selected floating-point precision (FP8, FP16, or FP32).
- **Star FP8 Converter** – Converts existing `.safetensors` checkpoints to FP8 (`float8_e4m3fn`) and saves the converted file in the standard ComfyUI output models folder.

### 🛠 Improvements

- Updated internal versioning to 1.9.2 across the project (pyproject, module version, docs, and example workflows).
- Documentation clean-up and consistency fixes.

---

## Version 1.9.1 (2025-12-01)

### ✨ New Nodes

- **Star Latent Resize** – Resizes existing latents to a target resolution using an advanced ratio/megapixel selector, with a custom mode for exact width/height while keeping model-friendly dimensions.

---

## Version 1.9.0 (2025-11-29)

### ✨ New & Featured Nodes

- **Star HighPass Filter** – High-pass based sharpening filter for enhancing fine image details and local contrast.
- **Star Black And White** – Flexible black-and-white conversion node with tonal controls for stylized monochrome output.
- **Star Radial Blur** – Radial blur effect node for focus/zoom style motion and creative depth effects.
- **Star Simple Filters** – Expanded image adjustment node providing sharpen, blur, saturation, contrast, brightness, temperature, and color matching.
- **Star PSD Saver Adv. Layers** – Advanced PSD export node for saving complex layer stacks for Photoshop post-processing.
- **Star Advanced Ratio/Latent** – Advanced aspect-ratio and latent megapixel helper for precise, resolution-safe image sizing.
- **Star Dynamic LoRA** – Dynamic LoRA loader that allows configuring multiple LoRAs with flexible weights and options in a single node.
- **Star Dynamic LoRA (Model Only)** – Variant of Star Dynamic LoRA that only applies LoRAs on the model branch, keeping CLIP untouched.
- **Star FlowMatch Option** – Additional FlowMatch-related options for compatible samplers.

---

## Version 1.7.0 (2024-11-20)

### 🎉 Major Release - Integration of StarBetaNodes

This release integrates all tested and stable nodes from the StarBetaNodes repository into the main StarNodes package.

### ✨ New Nodes Added

#### Qwen/WAN Image Editing Suite (8 nodes)
- **Star Qwen Image Ratio** - Aspect ratio selector for Qwen models with SD3-optimized dimensions
- **Star Qwen / WAN Ratio** - Unified ratio selector for Qwen and WAN video models with auto aspect ratio matching
- **Star Qwen Image Edit Inputs** - Multi-image stitcher for Qwen editing (up to 4 images)
- **Star Qwen Edit Encoder** - Advanced CLIP encoder optimized for Qwen image editing
- **Star Image Edit for Qwen/Kontext** - Dynamic prompt builder with customizable templates
- **Star Qwen Edit Plus Conditioner** - Enhanced conditioning for Qwen models
- **Star Qwen Rebalance Prompter** - Intelligent prompt rebalancing
- **Star Qwen Regional Prompter** - Region-based prompting system

#### Image Processing & Effects (2 nodes)
- **Star Apply Overlay (Depth)** - Blend filtered images using depth/mask with Gaussian blur
- **Star Simple Filters** - Comprehensive image adjustments with color matching (sharpen, blur, saturation, contrast, brightness, temperature)

#### AI Generation & Prompting (3 nodes)
- **Star Nano Banana (Gemini)** - Google Gemini 2.5 Flash image generation with 30+ templates
- **Star Ollama Sysprompter (JC)** - Structured prompt builder for Ollama with art styles
- **Star Sampler** - Advanced sampler with extensive configuration options

#### Utilities & Tools (2 nodes)
- **Star Save Folder String** - Flexible path builder with date-based organization
- **Star Duplicate Model Finder** - SHA256-based duplicate model scanner

### 📦 New Dependencies
- `google-generativeai>=0.8.3` - For Gemini image generation
- `color-matcher` - For advanced color matching in filters

### 📚 New Documentation
- `QwenEditPromptGuide.md` - Comprehensive guide for Qwen editing nodes
- `README_StarQwenRegionalPrompter.md` - Regional prompter documentation
- `SIMPLIFIED_REGIONAL_PROMPTER_V2.md` - Simplified regional prompter guide
- `editprompts.json` - Customizable prompt templates
- `styles.json` - Art style definitions for Ollama
- 15+ new markdown docs in `web/docs/` for all new nodes

### 🎨 New Web Assets
- Otter sprite images for UI enhancements
- JavaScript UI components for Qwen/Kontext nodes
- StarryLinks.js for enhanced node linking

### 🔧 Technical Improvements
- Added web server routes for serving editprompts.json and otter sprites
- Standardized all node categories with ⭐ emoji prefix
- Updated __init__.py with all new node registrations
- Enhanced README with comprehensive node listings

### 📂 New Configuration Files
- `googleapi.ini` - Google Gemini API configuration
- `star_save_folder_presets.json` - Folder preset configurations

### 🏷️ Category Organization
All nodes are now organized under these categories:
- ⭐StarNodes/Starters
- ⭐StarNodes/Sampler
- ⭐StarNodes/Qwen & Image Editing
- ⭐StarNodes/Image And Latent
- ⭐StarNodes/Text And Data
- ⭐StarNodes/IO
- ⭐StarNodes/InfiniteYou
- ⭐StarNodes/Conditioning
- ⭐StarNodes/Settings
- ⭐StarNodes/Helpers And Tools
- ⭐StarNodes/Color
- ⭐StarNodes/Prompts
- ⭐StarNodes/Image Generation

---

## Version 1.6.0

### New Nodes
- Star Random Image Loader - Load random images from folders with seed control
- Star Image Loader 1by1 - Sequential image loading with state persistence
- Star Save Panorama JPEG - Export JPEGs with XMP panorama metadata
- Star Frame From Video - Extract specific frames from video batches
- Star Icon Exporter - Multi-size PNG/ICO export with effects

---

## Version 1.5.0 and Earlier

See git history for previous version changes.
