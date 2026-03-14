# ComfyUI_StarNodes

Little Helper Nodes For ComfyUI

**Current Version:** 1.9.8

<img width="917" alt="image" src="https://github.com/user-attachments/assets/4bc1378e-d1cf-4063-9196-b056a58444ec" />

A collection of utility nodes designed to simplify and enhance your ComfyUI workflows.

## New in 1.9.8

### Text & Data
- ⭐ **Star Text Filter** — Added two new filter options:
  - `keep_from_start_to_end`: Keeps only the text from start word to end word (inclusive)
  - `remove_from_start_to_end`: Removes text from start word to end word (inclusive), keeps the rest

### IO & Image Loading
- ⭐ **Star Load Image+** — Added clipboard paste functionality:
  - Right-click context menu option "📋 Paste Clipboard Image"
  - Widget button "⭐ 📋 Paste Image" for quick clipboard pasting
  - Paste images directly into the node without manual file uploads

## New in 1.9.7

### Text & Data
- ⭐ **Star Prompt Picker** (`StarPromptPicker`) — Pick prompts from a text file (one prompt per line) or from a folder of single-prompt `.txt` files. Supports Random and One By One modes with saved progress.

### Sampling Utilities
- ⭐ **Distilled Optimizer (QWEN/ZIT)** (`StarDistilledOptimizerZIT`) — Two-pass distilled refinement options for ⭐ StarSampler (Unified). Works with Z-Image-Turbo and with Qwen-Image when using a Turbo/LightX LoRA.

## New in 1.9.5

### IO & Metadata
- ⭐ **Star Save Image+** (`StarSaveImagePlus`) — Save images with built-in folder/filename settings and store 5 extra metadata strings (`StarMetaData 1-5`) into the PNG.
- ⭐ **Star Load Image+** (`StarLoadImagePlus`) — Load images and read out the 5 extra metadata strings (`StarMetaData 1-5`) as separate outputs.

## New in 1.9.4

### Video & Animation
- ⭐ **Star Image Loop** (`StarImageLoop`) — Creates seamless looping video frames from images like panoramic images. Supports multiple dynamic image inputs that are joined horizontally to create longer slidess. Perfect for social media content from AI-generated or photographed panoramas.
- ⭐ **Star Video Loop** (`StarVideoLoop`) — Creates seamless looping video frames from video inputs. Videos are scrolled horizontally to create a slidingeffect with moving content. Supports multiple dynamic video inputs.

## New in 1.9.3

### Metadata & Workflow Sharing
- ⭐ **Star Meta Injector** (`StarMetaInjector`) — Transfers all PNG metadata (including ComfyUI workflow data) from a source image to a target image and saves it directly. Perfect for sharing workflows with custom preview images.

## New in 1.9.2

### Workflow Control & Preview
- ⭐ **Star Stop And Go** (`StarStopAndGo`) — Interactive workflow control node that lets you pause, preview, and decide whether to continue or stop your workflow. Works with any data type and supports user-select, timed pause, and bypass modes.

### Model Tools & Conversion
- ⭐ **Star Model Packer** (`StarModelPacker`) — Combines split safetensors model files into a single file and converts them to a chosen floating-point precision (FP8, FP16, or FP32).
- ⭐ **Star FP8 Converter** (`StarFP8Converter`) — Converts existing `.safetensors` checkpoints to FP8 (`float8_e4m3fn`) and writes them into the standard ComfyUI output models folder.

## New in 1.9.1

### Image & Latent Utilities
- ⭐ **Star Latent Resize** (`StarLatentResize`) — Resizes existing latents to a target resolution using an advanced ratio/megapixel selector, with a custom mode for exact width/height while keeping model-friendly dimensions.

## New in 1.9.0

### Image Filters & Effects
- ⭐ **Star HighPass Filter** (`StarHighPassFilter`) — High-pass based sharpening filter to enhance fine details and edge contrast.
- ⭐ **Star Black And White** (`StarBlackAndWhite`) — Flexible black-and-white conversion with tonal control for cinematic monochrome looks.
- ⭐ **Star Radial Blur** (`StarRadialBlur`) — Radial blur effect for focus/zoom style motion and creative depth effects.
- ⭐ **Star Simple Filters** (`StarSimpleFilters`) — Comprehensive image adjustment suite (sharpen, blur, saturation, contrast, brightness, temperature, color matching).

### Workflow & Ratio Utilities
- ⭐ **Star PSD Saver Adv. Layers** (`StarPSDSaverAdvLayers`) — Advanced PSD exporter with enhanced layer handling for complex Photoshop workflows.
- ⭐ **Star Advanced Ratio/Latent** (`StarAdvancedRatioLatent`) — Combined advanced aspect ratio and latent megapixel helper for precise, resolution-safe size selection.

### LoRA Utilities
- ⭐ **Star Dynamic LoRA** (`StarDynamicLoRA`) — Dynamic LoRA loader that lets you configure multiple LoRAs with flexible weights and options in a single node.
- ⭐ **Star Dynamic LoRA (Model Only)** (`StarDynamicLoRAModelOnly`) — Variant of Star Dynamic LoRA that only applies LoRAs to the model (no CLIP changes), ideal for more controlled style mixing.

### Sampling Utilities
- ⭐ **Star FlowMatch Option** (`StarFlowMatchOption`) — Additional FlowMatch-related sampling options for compatible samplers.

## New in 1.8.0

### Upscaling & Refinement
- ⭐ **Star SD Upscale Refiner** (`StarSDUpscaleRefiner`) — All-in-one SD1.5 upscaling and refinement node combining checkpoint loading, LoRA support, upscale models, tiled diffusion, ControlNet tile, and advanced optimizations (FreeU, PAG, Automatic CFG) into a single workflow.

### LoRA Utilities
- ⭐ **Star Random Lora Loader** (`StarRandomLoraLoader`) — Randomly selects a LoRA from your library (with subfolder and name filters) and can optionally apply it directly to MODEL/CLIP or output the LoRA path as a string.

## Previous Updates

### 1.7.0 – Qwen/WAN Image Editing & Utilities

#### Qwen/WAN Image Editing Suite
- ⭐ Star Qwen Image Ratio (`StarQwenImageRatio`) — Aspect ratio selector for Qwen models with SD3-optimized dimensions (1:1, 16:9, 9:16, 4:3, 3:4, etc.)
- ⭐ Star Qwen / WAN Ratio (`StarQwenWanRatio`) — Unified ratio selector for Qwen and WAN video models with auto aspect ratio matching
- ⭐ Star Qwen Image Edit Inputs (`StarQwenImageEditInputs`) — Multi-image stitcher for Qwen editing (up to 4 images)
- ⭐ Star Qwen Edit Encoder (`StarQwenEditEncoder`) — Advanced CLIP text encoder optimized for Qwen image editing with reference latents and caching
- ⭐ Star Image Edit for Qwen/Kontext (`StarImageEditQwenKontext`) — Dynamic prompt builder with customizable templates from editprompts.json
- ⭐ Star Qwen Edit Plus Conditioner (`StarQwenEditPlusConditioner`) — Enhanced conditioning specifically designed for Qwen models
- ⭐ Star Qwen Rebalance Prompter (`StarQwenRebalancePrompter`) — Intelligent prompt rebalancing for better results
- ⭐ Star Qwen Regional Prompter (`StarQwenRegionalPrompter`) — Region-based prompting system for precise control over different image areas

#### Image Processing & Effects
- ⭐ Star Apply Overlay (Depth) (`StarApplyOverlayDepth`) — Blend filtered images using depth/mask with Gaussian blur options
- ⭐ Star Simple Filters (`StarSimpleFilters`) — Comprehensive image adjustments with color matching (sharpen, blur, saturation, etc.)

#### AI Generation & Prompting
- ⭐ Star Nano Banana (Gemini) (`StarNanoBanana`) — Google Gemini 2.5 Flash image generation with 30+ templates
- ⭐ Star Ollama Sysprompter (JC) (`StarOllamaSysprompterJC`) — Structured prompt builder for Ollama with art styles
- ⭐ Star Sampler (`StarSampler`) — Advanced sampler with extensive configuration options

#### Utilities & Tools
- ⭐ Star Save Folder String (`StarSaveFolderString`) — Flexible path builder with date-based organization
- ⭐ Star Duplicate Model Finder (`StarDuplicateModelFinder`) — SHA256-based duplicate model scanner

### 1.6.0 – IO & Image Utilities

- ⭐ Star Random Image Loader — Load random images from folders with optional subfolders and seed control
- ⭐ Star Image Loader 1by1 — Sequentially loads images across runs with state saved in the folder
- ⭐ Star Save Panorama JPEG — Save JPEGs with embedded XMP panorama metadata for 360° viewers
- ⭐ Star Frame From Video — Extract specific frames from video batches
- ⭐ Star Icon Exporter — Export multi-size PNGs/ICO with effects

## Documentation

Detailed documentation for all nodes is available in the `web/docs` directory.

## Installation

### Via ComfyUI Manager (Recommended)
Search for "Starnodes" in ComfyUI Manager and install

### Manual Installation
1. Open CMD within your custom nodes folder
2. Run: `git clone https://github.com/Starnodes2024/ComfyUI_StarNodes`
3. Restart ComfyUI
