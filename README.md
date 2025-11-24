# ComfyUI_StarNodes

Little Helper Nodes For ComfyUI

**Current Version:** 1.8.1

<img width="917" alt="image" src="https://github.com/user-attachments/assets/4bc1378e-d1cf-4063-9196-b056a58444ec" />

A collection of utility nodes designed to simplify and enhance your ComfyUI workflows.

## New in 1.8.0

### Upscaling & Refinement
- ⭐ **Star SD Upscale Refiner** (`StarSDUpscaleRefiner`) — All-in-one SD1.5 upscaling and refinement node combining checkpoint loading, LoRA support, upscale models, tiled diffusion, ControlNet tile, and advanced optimizations (FreeU, PAG, Automatic CFG) into a single workflow.

### LoRA Utilities
- ⭐ **Star Random Lora Loader** (`StarRandomLoraLoader`) — Randomly selects a LoRA from your library (with subfolder and name filters) and can optionally apply it directly to MODEL/CLIP or output the LoRA path as a string.

## Previous Updates

### 1.7.0 – Qwen/WAN Image Editing & Utilities

#### Qwen/WAN Image Editing Suite
- ⭐ Star Qwen Image Ratio (`StarQwenImageRatio`) — Aspect ratio selector for Qwen models with SD3-optimized dimensions
- ⭐ Star Qwen / WAN Ratio (`StarQwenWanRatio`) — Unified ratio selector for Qwen and WAN video models with auto aspect ratio matching
- ⭐ Star Qwen Image Edit Inputs (`StarQwenImageEditInputs`) — Multi-image stitcher for Qwen editing (up to 4 images)
- ⭐ Star Qwen Edit Encoder (`StarQwenEditEncoder`) — Advanced CLIP encoder optimized for Qwen image editing
- ⭐ Star Image Edit for Qwen/Kontext (`StarImageEditQwenKontext`) — Dynamic prompt builder with customizable templates
- ⭐ Star Qwen Edit Plus Conditioner (`StarQwenEditPlusConditioner`) — Enhanced conditioning for Qwen models
- ⭐ Star Qwen Rebalance Prompter (`StarQwenRebalancePrompter`) — Intelligent prompt rebalancing for better results
- ⭐ Star Qwen Regional Prompter (`StarQwenRegionalPrompter`) — Region-based prompting for precise control

#### Image Processing & Effects
- ⭐ Star Apply Overlay (Depth) (`StarApplyOverlayDepth`) — Blend filtered images using depth/mask with Gaussian blur
- ⭐ Star Simple Filters (`StarSimpleFilters`) — Comprehensive image adjustments with color matching (sharpen, blur, saturation, etc.)

#### AI Generation & Prompting
- ⭐ Star Nano Banana (Gemini) (`StarNanoBanana`) — Google Gemini 2.5 Flash image generation with 30+ templates
- ⭐ Star Ollama Sysprompter (JC) (`StarOllamaSysprompterJC`) — Structured prompt builder for Ollama with art styles
- ⭐ Star Sampler (`StarSampler`) — Advanced sampler with extensive configuration options

#### Utilities & Tools
- ⭐ Star Save Folder String (`StarSaveFolderString`) — Flexible path builder with date-based organization
- ⭐ Star Duplicate Model Finder (`StarDuplicateModelFinder`) — SHA256-based duplicate model scanner

### 1.6.0 – IO & Image Utilities

- ⭐ Star Random Image Loader — Load random images from folders with seed control
- ⭐ Star Image Loader 1by1 — Sequential image loading with state persistence
- ⭐ Star Save Panorama JPEG — Export JPEGs with XMP panorama metadata
- ⭐ Star Frame From Video — Extract specific frames from video batches
- ⭐ Star Icon Exporter — Multi-size PNG/ICO export with effects

## Available Nodes

### ⭐StarNodes/Starters
- ⭐ SD(XL) Starter: Loads checkpoint with CLIP and VAE, creates empty latent with customizable resolution
- ⭐ FLUX Starter: Loads Unet with 2 CLIPs and creates empty latent
- ⭐ SD3.0/3.5 Starter: Loads Unet with 3 CLIPs and creates empty latent

### ⭐StarNodes/Sampler
- ⭐ StarSampler SD/SDXL: Advanced sampler for SD, SDXL, SD3.5 with model and conditioning passthroughs
- ⭐ StarSampler FLUX: Specialized sampler for Flux models with model and conditioning passthroughs
- ⭐ Detail Star Daemon: Enhances image details, compatible with Flux and all SD Models (Adapted from [original sources](https://github.com/muerrilla/sd-webui-detail-daemon))
- ⭐ Star FluxFill Inpainter: Specialized inpainting node for Flux models with optimized conditioning and noise mask handling
- ⭐ Star 3 LoRAs: Applies up to three LoRAs simultaneously to a model with individual weight controls for each

### ⭐StarNodes/Qwen & Image Editing
- ⭐ Star Qwen Image Ratio: Dropdown aspect ratio selector for Qwen models with SD3-optimized dimensions (1:1, 16:9, 9:16, 4:3, 3:4, etc.)
- ⭐ Star Qwen / WAN Ratio: Unified ratio selector supporting both Qwen and WAN video models with automatic aspect ratio matching
- ⭐ Star Qwen Image Edit Inputs: Prepares up to 4 input images for Qwen editing by intelligently stitching them into a single canvas
- ⭐ Star Qwen Edit Encoder: Advanced CLIP text encoder optimized for Qwen image editing with reference latents and caching
- ⭐ Star Image Edit for Qwen/Kontext: Dynamic prompt builder loading customizable templates from editprompts.json
- ⭐ Star Qwen Edit Plus Conditioner: Enhanced conditioning specifically designed for Qwen models
- ⭐ Star Qwen Rebalance Prompter: Intelligently rebalances prompts for optimal Qwen model performance
- ⭐ Star Qwen Regional Prompter: Region-based prompting system for precise control over different image areas
- ⭐ Star Apply Overlay (Depth): Blends filtered images over source using depth/mask with Gaussian blur options
- ⭐ Star Simple Filters: Comprehensive image adjustments (sharpen, blur, saturation, contrast, brightness, temperature) with advanced color matching
- ⭐ Star Nano Banana (Gemini): Google Gemini 2.5 Flash image generation/editing with 30+ prompt templates and flexible aspect ratios
- ⭐ Star Sampler: Advanced sampler with extensive configuration options for various workflows

### ⭐StarNodes/Image And Latent
- ⭐ Star Adaptive Detail Enhancer: Adaptively sharpens, denoises, and enhances image details using edge, face, and texture analysis. Great for portraits, art, and upscaling. See [StarDetailEnhancer.md](web/docs/StarDetailEnhancer.md).
- ⭐ Star Seven Inputs(img): Switch that automatically passes the first provided input image to the output
- ⭐ Star Seven Inputs(latent): Switch that automatically passes the first provided latent to the output
- ⭐ Star Face Loader: Specialized node for handling face-related operations. Image loader that works like the "load image" node but saves images in a special faces-folder for later use
- ⭐ Star Grid Composer: Compose multiple images into a grid layout with automatic sizing, captions, and customizable fonts/colors. Supports batch image and caption input via StarGridBatchers
- ⭐ Star Grid Image Batcher: Batch multiple images or image batches for use with Star Grid Composer, supporting up to 16 images
- ⭐ Star Grid Captions Batcher: Batch up to 16 caption strings for grid layouts in Star Grid Composer
- ⭐ Star Model Latent Upscaler: Complete pipeline for latent upscaling with model choice and VAE encoding/decoding
- ⭐ Star SD Upscale Refiner: All-in-one SD1.5 upscaling and refinement node with LoRA, upscale model, tiled diffusion, and ControlNet tile integration
- ⭐ StarWatermark: Adds customizable watermarks to images. Supports text, image, and advanced placement options for protecting or branding your outputs
- ⭐ Star 7 Layers 2 PSD: Saves up to seven images as layers in a single PSD file with automatic sizing based on the largest image dimensions
- ⭐ Starnodes Aspect Ratio Advanced: Enhanced version with additional options for aspect ratio calculation and resolution determination

### ⭐StarNodes/Text And Data
- ⭐ Star Seven Inputs(txt): Text concatenation with optional inputs. Works as automatic switch and concatenates multiple inputs
- ⭐ Star Text Filter: Cleans string text by removing text between two given words (default), removing text before a specific word, removing text after a specific word, removing empty lines, removing all whitespace, or stripping whitespace from line edges
- ⭐ Star Seven Wildcards: Advanced prompt maker with 7 inputs supporting wildcards and multiple random selections
- ⭐ Star Wildcards Advanced: Enhanced wildcard processing with support for folder paths, random selection, and multiple prompt inputs
- ⭐ Star Easy-Text-Storage: Save, load, and manage text snippets for reuse across workflows. Perfect for storing prompts, system messages, and other text content
- ⭐ Star Web Scraper (Headlines): Scrapes news headlines from websites for use in prompts or text generation
- ⭐ Star Ollama Sysprompter (JC): Builds structured prompts for Ollama with multiple art styles loaded from styles.json

### ⭐StarNodes/IO
- ⭐ Star Save Folder String: Flexible path builder for organized file saving with preset folders, date-based organization, and custom naming
- ⭐ Star Duplicate Model Finder: Scans ComfyUI models directory for duplicate files using SHA256 hashing with detailed reports
- ⭐ Star Random Image Loader: Load random images from folders with optional subfolders and seed control
- ⭐ Star Image Loader 1by1: Sequentially loads images across runs with state saved in the folder
- ⭐ Star Save Panorama JPEG: Save JPEGs with embedded XMP panorama metadata for 360° viewers
- ⭐ Star Frame From Video: Pick first/last/specific frame from an image batch (e.g., video)
- ⭐ Star Icon Exporter: Export multi-size PNGs and ICO with shaping, stroke, and shadow options

### ⭐StarNodes/InfiniteYou
- ⭐ Star InfiniteYou Apply: Apply face identity from a reference image to generated images
- ⭐ Star InfiniteYou Face Swap Mod: Modified version of the face swap node with additional control options
- ⭐ Star InfiniteYou Patch Saver: Save face identity data for later use
- ⭐ Star InfiniteYou Patch Loader: Load previously saved face identity data
- ⭐ Star InfiniteYou Patch Combine: Combine multiple face patches with weighted influence
- ⭐ Star InfiniteYou Advanced Patch Maker: Create advanced face patches with detailed control options

### ⭐StarNodes/Conditioning
- ⭐ Star Conditioning I/O: Allows saving and loading conditioning information for reuse across workflows

### ⭐StarNodes/Settings
- ⭐ Star Save Sampler Settings: Save customizable sampling settings for StarSamplers with support for both SD and Flux samplers
- ⭐ Star Load Sampler Settings: Load previously saved sampling settings for StarSamplers
- ⭐ Star Delete Sampler Settings: Delete saved sampling settings

### ⭐StarNodes/Helpers And Tools
- ⭐ Star Denoise Slider: Provides a simple slider interface to control the denoising strength for samplers
- ⭐ Starnodes Aspect Ratio: Calculates aspect ratio from an image or provides standard aspect ratios with customizable megapixel settings
- ⭐ Star Divisible Dimension: Ensures image dimensions are divisible by a specific value (useful for VAE compatibility)
- ⭐ Starnodes Aspect Video Ratio: Select a video aspect ratio from a dropdown, input width, and receive width/height as int/string plus formatted size (e.g., 750x422). Calculates height automatically from width and selected ratio.
- ⭐ Star Random Lora Loader: Randomly selects a LoRA from your library with subfolder/name filters and optional direct application to MODEL/CLIP or string output

### ⭐StarNodes/Color
- ⭐ Star Palette Extractor: Extracts dominant color palette from an image with various color format options

### ⭐StarNodes
- ⭐ Ollama Helper: Loads Ollama models from ollamamodels.txt for integration with Ollama nodes

*Note: You can add custom resolutions by editing the .json files in the node folder.

## Documentation

Detailed documentation for all nodes is available in the `web/docs` directory of this repository. Each node has its own markdown file with comprehensive information about:
- Inputs and outputs
- Usage instructions
- Features and capabilities
- Technical details
- Tips and notes

The documentation is automatically loaded by ComfyUI when you access the help for any node, based on your locale settings.

### Additional Documentation
- **QwenEditPromptGuide.md** - Comprehensive guide for using Qwen image editing nodes
- **README_StarQwenRegionalPrompter.md** - Detailed documentation for regional prompting
- **SIMPLIFIED_REGIONAL_PROMPTER_V2.md** - Simplified guide for regional prompter v2
- **editprompts.json** - Customizable prompt templates for Qwen/Kontext nodes
- **styles.json** - Art style definitions for Ollama Sysprompter

## Installation

### Via ComfyUI Manager (Recommended)
Search for "Starnodes" in ComfyUI Manager and install

### Manual Installation
1. Open CMD within your custom nodes folder
2. Run: `git clone https://github.com/Starnodes2024/ComfyUI_StarNodes`
3. Restart ComfyUI

Find the nodes under "⭐StarNodes" category or search for "star" in the node browser.

### Wildcards
You will find the wildcards in the wildcards folder of your ComfyUI main folder. If you add your own just copy the new files to this location.

## Wildcard Rules in the Star Wildcards Node

### Basic Wildcard Syntax
- Wildcards are defined using double underscores: `__wildcard_name__`
- The node looks for text files in the `wildcards` folder of your ComfyUI installation
- When a wildcard is encountered, a random line from the corresponding text file is selected

### Folder Structure
- Wildcards can be organized in subfolders
- To use a wildcard in a subfolder, use the syntax: `folder\__wildcard_name__`
- The system will look for the file at `[ComfyUI base path]/wildcards/folder/wildcard_name.txt`

### Random Options
- You can use curly braces `{}` with pipe symbols `|` to choose randomly between options
- Example: `{option1|option2|option3}` will randomly select one of the options
- You can even include wildcards inside these options: `{__wildcard1__|__wildcard2__}`

### Nested Wildcards
- Wildcards can be nested within other wildcards
- The system supports up to 10 levels of recursion to prevent infinite loops
- When a wildcard contains another wildcard, the nested wildcard is also processed

### Seed Behavior
- The node takes a seed parameter that determines the randomization
- Each prompt input (1-7) uses a different seed offset to ensure variety
- Each wildcard within a prompt also gets a unique seed based on its position

### Error Handling
- If a wildcard file doesn't exist, the wildcard name itself is used as text
- If a wildcard file exists but is empty, the wildcard name is used as fallback
- If there's an error processing a wildcard, it falls back to using the wildcard name

The Star Wildcards node allows you to combine up to 7 different prompts, each with their own wildcards, which are then joined together with spaces to create the final output string.

- Supports recursive wildcard processing up to 10 layers

## Requirements

### Core Dependencies
All dependencies are listed in `requirements.txt` and will be installed automatically:
- **google-generativeai>=0.8.3** - Required for Star Nano Banana (Gemini) node
- **color-matcher** - Required for Star Simple Filters color matching features
- **insightface** - Required for InfiniteYou face swap nodes
- **onnxruntime** / **onnxruntime-gpu** - Required for face detection and processing
- **psd-tools>=1.10.0** - Required for PSD export nodes
- **beautifulsoup4**, **newspaper3k**, **lxml** - Required for web scraping nodes

### Google Gemini API Key
To use the **Star Nano Banana** node:
1. Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create or edit `googleapi.ini` in the starnodes folder
3. Add your key: `[GOOGLE]` / `api_key = YOUR_API_KEY_HERE`

### InfiniteYou Setup
For InfiniteYou Insightface is a requirement. If you are having trouble installing it (windows) here is how to fix that problem:
1. Download that insightface wheel that fits your python version from: 
https://github.com/Gourieff/Assets/tree/main/Insightface
2. open command and input:
PATH_TO_YOUR_COMFYUI\.venv\Scripts\python.exe -m pip install PATH_TO_DOWNLOADED_WHEEL\insightface-0.7.3-cp312-cp312-win_amd64.whl onnxruntime
3. Restart ComfyUI
Also this Video could help you if you having problems:
https://www.youtube.com/watch?v=vCCVxGtCyho&ab_channel=DataLeveling

You will need the InfiniteYou Models from Bytedance:
https://huggingface.co/vuongminhkhoi4/ComfyUI_InfiniteYou/tree/main
in models/infiniteyou place:
aes_stage2_img_proj.bin
sim_stage1_img_proj.bin

in models/controlnet place:
sim_stage1_control_net.safetensors
aes_stage2_control.safetensors