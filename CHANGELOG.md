# Changelog - ComfyUI StarNodes

## Version 2.0.0 (2026-04-10)

### 🎉 Major Release - New Integrated Nodes + Cleanup

This release integrates two powerful custom node packages directly into StarNodes and removes deprecated nodes to streamline the codebase.

### ✨ New Integrated Nodes (9 total)

#### LTX Video Toolz Suite (8 nodes)
- **Star LTX Video Settings** (`StarLTXVideoSettings`) - Comprehensive video dimension and frame calculator for LTX video generation
  - Video size presets (HD/FHD/Custom)
  - Multiple aspect ratios (1:1, 16:9, 21:9, etc.)
  - Smart input image ratio detection
  - Frame calculation based on FPS and duration
  - Divisibility constraints for LTX requirements
- **Star VAE LTXV Save** (`StarVAELTXVSave`) - Advanced VAE encoder for LTX video with quality presets and latent saving
- **Star VAE LTXV Load** (`StarVAELTXVLoad`) - VAE decoder for LTX video latents
- **Star LTX Image Cut** (`StarLTXImageCut`) - Smart image cropping tool with aspect ratio preservation
- **Star Multi Inputs to One** (`StarMultiInputsToOne`) - Combine multiple dynamic inputs into single output
- **Star LTXV Get Last Frame** (`StarLTXVGetLastFrame`) - Extract last frame from LTX video latents
- **Star LTXV Load Last Image** (`StarLTXVLoadLastImage`) - Load and process last generated image
- **Star Video Joiner** (`StarVideoJoiner`) - Join multiple video files into seamless video

#### Music Generation Suite (1 node)
- **Star ACE Step 1.5** (`StarACEStep`) - Professional music generation using ACE Step 1.5 API
  - Full control over duration, BPM, key/scale, time signature
  - Lyrics support in 50+ languages
  - Thinking mode with LM-enhanced generation
  - Multiple output formats (MP3, WAV, FLAC)
  - Batch generation (up to 8 songs)
  - Sample mode for natural language descriptions
  - Automatic subfolder organization

### ❌ Removed Nodes (8 total)

#### InfiniteYou Face Swap Suite (4 nodes)
- **StarInfiniteYou** - Removed due to insightface dependency issues
- **StarInfiniteYouFaceSwapMod** - Removed due to insightface dependency issues
- **StarInfiniteYouPatch** - Removed due to insightface dependency issues
- **StarInfiniteYouAdvancedPatchMaker** - Removed due to insightface dependency issues

#### Misc Nodes (3 nodes)
- **StarFaceLoader** - Deprecated and unused
- **StarGeminiRefiner** - Removed to reduce external API dependencies
- **StarFlowmatchOption** - Deprecated experimental feature

#### External API Nodes (1 node)
- **StarNanoBanana** - Removed to reduce external API dependencies

### 🗑️ Removed Files & Folders
- Complete `infiniteyou/` folder and all related files (12 files)
- Root-level duplicate files: `starinfiniteyou_core.py`, `starinfiniteyou_utils.py`, `ollamahelper.py`
- Unused node files: `starfaceloader.py`, `star_gemini_refiner.py`, `star_flowmatch_option.py`, `star_nano_banana.py`

### 📦 Dependencies Changes

**Added:**
- `soundfile>=0.12.0` - For ACE Step music generation audio processing

**Removed:**
- `google-generativeai>=0.8.3` - No longer needed after removing Gemini-based nodes
- `onnxruntime-gpu` - No longer needed after removing InfiniteYou nodes
- `insightface` - No longer needed (was already commented out)
- `facexlib==0.3.0` - No longer needed (was already commented out)

### 🐛 Bug Fixes
- Fixed duplicate imports of `StarShowLastFrame` and `StarAspectVideoRatio` in `__init__.py`
- Restored `star_flux2_conditioner.py` to root directory (was accidentally deleted during cleanup)

### 📊 Statistics
- **Nodes:** 85 → 86 (+1 net: -8 removed, +9 integrated)
- **Files removed:** ~60 Python files from root cleanup
- **Dependencies:** +1 added, -4 removed (net -3)
- **New folders:** `ltx_video/` and `music/`

### ⚠️ Migration Notes
If you were using any of the removed InfiniteYou nodes, you will need to:
1. Remove them from your workflows
2. Consider alternative face swap solutions from other ComfyUI custom nodes
3. Update your requirements if you had manually installed insightface

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
