# ⭐ StarNodes Example Workflows Guide

## Overview
This document provides recommended example workflows for StarNodes v1.9.2, organized by use case and complexity level.

---

## 🎨 Basic Image Generation Workflows

### 1. FLUX Basic Generation
**Difficulty:** Beginner  
**Purpose:** Simple FLUX model image generation

**Workflow:**
```
FLUX Start Settings → StarSampler (Unified) → VAE Decode → Save Image
```

**Key Nodes:**
- `⭐ FLUX Start Settings` - Configure FLUX model, prompts, and basic settings
- `⭐ StarSampler (Unified)` - Unified sampler for all model types
- Standard ComfyUI VAE Decode and Save Image

**Use Cases:**
- Quick FLUX generations
- Testing prompts
- Basic creative work

---

### 2. SDXL Basic Generation
**Difficulty:** Beginner  
**Purpose:** SDXL model image generation with refinement

**Workflow:**
```
SDXL Start Settings → StarSampler (Unified) → VAE Decode → Save Image
```

**Key Nodes:**
- `⭐ SDXL Start Settings` - SDXL-specific configuration
- `⭐ StarSampler (Unified)` - Handles SDXL sampling

**Use Cases:**
- SDXL model workflows
- Portrait and landscape generation
- High-quality outputs

---

### 3. SD3.5 Generation
**Difficulty:** Beginner  
**Purpose:** Latest SD3.5 model generation

**Workflow:**
```
SD3.5 Start Settings → StarSampler (Unified) → VAE Decode → Save Image
```

**Key Nodes:**
- `⭐ SD3.5 Start Settings` - SD3.5 configuration
- `⭐ StarSampler (Unified)` - SD3.5 sampling

---

## 🚀 Advanced Upscaling Workflows

### 4. SD Upscale Refiner (Complete Pipeline)
**Difficulty:** Intermediate  
**Purpose:** High-quality image upscaling with refinement

**Workflow:**
```
Load Image → ⭐ Star SD Upscale Refiner → Save Image
```

**Configuration:**
- Select upscale model (4x recommended)
- Choose LoRA for style enhancement (optional)
- Configure tile size for VRAM management
- Set denoise strength (0.3-0.5 for subtle refinement)

**Key Features:**
- FreeU v2 support for enhanced details
- PAG (Perturbed Attention Guidance) for quality
- Tiled VAE decoding for large images
- Custom sampler integration

**Use Cases:**
- Upscaling generated images
- Enhancing photo quality
- Preparing images for print
- Detail enhancement

---

### 5. Model Latent Upscaler
**Difficulty:** Intermediate  
**Purpose:** Latent-space upscaling with model selection

**Workflow:**
```
Load Image → ⭐ Star Model Latent Upscaler → VAE Decode → Save Image
```

**Key Nodes:**
- `⭐ Star Model Latent Upscaler` - Upscale in latent space with VAE control

**Use Cases:**
- Fast upscaling
- Latent-space manipulation
- VAE experimentation

---

## 🎭 Grid Composition Workflows

### 6. Image Grid with Captions
**Difficulty:** Intermediate  
**Purpose:** Create professional image grids with text captions

**Workflow:**
```
[Multiple Image Sources]
    ↓
⭐ Star Grid Image Batcher (collects up to 16 images)
    ↓
⭐ Star Grid Captions Batcher (optional captions)
    ↓
⭐ Star Grid Composer
    ↓
Save Image
```

**Configuration:**
- Grid layout: 2x2, 3x3, 4x4, or custom
- Caption font, size, and color
- Background color
- Spacing and padding

**Use Cases:**
- Portfolio presentations
- Before/after comparisons
- Style exploration grids
- Social media posts
- Tutorial images

---

### 7. Dynamic Image Batching
**Difficulty:** Beginner  
**Purpose:** Batch multiple images for processing

**Workflow:**
```
⭐ Star Image Input (Dynamic) → [Processing Nodes] → Output
```

**Key Features:**
- Auto-adds inputs when connected
- Auto-removes unused inputs
- Supports unlimited images (dynamic)

**Use Cases:**
- Variable number of inputs
- Flexible workflows
- Batch processing

---

## 🎨 InfiniteYou Character Consistency Workflows

### 8. Character Consistency (Basic)
**Difficulty:** Advanced  
**Purpose:** Maintain consistent character across generations

**Workflow:**
```
⭐ Star InfiniteYou Patch Loader
    ↓
[Connect to Model/Conditioning]
    ↓
StarSampler (Unified)
    ↓
VAE Decode → Save Image
```

**Key Nodes:**
- `⭐ Star InfiniteYou Patch Loader` - Load character patch
- Patch files stored in `models/infiniteyou/`

**Use Cases:**
- Consistent character generation
- Character in different scenes
- Story boarding

---

### 9. Multiple Character Patches
**Difficulty:** Advanced  
**Purpose:** Combine multiple character patches

**Workflow:**
```
⭐ Star InfiniteYou Patch Combine
    ↓
[Apply to generation pipeline]
```

**Key Features:**
- Combine up to 3 patches
- Adjustable patch strength
- Device selection (CPU/GPU)

---

### 10. Face Swap with InfiniteYou
**Difficulty:** Advanced  
**Purpose:** Swap faces while maintaining consistency

**Workflow:**
```
Source Image → ⭐ Star InfiniteYou Face Swap Mod → Output
```

**Use Cases:**
- Face replacement
- Character adaptation
- Identity transfer

---

## 🖼️ Qwen Image Editing Workflows

### 11. Qwen Image Editing
**Difficulty:** Intermediate  
**Purpose:** AI-powered image editing with Qwen models

**Workflow:**
```
⭐ Qwen Image Start Settings
    ↓
⭐ Star Qwen Image Edit Inputs (up to 4 images)
    ↓
⭐ Star Qwen Edit Encoder
    ↓
StarSampler (Unified)
    ↓
VAE Decode → Save Image
```

**Key Features:**
- Multi-image input (1-4 images)
- Automatic grid stitching
- Qwen-specific aspect ratios
- Edit-aware conditioning

**Use Cases:**
- Image-to-image editing
- Style transfer
- Image variations
- Guided generation

---

### 12. Regional Prompting with Qwen
**Difficulty:** Advanced  
**Purpose:** Control different regions with different prompts

**Workflow:**
```
CLIP Text Encode (base prompt)
    ↓
⭐ Star Qwen Regional Prompter
    ↓
[Apply to sampler]
```

**Configuration:**
- Define up to 4 regions
- Set coordinates (x1, y1, x2, y2)
- Individual prompts per region
- Strength control per region

**Use Cases:**
- Complex compositions
- Multi-subject scenes
- Precise control over image areas
- Architectural rendering

---

## 📝 Advanced Prompting Workflows

### 13. Structured Prompting (Qwen Rebalance)
**Difficulty:** Intermediate  
**Purpose:** Create highly structured, layered prompts

**Workflow:**
```
⭐ Star Qwen-Rebalance-Prompter
    ↓
Text to Conditioning
    ↓
StarSampler
```

**Key Features:**
- Foreground/Midground/Background layers
- Composition presets
- Color tone control
- Lighting mood
- Visual guidance

**Use Cases:**
- Complex scene composition
- Photorealistic generation
- Cinematic shots
- Architectural visualization

---

### 14. AI-Assisted Prompting (Ollama)
**Difficulty:** Intermediate  
**Purpose:** Generate enhanced prompts using local LLM

**Workflow:**
```
⭐ Star Ollama Sysprompter (JC)
    ↓
Text to Conditioning
    ↓
StarSampler
```

**Requirements:**
- Ollama installed locally
- Compatible LLM model

**Use Cases:**
- Prompt enhancement
- Style application
- Creative prompt generation
- Consistent style across generations

---

### 15. Wildcard Prompting
**Difficulty:** Beginner  
**Purpose:** Random variations using wildcard files

**Workflow:**
```
⭐ Star Seven Wildcards (or Wildcards Advanced)
    ↓
Text to Conditioning
    ↓
StarSampler
```

**Key Features:**
- Load wildcard files from `wildcards/` folder
- Random selection per generation
- Supports nested wildcards
- Batch generation with variations

**Use Cases:**
- Generating variations
- Random character traits
- Style exploration
- Batch diverse outputs

---

## 💾 Output and Export Workflows

### 16. PSD Layer Export
**Difficulty:** Intermediate  
**Purpose:** Export multiple images as Photoshop layers

**Workflow:**
```
[Multiple Image Generations]
    ↓
⭐ Star PSD Saver (Dynamic)
    ↓
[PSD file saved to output folder]
```

**Key Features:**
- Up to 7 layers (PSD Saver 2) or unlimited (Dynamic)
- Auto-sizing based on largest image
- Layer masks support
- Centered composition

**Use Cases:**
- Post-processing in Photoshop
- Layer-based editing
- Compositing
- Professional workflows

---

### 17. Panorama Export
**Difficulty:** Beginner  
**Purpose:** Save images with panorama metadata

**Workflow:**
```
[Generated Image]
    ↓
⭐ Star Save Panorama JPEG
```

**Key Features:**
- XMP metadata for 360° viewers
- Projection type selection
- JPEG optimization

**Use Cases:**
- 360° panoramas
- VR content
- Immersive images

---

### 18. Organized File Saving
**Difficulty:** Beginner  
**Purpose:** Save files with organized folder structure

**Workflow:**
```
⭐ Star Save Folder String
    ↓
[Use path in Save Image node]
```

**Key Features:**
- Dynamic folder creation
- Date/time stamps
- Custom naming patterns
- Preset management

**Use Cases:**
- Organized output
- Project-based saving
- Automated workflows

---

## 🎨 Creative and Specialized Workflows

### 19. Icon Generation and Export
**Difficulty:** Intermediate  
**Purpose:** Generate and export multi-size icons

**Workflow:**
```
[Generated Image]
    ↓
⭐ Star Icon Exporter
```

**Key Features:**
- Multiple size exports (16x16 to 512x512)
- ICO format support
- Batch icon generation

**Use Cases:**
- App icons
- Website favicons
- UI elements

---

### 20. Color Palette Extraction
**Difficulty:** Beginner  
**Purpose:** Extract color palettes from images

**Workflow:**
```
Load Image
    ↓
⭐ Star Palette Extractor
    ↓
[Use colors in subsequent generations]
```

**Use Cases:**
- Color matching
- Style transfer
- Brand consistency
- Color analysis

---

### 21. Watermarking
**Difficulty:** Beginner  
**Purpose:** Add watermarks to generated images

**Workflow:**
```
[Generated Image]
    ↓
⭐ StarWatermark
    ↓
Save Image
```

**Key Features:**
- Text or image watermarks
- Position control
- Opacity adjustment
- Multiple watermark support

**Use Cases:**
- Copyright protection
- Branding
- Portfolio watermarking

---

### 22. Simple Image Filters
**Difficulty:** Beginner  
**Purpose:** Apply quick adjustments to images

**Workflow:**
```
Load Image
    ↓
⭐ Star Simple Filters
    ↓
Save Image
```

**Filters:**
- Sharpen
- Blur
- Saturation
- Contrast
- Brightness
- Temperature
- Color matching

**Use Cases:**
- Quick adjustments
- Color correction
- Image enhancement

---

### 23. Video Frame Extraction
**Difficulty:** Beginner  
**Purpose:** Extract frames from video files

**Workflow:**
```
⭐ Star Frame From Video
    ↓
[Process frames]
```

**Use Cases:**
- Video to image
- Frame analysis
- Animation reference

---

## 🤖 AI Generation Workflows

### 24. Gemini Image Generation (Nano Banana)
**Difficulty:** Intermediate  
**Purpose:** Generate images using Google's Gemini AI

**Workflow:**
```
⭐ Star Nano Banana (Gemini Image Gen)
    ↓
[Image output]
```

**Requirements:**
- Google API key
- Gemini API access

**Use Cases:**
- Alternative AI generation
- Gemini-specific features
- Multi-model workflows

---

## 🛠️ Utility Workflows

### 25. Aspect Ratio Management
**Difficulty:** Beginner  
**Purpose:** Calculate and manage aspect ratios

**Workflows:**
- `⭐ Starnodes Aspect Ratio` - Basic ratio calculation
- `⭐ Starnodes Aspect Ratio Advanced` - Advanced options
- `⭐ Starnodes Aspect Video Ratio` - Video-specific ratios

**Use Cases:**
- Resolution calculation
- Model-specific sizing
- Batch generation with consistent ratios

---

### 26. Duplicate Model Finder
**Difficulty:** Beginner  
**Purpose:** Find duplicate models in your collection

**Workflow:**
```
⭐ Star Duplicate Model Finder
    ↓
[Review results]
```

**Use Cases:**
- Clean up model folder
- Find duplicate checkpoints
- Organize model library

---

### 27. Text Storage and Management
**Difficulty:** Beginner  
**Purpose:** Store and retrieve text snippets

**Workflows:**
- `⭐ Star Easy-Text-Storage` - Save/load text
- `⭐ Star Seven Inputs (txt)` - Multiple text inputs
- `⭐ Star Text Filter` - Filter and process text

**Use Cases:**
- Prompt libraries
- Reusable text snippets
- Workflow templates

---

## 📚 Recommended Workflow Combinations

### Complete Generation Pipeline
```
FLUX Start → StarSampler → VAE Decode → Star Simple Filters → StarWatermark → Save Image
```

### Professional Upscale Pipeline
```
Load Image → Star SD Upscale Refiner → Star Simple Filters → Star PSD Saver
```

### Character Consistency Pipeline
```
InfiniteYou Patch → SDXL Start → StarSampler → Upscale → PSD Export
```

### Batch Variation Pipeline
```
Star Wildcards → Multiple Seeds → Grid Composer → Save
```

---

## 💡 Tips for Creating Workflows

1. **Start Simple**: Begin with basic workflows and add complexity gradually
2. **Save Templates**: Save successful workflows as templates
3. **Use Dynamic Nodes**: Leverage dynamic input nodes for flexibility
4. **Organize Outputs**: Use Star Save Folder String for organized file management
5. **Batch Process**: Combine Grid Composer with batch generation for comparisons
6. **Test Settings**: Use StarSampler Settings nodes to save/load configurations
7. **Document**: Keep notes on successful parameter combinations

---

## 🔗 Workflow Resources

### Example Workflow Files
Check the `example_workflows/` folder for:
- Basic generation templates
- Advanced composition examples
- InfiniteYou character workflows
- Upscaling pipelines

### Additional Documentation
- `README.md` - Full node list and features
- `web/docs/` - Individual node documentation
- `CHANGELOG.md` - Version history and changes

---

## 📞 Support and Community

For questions, issues, or sharing workflows:
- GitHub Issues: Report bugs and request features
- Community Workflows: Share your creations
- Documentation: Contribute improvements

---

**Version:** 1.9.2  
**Last Updated:** November 2025
