# Quick Start Guide - StarNodes 1.7.0

## 🚀 What's New in 1.7.0

StarNodes 1.7.0 adds **15 powerful new nodes** focused on Qwen image editing, AI generation, and advanced utilities!

## 🎯 Quick Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Google Gemini (Optional)
For **Star Nano Banana** and **Star Gemini Refiner** nodes:
1. Get API key from https://makersuite.google.com/app/apikey
2. Edit `googleapi.ini` in the comfyui_starnodes folder:
```ini
[API_KEY]
key = YOUR_API_KEY_HERE
```
Or use external file pointer or environment variable (see googleapi.ini for details)

### 3. Restart ComfyUI
All nodes will appear under **⭐StarNodes** categories!

---

## 🌟 Featured New Nodes

### 🖼️ Qwen Image Editing Suite

#### Star Qwen Image Ratio
**Category**: Image And Latent  
**Purpose**: Quick aspect ratio selector for Qwen models  
**Use Case**: Start Qwen workflows with optimized dimensions
```
Outputs: LATENT, width, height
Ratios: 1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3, 5:7, 7:5, custom
```

#### Star Qwen Image Edit Inputs
**Category**: Image And Latent  
**Purpose**: Stitch up to 4 images for Qwen editing  
**Use Case**: Before/after comparisons, multi-image editing
```
Inputs: 1-4 images
Outputs: Stitched image, latent, dimensions
```

#### Star Qwen Edit Encoder
**Category**: Conditioning  
**Purpose**: Advanced CLIP encoding for Qwen  
**Use Case**: High-quality text conditioning with caching
```
Features: Reference latents, performance optimization, timing controls
```

#### Star Image Edit for Qwen/Kontext
**Category**: Prompts  
**Purpose**: Template-based prompt builder  
**Use Case**: Quick access to 30+ editing templates
```
Templates: Style transfer, color enhancement, background change, etc.
Customizable via editprompts.json
```

#### Star Qwen Regional Prompter
**Category**: Conditioning  
**Purpose**: Region-based prompting  
**Use Case**: Control different image areas independently
```
Features: Multiple regions, blend modes, attention control
```

---

### 🎨 Image Processing

#### Star Apply Overlay (Depth)
**Category**: Image And Latent  
**Purpose**: Blend images using depth/mask  
**Use Case**: Apply filters with depth-aware blending
```
Inputs: Source, filtered image, depth/mask
Features: Gaussian blur, strength control, invert mask
```

#### Star Simple Filters
**Category**: Image And Latent  
**Purpose**: Comprehensive image adjustments  
**Use Case**: Quick color/tone corrections
```
Filters: Sharpen, blur, saturation, contrast, brightness, temperature
Advanced: Color matching (8 methods including MKL, Reinhard, MVGD)
```

---

### 🤖 AI Generation

#### Star Nano Banana (Gemini)
**Category**: Image Generation  
**Purpose**: Google Gemini 2.5 Flash generation  
**Use Case**: AI image generation/editing with templates
```
Features:
- 30+ prompt templates
- Up to 5 input images
- Flexible aspect ratios (1:1, 16:9, 9:16, 4:3, 3:4)
- 1-15 megapixel output
- Requires Google Gemini API key
```

#### Star Ollama Sysprompter (JC)
**Category**: Prompts  
**Purpose**: Structured Ollama prompts  
**Use Case**: Generate styled prompts for Ollama models
```
Features:
- Multiple art styles from styles.json
- Token limit control
- Custom style support
- System prompt building
```

---

### 🛠️ Utilities

#### Star Save Folder String
**Category**: IO  
**Purpose**: Flexible path builder  
**Use Case**: Organize outputs with date-based folders
```
Features:
- Preset folders
- Date-based organization
- Custom naming
- Cross-platform paths
```

#### Star Duplicate Model Finder
**Category**: Helpers And Tools  
**Purpose**: Find duplicate models via SHA256  
**Use Case**: Clean up model directory, save disk space
```
Features:
- Scans all models or specific folders
- Hash-based detection
- Detailed reports
- Cache for performance
```

#### Star Sampler
**Category**: Sampler  
**Purpose**: Advanced sampling with detail control  
**Use Case**: Fine-tuned generation with detail daemon
```
Features:
- Detail schedule support
- Flux and SD compatibility
- Extensive configuration
- Model/conditioning passthrough
```

---

## 📚 Workflow Examples

### Example 1: Qwen Image Editing
```
1. Star Qwen Image Ratio → Select aspect ratio
2. Load Qwen Model → Your Qwen checkpoint
3. Star Image Edit for Qwen/Kontext → Choose template
4. Star Qwen Edit Encoder → Encode prompt
5. Star Sampler → Generate
```

### Example 2: Gemini Generation
```
1. Star Nano Banana:
   - Add API key to googleapi.ini
   - Select template or write custom prompt
   - Choose ratio and megapixels
   - Optional: Add reference images
2. Output: Generated image
```

### Example 3: Color Matching
```
1. Load Image → Your image
2. Load Image → Reference image (for color)
3. Star Simple Filters:
   - Select color_match_method (e.g., "mkl")
   - Connect reference image
   - Adjust filter_strength
4. Output: Color-matched image
```

### Example 4: Depth-Based Overlay
```
1. Load Image → Source image
2. Apply Filter → Create filtered version
3. Load Depth Map → Or use depth estimation
4. Star Apply Overlay (Depth):
   - Connect all inputs
   - Adjust strength and blur
5. Output: Blended result
```

---

## 🎨 Customization

### Edit Prompt Templates
Edit `editprompts.json` to add custom templates:
```json
{
  "Qwen": {
    "My Custom Task": {
      "template": "Your template with {placeholders}",
      "params": ["param1", "param2"]
    }
  }
}
```

### Add Art Styles
Edit `styles.json` to add Ollama styles:
```json
{
  "My Style": {
    "name": "My Style Name",
    "prompt": "Style description"
  }
}
```

### Configure Save Presets
Edit `star_save_folder_presets.json`:
```json
{
  "My Preset": "path/to/folder"
}
```

---

## 🔍 Finding Nodes

All new nodes are under **⭐StarNodes** categories:
- Right-click → Add Node → ⭐StarNodes
- Or search: "star qwen", "star nano", "star simple", etc.

---

## 📖 Documentation

Each node has detailed documentation:
1. Right-click node → Help
2. Or check `web/docs/Star*.md` files

Additional guides:
- `QwenEditPromptGuide.md` - Comprehensive Qwen guide
- `README_StarQwenRegionalPrompter.md` - Regional prompting
- `SIMPLIFIED_REGIONAL_PROMPTER_V2.md` - Simplified guide

---

## ⚠️ Troubleshooting

### "Can't import color-matcher"
```bash
pip install color-matcher
```

### "Google API key not found"
1. Create/edit `googleapi.ini` in comfyui_starnodes folder
2. Add your API key under `[API_KEY]` section with `key = YOUR_API_KEY_HERE`

### "Qwen model not loading"
- Ensure you have Qwen-compatible models
- Check model path in ComfyUI

### Nodes not appearing
1. Restart ComfyUI completely
2. Check console for import errors
3. Verify `requirements.txt` installed

---

## 🎉 Enjoy StarNodes 1.7.0!

For support and updates:
- GitHub: https://github.com/Starnodes2024/ComfyUI_StarNodes
- Check CHANGELOG.md for detailed changes
- See README.md for complete node list
