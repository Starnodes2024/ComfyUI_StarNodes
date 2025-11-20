# 🌟 StarNodes 1.7.0 Release Notes

**Release Date**: November 20, 2024  
**Type**: Major Feature Release  
**Status**: Stable

---

## 🎉 Overview

StarNodes 1.7.0 represents a major milestone, integrating all tested and stable nodes from the StarBetaNodes repository. This release adds **15 powerful new nodes** focused on:

- 🖼️ **Qwen Image Editing** - Complete suite for Qwen model workflows
- 🤖 **AI Generation** - Google Gemini integration with templates
- 🎨 **Image Processing** - Advanced filters and depth-based effects
- 🛠️ **Utilities** - Path management and duplicate detection

**Total Nodes**: 70+ (15 new in this release)

---

## ✨ Highlights

### 🏆 Major Features

#### 1. Complete Qwen Editing Suite (8 nodes)
Transform your Qwen workflows with specialized nodes:
- **Aspect Ratio Management** - Optimized dimensions for Qwen models
- **Multi-Image Stitching** - Combine up to 4 images intelligently
- **Advanced Encoding** - Performance-optimized CLIP encoding
- **Template System** - 30+ ready-made editing prompts
- **Regional Control** - Precise area-specific prompting
- **Smart Rebalancing** - Automatic prompt optimization

#### 2. Google Gemini Integration
- **Star Nano Banana** brings Google's Gemini 2.5 Flash to ComfyUI
- 30+ built-in prompt templates for common tasks
- Support for up to 5 reference images
- Flexible output sizes (1-15 megapixels)
- Multiple aspect ratios

#### 3. Professional Image Processing
- **Color Matching** - 8 advanced algorithms (MKL, Reinhard, MVGD, etc.)
- **Depth-Based Blending** - Intelligent overlay with depth awareness
- **Comprehensive Filters** - Sharpen, blur, saturation, contrast, and more
- **Non-Destructive** - Blend strength control for all effects

#### 4. Enhanced Utilities
- **Smart Path Builder** - Organize outputs with date-based folders
- **Duplicate Finder** - SHA256-based model deduplication
- **Advanced Sampler** - Detail daemon integration for better quality

---

## 📦 What's Included

### New Nodes (15)

#### Image Editing & Latent
1. **Star Qwen Image Ratio** - Aspect ratio selector with SD3 optimization
2. **Star Qwen / WAN Ratio** - Unified ratio selector for Qwen/WAN models
3. **Star Qwen Image Edit Inputs** - Multi-image stitcher (up to 4 images)
4. **Star Apply Overlay (Depth)** - Depth-aware image blending
5. **Star Simple Filters** - Comprehensive adjustments + color matching

#### Conditioning & Prompts
6. **Star Qwen Edit Encoder** - Advanced CLIP encoder for Qwen
7. **Star Qwen Edit Plus Conditioner** - Enhanced Qwen conditioning
8. **Star Qwen Regional Prompter** - Region-based prompting system
9. **Star Qwen Rebalance Prompter** - Intelligent prompt optimization
10. **Star Image Edit for Qwen/Kontext** - Template-based prompt builder
11. **Star Ollama Sysprompter (JC)** - Structured Ollama prompts

#### Generation & Sampling
12. **Star Nano Banana (Gemini)** - Google Gemini image generation
13. **Star Sampler** - Advanced sampler with detail control

#### Utilities
14. **Star Save Folder String** - Flexible path builder
15. **Star Duplicate Model Finder** - Model deduplication tool

### New Configuration Files
- `editprompts.json` - Customizable prompt templates (30+ included)
- `styles.json` - Art style definitions for Ollama
- `googleapi.ini` - Google Gemini API configuration
- `star_save_folder_presets.json` - Folder presets

### New Documentation (20+ files)
- **Guides**: QwenEditPromptGuide.md, README_StarQwenRegionalPrompter.md
- **Node Docs**: 17 markdown files in web/docs/
- **Quick Start**: QUICK_START_1.7.0.md
- **Migration**: MIGRATION_1.7.0.md
- **Changelog**: CHANGELOG.md

### Web Assets
- JavaScript UI components for enhanced node interaction
- 9 otter sprite images for UI enhancements
- StarryLinks.js for advanced node linking

---

## 🔧 Technical Improvements

### Architecture
- ✅ Standardized all node categories with ⭐ emoji
- ✅ Added web server routes for dynamic content
- ✅ Enhanced __init__.py with proper node registration
- ✅ Improved error handling and fallbacks

### Performance
- ✅ Caching support in Qwen Edit Encoder
- ✅ SHA256-based duplicate detection with caching
- ✅ Optimized image processing pipelines
- ✅ Lazy loading for optional dependencies

### Compatibility
- ✅ Cross-platform path handling
- ✅ Backward compatible with existing workflows
- ✅ Graceful degradation for missing dependencies
- ✅ ComfyUI Manager compatible

---

## 📋 Requirements

### Core Dependencies (Auto-installed)
```
google-generativeai>=0.8.3  # For Gemini
color-matcher               # For color matching
insightface                 # For face operations
onnxruntime                 # For face detection
psd-tools>=1.10.0          # For PSD export
beautifulsoup4             # For web scraping
```

### Optional Requirements
- **Google Gemini API Key** - For Star Nano Banana
- **Qwen Models** - For Qwen-specific nodes
- **Ollama** - For Ollama Sysprompter

---

## 🚀 Getting Started

### Installation

#### Via ComfyUI Manager (Recommended)
1. Open ComfyUI Manager
2. Search for "StarNodes"
3. Click Install/Update
4. Restart ComfyUI

#### Manual Installation
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Starnodes2024/ComfyUI_StarNodes
cd ComfyUI_StarNodes
pip install -r requirements.txt
# Restart ComfyUI
```

### Quick Setup
1. **Install dependencies**: `pip install -r requirements.txt`
2. **Configure Gemini** (optional): Edit `googleapi.ini` with your API key
3. **Restart ComfyUI**: All nodes appear under ⭐StarNodes categories
4. **Start creating**: Right-click → Add Node → ⭐StarNodes

---

## 📚 Learning Resources

### Documentation
- **README.md** - Complete node reference
- **QUICK_START_1.7.0.md** - Fast-track guide for new features
- **CHANGELOG.md** - Detailed version history
- **web/docs/** - Individual node documentation

### Guides
- **QwenEditPromptGuide.md** - Master Qwen image editing
- **README_StarQwenRegionalPrompter.md** - Regional prompting techniques
- **SIMPLIFIED_REGIONAL_PROMPTER_V2.md** - Beginner-friendly guide

### In-App Help
Right-click any node → Help for context-sensitive documentation

---

## 🎯 Use Cases

### Professional Workflows
- **Image Editing Studios** - Qwen suite for batch editing
- **AI Artists** - Gemini integration for creative generation
- **Content Creators** - Template system for consistent results
- **Photographers** - Color matching and depth-based effects

### Specific Applications
- **Style Transfer** - Using Qwen Regional Prompter
- **Color Grading** - With Simple Filters color matching
- **Batch Processing** - Using Save Folder String organization
- **Model Management** - Duplicate Finder for cleanup
- **AI Generation** - Gemini templates for quick results

---

## 🔄 Migration from 1.6.0

### Automatic
- All existing workflows continue to work
- New nodes available immediately
- No breaking changes

### Manual Steps (Optional)
1. Update `requirements.txt`: `pip install -r requirements.txt`
2. Configure Gemini API if using Star Nano Banana
3. Explore new nodes in ⭐StarNodes categories

---

## 🐛 Known Issues & Limitations

### Current Limitations
- **Gemini API**: Requires internet connection and API key
- **Color Matching**: Some methods require significant memory
- **Qwen Models**: Requires compatible Qwen checkpoints
- **Regional Prompter**: Learning curve for advanced features

### Workarounds
- All documented in node-specific help files
- See TROUBLESHOOTING section in QUICK_START_1.7.0.md

---

## 🙏 Acknowledgments

### Contributors
- StarNodes Team
- StarBetaNodes Testers
- ComfyUI Community

### Special Thanks
- Google for Gemini API
- Qwen team for amazing models
- ComfyUI developers
- All beta testers who provided feedback

---

## 📞 Support

### Getting Help
- **Documentation**: Check web/docs/ for node-specific help
- **GitHub Issues**: Report bugs and request features
- **Community**: ComfyUI Discord and forums

### Reporting Issues
When reporting issues, please include:
- ComfyUI version
- StarNodes version (1.7.0)
- Node name and category
- Error message (if any)
- Steps to reproduce

---

## 🔮 Future Plans

### Upcoming Features
- Additional Qwen model support
- More Gemini templates
- Enhanced regional prompting
- Performance optimizations
- More color matching algorithms

### Community Requests
We're listening! Submit feature requests on GitHub.

---

## 📄 License

See LICENSE file in repository.

---

## 🎊 Thank You!

Thank you for using StarNodes 1.7.0! We hope these new features enhance your ComfyUI workflows.

**Happy Creating! ⭐**

---

*For detailed technical changes, see CHANGELOG.md*  
*For quick start guide, see QUICK_START_1.7.0.md*  
*For migration details, see MIGRATION_1.7.0.md*
