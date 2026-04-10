# StarNodes v2.0.0 - Release Notes

## 🎉 What's New

### New Nodes (9 total)

#### Music Generation
- **Star ACE Step Music Generator** - Professional music generation with ACE Step 1.5 API

#### LTX Video Suite (8 nodes)
- Star LTX Video Settings
- Star VAE LTXV Save
- Star VAE LTXV Load
- Star LTX Image Cut
- Star Multi Inputs to One
- Star LTXV Get Last Frame
- Star LTXV Load Last Image
- Star Video Joiner

## 🗑️ Removed Nodes (9 total)

### InfiniteYou Suite (4 nodes)
- StarInfiniteYou
- StarInfiniteYouFaceSwapMod
- StarInfiniteYouPatch
- StarInfiniteYouAdvancedPatchMaker

**Reason:** InsightFace dependency conflicts

### Other Removals (5 nodes)
- StarFaceLoader (deprecated)
- StarGeminiRefiner (external API reduction)
- StarFlowmatchOption (experimental/deprecated)
- StarNanoBanana (external API reduction)

## 🔧 Improvements

- Fixed duplicate import bugs
- Cleaned up dependencies
- Improved code organization
- Better error handling
- Updated documentation

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

1. Copy this folder to `ComfyUI/custom_nodes/comfyui_starnodes/`
2. Install dependencies
3. Restart ComfyUI
4. Find nodes under "StarNodes" category

## 📚 Documentation

- **RELEASE_SUMMARY_2.0.0.md** - Complete release documentation
- **NODES_LIST_V2.md** - All 86 nodes listed by category
- **README.md** - Main project documentation
- **CHANGELOG.md** - Version history

## ⚠️ Breaking Changes

If you used any removed nodes, please check RELEASE_SUMMARY_2.0.0.md for migration alternatives.

---

**Version:** 2.0.0  
**Release Date:** April 10, 2026  
**Total Nodes:** 86
