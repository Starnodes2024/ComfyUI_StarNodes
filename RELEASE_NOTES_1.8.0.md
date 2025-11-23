# 🌟 StarNodes v1.8.0 Release Notes

**Release Date:** November 23, 2025

---

## 🎉 What's New

### Major Fixes
- **✅ Fixed Dynamic Input/Output Nodes** - All dynamic nodes now properly add/remove inputs when connecting/disconnecting:
  - ⭐ Star Image Input (Dynamic) - Now correctly targets `StarImageSwitch` and uses proper input naming
  - ⭐ Star Latent Input (Dynamic) - Fixed to target `StarLatentSwitch` with correct naming
  - ⭐ Star Grid Image Batcher - Fixed input removal logic to prevent index corruption
  - ⭐ Star Grid Captions Batcher - Fixed input removal logic
  - ⭐ Star PSD Saver (Dynamic) - Fixed layer/mask pair management

### Documentation Improvements
- **📚 Complete Documentation Coverage** - All nodes now have comprehensive documentation in `web/docs/`
- **📖 New Example Workflows Guide** - Added `EXAMPLE_WORKFLOWS.md` with 27 detailed workflow examples
- **✨ Enhanced Node Documentation** - Updated and completed missing documentation files

### Code Quality
- **🧹 Cleaned Up Codebase** - Removed obsolete and empty files
- **📝 Better Code Organization** - Improved file structure and naming consistency

---

## 🔧 Bug Fixes

### Dynamic Input Nodes
**Problem:** Dynamic input nodes (Image Input, Latent Input, Grid Batchers, PSD Saver) were not automatically adding/removing inputs when connections changed.

**Root Causes:**
1. Wrong node name targeting in JavaScript files
2. Mismatched input naming patterns between Python and JavaScript
3. Index corruption during input removal operations

**Solutions:**
- Updated JavaScript files to target correct node class names
- Fixed input naming patterns to match Python definitions
- Implemented proper index management for input removal
- Added reverse-order removal to prevent index shifting issues

**Impact:** All dynamic nodes now work as intended, providing a much better user experience.

---

## 📚 Documentation

### New Documentation Files
- `EXAMPLE_WORKFLOWS.md` - Comprehensive guide with 27 workflow examples
- `StarQwenRebalancePrompter.md` - Complete documentation for structured prompting
- `RELEASE_PREP_1.8.0.md` - Internal release preparation checklist

### Updated Documentation
- All node documentation files verified and updated
- Removed empty/placeholder documentation files
- Improved consistency across all docs

---

## 🗑️ Removed Files

### Obsolete Files Removed
- `star_advanced_enhancer.py` - Empty file, not in use
- `_temp_extract_nodes.py` - Temporary development script
- `StarAdvancedEnhancer.md` - Documentation for non-existent node

### Files Kept (Not Duplicates)
- `starupscale.py` - Simple latent upscaler (different from SD Upscale Refiner)
- `star_sd_upscale_refiner.py` - Advanced upscaling with refinement features
- Both serve different purposes and are actively used

---

## 📊 Node Statistics

- **Total Nodes:** 70+
- **Categories:** 13
- **Documentation Files:** 68
- **Example Workflows:** 27

---

## 🎯 Categories Overview

```
⭐StarNodes/
├── Conditioning (5 nodes)
├── Color (1 node)
├── Helpers And Tools (6 nodes)
├── Image And Latent (24 nodes)
├── Image Generation (1 node)
├── InfiniteYou (7 nodes)
├── IO (1 node)
├── Prompts (4 nodes)
├── Sampler (5 nodes)
├── Settings (2 nodes)
├── Text And Data (6 nodes)
├── Upscale (2 nodes)
└── Video (1 node)
```

---

## 🚀 Highlighted Workflows

### 1. Complete Generation Pipeline
```
FLUX/SDXL/SD3.5 Start → StarSampler (Unified) → VAE Decode → Save
```

### 2. Professional Upscaling
```
Load Image → Star SD Upscale Refiner → Save
```

### 3. Grid Composition
```
Multiple Images → Grid Image Batcher → Grid Composer → Save
```

### 4. Character Consistency
```
InfiniteYou Patch Loader → Generation Pipeline
```

### 5. Structured Prompting
```
Qwen Rebalance Prompter → Conditioning → Generation
```

---

## 📖 Documentation Resources

- **README.md** - Main documentation and feature list
- **EXAMPLE_WORKFLOWS.md** - 27 detailed workflow examples
- **web/docs/** - Individual node documentation (68 files)
- **CHANGELOG.md** - Complete version history
- **MIGRATION_1.7.0.md** - Migration guide from previous versions

---

## 🔄 Migration from 1.7.0

### Breaking Changes
**None** - This release is fully backward compatible

### Recommended Actions
1. **Update Workflows** - Dynamic input nodes will now work correctly
2. **Review Documentation** - Check new example workflows for ideas
3. **Clean Cache** - Clear browser cache to load updated JavaScript files

### Deprecated Nodes
- `⭐ StarSampler FLUX (DEPRECATED)` - Use StarSampler (Unified) instead
- `⭐ StarSampler SD (DEPRECATED)` - Use StarSampler (Unified) instead

---

## 🐛 Known Issues

### Minor Issues
- Some category names could be more user-friendly (planned for future release)
- InfiniteYou patch files require specific folder structure

### Workarounds
- All issues have documented workarounds in respective node documentation
- Check `web/docs/` for specific node troubleshooting

---

## 🎓 Getting Started

### For New Users
1. Read `README.md` for overview
2. Check `EXAMPLE_WORKFLOWS.md` for workflow ideas
3. Start with basic workflows (FLUX/SDXL Start → StarSampler)
4. Explore advanced features gradually

### For Existing Users
1. Clear browser cache to load updated JavaScript
2. Test dynamic input nodes (they now work correctly!)
3. Explore new example workflows
4. Check updated documentation for new tips

---

## 🙏 Acknowledgments

Thank you to all users who reported issues and provided feedback. Special thanks to the ComfyUI community for continued support.

---

## 📞 Support

- **Issues:** Report bugs on GitHub
- **Documentation:** Check `web/docs/` folder
- **Examples:** See `EXAMPLE_WORKFLOWS.md`
- **Community:** Share your workflows and creations

---

## 🔮 What's Next (v1.9.0 Preview)

Planned features for next release:
- Category reorganization for better UX
- Additional example workflow files
- Performance optimizations
- New utility nodes
- Enhanced documentation

---

## 📝 Full Changelog

See `CHANGELOG.md` for complete version history and detailed changes.

---

**Enjoy StarNodes v1.8.0!** ⭐

*If you find this extension useful, please star the repository and share your creations with the community!*
