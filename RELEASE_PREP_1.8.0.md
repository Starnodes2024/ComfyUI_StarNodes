# StarNodes v1.8.0 Release Preparation Checklist

## Status: IN PROGRESS

### 1. Documentation Review ✅ NEEDS ATTENTION

#### Missing or Empty Documentation Files:
- `StarQwenRebalancePrompter.md` - **0 bytes** (EMPTY - needs content)
- `StarAdvancedEnhancer.md` - **2 bytes** (EMPTY - needs content)
- Missing docs for:
  - `StarLatentSwitch2` (Star Latent Input 2 - Optimized)
  - `StarRandomImageLoader` (has doc but verify completeness)
  - `StarSavePanoramaJPEG` (has doc but verify completeness)
  - `StarApplyOverlayDepth` (has doc but verify completeness)

#### Documentation Files to Verify/Update:
- All InfiniteYou nodes (multiple patch loaders/combiners)
- Deprecated nodes (FLUX/SD StarSamplers) - add deprecation notice
- New unified StarSampler documentation

### 2. Category Optimization 🔄 NEEDS REVIEW

#### Current Categories:
```
⭐StarNodes/
├── Conditioning (5 nodes)
├── Color (1 node)
├── Helpers And Tools (6 nodes)
├── Image And Latent (24 nodes) ⚠️ TOO LARGE
├── Image Generation (1 node)
├── InfiniteYou (7 nodes)
├── IO (1 node)
├── Prompts (4 nodes)
├── Sampler (5 nodes)
├── Settings (2 nodes)
├── Text And Data (6 nodes)
├── Upscale (1 node)
└── Video (1 node)
```

#### Recommended Category Restructuring:
```
⭐StarNodes/
├── 🎨 Image/
│   ├── Loaders (Image Input, Image Loader 1by1, Random Image Loader, Face Loader)
│   ├── Processors (Grid Composer, Grid Batchers, Icon Exporter, Simple Filters)
│   ├── Savers (PSD Saver, PSD Saver 2, Save Panorama JPEG)
│   └── Utilities (Image2Latent, Palette Extractor, Frame From Video)
├── 📐 Latent/
│   ├── Generators (Qwen Image Ratio, Qwen WAN Ratio, Qwen Image Edit Inputs)
│   └── Processors (Latent Input, Latent Input 2)
├── 🎯 Sampling/
│   ├── Samplers (StarSampler Unified, FluxFill Inpainter)
│   ├── Settings (Load/Save Sampler Settings)
│   └── Deprecated (FLUX StarSampler, SD StarSampler)
├── 🔧 Conditioning/
│   ├── Encoders (Qwen Edit Encoder, QwenEdit+ Conditioner)
│   ├── Regional (Qwen Regional Prompter)
│   └── IO (Conditioning Loader/Saver)
├── 📝 Prompts & Text/
│   ├── Generators (Ollama Sysprompter, Qwen Rebalance Prompter, Image Edit Qwen/Kontext)
│   ├── Wildcards (Seven Wildcards, Wildcards Advanced)
│   ├── Utilities (Text Filter, Seven Inputs, Easy-Text-Storage)
│   └── Scrapers (Web Scraper Headlines)
├── 🎭 InfiniteYou/
│   ├── Patch Loaders
│   ├── Patch Combiners
│   ├── Face Swap
│   └── Patch Maker/Saver
├── ⚙️ Utilities/
│   ├── Aspect Ratios (Aspect Ratio, Aspect Ratio Advanced, Aspect Video Ratio)
│   ├── Helpers (Denoise Slider, Divisible Dimension, Show Last Frame, Duplicate Model Finder)
│   └── IO (Save Folder String)
├── 🚀 Upscale/
│   ├── Model Latent Upscaler
│   └── SD Upscale Refiner
├── 🎬 Starters/
│   ├── FLUX Start
│   ├── SDXL Start
│   ├── SD3.5 Start
│   └── Qwen Image Start
└── 🤖 AI Generation/
    └── Nano Banana (Gemini)
```

### 3. Version Updates 📝 PENDING

Files that need version update to 1.8.0:
- [ ] `__init__.py` (line 215: `__version__ = "1.7.0"`)
- [ ] `pyproject.toml` (if exists)
- [ ] `README.md` (version badge/mention)
- [ ] `CHANGELOG.md` (add 1.8.0 entry)
- [ ] Create `RELEASE_NOTES_1.8.0.md`

### 4. Files to Remove 🗑️ PENDING

#### Development/Debug Files:
- `_temp_extract_nodes.py` (temporary script)
- `_nodes_list.txt` (if exists)
- `.tracking` (tracking file)
- `__pycache__/` (Python cache - should be in .gitignore)

#### Obsolete/Duplicate Files:
- `star_infiniteyou_patch_fixed.py` (if obsolete)
- `star_infiniteyou_patch_modified.py` (if obsolete)
- `star_infiniteyou_apply.py` (if exists and obsolete)
- `star_infiniteyou_face_swap.py` (if replaced by _mod version)
- `star_advanced_enhancer.py` (2 bytes - empty)
- `divisibledimensions.py` (if duplicate of StarDivisibleDimension.py)

#### Configuration Files (User shouldn't see):
- `googleapi.ini` (should be .gitignore or template)
- `ollamamodels.txt` (should be in docs or .gitignore)
- `sites.txt` (should be in docs or .gitignore)

### 5. Example Workflows 📚 PENDING

Create `EXAMPLE_WORKFLOWS.md` with:

#### Recommended Example Workflows:
1. **Basic Image Generation**
   - FLUX Start → StarSampler → Save Image
   - SDXL Start → StarSampler → Save Image

2. **Advanced Upscaling**
   - Load Image → Star SD Upscale Refiner → Save

3. **Grid Composition**
   - Multiple Images → Star Grid Image Batcher → Star Grid Composer → Save
   - With Captions: + Star Grid Captions Batcher

4. **InfiniteYou Character Consistency**
   - Star InfiniteYou Patch Loader → Apply to generation
   - Face Swap workflow

5. **Qwen Image Editing**
   - Qwen Image Start → Qwen Edit Encoder → Generation
   - Regional Prompting with Qwen Regional Prompter

6. **Wildcard Prompting**
   - Star Seven Wildcards → Text to conditioning
   - Star Wildcards Advanced for complex prompts

7. **PSD Layer Export**
   - Multiple generations → Star PSD Saver → Photoshop editing

8. **Dynamic Prompting with Ollama**
   - Star Ollama Sysprompter → Enhanced prompts
   - Star Qwen Rebalance Prompter for composition

9. **Aspect Ratio Workflows**
   - Aspect Ratio nodes → Proper sizing for different models

10. **FluxFill Inpainting**
    - Star FluxFill Inpainter → Inpainting workflow

### 6. Additional Release Tasks 📋

#### Pre-Release Checklist:
- [ ] Run tests on all dynamic input nodes (fixed in this release!)
- [ ] Verify all dependencies in `requirements.txt`
- [ ] Update `README.md` with new features
- [ ] Create migration guide if needed
- [ ] Check all import statements
- [ ] Verify web/js files are properly loaded
- [ ] Test on fresh ComfyUI installation
- [ ] Create release tag in git
- [ ] Prepare release announcement

#### What's New in 1.8.0:
- ✅ Fixed dynamic input/output nodes (Star Image Input, Star Latent Input, Grid Batchers, PSD Saver)
- 🆕 [Add other new features here]
- 🔧 [Add improvements here]
- 🐛 [Add bug fixes here]

#### Known Issues:
- Deprecated nodes (FLUX/SD StarSamplers) - users should migrate to Unified StarSampler
- [Add any other known issues]

---

## Next Steps:
1. Review and complete missing documentation
2. Implement category restructuring (if approved)
3. Update version numbers
4. Remove obsolete files
5. Create example workflows documentation
6. Final testing
7. Create release notes
8. Tag release
