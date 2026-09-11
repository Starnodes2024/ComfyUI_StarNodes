# StarNodes - Complete Node Inventory (v2.5.0 Cleanup)

Generated: 2026-07-31
Updated: 2026-07-31 (cleanup step 2 - user-marked node removals applied)

## Summary

- **Total registered nodes**: 96 (was 111; 15 user-marked nodes removed)
- **Source files with registered nodes**: 72
- **Step 1 deleted unregistered files**: 6
- **Step 1 deleted orphaned files**: 1
- **Step 1 deleted deprecated samplers**: 2
- **Step 2 deleted Python files**: 14
- **Step 2 edited Python files**: 1 (star_sd_upscale_refiner.py - removed basic class, kept Advanced)
- **Step 2 deleted JSON files**: 3 (sd3ratios.json x2, styles.json)
- **Step 2 deleted JS files**: 6
- **Step 2 deleted docs (.md) files**: 33
- **Step 2 deleted example workflow**: 1
- **Helper files** (no nodes, used by other modules): 9

---

## Registered Nodes (imported in `__init__.py`)

### Samplers

| # | Node Name | Display Name | Source File |
|---|-----------|-------------|------------|
| 1 | `FluxStartSettings` | ⭐ Star FLUX Start(t) Settings | `samplers/FluxStart.py` |
| 2 | `SDXLStartSettings` | ⭐ Star SD(XL) Start(t) Settings | `samplers/SDXLStart.py` |
| 4 | `QwenImageStartSettings` | ⭐ Star Qwen Image Start(t) Settings | `qwen/QwenImageStart.py` |
| 5 | `StarSampler` | ⭐ StarSampler (Unified) | `samplers/star_sampler.py` |
| 6 | `StarSplitSamplerInfo` | ⭐ Star Split Sampler Info | `samplers/star_split_sampler_info.py` |
| 7 | `StarSaveSamplerSettings` | ⭐ Star Save Sampler Settings | `samplers/starsamplersettings_nodes.py` |
| 8 | `StarLoadSamplerSettings` | ⭐ Star Load Sampler Settings | `samplers/starsamplersettings_nodes.py` |
| 9 | `StarDeleteSamplerSettings` | ⭐ Star Delete Sampler Settings | `samplers/starsamplersettings_nodes.py` |

### Image Tools

| # | Node Name | Display Name | Source File |
|---|-----------|-------------|------------|
| 13 | `Starnodes_Aspect_Ratio_Advanced` | ⭐ Starnodes Aspect Ratio Advanced | `image_tools/aspect_ratio_advanced.py` |
| 15 | `StarDivisibleDimension` | ⭐ Star Divisible Dimension | `image_tools/StarDivisibleDimension.py` |
| 16 | `Starupscale` | ⭐ Star Model Latent Upscaler | `image_tools/starupscale.py` |
| 19 | `Star_Image2Latent` | ⭐ Star Image2Latent | `image_tools/StarImage2Latent.py` |
| 20 | `StarImageShifter` | ⭐ Star Image Shifter | `image_tools/StarImageShifter.py` |
| 21 | `StarBoxDrawer` | ⭐ Star Box Drawer | `image_tools/StarBoxDrawer.py` |
| 22 | `AdaptiveDetailEnhancement` | ⭐ Star Adaptive Detail Enhancement | `image_tools/star_detailenhancer.py` |
| 23 | `StarApplyOverlayDepth` | ⭐ Star Apply Overlay (Depth) | `image_tools/star_apply_overlay_depth.py` |
| 24 | `StarBlackWhite` | ⭐ Star Black & White | `image_tools/star_black_white.py` |
| 25 | `StarFrameFromVideo` | ⭐ Star Frame From Video | `image_tools/star_frame_from_video.py` |
| 26 | `StarHDREffects` | ⭐ Star HDR Effects | `image_tools/star_hdr_effects.py` |
| 27 | `StarSpecialFilters` | ⭐ Star HighPass Filters | `image_tools/star_highpass.py` |
| 28 | `StarImageCompare` | ⭐ Star Image Compare | `image_tools/star_image_compare.py` |
| 29 | `StarImageLoader1by1` | ⭐ Star Image Loader 1by1 | `image_tools/star_image_loader_1by1.py` |
| 30 | `StarImageLoaderOptions` | ⭐ Star Image Loader Options | `image_tools/star_image_loader_options.py` |
| 31 | `StarImageLoop` | ⭐ Star Image Loop | `image_tools/star_image_loop.py` |
| 32 | `StarKrea2Unbound` | ⭐ Star Krea2 Unbound | `image_tools/star_krea2_unbound.py` |
| 33 | `StarAdvanvesRatioLatent` | ⭐ Star Advanved Ratio/Latent | `image_tools/star_latent_ratio_megapixel.py` |
| 34 | `StarLatentResize` | ⭐ Star Latent Resize | `image_tools/star_latent_resize.py` |
| 35 | `StarLoadImagePlus` | ⭐ Star Load Image+ | `image_tools/star_load_image_plus.py` |
| 36 | `⭐ Star Save Image+` | ⭐ Star Save Image+ | `image_tools/star_save_image_plus.py` |
| 37 | `StarLucidaRMBG` | ⭐ Star Lucida RMBG | `image_tools/star_lucida_rmbg.py` |
| 38 | `StarMetadataSaverOption` | ⭐ Star Metadata Saver Option | `image_tools/star_metadata_saver_option.py` |
| 39 | `StarMetaInjector` | ⭐ Star Meta Injector | `image_tools/star_meta_injector.py` |
| 40 | `StarOllamaPromptHelper` | ⭐ Star Ollama Prompt Helper | `image_tools/star_ollama_prompt_helper.py` (optional, try/except) |
| 41 | `StarPanoramaViewer` | ⭐ Star 360 Parallax Viewer | `image_tools/star_panorama_viewer.py` |
| 42 | `StarPanoramaViewerPro` | ⭐ Star 360 Parallax Viewer Pro | `image_tools/star_panorama_viewer_pro.py` |
| 43 | `StarRadialBlur` | ⭐ Star Radial Blur | `image_tools/star_radial_blur.py` |
| 44 | `StarRandomImageLoader` | ⭐ Star Random Image Loader | `image_tools/star_random_image_loader.py` |
| 45 | `StarRealisticFilmGrain` | ⭐ Star Realistic Film Grain | `image_tools/star_realistic_film_grain.py` |
| 47 | `StarSDUpscaleRefinerAdvanced` | ⭐ Star SD Upscale Refiner Advanced | `image_tools/star_sd_upscale_refiner.py` |
| 48 | `StarSimpleFilters` | ⭐ Star Simple Filters | `image_tools/star_simple_filters.py` |
| 49 | `Star_Size_Calculator_By_Side` | ⭐ Star Size Calculator by Side | `image_tools/star_size_calculator_by_side.py` (optional, try/except) |
| 50 | `StarTiledPiDUpscaler` | ⭐ Star Tiled PiD Upscaler | `image_tools/star_tiled_pid_upscaler.py` |
| 51 | `StarTiledSeedVRUpscaler` | ⭐ Star Tiled SeedVR Upscaler | `image_tools/star_tiled_seedvr_upscaler.py` |
| 52 | `StarVideoLoop` | ⭐ Star Video Loop | `image_tools/star_video_loop.py` |

### Text I/O

| # | Node Name | Display Name | Source File |
|---|-----------|-------------|------------|
| 55 | `StarTextInput` | ⭐ Star Text Inputs (Concatenate) | `text_io/startextinput.py` |
| 56 | `StarTextFilter` | ⭐ Star Text Filter | `text_io/startextfilter.py` |
| 57 | `StarEasyTextStorage` | ⭐ Star Easy-Text-Storage | `text_io/startextstorage.py` |
| 58 | `StarDenoiseSlider` | ⭐ Star Denoise Slider | `text_io/StarDenoiseSlider.py` |
| 59 | `StarPaletteExtractor` | ⭐ Star Palette Extractor | `text_io/StarPaletteExtractor.py` (conditional) |
| 62 | `StarPSDSaverAdvLayers` | ⭐ Star PSD Saver Adv. Layers | `text_io/StarPSDSaverAdvLayers.py` (conditional) |
| 63 | `StarWatermark` | ⭐ Star Watermark | `text_io/starwatermark.py` |
| 64 | `StarConditioningSaver` | ⭐ Star Conditioning Saver | `text_io/star_conditioning_io.py` |
| 65 | `StarConditioningLoader` | ⭐ Star Conditioning Loader | `text_io/star_conditioning_io.py` |
| 66 | `Star_Show_Last_Frame` | ⭐ Star Show Last Frame | `text_io/StarShowLastFrame.py` |
| 67 | `StarIconExporter` | ⭐ Star Icon Exporter | `text_io/star_icon_exporter.py` |
| 68 | `StarPromptPicker` | ⭐ Star Prompt Picker | `text_io/star_prompt_picker.py` |
| 69 | `StarSaveFolderString` | ⭐ Star Save Folder String | `text_io/star_save_folder_string.py` |
| 70 | `StarSavePanoramaJPEG` | ⭐ Star Save Panorama JPEG | `text_io/star_save_panorama_jpeg.py` |
| 71 | `StarSavePanoramaJPEGPlus` | ⭐ Star Save Panorama JPG+ | `text_io/star_save_panorama_jpeg_plus.py` |
| 72 | `StarDuplicateModelFinder` | ⭐ Star Duplicate Model Finder | `text_io/star_duplicate_model_finder.py` |

### Misc

| # | Node Name | Display Name | Source File |
|---|-----------|-------------|------------|
| 74 | `StarWildcardsAdvanced` | ⭐ Star Wildcards Advanced | `misc/starwildsadv.py` |
| 75 | `Star3LoRAs` | ⭐ Star 3 LoRAs | `misc/StarLora.py` |
| 76 | `StarDynamicLora` | ⭐ Star Dynamic LoRA | `misc/star_dynamic_lora.py` |
| 77 | `StarDynamicLoraModelOnly` | ⭐ Star Dynamic LoRA (Model Only) | `misc/star_dynamic_lora.py` |
| 78 | `StarLoraWeightNormalizer` | ⭐ Star Dynamic LoRA Weight | `misc/star_lora_weight_normalizer.py` |
| 79 | `StarFluxFiller` / `FluxFillSampler` | ⭐ Star FluxFill Inpainter | `misc/StarFluxFiller.py` |
| 80 | `StarFlux2Inpainter` | ⭐ Star Flux2/Qwen-Image-Edit Inpainter | `misc/StarFlux2Inpainter.py` |
| 81 | `DetailStarDaemon` | ⭐ Star Detail Daemon | `misc/detailstardaemon.py` |
| 82 | `StarDistilledOptimizerZIT` | ⭐ Star Distilled Optimizer (QWEN/ZIT) | `misc/star_distilled_optimizer_zit.py` |
| 83 | `StarFP8Converter` | ⭐ Star FP8 Converter | `misc/star_fp8_converter.py` |
| 84 | `StarModelPacker` | ⭐ Star Model Packer | `misc/star_model_packer.py` |
| 85 | `StarStopAndGo` | ⭐ Star Stop And Go | `misc/star_stop_and_go.py` |
| 86 | `StarShowEverything` | ⭐ Star Show Everything | `misc/star_show_everything.py` |
| 87 | `StarOutputCleaner` | ⭐ Star Output Cleaner | `misc/star_output_cleaner.py` (optional, try/except) |

### Qwen

| # | Node Name | Display Name | Source File |
|---|-----------|-------------|------------|
| 90 | `StarQwenImageEditInputs` | ⭐ Star Qwen Image Edit Inputs | `qwen/star_qwen_image_edit_inputs.py` |
| 91 | `StarQwenEditEncoder` | ⭐ Star Qwen Edit Encoder | `qwen/star_qwen_edit_encoder.py` |
| 92 | `StarImageEditQwenKontext` | ⭐ Star Image Edit for Qwen/Kontext | `qwen/star_image_edit_qwen_kontext.py` |
| 93 | `StarQwenEditPlusConditioner` | ⭐ Star QwenEdit+ Conditioner | `qwen/star_qwen_edit_plus_conditioner.py` |
| 94 | `StarQwenRebalancePrompter` | ⭐ Star Qwen-Rebalance-Prompter | `qwen/star_qwen_rebalance_prompter.py` |
| 95 | `StarQwenRegionalPrompter` | ⭐ Star Qwen Regional Prompter | `qwen/star_qwen_regional_prompter.py` |

### Grid

| # | Node Name | Display Name | Source File |
|---|-----------|-------------|------------|
| 96 | `StarGridComposer` | ⭐ Star Grid Composer | `grid/stargridcomposer.py` |
| 97 | `StarGridImageBatcher` | ⭐ Star Grid Image Batcher | `grid/stargridbatchers.py` |
| 98 | `StarGridCaptionsBatcher` | ⭐ Star Grid Captions Batcher | `grid/stargridbatchers.py` |

### External

| # | Node Name | Display Name | Source File |
|---|-----------|-------------|------------|
| 99 | `StarNewsScraper` | ⭐ Star News Scraper | `external/StarNewsScraper.py` |

### LTX Video

| # | Node Name | Display Name | Source File |
|---|-----------|-------------|------------|
| 101 | `StarVAE_LTXV_Save` | ⭐ Star VAE LTXV Save | `ltx_video/starvae_ltxv_save.py` |
| 102 | `StarVAE_LTXV_Load` | ⭐ Star VAE LTXV Load | `ltx_video/starvae_ltxv_load.py` |
| 103 | `StarLTXVideoSettings` | ⭐ Star LTX Video Settings | `ltx_video/star_ltx_video_settings.py` |
| 104 | `LTXImageCut` | ⭐ Star LTX Image Cut | `ltx_video/ltx_image_cut.py` |
| 105 | `StarMultiInputsToOne` | ⭐ Star Multi Inputs To One | `ltx_video/star_multi_inputs_to_one.py` |
| 106 | `StarLTXVGetLastFrame` | ⭐ Star LTXV Get Last Frame | `ltx_video/star_ltxv_get_last_frame.py` |
| 107 | `StarLTXVLoadLastImage` | ⭐ Star LTXV Load Last Image From Folder | `ltx_video/star_ltxv_load_last_image.py` |
| 108 | `StarVideoJoiner` | ⭐ Star Video Joiner | `ltx_video/star_video_joiner.py` |
| 109 | `LTXVSulphurAllInOne` | ⭐ Star LTXV All-in-One (2-Pass) | `ltx_video/ltxv_sulphur_aio.py` |

### Video Tools

| # | Node Name | Display Name | Source File |
|---|-----------|-------------|------------|
| 111 | `StarVideoCompressor` | ⭐ Star Video Compressor | `video_tools/star_video_compressor.py` |
| 112 | `StarVideoLoader` | ⭐ Star Video Loader | `video_tools/star_video_loader.py` |
| 113 | `StarSlideshowMaker` | ⭐ Star Slideshow Maker | `video_tools/star_slideshow_maker.py` |

### Root-level

| # | Node Name | Display Name | Source File |
|---|-----------|-------------|------------|
| 114 | `StarFlux2Conditioner` | ⭐ Star Flux2 Conditioner | `star_flux2_conditioner.py` |

---

## Unregistered Files With Nodes (NOT imported in `__init__.py`) — DELETED

*All unregistered files have been deleted in step 1. The table below is kept for historical reference.*

| # | File | Node Name(s) | Display Name | Notes |
|---|------|-------------|-------------|-------|
| 1 | `external/ollamahelper.py` | `OllamaModelChooser` | ⭐ Star Ollama Helper | Not imported anywhere. Dead code. |
| 2 | `misc/star_flux2_condition.py` | `StarFlux2Condition` | ⭐ Star Flux2 Conditioner | Duplicate of `star_flux2_conditioner.py` at root. Not imported anywhere. Dead code. |
| 3 | `misc/star_fp8_model_patch_loader.py` | `StarFP8ModelPatchLoader` | ⭐ Star FP8 Model Patch Loader | Not imported anywhere. Dead code. |
| 4 | `music/examples.py` | `VHS_VideoCombine`, `VHS_LoadVideo`, etc. (50+ VHS nodes) | VHS Video Combine etc. | These are references to ComfyUI-VideoHelperSuite nodes, not actual StarNodes. Not imported anywhere. Dead code. |
| 5 | `samplers/star_face_detailer.py` | `StarFaceDetailerPlus` | ⭐ Star Face Detailer+ | Not imported anywhere. Dead code. |
| 6 | `text_io/star_everything_to_int_str.py` | `StarEverythingToIntStr` | ⭐ Star Everything to INT/STR | Not imported anywhere. Dead code. |

---

## Orphaned/Unused Files (no nodes, not imported by any other file)

*All orphaned files have been deleted in step 1.*

---

## Helper Files (no nodes, used by other modules - DO NOT DELETE)

| # | File | Used By |
|---|------|---------|
| 1 | `image_tools/divisibledimensions.py` | `image_tools/StarDivisibleDimension.py` |
| 2 | `image_tools/metadata_utils.py` | `image_tools/star_load_image_plus.py`, `image_tools/star_save_image_plus.py` |
| 3 | `samplers/starsamplersettings.py` | `samplers/starsamplersettings_nodes.py` |
| 4 | `star_progress.py` | `image_tools/star_panorama_viewer_pro.py`, `image_tools/star_sd_upscale_refiner.py`, `image_tools/star_tiled_pid_upscaler.py`, `image_tools/star_tiled_seedvr_upscaler.py`, `ltx_video/ltxv_sulphur_aio.py`, `samplers/star_sampler.py` |
| 5 | `video_tools/star_nodes_common.py` | `video_tools/star_video_compressor.py`, `video_tools/star_video_loader.py` |
| 6 | `lucida/birefnet.py` | `lucida/pipeline.py`, `image_tools/star_lucida_rmbg.py` (indirectly) |
| 7 | `lucida/_compat.py` | `lucida/birefnet.py` |
| 8 | `lucida/BiRefNet_config.py` | `lucida/birefnet.py`, `lucida/pipeline.py` |
| 9 | `lucida/pipeline.py` | `image_tools/star_lucida_rmbg.py` |

---

## Cleanup Actions Applied (Step 1)

### Deleted unregistered dead-code files (6):
- `external/ollamahelper.py`
- `misc/star_flux2_condition.py` (duplicate of `star_flux2_conditioner.py`)
- `misc/star_fp8_model_patch_loader.py`
- `music/examples.py` (VHS references, not real nodes)
- `samplers/star_face_detailer.py`
- `text_io/star_everything_to_int_str.py`

### Deleted orphaned file (1):
- `external/headline_scraper.py`

### Deleted deprecated samplers (2):
- `samplers/fluxstarsampler.py` (Fluxstarsampler - DEPRECATED)
- `samplers/sdstarsampler.py` (SDstarsampler - DEPRECATED)
- Removed corresponding imports and NODE_CLASS_MAPPINGS/NODE_DISPLAY_NAME_MAPPINGS entries from `__init__.py`
- Removed dead imports of these samplers from `starsamplersettings_nodes.py`

### Step 2: User-marked node removals (15 nodes, 14 files deleted, 1 file edited)

**Deleted Python files (14):**
- `samplers/SD35Start.py` (SD35StartSettings)
- `image_tools/aspect_ratio.py` (Starnodes_Aspect_Ratio)
- `image_tools/StarAspectVideoRatio.py` (Starnodes_Aspect_Video_Ratio)
- `image_tools/starlatentinput.py` (StarLatentSwitch)
- `image_tools/StarLatentInput2.py` (StarLatentSwitch2)
- `text_io/StarNode.py` (StarImageSwitch)
- `text_io/StarNode2.py` (StarImageSwitch2)
- `text_io/StarPSDSaver.py` (StarPSDSaver)
- `text_io/StarPSDSaver2.py` (StarPSDSaver2)
- `misc/starwilds.py` (StarFiveWildcards)
- `qwen/star_qwen_image_ratio.py` (StarQwenImageRatio)
- `qwen/star_qwen_wan_ratio.py` (StarQwenWanRatio)
- `external/star_ollama_sysprompter_jc.py` (StarOllamaSysprompterJC)
- `music/acestep_node.py` (ACEStepMusicGenerator)

**Edited Python file (1):**
- `image_tools/star_sd_upscale_refiner.py` - removed StarSDUpscaleRefiner class, kept StarSDUpscaleRefinerAdvanced

**Deleted JSON files (3):**
- `sd3ratios.json` (root)
- `samplers/sd3ratios.json`
- `styles.json`

**Deleted JS files (6):**
- `web/js/star_latent_input_dynamic.js`
- `web/js/star_image_input_dynamic.js`
- `web/js/star_psd_saver_dynamic.js`
- `web/js/star_face_detailer.js` (already orphaned from step 1)
- `web/js/starnodes_appearance.bak`
- `web/js/zz_star_save_image_plus_suggestions.js.txt`

**Deleted docs (.md) files (33):**
- 14 docs for step 2 removed nodes
- 7 docs for step 1 removed nodes (Fluxstarsampler, SDstarsampler, OllamaModelChooser, etc.)
- 12 docs for non-existent/never-registered nodes (StarApplyInfiniteYou, StarGeminiRefiner, etc.)

**Deleted example workflow (1):**
- `example_workflows/Starnodes AceStep Auto Songwriter.json`

**Updated files:**
- `__init__.py` - removed 14 import lines and corresponding NODE_CLASS_MAPPINGS/NODE_DISPLAY_NAME_MAPPINGS entries
- `music/__init__.py` - removed acestep_node import

### Kept as-is (per user request):
- Node name inconsistency in `star_save_image_plus.py` (emoji in node key) - unchanged
