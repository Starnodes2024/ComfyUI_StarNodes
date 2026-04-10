# StarNodes - Complete Node List (Version 1.9.9)

## Sampler Nodes
1. **FluxStart** - `samplers/FluxStart.py`
2. **SDXLStart** - `samplers/SDXLStart.py`
3. **SD35Start** - `samplers/SD35Start.py`
4. **FluxStarSampler** - `samplers/fluxstarsampler.py`
5. **SDStarSampler** - `samplers/sdstarsampler.py`
6. **StarSampler** - `samplers/star_sampler.py`
7. **StarSamplerSettings** - `samplers/starsamplersettings_nodes.py`

## Qwen Image Nodes
8. **QwenImageStart** - `qwen/QwenImageStart.py`
9. **StarQwenImageRatio** - `qwen/star_qwen_image_ratio.py`
10. **StarQwenWanRatio** - `qwen/star_qwen_wan_ratio.py`
11. **StarQwenImageEditInputs** - `qwen/star_qwen_image_edit_inputs.py`
12. **StarQwenEditEncoder** - `qwen/star_qwen_edit_encoder.py`
13. **StarImageEditQwenKontext** - `qwen/star_image_edit_qwen_kontext.py`
14. **StarQwenEditPlusConditioner** - `qwen/star_qwen_edit_plus_conditioner.py`
15. **StarQwenRebalancePrompter** - `qwen/star_qwen_rebalance_prompter.py`
16. **StarQwenRegionalPrompter** - `qwen/star_qwen_regional_prompter.py`

## Image Tools
17. **StarUpscale** - `image_tools/starupscale.py`
18. **AspectRatio** - `image_tools/aspect_ratio.py`
19. **AspectRatioAdvanced** - `image_tools/aspect_ratio_advanced.py`
20. **StarLatentInput** - `image_tools/starlatentinput.py`
21. **StarLatentInput2** - `image_tools/StarLatentInput2.py`
22. **StarImage2Latent** - `image_tools/StarImage2Latent.py`
23. **StarDivisibleDimension** - `image_tools/StarDivisibleDimension.py`
24. **StarDetailEnhancer** (AdaptiveDetailEnhancement) - `image_tools/star_detailenhancer.py`
25. **StarRandomImageLoader** - `image_tools/star_random_image_loader.py`
26. **StarFrameFromVideo** - `image_tools/star_frame_from_video.py`
27. **StarImageLoader1by1** - `image_tools/star_image_loader_1by1.py`
28. **StarLatentRatioMegapixel** - `image_tools/star_latent_ratio_megapixel.py`
29. **StarLatentResize** - `image_tools/star_latent_resize.py`
30. **StarApplyOverlayDepth** - `image_tools/star_apply_overlay_depth.py`
31. **StarAspectVideoRatio** - `image_tools/StarAspectVideoRatio.py`
32. **StarSimpleFilters** - `image_tools/star_simple_filters.py`
33. **StarRadialBlur** - `image_tools/star_radial_blur.py`
34. **StarBlackWhite** - `image_tools/star_black_white.py`
35. **StarHDREffects** - `image_tools/star_hdr_effects.py`
36. **StarSDUpscaleRefiner** - `image_tools/star_sd_upscale_refiner.py`
37. **StarHighpass** (Special Filters) - `image_tools/star_highpass.py`
38. **StarMetaInjector** - `image_tools/star_meta_injector.py`
39. **StarImageLoop** - `image_tools/star_image_loop.py`
40. **StarVideoLoop** - `image_tools/star_video_loop.py`
41. **StarSaveImagePlus** - `image_tools/star_save_image_plus.py`
42. **StarLoadImagePlus** - `image_tools/star_load_image_plus.py`
43. **StarSizeCalculatorBySide** - `image_tools/star_size_calculator_by_side.py` (conditional import)

## Text I/O Nodes
44. **StarNode** - `text_io/StarNode.py`
45. **StarNode2** - `text_io/StarNode2.py`
46. **StarTextFilter** - `text_io/startextfilter.py`
47. **StarTextInput** - `text_io/startextinput.py`
48. **StarTextStorage** - `text_io/startextstorage.py`
49. **StarPSDSaver** - `text_io/StarPSDSaver.py`
50. **StarPSDSaver2** - `text_io/StarPSDSaver2.py`
51. **StarPSDSaverAdvLayers** - `text_io/StarPSDSaverAdvLayers.py`
52. **StarWatermark** - `text_io/starwatermark.py`
53. **StarDenoiseSlider** - `text_io/StarDenoiseSlider.py`
54. **StarPaletteExtractor** - `text_io/StarPaletteExtractor.py`
55. **StarShowLastFrame** - `text_io/StarShowLastFrame.py`
56. **StarSavePanoramaJpeg** - `text_io/star_save_panorama_jpeg.py`
57. **StarIconExporter** - `text_io/star_icon_exporter.py`
58. **StarSaveFolderString** - `text_io/star_save_folder_string.py`
59. **StarDuplicateModelFinder** - `text_io/star_duplicate_model_finder.py`
60. **StarPromptPicker** - `text_io/star_prompt_picker.py`

## Conditioning I/O
61. **StarConditioningIO** - `text_io/star_conditioning_io.py`

## Misc Nodes
62. **StarWilds** - `misc/starwilds.py`
63. **StarWildsAdv** - `misc/starwildsadv.py`
64. **StarFluxFiller** - `misc/StarFluxFiller.py`
65. **DetailStarDaemon** - `misc/detailstardaemon.py`
remove: 66. **StarFaceLoader** - `misc/starfaceloader.py`
67. **StarLora** - `misc/StarLora.py`
68. **StarDynamicLora** - `misc/star_dynamic_lora.py`
69. **StarLoraWeightNormalizer** - `misc/star_lora_weight_normalizer.py`
remove: 70. **StarGeminiRefiner** - `misc/star_gemini_refiner.py`
71. **StarFP8Converter** - `misc/star_fp8_converter.py`
remove: 72. **StarFlowmatchOption** - `misc/star_flowmatch_option.py`
73. **StarDistilledOptimizerZit** - `misc/star_distilled_optimizer_zit.py`
74. **StarModelPacker** - `misc/star_model_packer.py`
75. **StarStopAndGo** - `misc/star_stop_and_go.py`

## Grid Nodes
76. **StarGridComposer** - `grid/stargridcomposer.py`
77. **StarGridBatchers** - `grid/stargridbatchers.py`

## External API Nodes
78. **StarNewsScraper** - `external/StarNewsScraper.py`
79. **StarOllamaSysprompterJC** - `external/star_ollama_sysprompter_jc.py`
remove: 80. **StarNanoBanana** - `external/star_nano_banana.py`

## InfiniteYou Nodes (InsightFace dependent - conditional imports)
remove: 81. **StarInfiniteYou** - `infiniteyou/starinfiniteyou.py`
remove: 82. **StarInfiniteYouFaceSwapMod** - `infiniteyou/star_infiniteyou_face_swap_mod.py`
remove: 83. **StarInfiniteYouPatch** - `infiniteyou/star_infiniteyou_patch.py`
remove: 84. **StarInfiniteYouAdvancedPatchMaker** - `infiniteyou/star_infiniteyou_advanced_patch_maker.py`

## Music Nodes
86. **StarACEStepMusicGenerator** - music/acestep_node.py

## Flux2 Conditioner
85. **StarFlux2Conditioner** - `star_flux2_conditioner.py`

---

## Notes for Version 2.0 Cleanup

### Conditional Imports (May fail if dependencies missing):
remove: - **InsightFace nodes** (81-84): Require `insightface` library
 **StarSizeCalculatorBySide** (43): Has try/except wrapper

### Duplicate Imports Found:
remove the double import: - Lines 75-78 in __init__.py: `StarShowLastFrame` and `StarAspectVideoRatio` imported twice

### Files in Directory Not in __init__.py:
These files exist but are NOT registered in NODE_CLASS_MAPPINGS:
- `aspect_ratio.py` (root level - old version?)
- `aspect_ratio_advanced.py` (root level - old version?)
- `divisibledimensions.py`
- `headline_scraper.py`
remove: - `ollamahelper.py`
remove: - `starinfiniteyou_core.py` (utility file)
remove: - `starinfiniteyou_resampler.py`
remove: - `starinfiniteyou_utils.py` (utility file)
- `starsamplersettings.py`
remove: - `star_infiniteyou_apply.py`
remove: - `star_infiniteyou_face_swap.py`
remove: - `star_infiniteyou_patch_fixed.py`
remove: - `star_infiniteyou_patch_modified.py`
remove: - `star_infiniteyou_patch_saver.py`
- `update_node_colors.py` (utility script)

### Potential Candidates for Removal in v2.0:
1. Old/unused Python files not in __init__.py
2. Duplicate node registrations
3. Deprecated nodes (check MIGRATION_1.7.0.md for hints)
4. Unused dependencies in requirements.txt

### Dependencies to Review:
Check `requirements.txt` for packages only used by nodes being removed.

---

**Total Active Nodes: 86**
**Version: 2.0.0**
**Generated: 2026-04-10**
