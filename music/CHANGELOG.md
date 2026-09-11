# Changelog

All notable changes to the ACE Step ComfyUI Custom Node will be documented in this file.

## [1.1.0] - 2026-04-07

### Fixed
- **Audio Output Format**: Fixed compatibility with ComfyUI audio nodes by adding proper waveform data
  - Node now loads generated audio files and provides waveform tensor
  - Includes sample_rate for proper audio processing
  - Resolves "KeyError: 'waveform'" when connecting to PreviewAudio or other audio nodes
- **Waveform Batch Dimensions**: Fixed tensor shape for proper batch processing
  - Added `unsqueeze(0)` to ensure waveforms have correct shape `[1, C, S]`
  - Proper concatenation along batch dimension for multiple audio files
  - Prevents dimension mismatch errors in audio preview nodes
- **Audio/Metadata Separation**: Split outputs to avoid conflicts with ComfyUI audio nodes
  - Audio output now contains only `waveform` and `sample_rate` (clean format)
  - Metadata (files, task_id, etc.) moved to separate STRING output
  - Prevents issues with audio nodes that are strict about dictionary keys
- **Sample Rate Consistency**: Added validation for consistent sample rates across batch
  - Warns if multiple sample rates detected in batch
  - Uses first sample rate for the entire batch
- **JSON Serialization Error**: Fixed "TypeError: Object of type function is not JSON serializable"
  - Replaced lambda function with direct function call in INPUT_TYPES
  - Node now loads properly in ComfyUI without serialization errors
- **Model Selection Validation**: Added backward compatibility for empty model string
  - Existing workflows with `model: ''` now work without validation errors
  - Empty string treated same as "auto (use server default)"

### Added
- **Model Selection**: Fixed dropdown list with all available ACE Step models
  - Includes all standard models (turbo, base, sft, turbo-shift3)
  - **NEW**: Added XL model variants (xl-base, xl-sft, xl-turbo)
  - Total 9 model options plus auto/default selection
  - Select "" (empty) or "auto (use server default)" to use server's configured default
  - Removed dynamic model loading for better stability
- **Subfolder Support**: New `subfolder` parameter for organized file management
  - Default subfolder: "ACE-Step-1.5"
  - Files are now saved to `ComfyUI/output/ACE-Step-1.5/` by default
  - Can be customized or disabled (empty string for root output folder)
  - Subfolder is automatically created if it doesn't exist

### Changed
- **Audio Loading Method**: Implemented VHS-style audio reload workaround
  - Audio files are now saved, then reloaded using torchaudio with soundfile backend
  - Uses explicit backend specification to avoid torchcodec dependency
  - Multiple fallback methods for maximum compatibility
  - Ensures 100% compatibility with ComfyUI audio nodes (PreviewAudio, SaveAudio, etc.)
  - Fixes issues with audio preview showing 0:01 length or no playback
  - Proper waveform shape: `[1, C, S]` for single files, `[B, C, S]` for batch
- **Key Scale Input**: Changed from text input to dropdown selector
  - 34 musical key options (all major and minor keys)
  - Includes enharmonic equivalents (C#/Db, D#/Eb, F#/Gb, G#/Ab, A#/Bb)
  - Prevents typos and ensures valid key selection
- Audio files now include `subfolder` field in metadata
- Return format now includes both waveform data and file metadata
- Improved error handling for audio loading with fallback to empty waveform

### Dependencies
- Added `torch` and `torchaudio` for audio waveform processing

## [1.0.0] - 2026-04-07

### Added
- Initial release of ACE Step 1.5 ComfyUI custom node
- Full integration with ACE Step 1.5 local API
- Support for all major generation parameters:
  - Text prompts and lyrics input
  - Thinking mode with LM-enhanced generation
  - Sample mode for auto-generation from descriptions
  - Format enhancement for prompts and lyrics
  - Music attributes control (BPM, key/scale, time signature, duration)
  - Multi-language support (50+ languages)
  - Batch generation (up to 8 files)
  - Multiple output formats (MP3, WAV, FLAC)
  - Seed control for reproducibility
  - Advanced LM parameters (temperature, CFG scale, top-p)
  - Chain-of-Thought caption and language detection
  - Custom model selection
  - API authentication support
- Automatic task polling and status monitoring
- Audio file download and saving to ComfyUI output directory
- Comprehensive error handling and logging
- Detailed console output for generation progress

### Documentation
- Complete README with usage instructions
- Quick Start guide for beginners
- Troubleshooting section
- Example workflows and prompts
- GPU requirements and performance tips
- API server configuration guide

### Features
- Async task submission and polling
- Configurable timeout and poll intervals
- Support for all ACE Step 1.5 API endpoints
- Metadata preservation (BPM, key, time signature, seeds, models used)
- Batch file handling with proper naming conventions

## Future Enhancements (Planned)

### Potential Features
- [ ] Reference audio input support
- [ ] Cover generation mode
- [ ] Repainting/editing capabilities
- [ ] Audio preview in ComfyUI interface
- [ ] Workflow presets for common use cases
- [ ] Integration with ComfyUI audio nodes
- [ ] Real-time generation progress bar
- [ ] Queue management interface
- [ ] Metadata display in node output
- [ ] Audio waveform visualization
