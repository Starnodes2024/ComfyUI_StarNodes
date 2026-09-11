# Star Sampler Settings

## Description
The StarSamplerSettings utility provides a persistent storage and management system for sampler configurations used by StarNodes samplers. It allows saving, loading, and deleting sampler settings presets, making it easy to reuse optimal configurations across different workflows.

## Usage Context
This utility class is used internally by various StarNodes sampler nodes, particularly:
- SDstarsampler
- Fluxstarsampler

It is not a standalone node but provides the backend functionality for sampler preset management.

## Features

### Settings Storage
- Maintains settings in two locations for redundancy:
  - Within the StarNodes directory (`custom_nodes/comfyui_starnodes/samplersettings.json`)
  - In the main ComfyUI directory (`samplersettings.json`)
- Automatically merges settings from both locations, with the main ComfyUI directory taking precedence

### Preset Management
- **Save Presets**: Store sampler configurations with custom names
- **Load Presets**: Retrieve previously saved configurations
- **Delete Presets**: Remove unwanted configurations
- **List Presets**: Get all available preset names

### Sampler Support
The utility supports two main sampler types:
1. **SDstarsampler** with settings:
   - seed
   - control_after_generate
   - sampler_name
   - scheduler
   - steps
   - cfg
   - denoise

2. **Fluxstarsampler** with settings:
   - seed
   - control_after_generate
   - sampler
   - scheduler
   - steps
   - guidance
   - max_shift
   - base_shift
   - denoise
   - use_teacache

## Implementation Details
- Uses Python's `pathlib` for robust path handling
- Implements error handling for file operations
- Provides clear feedback messages for all operations
- Maintains type consistency for saved settings

## Technical Notes
- Settings are stored in JSON format for easy inspection and manual editing if needed
- The utility automatically creates necessary directories and files if they don't exist
- When loading settings, it intelligently merges from both storage locations
- The system is designed to be resilient to file corruption or access issues

This utility enhances the StarNodes sampler experience by allowing users to save their preferred sampler configurations and quickly apply them to new workflows, improving consistency and workflow efficiency.
