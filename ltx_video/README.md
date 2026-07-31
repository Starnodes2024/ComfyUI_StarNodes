# ComfyUI Star LTX Toolz

Custom nodes for ComfyUI focused on LTX video generation and VAE management.

## Nodes

### ⭐ Star LTX Video Settings

A comprehensive node for calculating video dimensions and frame counts for LTX video generation with proper divisibility constraints.

#### Features

- **Video Size Presets**: Choose between HD (1280-based), FHD (1920-based), or Custom dimensions
- **Aspect Ratio Selection**: Multiple predefined ratios for both HD and FHD
- **Smart Input Image Ratio**: Automatically calculate dimensions based on input image aspect ratio
- **Frame Calculation**: Automatic frame count calculation based on FPS and duration
- **Divisibility Constraints**: Ensures all outputs meet LTX requirements

#### Inputs

**Required:**
- `video_size`: Select video resolution base
  - **HD**: 1280-based dimensions
  - **FHD**: 1920-based dimensions  
  - **Custom**: User-defined width and height
  
- `ratio`: Aspect ratio selection
  - Available ratios: 1:1, 4:3, 3:2, 16:10, 16:9, 21:9, 3:4, 2:3, 10:16, 9:16, 9:21
  
- `custom_width`: Width in pixels (32-8192, step 32) - used when video_size is "Custom"
- `custom_height`: Height in pixels (32-8192, step 32) - used when video_size is "Custom"
- `best_size_from_input`: Boolean - when enabled, calculates dimensions fitting the input image's aspect ratio
- `fps`: Frames per second (1-120, default 24)
- `seconds`: Video duration in seconds (1-60, default 15)

**Optional:**
- `image`: Input image (used with "Best Size From Input" feature)

#### Outputs

All outputs are provided as both **INT** and **FLOAT** types:

1. **width** / **width_float**: Final video width (divisible by 32 + 1)
2. **height** / **height_float**: Final video height (divisible by 32 + 1)
3. **width_50%** / **width_50%_float**: Half width (divisible by 32 + 1)
4. **height_50%** / **height_50%_float**: Half height (divisible by 32 + 1)
5. **frames** / **frames_float**: Total frame count (divisible by 8 + 1)
6. **fps** / **fps_float**: Frames per second value
7. **seconds** / **seconds_float**: Video duration in seconds

#### Divisibility Rules

The node enforces LTX video generation requirements:
- **Width & Height**: Must be divisible by 32 + 1
- **Frame Count**: Must be divisible by 8 + 1
- **Frame Calculation**: `(fps × seconds + 1)` rounded to nearest valid value

#### Predefined Dimensions

**HD Ratios (1280-based):**
- 1:1 → 1280×1280
- 4:3 → 1280×960
- 3:2 → 1280×853
- 16:10 → 1280×800
- 16:9 → 1280×720
- 21:9 → 1280×548
- 3:4 → 960×1280
- 2:3 → 853×1280
- 10:16 → 800×1280
- 9:16 → 720×1280
- 9:21 → 548×1280

**FHD Ratios (1920-based):**
- 1:1 → 1920×1920
- 4:3 → 1920×1440
- 3:2 → 1920×1280
- 16:10 → 1920×1200
- 16:9 → 1920×1080
- 21:9 → 1920×823
- 3:4 → 1440×1920
- 2:3 → 1280×1920
- 10:16 → 1200×1920
- 9:16 → 1080×1920
- 9:21 → 823×1920

#### Usage Examples

**Example 1: Standard FHD 16:9 video**
- video_size: FHD
- ratio: 16:9
- fps: 24
- seconds: 15
- Result: 1921×1081 pixels, 361 frames

**Example 2: Custom dimensions**
- video_size: Custom
- custom_width: 1024
- custom_height: 576
- fps: 30
- seconds: 10
- Result: 1025×577 pixels, 305 frames

**Example 3: Match input image ratio**
- video_size: FHD
- best_size_from_input: enabled
- image: [connected image]
- fps: 24
- seconds: 5
- Result: Dimensions matching closest FHD ratio to input image, 121 frames

---

### ⭐ LTX Image Cut

A utility node for cutting and extracting frames from image batches (video frames) while maintaining the original frame count for audio synchronization.

#### Features

- **Cut First Frames**: Replace unwanted frames at the beginning with copies of the first content frame
- **Cut Last Frames**: Replace unwanted frames at the end with copies of the last content frame
- **Frame Count Preservation**: Maintains original frame count for audio sync
- **Frame Extraction**: Extract specific frames (first, last, or by number)
- **Batch Processing**: Process entire image batches efficiently

#### Inputs

**Required:**
- `images`: Input image batch (video frames)
- `cut_first_frames`: Number of frames to replace at the start (default: 0, min: 0)
  - Set to 0 to keep all original frames from the start
  - These frames will be replaced with copies of the first content frame
- `cut_last_frames`: Number of frames to replace at the end (default: 10, min: 0)
  - Set to 0 to keep all original frames at the end
  - These frames will be replaced with copies of the last content frame
- `export_frame_number`: Frame number to extract as selected frame (default: 0, min: 0)
  - Frame numbering starts at 0 (first frame)

#### Outputs

All outputs are optional IMAGE types:

1. **images**: The processed image batch with the same frame count as input
   - Cut frames at start are replaced with copies of the first content frame
   - Cut frames at end are replaced with copies of the last content frame
2. **first_image**: The first content frame (after cutting)
3. **last_image**: The last content frame (before cutting)
4. **selected_frame**: The frame at the specified `export_frame_number` from the original batch

#### Behavior

- **Audio Synchronization**: The output always has the same number of frames as the input
- Cut frames are replaced with static copies to maintain timing
- If cutting would remove all frames, the middle frame is used as content
- If `export_frame_number` exceeds the total frame count, the last frame is returned
- Frame indices are 0-based (first frame = 0)

#### Usage Examples

**Example 1: Cut first and last frames (audio sync maintained)**
- images: [100 frame batch]
- cut_first_frames: 5
- cut_last_frames: 10
- Result: 100 frames total
  - Frames 0-4: copies of frame 5 (first content frame)
  - Frames 5-89: original content
  - Frames 90-99: copies of frame 89 (last content frame)

**Example 2: Extract specific frame without cutting**
- images: [100 frame batch]
- cut_first_frames: 0
- cut_last_frames: 0
- export_frame_number: 50
- Result: 100 frames (unchanged) + frame 50 as selected_frame

**Example 3: Cut intro and outro for audio sync**
- images: [361 frame batch] (15 seconds at 24fps)
- cut_first_frames: 24 (remove 1 second intro)
- cut_last_frames: 48 (remove 2 seconds outro)
- Result: 361 frames total (maintains 15 second duration for audio)
  - First 24 frames: static copy of frame 24
  - Frames 24-312: original content (13 seconds)
  - Last 48 frames: static copy of frame 312

---

### StarVAE LTXV Load

Loads separate video and audio VAE models for LTX video generation.

#### Inputs
- `video_vae`: Video VAE checkpoint file
- `audio_vae`: Audio VAE checkpoint file

#### Outputs
- `video_vae`: Loaded video VAE model
- `audio_vae`: Loaded audio VAE model

---

### StarVAE LTXV Save

Saves video outputs with audio integration using LTX VAE models.

---

## Installation

1. Clone or download this repository into your ComfyUI custom_nodes folder:
   ```
   ComfyUI/custom_nodes/comfyui-Star-LTXtoolz/
   ```

2. Restart ComfyUI

3. The nodes will appear in the "⭐StarNodes" category

## Requirements

- ComfyUI
- PyTorch
- Standard ComfyUI dependencies

## License

This project is provided as-is for use with ComfyUI.

## Support

For issues or questions, please refer to the ComfyUI community resources.
