# ⭐ Star Sound Mixer

## Overview

The **Star Sound Mixer** node lets you mix multiple audio inputs with individual volume control. Each connected audio input gets its own volume slider (0-100%) shown directly in the node body, making it easy to balance levels — like a mini mixing desk inside ComfyUI.

## Features

- **Dynamic Audio Inputs**: Start with one audio input — a new slot appears automatically each time you connect one (up to 12)
- **Per-Channel Volume Sliders**: Each connected audio input gets a styled DOM slider showing "Audio N" with a 0-100% volume control (default 100%)
- **Automatic Resampling**: All audio is resampled to the first connected input's sample rate before mixing
- **Peak Protection**: If the mixed signal exceeds 0 dBFS (|amplitude| > 1.0), it is normalized to prevent clipping
- **Smart Padding**: Inputs of different lengths and channel counts are zero-padded to match, so everything sums cleanly

## Inputs

### Dynamic Audio Slots (AUDIO)
- **audio_1** (AUDIO): First audio input — sets the reference sample rate for all subsequent inputs
- **audio_2** … **audio_12** (AUDIO): Additional audio inputs, appear automatically when the previous slot is connected

### Volume Sliders (FLOAT, 0.0-1.0)
- **volume_1**: Volume for audio_1 (default 100%)
- **volume_2** … **volume_12**: Volume for each additional audio input, appears alongside its audio slot

Each volume slider is rendered as a styled DOM widget inside the node body, with a green-themed slider thumb and a percentage readout.

## Output

- **audio** (AUDIO): The mixed audio stream

## How It Works

### Mixing Process
1. The first connected audio input sets the reference sample rate
2. Each subsequent audio input is resampled to match the reference sample rate (using `torchaudio.functional.resample`)
3. Each input is multiplied by its volume slider value (0.0 = silent, 1.0 = full volume)
4. All inputs are zero-padded to the same length and channel count, then summed sample-by-sample
5. If the peak amplitude of the mixed signal exceeds 1.0, the entire signal is normalized to 1.0 to prevent clipping

### Dynamic Input Growth
The node uses ComfyUI's autogrow input system. When you connect a cable to the last visible audio slot:
- A new `audio_N` input slot is added below it (up to max 12)
- A volume slider widget for the new slot appears in the node body
- When you disconnect an audio input (that isn't the last one), its slider is removed and trailing empty slots are pruned

### Volume Slider UI
The volume sliders are custom DOM widgets (`star_sound_mixer.js`) — not standard ComfyUI slider widgets. They feature:
- A green-themed slider with "Audio N" label
- Real-time percentage readout (0% to 100%)
- Smooth interaction with mouse/touch

## Usage Example

### Basic Workflow
```
[Load Audio 1] ──> [Star Sound Mixer] ──> [Video Combine / Save Audio]
[Load Audio 2] ──>   (slot appears after connecting audio_1)
[Load Audio 3] ──>   (slot appears after connecting audio_2)
```

### Common Use Cases

1. **Background Music + Voiceover**: Mix a music track at 40% with a voiceover at 100%
2. **Layered Sound Effects**: Combine multiple sound effects at different volumes
3. **Video Soundtrack Mixing**: Mix audio from multiple video segments before encoding
4. **Audio Ducking**: Set one audio source to a low volume while another plays at full volume

## Tips

- The first connected audio sets the sample rate — connect your primary audio first
- Volume sliders default to 100% — adjust each channel to taste
- If the mixed output sounds distorted, lower individual volumes — the node normalizes to prevent hard clipping, but heavy mixing can still sound compressed
- Inputs of different lengths are padded with silence — shorter audio tracks will have silence at the end
- You can use this node with just one audio input as a simple volume control

## Category

Located in: **⭐StarNodes/Video**

## Technical Details

- Uses `torchaudio.functional.resample` for sample rate conversion
- Zero-pads both time and channel dimensions for mismatched inputs
- Peak normalization prevents clipping when mixed amplitude exceeds 1.0
- Volume is applied as a simple scalar multiplication before summing
- Dynamic inputs implemented via ComfyUI's autogrow dict pattern (max 12)
- Custom DOM widgets (`star_sound_mixer.js`) for the slider UI
