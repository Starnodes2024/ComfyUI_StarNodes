# ⭐ Star Sound Mixer

## Overview

The **Star Sound Mixer** node lets you mix multiple audio inputs with individual volume control and per-track start pause. Each connected audio input gets its own volume slider (0-100%) shown directly in the node body, and every track after the first gets a "Pause" value in seconds so it can start later than the master track — like a mini mixing desk inside ComfyUI.

## Features

- **Dynamic Audio Inputs**: Start with one audio input — a new slot appears automatically each time you connect one (up to 12)
- **Per-Channel Volume Sliders**: Each connected audio input gets a styled DOM slider showing "Audio N" with a 0-100% volume control (default 100%)
- **Per-Track Start Pause**: Every audio input from `audio_2` onward gets a "Pause" number input (seconds, default 0) that delays the start of that track relative to the first
- **Master Track Length**: The first connected audio (`audio_1`) defines the total output length — other tracks are aligned to it (truncated if they overflow, padded with silence if they end early)
- **Automatic Resampling**: All audio is resampled to the first connected input's sample rate before mixing
- **Peak Protection**: If the mixed signal exceeds 0 dBFS (|amplitude| > 1.0), it is normalized to prevent clipping
- **Smart Padding**: Inputs of different channel counts are zero-padded to match, so everything sums cleanly

## Inputs

### Dynamic Audio Slots (AUDIO)
- **audio_1** (AUDIO): First audio input — the master track. Sets the reference sample rate and the total output length. Always starts at 00:00:00.
- **audio_2** … **audio_12** (AUDIO): Additional audio inputs, appear automatically when the previous slot is connected. Each can be delayed with its own start pause.

### Volume Sliders (FLOAT, 0.0-1.0)
- **volume_1**: Volume for audio_1 (default 100%)
- **volume_2** … **volume_12**: Volume for each additional audio input, appears alongside its audio slot

### Start Pause (FLOAT, seconds, 0.0-3600.0)
- **start_pause_2** … **start_pause_12**: Pause before that audio starts, in seconds (default 0.0). Only available for `audio_2` onward — the first track always starts at 0. The pause is applied after resampling, so it is sample-accurate relative to the master track's sample rate.

Each volume slider is rendered as a styled DOM widget inside the node body, with a green-themed slider thumb and a percentage readout. The start pause is a small number input shown next to the volume slider (labelled "Pause … s").

## Output

- **audio** (AUDIO): The mixed audio stream. Its length equals the first track's length.

## How It Works

### Mixing Process
1. The first connected audio input (`audio_1`) sets the reference sample rate and the output length
2. Each subsequent audio input is resampled to match the reference sample rate (using `torchaudio.functional.resample`)
3. Each input is multiplied by its volume slider value (0.0 = silent, 1.0 = full volume)
4. For tracks from `audio_2` onward, `start_pause_N` seconds of silence are prepended (after resampling, so the offset is sample-accurate)
5. Every non-master track is aligned to the master length: samples beyond the master length are dropped, shortfalls are padded with silence
6. All inputs are summed sample-by-sample (channel counts are zero-padded to match)
7. If the peak amplitude of the mixed signal exceeds 1.0, the entire signal is normalized to 1.0 to prevent clipping

### Dynamic Input Growth
The node uses ComfyUI's autogrow input system. When you connect a cable to the last visible audio slot:
- A new `audio_N` input slot is added below it (up to max 12)
- A volume slider widget and a start pause input for the new slot appear in the node body
- When you disconnect an audio input (that isn't the last one), its widgets are removed and trailing empty slots are pruned

### Volume Slider + Pause UI
The volume sliders and pause inputs are custom DOM widgets (`star_sound_mixer.js`) — not standard ComfyUI slider widgets. They feature:
- A green-themed slider with "Audio N" label
- Real-time percentage readout (0% to 100%)
- A "Pause" number input (seconds) next to the slider for tracks 2 and up
- Smooth interaction with mouse/touch

## Usage Example

### Basic Workflow
```
[Load Audio 1] ──> [Star Sound Mixer] ──> [Video Combine / Save Audio]
[Load Audio 2] ──>   (slot appears after connecting audio_1, set Pause to delay it)
[Load Audio 3] ──>   (slot appears after connecting audio_2)
```

### Common Use Cases

1. **Background Music + Voiceover**: Mix a music track (audio_1) at 40% with a voiceover (audio_2) at 100% starting after a 3-second pause
2. **Layered Sound Effects**: Combine multiple sound effects at different volumes and start times
3. **Video Soundtrack Mixing**: Mix audio from multiple video segments, delaying each to match its scene
4. **Audio Ducking**: Set one audio source to a low volume while another plays at full volume
5. **Delayed Narration**: Keep music as the master track and start a narration a few seconds in

## Tips

- The first connected audio sets the sample rate and the output length — connect your primary/longest audio first
- Volume sliders default to 100% — adjust each channel to taste
- Start pause defaults to 0 seconds — set it for any track that should begin later than the master
- Because the master track defines the output length, a delayed track that would run past the end is truncated to the master length. Make the master track long enough to fit delayed tracks.
- If the mixed output sounds distorted, lower individual volumes — the node normalizes to prevent hard clipping, but heavy mixing can still sound compressed
- You can use this node with just one audio input as a simple volume control

## Category

Located in: **⭐StarNodes/Video**

## Technical Details

- Uses `torchaudio.functional.resample` for sample rate conversion
- Start pause is applied after resampling by prepending `round(pause_seconds * sample_rate)` zero samples
- Non-master tracks are truncated/padded to the master track's length so the output length is always the first track's length
- Zero-pads channel dimensions for mismatched inputs
- Peak normalization prevents clipping when mixed amplitude exceeds 1.0
- Volume is applied as a simple scalar multiplication before summing
- Dynamic inputs implemented via ComfyUI's autogrow dict pattern (max 12)
- Custom DOM widgets (`star_sound_mixer.js`) for the slider + pause UI
