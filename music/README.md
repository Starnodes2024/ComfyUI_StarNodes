# ACE Step 1.5 ComfyUI Custom Node

A powerful ComfyUI custom node for generating high-quality music using the ACE Step 1.5 local API. This node provides full control over all major generation parameters and supports commercial-grade music generation directly within your ComfyUI workflows.

## Features

- **Full API Integration**: Complete access to ACE Step 1.5's music generation capabilities
- **Dynamic Model Selection**: Automatically fetches and displays available models from your API server
- **High-Quality Output**: Supports thinking mode with LM-enhanced generation for best quality
- **Flexible Control**: Control duration, BPM, key/scale, time signature, and more
- **Multi-Language Support**: Generate lyrics in 50+ languages
- **Batch Generation**: Generate up to 8 songs simultaneously
- **Multiple Formats**: Output in MP3, WAV, or FLAC
- **Sample Mode**: Auto-generate music from natural language descriptions
- **Format Enhancement**: Use LM to enhance and format your prompts and lyrics
- **Organized Output**: Automatic subfolder organization (default: ACE-Step-1.5)

## Requirements

### ACE Step 1.5 API Server

This node requires a running ACE Step 1.5 API server. Follow these steps to set it up:

1. **Install ACE Step 1.5**:
   ```bash
   # Clone the repository
   git clone https://github.com/ACE-Step/ACE-Step-1.5.git
   cd ACE-Step-1.5
   
   # Install dependencies
   curl -LsSf https://astral.sh/uv/install.sh | sh  # macOS/Linux
   # OR for Windows: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   
   uv sync
   ```

2. **Start the API Server**:
   ```bash
   # Windows
   start_api_server.bat
   
   # Linux
   chmod +x start_api_server.sh && ./start_api_server.sh
   
   # macOS
   chmod +x start_api_server_macos.sh && ./start_api_server_macos.sh
   
   # Or using uv directly
   uv run acestep-api
   ```

3. **Verify the Server**:
   - Default URL: `http://localhost:8001`
   - Check health: `http://localhost:8001/health`

### Python Dependencies

The node requires the following Python packages (usually already available in ComfyUI):

```
requests
```

If not available, install with:
```bash
pip install requests
```

## Installation

1. Navigate to your ComfyUI custom nodes directory:
   ```bash
   cd ComfyUI/custom_nodes/
   ```

2. The node is already installed at:
   ```
   comfyui_ACESTEP_starapi/
   ```

3. Restart ComfyUI

4. The node will appear under: **StarNodes → ACE Step → ACE Step Music Generator**

## Usage

### Basic Usage

1. **Add the Node**: In ComfyUI, add the "ACE Step Music Generator" node from the StarNodes/ACE Step category

2. **Configure API Connection**:
   - Set `api_url` to your ACE Step API server (default: `http://localhost:8001`)
   - If your server requires authentication, set the `api_key` parameter

3. **Set Your Prompt**:
   - Enter a music description in the `prompt` field
   - Example: "upbeat pop song with electric guitar and drums"

4. **Optional - Add Lyrics**:
   - Enter lyrics in the `lyrics` field
   - Use standard song structure markers: `[Verse 1]`, `[Chorus]`, `[Bridge]`, etc.

5. **Run**: Execute the workflow and wait for generation to complete

### Advanced Parameters

#### Core Settings

- **thinking** (default: `True`): Enable LM-enhanced generation for higher quality
- **sample_mode** (default: `False`): Auto-generate caption/lyrics from `sample_query`
- **sample_query**: Natural language description (e.g., "a soft Bengali love song")
- **use_format** (default: `False`): Use LM to enhance your prompt and lyrics

#### Music Attributes

- **audio_duration** (10-600 seconds, default: 30): Length of generated audio
- **bpm** (30-300, default: 120): Tempo in beats per minute
- **key_scale** (default: "C Major"): Musical key - dropdown selector with all major and minor keys
  - Major keys: C, C#/Db, D, D#/Eb, E, F, F#/Gb, G, G#/Ab, A, A#/Bb, B
  - Minor keys: All corresponding minor keys
  - Total: 34 key options (including enharmonic equivalents)
- **time_signature** (default: "4"): Time signature (2/4, 3/4, 4/4, 6/8)
- **vocal_language**: Language for lyrics (en, zh, ja, ko, es, fr, de, it, pt, ru, ar, hi, bn, th, vi, id, tr, nl, pl)

#### Generation Control

- **inference_steps** (1-200, default: 8): Number of diffusion steps
  - Turbo models: 1-20 (recommended: 8)
  - Base models: 1-200 (recommended: 32-64)
- **guidance_scale** (1.0-20.0, default: 7.0): Prompt guidance strength (base models only)
- **batch_size** (1-8, default: 1): Number of audio files to generate
- **use_random_seed** (default: `True`): Use random seed for variation
- **seed** (default: -1): Fixed seed when `use_random_seed` is False
- **audio_format** (default: "mp3"): Output format (mp3, wav, flac)

#### LM Parameters

- **lm_temperature** (0.1-2.0, default: 0.85): Sampling temperature for LM
- **lm_cfg_scale** (1.0-10.0, default: 2.5): CFG scale for LM
- **lm_top_p** (0.0-1.0, default: 0.9): Top-p sampling for LM
- **use_cot_caption** (default: `True`): Use Chain-of-Thought for caption enhancement
- **use_cot_language** (default: `True`): Use CoT for language detection

#### Advanced Settings

- **model**: Select which DiT model to use
  - **Available Models**:
    - `""` (empty) or `"auto (use server default)"` - Uses server's configured default model
    - **Standard Models**:
      - `acestep-v15-turbo` - Fast generation (recommended for quick iterations)
      - `acestep-v15-base` - Balanced quality and speed
      - `acestep-v15-sft` - Supervised fine-tuned model
      - `acestep-v15-turbo-shift3` - Turbo variant with shift3
    - **XL Models** (higher quality, slower):
      - `acestep-v15-xl-base` - XL base model
      - `acestep-v15-xl-sft` - XL supervised fine-tuned
      - `acestep-v15-xl-turbo` - XL turbo variant shown
- **poll_interval** (0.5-10.0 seconds, default: 2.0): Status check interval
- **timeout** (60-3600 seconds, default: 600): Maximum wait time
- **subfolder** (default: "ACE-Step-1.5"): Subfolder name for organizing generated audio files
  - Files will be saved to: `ComfyUI/output/ACE-Step-1.5/`
  - Set to empty string `""` to save directly in output folder

### Example Workflows

#### Example 1: Simple Pop Song

```
Prompt: "upbeat pop song with catchy melody"
Lyrics: "[Verse 1]\nWalking down the street\nFeeling the beat\n[Chorus]\nThis is my song\nSing along"
thinking: True
audio_duration: 30
bpm: 120
vocal_language: en
```

#### Example 2: Instrumental Jazz

```
Prompt: "smooth jazz piano trio, relaxing atmosphere"
Lyrics: (leave empty)
thinking: True
audio_duration: 60
bpm: 90
key_scale: "Bb Major"
```

#### Example 3: Sample Mode Generation

```
sample_mode: True
sample_query: "a romantic Chinese ballad with piano and strings"
thinking: True
audio_duration: 45
vocal_language: zh
```

#### Example 4: High-Quality Long Form

```
Prompt: "epic orchestral soundtrack with dramatic crescendos"
thinking: True
audio_duration: 180
inference_steps: 8
batch_size: 2
use_random_seed: True
```

## Output

The node returns **two outputs**:

### 1. Audio Output (AUDIO type)

Clean audio dictionary compatible with all ComfyUI audio nodes:

```python
{
    "waveform": torch.Tensor,  # Shape: [1, Channels, Samples]
    "sample_rate": 44100,      # Sample rate in Hz
}
```

**Waveform Format:**
- Shape: `[1, C, S]` where C=channels (usually 2), S=samples
- Type: `torch.Tensor`
- For single audio: `[1, 2, Samples]`
- For multiple audio (batch): Files are concatenated along time dimension
- Compatible with: PreviewAudio, SaveAudio, and all standard ComfyUI audio nodes

### 2. Metadata Output (STRING type)

JSON string containing generation details:

```json
{
  "files": [
    {
      "filename": "acestep_<task_id>_0.mp3",
      "subfolder": "ACE-Step-1.5",
      "filepath": "/path/to/output/ACE-Step-1.5/acestep_<task_id>_0.mp3",
      "prompt": "upbeat pop song...",
      "lyrics": "...",
      "metas": {
        "bpm": 120,
        "duration": 30,
        "keyscale": "C Major",
        "timesignature": "4"
      },
      "seed": "12345,67890",
      "lm_model": "acestep-5Hz-lm-0.6B",
      "dit_model": "acestep-v15-turbo"
    }
  ],
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "batch_size": 1,
  "sample_rate": 44100
}
```

**Audio Files Location:**
- Default: `ComfyUI/output/ACE-Step-1.5/acestep_<task_id>_<index>.<format>`
- Custom subfolder: `ComfyUI/output/<subfolder>/acestep_<task_id>_<index>.<format>`
- No subfolder: `ComfyUI/output/acestep_<task_id>_<index>.<format>`

**Usage:**
- Connect the **audio** output to PreviewAudio, SaveAudio, or other audio processing nodes
- Use the **metadata** output for debugging or displaying generation information

## Troubleshooting

### Connection Issues

**Problem**: "Failed to submit task: Connection refused"

**Solution**: 
- Ensure ACE Step API server is running
- Check the API URL (default: `http://localhost:8001`)
- Verify with: `curl http://localhost:8001/health`

### Authentication Errors

**Problem**: "401 Unauthorized"

**Solution**:
- If your server requires authentication, set the `api_key` parameter
- Check your server's `ACESTEP_API_KEY` environment variable

### Timeout Issues

**Problem**: "Task timeout after X seconds"

**Solution**:
- Increase the `timeout` parameter (especially for long audio)
- Check server logs for errors
- Verify server has sufficient GPU memory

### Quality Issues

**Problem**: Generated music quality is poor

**Solution**:
- Enable `thinking` mode for LM-enhanced generation
- Increase `inference_steps` (try 16-32 for base models)
- Use more detailed prompts
- Try `use_format=True` to enhance your prompts

### Memory Issues

**Problem**: Server crashes or OOM errors

**Solution**:
- Reduce `batch_size`
- Reduce `audio_duration`
- Enable CPU offloading in ACE Step server config:
  ```bash
  export ACESTEP_OFFLOAD_TO_CPU=true
  ```

## GPU Requirements

Recommended GPU VRAM based on model:

| VRAM | Recommended Configuration |
|------|---------------------------|
| ≤6GB | DiT only, INT8 quantization, CPU offload |
| 6-8GB | 0.6B LM with PyTorch backend |
| 8-16GB | 0.6B or 1.7B LM with vLLM backend |
| 16-24GB | 1.7B LM, no offload needed |
| ≥24GB | 4B LM, best quality |

## API Server Configuration

Configure your ACE Step server via `.env` file:

```bash
# Server settings
ACESTEP_API_HOST=127.0.0.1
ACESTEP_API_PORT=8001
ACESTEP_API_KEY=your-secret-key  # Optional

# Model settings
ACESTEP_CONFIG_PATH=acestep-v15-turbo
ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-1.7B
ACESTEP_LM_BACKEND=vllm

# Performance
ACESTEP_OFFLOAD_TO_CPU=false
ACESTEP_USE_FLASH_ATTENTION=true
```

## Performance Tips

1. **Use thinking mode** for best quality (adds ~2-5s processing time)
2. **Batch generation** is efficient for multiple variations
3. **Turbo models** are faster (8 steps) vs base models (32-64 steps)
4. **Enable flash attention** in server config for speed boost
5. **Use vLLM backend** for LM when you have sufficient VRAM

## Links

- **ACE Step 1.5 GitHub**: https://github.com/ACE-Step/ACE-Step-1.5
- **ACE Step Documentation**: https://github.com/ACE-Step/ACE-Step-1.5/tree/main/docs
- **API Documentation**: https://github.com/ACE-Step/ACE-Step-1.5/blob/main/docs/en/API.md
- **Hugging Face**: https://huggingface.co/ACE-Step/Ace-Step1.5
- **Project Page**: https://ace-step.github.io/ace-step-v1.5.github.io/

## License

This custom node is provided as-is. ACE Step 1.5 is licensed under MIT License.

## Credits

- **ACE Step 1.5**: Developed by ACE Studio and StepFun
- **Custom Node**: Created for ComfyUI integration
- **ComfyUI**: https://github.com/comfyanonymous/ComfyUI

## Support

For issues related to:
- **This custom node**: Open an issue in your ComfyUI custom nodes repository
- **ACE Step 1.5**: Visit https://github.com/ACE-Step/ACE-Step-1.5/issues
- **API questions**: Check the API documentation at https://github.com/ACE-Step/ACE-Step-1.5/blob/main/docs/en/API.md
