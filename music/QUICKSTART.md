# Quick Start Guide - ACE Step ComfyUI Node

## Prerequisites

Before using this node, you need to have the ACE Step 1.5 API server running.

### Step 1: Install ACE Step 1.5

If you haven't already installed ACE Step 1.5, navigate to your ACE Step directory:

```bash
cd E:\AI\05_ACE-Step-1.5
```

### Step 2: Start the API Server

**Windows:**
```bash
start_api_server.bat
```

**Linux/macOS:**
```bash
chmod +x start_api_server.sh
./start_api_server.sh
```

The server will start on `http://localhost:8001` by default.

### Step 3: Verify Server is Running

Open your browser and visit:
```
http://localhost:8001/health
```

You should see a response like:
```json
{
  "data": {
    "status": "ok",
    "service": "ACE-Step API",
    "version": "1.0"
  },
  "code": 200
}
```

## Using the Node in ComfyUI

### Step 1: Add the Node

1. Open ComfyUI
2. Right-click on the canvas
3. Navigate to: **Add Node → StarNodes → ACE Step → ACE Step Music Generator**

### Step 2: Basic Configuration

**Minimum required settings:**
- `api_url`: `http://localhost:8001` (default)
- `prompt`: Your music description
- `model`: Select from available models (auto-populated from your server)

**Example:**
```
prompt: "energetic rock song with electric guitar solo"
model: "auto (use server default)"  # or select a specific model
```

**Model Selection:**
The node automatically queries your API server and shows available models in a dropdown. The default model is marked for easy identification.

### Step 3: Run Your First Generation

1. Click "Queue Prompt" in ComfyUI
2. Watch the console for progress:
   ```
   [ACE Step] Starting music generation...
   [ACE Step] Task submitted successfully. Task ID: xxx
   [ACE Step] Task in progress...
   [ACE Step] Task completed successfully!
   [ACE Step] Saved: /path/to/output/acestep_xxx_0.mp3
   ```

3. Find your generated audio in ComfyUI's output directory

## Common Use Cases

### 1. Simple Music Generation

**Settings:**
```
prompt: "upbeat pop song"
thinking: True
audio_duration: 30
```

### 2. Song with Lyrics

**Settings:**
```
prompt: "emotional ballad with piano"
lyrics: "[Verse 1]
In the quiet of the night
Stars are shining bright
[Chorus]
This is our moment
Don't let it go"
thinking: True
vocal_language: en
audio_duration: 45
```

### 3. Instrumental Music

**Settings:**
```
prompt: "smooth jazz piano trio"
lyrics: (leave empty)
thinking: True
audio_duration: 60
bpm: 90
key_scale: "Bb Major"
```

### 4. Quick Generation from Description

**Settings:**
```
sample_mode: True
sample_query: "a cheerful children's song with xylophone"
thinking: True
audio_duration: 30
```

### 5. Batch Generation (Multiple Variations)

**Settings:**
```
prompt: "electronic dance music"
thinking: True
batch_size: 4
use_random_seed: True
audio_duration: 30
```

## Parameter Quick Reference

### Essential Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `api_url` | `http://localhost:8001` | ACE Step API server URL |
| `prompt` | - | Music description (required) |
| `thinking` | `True` | Use LM for better quality |
| `audio_duration` | `30` | Length in seconds (10-600) |

### Quality Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `inference_steps` | `8` | More steps = better quality (slower) |
| `lm_temperature` | `0.85` | Creativity (0.1-2.0) |
| `use_cot_caption` | `True` | Enhance prompt with AI |

### Music Attributes

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bpm` | `120` | Tempo (30-300) |
| `key_scale` | `"C Major"` | Musical key |
| `time_signature` | `"4"` | 2/4, 3/4, 4/4, or 6/8 |
| `vocal_language` | `"en"` | Lyrics language |

## Tips for Best Results

1. **Enable thinking mode** - Always use `thinking: True` for best quality
2. **Be descriptive** - Detailed prompts produce better results
   - Good: "upbeat pop song with electric guitar, drums, and catchy vocal melody"
   - Basic: "pop song"

3. **Use proper lyrics format** - Structure your lyrics with markers:
   ```
   [Intro]
   [Verse 1]
   ...
   [Chorus]
   ...
   [Bridge]
   ...
   [Outro]
   ```

4. **Match duration to content** - Longer songs need more time:
   - Short clip: 10-30 seconds
   - Full verse + chorus: 30-60 seconds
   - Complete song: 120-180 seconds

5. **Experiment with seeds** - For variations:
   - `use_random_seed: True` - Different each time
   - `use_random_seed: False` + fixed `seed` - Reproducible results

## Troubleshooting

### "Connection refused" Error

**Problem:** Can't connect to API server

**Solution:**
1. Make sure ACE Step API server is running
2. Check the URL is correct: `http://localhost:8001`
3. Try accessing `http://localhost:8001/health` in your browser

### "Task timeout" Error

**Problem:** Generation takes too long

**Solution:**
1. Increase `timeout` parameter (default: 600 seconds)
2. Reduce `audio_duration` for faster generation
3. Check server logs for errors

### Poor Quality Output

**Problem:** Generated music doesn't sound good

**Solution:**
1. Enable `thinking: True`
2. Use more detailed prompts
3. Try `use_format: True` to enhance your prompt
4. Increase `inference_steps` to 16 or 32

### Server Out of Memory

**Problem:** Server crashes during generation

**Solution:**
1. Reduce `batch_size` to 1
2. Reduce `audio_duration`
3. Enable CPU offloading in server config:
   ```bash
   # In ACE Step .env file
   ACESTEP_OFFLOAD_TO_CPU=true
   ```

## Advanced Features

### Using Specific Models

If you have multiple models loaded on your server:

```
model: "acestep-v15-turbo"
```

Check available models:
```bash
curl http://localhost:8001/v1/models
```

### Format Enhancement

Let the LM improve your prompt and lyrics:

```
prompt: "rock song"
lyrics: "Walking down the street"
use_format: True
thinking: True
```

### Sample Mode

Auto-generate everything from a description:

```
sample_mode: True
sample_query: "a romantic French chanson with accordion"
thinking: True
```

## Next Steps

- Read the full [README.md](README.md) for detailed parameter documentation
- Check [ACE Step API Documentation](https://github.com/ACE-Step/ACE-Step-1.5/blob/main/docs/en/API.md)
- Experiment with different prompts and settings
- Join the [ACE Step Discord](https://discord.gg/PeWDxrkdj7) for community support

## Example Prompts

Here are some example prompts to get you started:

**Pop:**
- "upbeat pop song with catchy chorus and electronic beats"
- "emotional pop ballad with piano and strings"

**Rock:**
- "energetic rock anthem with electric guitar solo"
- "indie rock song with jangly guitars and driving drums"

**Electronic:**
- "deep house track with groovy bassline"
- "ambient electronic music with atmospheric pads"

**Jazz:**
- "smooth jazz with saxophone and walking bass"
- "bebop jazz piano trio"

**Classical:**
- "romantic piano piece in the style of Chopin"
- "epic orchestral soundtrack with dramatic crescendos"

**World Music:**
- "traditional Chinese music with erhu and guzheng"
- "flamenco guitar with passionate rhythms"

Happy music making! 🎵
