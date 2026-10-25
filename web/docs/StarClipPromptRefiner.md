# Star CLIP Prompt Refiner

## Description
A single node that loads a text encoder from `models/text_encoders` and uses it as a local language model to create or refine prompts — similar to ComfyUI's native **Generate Text** node, but with the model loader built in, dynamic reference image slots, and shared StarNodes system prompt presets. No Ollama or external server needed; everything runs locally through the ComfyUI text encoder stack.

## Requirements
- A text generation capable model in `models/text_encoders`, e.g. `qwen3-vl-4b-heretic_int8.safetensors` (default) or any Qwen3-VL / Gemma instruct checkpoint
- For image input: a vision-language model (Qwen3-VL)

## Inputs

### Required
- **clip_name**: Dropdown of all models in `models/text_encoders`
  - Default: `qwen3-vl-4b-heretic_int8.safetensors`
  - The model is loaded, used for generation and offloaded again automatically by ComfyUI's model management
- **prompt**: Your user prompt text
  - This is the input you want the model to process, expand, or refine
- **system_prompt_preset**: Dropdown of preset system prompts
  - **Custom** (default, first option): Use your own system prompt text
  - Presets are loaded from the shared text storage (`json/startext.json`), sorted alphabetically — the same file used by the **Star Easy-Text-Storage** node and the **Star Ollama Prompt Helper**. Add, edit, or remove entries with that node and they appear here after refreshing the page.
- **system_prompt**: Custom system prompt text (used when preset is "Custom")
  - Replaces the system prompt in the model's chat template
  - If left empty while "Custom" is selected, the **Flux2(Klein) SysPrompt** preset is used as the default
- **use_seed**: Toggle for seed behavior
  - **On (default)**: The seed below is used; the node caches its result and only re-runs when an input changes
  - **Off**: A random seed is used every run, so the node always executes again even if the inputs did not change
- **seed**: Random seed for reproducibility (only used when `use_seed` is on)
  - Supports the standard **Control After Generate** modes (fixed / random / increment / decrement)
- **temperature**: Controls creativity of the output (default: 0.7)
  - Lower values = more focused/deterministic, higher values = more creative
- **top_k**: Limits sampling to the top-K most likely tokens (default: 64, 0 disables)
- **top_p**: Nucleus sampling threshold (default: 0.95)
- **min_p**: Minimum token probability relative to the most likely token (default: 0.05, 0 disables)
- **repetition_penalty**: Penalizes repeated tokens (default: 1.05, 1.0 disables)
- **max_length**: Maximum number of tokens to generate (default: 2048)
- **thinking**: Thinking / reasoning mode toggle (default: on)
  - The reasoning process is returned on the `think_output`
  - **Note for Qwen3-VL "heretic" models:** turning this off injects an empty think block into the template, which can confuse these models (they may echo the prompt instead of answering). Keep it on if the output looks wrong.

### Optional
- **clip_override**: CLIP input for an already loaded text encoder
  - When connected, this encoder is used for generation and the **clip_name** dropdown is ignored — lets you reuse a text encoder that is already loaded in the workflow instead of loading a second copy
  - When not connected, the model from the **clip_name** dropdown is loaded normally
- **image1 … image4**: IMAGE inputs for vision language models
  - Starts with one slot; connecting an image automatically adds the next slot (up to 4)
  - Disconnecting removes trailing empty slots again
  - Each input can also be an image batch — every frame is used as a reference image
  - Refer to the images in your prompt as "this image", "the first image", etc.
- **video**: IMAGE input for video frames (e.g. from a Load Video node)
  - Video frames as an image batch; assumed to be 24 FPS and subsampled to 1 FPS internally
  - Only used by models that support video (e.g. Gemma 4) — ignored by text-only encoders
- **audio**: AUDIO input (e.g. from a Load Audio node)
  - Only used by models that support audio (e.g. Gemma 4) — ignored by text-only encoders

## Outputs
- **result**: The generated/refined text (STRING)
  - Connect to a text widget on any node, or to a CLIP Text Encode node
- **think_output**: The model's reasoning process when `thinking` is enabled (STRING, empty otherwise)

## Usage

### Basic Prompt Refinement
1. Add the **Star CLIP Prompt Refiner** node
2. Keep the default model or pick another one from the **clip_name** dropdown
3. Choose a **system_prompt_preset** (or "Custom" and write your own)
4. Type your rough prompt idea in the **prompt** field
5. Run the workflow — the refined prompt appears on the **result** output
6. Connect the output to your text-to-image node

### Refining with Reference Images
1. Connect an image to **image1** — a new slot **image2** appears automatically
2. Add up to 4 reference images
3. Ask the model about the images in the prompt, e.g. "Describe the style of these images"

### Fresh Result Every Run
- Turn **use_seed** off if you want a new variation every time you queue the workflow, even when nothing else changed

## System Prompts File
Preset system prompts are loaded from the shared text storage file `json/startext.json`, the same file used by the **Star Easy-Text-Storage** node and the **⭐ Star Ollama Prompt Helper**. Add, edit, or remove entries with the storage node and they become available as presets here after a page refresh.

## Notes
- The node uses the same generation pipeline as ComfyUI's native **Generate Text** node (`clip.tokenize` → `clip.generate` → `clip.decode`)
- No context/history is kept between runs — each execution is independent
- The model type (Qwen3-VL 4B/8B, Gemma, etc.) is auto-detected from the checkpoint
