# Star Ollama Prompt Helper

## Description
A user-friendly single node that connects to a local Ollama instance to create or refine prompts for image generation. It uses preset system prompts optimized for various text-to-image models, or lets you write your own custom system prompt.

## Requirements
- [Ollama](https://ollama.com) installed and running locally
- At least one model pulled (e.g., `ollama pull llama3.1`)
- Python `ollama` package installed (`pip install ollama`)

## Inputs

### Required
- **Local Address**: The URL of your Ollama server (default: `http://127.0.0.1:11434`)
  - Click the **🔄 Refresh Models** button after changing the address to load available models
- **Model**: Dropdown of available models from the Ollama server
  - Populated automatically on node creation or when refresh is clicked
  - If empty, make sure Ollama is running and models are installed
- **Free Ram**: Toggle for model memory management
  - **On (default)**: Model is unloaded immediately after generation (`keep_alive=0`)
  - **Off**: Model stays loaded in memory for 5 minutes after generation (`keep_alive=5m`)
- **System Prompt Preset**: Dropdown of preset system prompts
  - **Custom** (default, first option): Use your own system prompt text
  - 15 preset prompts sorted alphabetically, each optimized for a specific model or task:
    - AceStep 1.5 System Prompt
    - AceStep Music Marker LLM
    - BOOGU Image Prompt Refiner
    - Ernie Image Prompt Architect
    - Flux Klein Edit
    - Flux2 Klein HQ Photo System Prompt
    - Flux2 Klein System Prompt
    - Gemma4 Refiner Chinese
    - Ideogram4 Image Prompt Refiner (JSON)
    - Krea2 360 Pano Maker
    - Krea2 Image Prompt Refiner
    - LLM Z-Image Prompt
    - LTX2 System Prompt
    - LTX2.3 Prompt Refiner
    - LTXV 2.3 I2V System Prompt
- **System Prompt**: Custom system prompt text (only visible when preset is "Custom")
  - Defines the role and behavior of the model
- **Prompt**: Your user prompt text
  - This is the input you want the model to process, expand, or refine
- **Temperature**: Controls creativity of the output
  - Range: 0.0 to 2.0, default: 0.8
  - Lower values = more focused/deterministic, higher values = more creative
- **Seed**: Random seed for reproducibility
  - Set to 0 to use random seed (no seed passed to Ollama)
  - Same seed + same inputs = same output
- **Control After Generate**: Seed behavior after each run
  - **fixed**: Keep the same seed
  - **random**: Generate a new random seed
  - **increment**: Increase seed by 1
  - **decrement**: Decrease seed by 1

### Optional
- **image**: IMAGE input for vision language models (e.g., LLaVA, Llama 3.2 Vision)
  - When connected, images are base64-encoded and sent alongside the user prompt
  - Supports image batches (multiple images sent together)
  - Refer to the image in your prompt as "this image", "the photo", etc.
  - Make sure the selected model supports vision, otherwise it may hallucinate

## Outputs
- **result**: The text response from the Ollama model (STRING)
  - Connect to a text widget on any node, or to a CLIP Text Encode node

## Usage

### Basic Prompt Refinement
1. Set up Ollama locally and pull a model (e.g., `ollama pull llama3.1`)
2. Add the **Star Ollama Prompt Helper** node
3. Click **🔄 Refresh Models** to load available models
4. Select a model from the dropdown
5. Choose a system prompt preset (e.g., "Krea2 Image Prompt Refiner")
6. Type your rough prompt idea in the **Prompt** field
7. Run the workflow — the refined prompt appears in the **result** output
8. Connect the output to your text-to-image node

### Custom System Prompt
1. Set **System Prompt Preset** to "Custom"
2. Write your own system prompt in the **System Prompt** field
3. Use the **Prompt** field for your user input

### Memory Management
- Keep **Free Ram** on if you have limited VRAM and need to run other models
- Turn **Free Ram** off if you're running multiple generations in sequence (avoids reload overhead)

## System Prompts File
Preset system prompts are loaded from `image_tools/systemprompts.json`. You can edit this file to add, modify, or remove presets. The file is sorted alphabetically by key name.

## Notes
- The node uses the Ollama Chat API (not Generate API)
- No context/history is kept between runs — each execution is independent
- The `think` and `format` options from the original Ollama nodes are removed for simplicity
- Seed is only passed to Ollama when greater than 0
