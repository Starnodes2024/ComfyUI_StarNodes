# Starnode Ollama Helper

## Description
The Starnode Ollama Helper is a utility node that simplifies integration with Ollama language models in ComfyUI workflows. It provides a convenient way to select from available Ollama models and define system instructions for use with Ollama-compatible nodes. This node is particularly useful for workflows that leverage local language models for text generation, prompt enhancement, or other AI-assisted tasks.

## Inputs

### Required
- **Model**: Selection of available Ollama models from the ollamamodels.txt file
- **Instructions**: System instructions to guide the Ollama model's behavior and output

## Outputs
- **Instructions (System)**: The system instructions passed through for use with Ollama nodes
- **Ollama Model**: The selected model name passed through for use with Ollama nodes

## Usage
1. Create or modify the ollamamodels.txt file in the comfyui_starnodes directory with your available Ollama models (one per line)
2. Select the desired model from the dropdown list
3. Enter system instructions to guide the model's behavior
4. Connect the outputs to compatible Ollama integration nodes

## Features

### Dynamic Model Selection
- Automatically loads available models from the ollamamodels.txt file
- Updates the dropdown list whenever the file is modified
- Provides a clear error message if no models are found

### Customizable System Instructions
- Includes a default prompt template optimized for image prompt generation
- Supports multiline text input for complex instructions
- Allows complete customization of the system message

### Seamless Integration
- Compatible with various Ollama integration nodes
- Simplifies workflow creation by centralizing model selection
- Enables quick switching between different Ollama models

## Technical Details
- The node reads available models from the ollamamodels.txt file in the comfyui_starnodes directory
- If the file doesn't exist or is empty, "No models found" will be displayed
- The first model in the list is automatically set as the default
- The node passes through both the selected model name and instructions without modification

## Notes
- You must have Ollama installed separately and models downloaded through the Ollama interface
- The ollamamodels.txt file should contain one model name per line (e.g., llama3, mistral, etc.)
- This node does not directly communicate with Ollama; it only provides selection and configuration
- For best results, pair this node with dedicated Ollama integration nodes
- The default system instruction is optimized for generating image prompts
