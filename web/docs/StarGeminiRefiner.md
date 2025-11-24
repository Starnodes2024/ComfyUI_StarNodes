# ⭐ Star Gemini Refiner

This node uses Google's Gemini models to refine and enhance image generation prompts. It is designed to take a simple input idea and transform it into a detailed, high-quality prompt suitable for advanced image generation models like "Gemono 3 Pro".

## Inputs

- **text_input**: The base text or concept you want to refine.
- **system_prompt**: Instructions for the AI on how to refine the prompt. Pre-filled with an optimized instruction for image prompt engineering.
- **model**: Select the Gemini model to use (e.g., `gemini-2.0-flash-exp`, `gemini-1.5-pro`).

## Outputs

- **refined_prompt**: The enhanced and detailed prompt string ready for use with image generation nodes.

## Setup

This node requires a Google Gemini API key. It shares the configuration with the `StarNanoBanana` node.
Ensure your `googleapi.ini` file is set up correctly in the node directory or the fallback path.

## Usage

1. Connect a string to `text_input` (or type it manually).
2. Connect the `refined_prompt` output to your CLIP Text Encode node or any other node requiring a string prompt.
