# ComfyUI_StarNodes

Little Helper Nodes For ComfyUI
 
![nodes1](https://github.com/user-attachments/assets/7e858aeb-3cf7-4675-bc44-473ec345ef1c)

A collection of utility nodes designed to simplify and enhance your ComfyUI workflows.

## Available Nodes

### Input & Control Nodes
- ⭐ Star Seven Inputs(img): Switch that automatically passes the first provided input image to the output
- ⭐ Star Seven Inputs(txt): Text concatenation with optional inputs. Works as automatic switch and concatenates multiple inputs
- ⭐ Star Seven Inputs(latent): Switch that automatically passes the first provided latent to the output
- ⭐ Star Text Filter: Cleans string text by removing blanks, empty lines or text between two given words
- ⭐ Star Face Loader: Specialized node for handling face-related operations. Image loader that works like the "load image" node but saves images in a special faces-folder for later use.

### Model & Sampling Nodes
- ⭐ StarSampler SD/SDXL: Advanced sampler for SD, SDXL, SD3.5 with model and conditioning passthroughs
- ⭐ StarSampler FLUX: Specialized sampler for Flux models with model and conditioning passthroughs
- ⭐ Detail Star Daemon: Enhances image details, compatible with Flux and all SD Models (Adapted from [original sources](https://github.com/muerrilla/sd-webui-detail-daemon))
- ⭐ Star FluxFill Inpainter 🆕: Specialized inpainting node for Flux models with optimized conditioning and noise mask handling
- ⭐ Star 3 LoRAs 🆕: Applies up to three LoRAs simultaneously to a model with individual weight controls for each. NEW!

### Starter Nodes
- ⭐ SD(XL) Starter: Loads checkpoint with CLIP and VAE, creates empty latent with customizable resolution
- ⭐ FLUX Starter: Loads Unet with 2 CLIPs and creates empty latent
- ⭐ SD3.0/3.5 Starter: Loads Unet with 3 CLIPs and creates empty latent

### Upscaling & Processing
- ⭐ Star Model Latent Upscaler: Complete pipeline for latent upscaling with model choice and VAE encoding/decoding

### Text & Prompt Generation
- ⭐ Star Seven Wildcards: Advanced prompt maker with 7 inputs supporting wildcards and multiple random selections
- ⭐ Star Easy-Text-Storage 🆕: Save, load, and manage text snippets for reuse across workflows. Perfect for storing prompts, system messages, and other text content.
- ⭐ Ollama Helper: Loads Ollama models from ollamamodels.txt for integration with Ollama nodes

### Web & Data 🆕
- ⭐ Star Web Scraper (Headlines) 📰: Scrapes news headlines from websites. Includes URL management with saved sites for quick access. NEW!
- ⭐ Star 7 Layers 2 PSD 🆕: Saves up to seven images as layers in a single PSD file with automatic sizing based on the largest image dimensions. NEW!

*Note: You can add custom resolutions by editing the .json files in the node folder.

## Installation

### Via ComfyUI Manager (Recommended)
Search for "Starnodes" in ComfyUI Manager and install

### Manual Installation
1. Open CMD within your custom nodes folder
2. Run: `git clone https://github.com/Starnodes2024/ComfyUI_StarNodes`
3. Restart ComfyUI

Find the nodes under "⭐StarNodes" category or search for "star" in the node browser.

### Wildcards
For the Star Seven Wildcards node, you'll find many wildcards in the "wildcards" subfolder. Copy this folder to your main ComfyUI directory to use them.

## Wildcard Rules in the Star Wildcards Node

### Basic Wildcard Syntax
- Wildcards are defined using double underscores: `__wildcard_name__`
- The node looks for text files in the `wildcards` folder of your ComfyUI installation
- When a wildcard is encountered, a random line from the corresponding text file is selected

### Folder Structure
- Wildcards can be organized in subfolders
- To use a wildcard in a subfolder, use the syntax: `folder\__wildcard_name__`
- The system will look for the file at `[ComfyUI base path]/wildcards/folder/wildcard_name.txt`

### Random Options
- You can use curly braces `{}` with pipe symbols `|` to choose randomly between options
- Example: `{option1|option2|option3}` will randomly select one of the options
- You can even include wildcards inside these options: `{__wildcard1__|__wildcard2__}`

### Nested Wildcards
- Wildcards can be nested within other wildcards
- The system supports up to 10 levels of recursion to prevent infinite loops
- When a wildcard contains another wildcard, the nested wildcard is also processed

### Seed Behavior
- The node takes a seed parameter that determines the randomization
- Each prompt input (1-7) uses a different seed offset to ensure variety
- Each wildcard within a prompt also gets a unique seed based on its position

### Error Handling
- If a wildcard file doesn't exist, the wildcard name itself is used as text
- If a wildcard file exists but is empty, the wildcard name is used as fallback
- If there's an error processing a wildcard, it falls back to using the wildcard name

The Star Wildcards node allows you to combine up to 7 different prompts, each with their own wildcards, which are then joined together with spaces to create the final output string.

- Supports recursive wildcard processing up to 10 layers
