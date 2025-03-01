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

### Starter Nodes
- ⭐ SD(XL) Starter: Loads checkpoint with CLIP and VAE, creates empty latent with customizable resolution
- ⭐ FLUX Starter: Loads Unet with 2 CLIPs and creates empty latent
- ⭐ SD3.0/3.5 Starter: Loads Unet with 3 CLIPs and creates empty latent

### Upscaling & Processing
- ⭐ Star Model Latent Upscaler: Complete pipeline for latent upscaling with model choice and VAE encoding/decoding

### Text & Prompt Generation
- ⭐ Star Seven Wildcards: Advanced prompt maker with 7 inputs supporting wildcards and multiple random selections
- ⭐ Ollama Helper: Loads Ollama models from ollamamodels.txt for integration with Ollama nodes

### Web & Data 🆕
- ⭐ Star Web Scraper (Headlines) 📰: Scrapes news headlines from websites. Includes URL management with saved sites for quick access. NEW!

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
- Supports recursive wildcard processing up to 10 layers
