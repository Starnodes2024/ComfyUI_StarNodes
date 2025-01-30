# ComfyUI_StarNodes

 Little Helper Nodes For ComfyUI
 
![nodes1](https://github.com/user-attachments/assets/7e858aeb-3cf7-4675-bc44-473ec345ef1c)

This are little nodes that are helping to make big workflows a bit smaller. 

This are nodes i made for my daily work with ComfyUi to make my life a bit easier. Maybe they can help you too.

- ⭐ NEW Star Text Filter: Simple filter to clean string text, remove blanks, empty lines or text beween two given words.

- ⭐ NEW Star Seven Inputs(img): Automatic input image switch. Always pass the first provided input image to the output

- ⭐ NEW Star Seven Inputs(txt): Text Concat with optional inputs. Works as automatic switch too. If more than one inputs are provided it will concatenate the inputs.

- ⭐ Detail Star Deamon is adapted from the original sources. This can improve the details of your generated images and can be connected to the Starsamplers. Works with Flux and all SD Models.
Read more about the settings and how it works on the original sources: https://github.com/muerrilla/sd-webui-detail-daemon / https://github.com/Jonseed/ComfyUI-Detail-Daemon

- ⭐ StarSampler SD / SDXL: A Ksampler for SD, SDXL, SD3.5 with settings and outputs for model and conditioning passtrough. Optional: Connector for Detail Star Deamon

- ⭐ StarSampler FLUX: A Ksampler for Flux with settings and outputs for model and conditioning passtrough. Optional: Connector for Detail Star Deamon

- ⭐ SD(XL) Starter : Is loading checkpoint (with CLIP and VAE) and create an empty latend (you can choose resolutions or set your own)

- ⭐ FLUX Starter : Is loading Unet (Diffusion Model), 2 Clips and create an empty latend (you can choose resolutions  or set your own)

- ⭐ SD3.0/3.5 Starter: Is loading Unet (Diffusion Model), 3 Clips and create an empty latend (you can choose resolutions or set your own)

*you can add more ratios if you edit the .json-files in the node folder.


- ⭐ Star Model Latent Upscaler: Decode an input Latent and Vae to an Image, upscale with choosen model, resize to given size, decode output image back to laten with selected VAE.

- ⭐ Star Seven Wildcards: A prompt maker that use 7 inputs with wildcards to create prompts with different wildcards and multiple random inputs from one wildcard used. Also works without wildcards as text concatenate.
  You will find many wildcards in the "wildcards" subfolder of the node. To use them just copy the whole folder to your main ComfyUI directory.
  NEW: Recursive wildcard processing up to 10 layers. 

- ⭐ Input Image Chooser: A simple switch for big workflows to switch between 8 input images 

- ⭐ Ollama Helper: Is loading your Ollama models from ollamamodels.txt file in the nodes folder and pass it and a system prompt to Ollama nodes.
This helps when ollama models are not show up in the list.

## Install:
Just search for Starnodes in ComfyUI Manager

Manual install:
1. Open CMD within your custom nodes folder
2. Type: git clone https://github.com/Starnodes2024/ComfyUI_StarNodes
3.  Restart ComfyUI

You will find the Nodes under "Starnodes" or search for "star"  
