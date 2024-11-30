# ComfyUI_StarNodes

 Little Helper Nodes For ComfyUI



This are little nodes that are helping to make big workflows a bit smaller. 

This are nodes i made for my daily work with ComfyUi to make my life a bit easier. Maybe they can help you too.

![nodes1](https://github.com/user-attachments/assets/c0a83f7d-8089-4572-9ca8-a337d4031fbf)


- SD(XL) Starter : Is loading checkpoint (with CLIP and VAE) and create an empty latend (you can choose resolutions or set your own)

- FLUX Starter : Is loading Unet (Diffusion Model), 2 Clips and create an empty latend (you can choose resolutions  or set your own)

- SD3.0/3.5 Starter: Is loading Unet (Diffusion Model), 3 Clips and create an empty latend (you can choose resolutions or set your own)

*you can add more ratios if you edit the .json-files in the node folder.


- Star Model Latent Upscaler: Decode an input Latent and Vae to an Image, upscale with choosen model, resize to given size, decode output image back to laten with selected VAE.

- Star Seven Wildcards: A prompt maker that use 7 inputs with wildcards to create prompts with different wildcards and multiple random inputs from one wildcard used. Also works without wildcards as text concatenate.

- Input Image Chooser: A simple switch for big workflows to switch between 8 input images 

- Ollama Helper: Is loading your Ollama models from ollamamodels.txt file in the nodes folder and pass it and a system prompt to Ollama nodes.
This helps when ollama models are not show up in the list.

## Install:

1. Open CMD within your custom nodes folder

2. Type: git clone https://github.com/Starnodes2024/ComfyUI_StarNodes

3.  Restart ComfyUI

You will find the Nodes under "Starnodes" or search for "star"  
