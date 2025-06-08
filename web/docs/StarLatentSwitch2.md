# Star Latent Input 2 (Optimized)

## Description
An optimized version of the Star Latent Input node that automatically passes the first provided latent to the output. This node serves as an automatic switch for latent inputs with improved performance.

## Inputs
- **Latent 1** (optional): First priority latent input
- **Latent 2** (optional): Second priority latent input
- **Latent 3** (optional): Third priority latent input
- **Latent 4** (optional): Fourth priority latent input
- **Latent 5** (optional): Fifth priority latent input

## Outputs
- **latent_out**: The first connected latent from the inputs, or a default empty latent if no latents are connected

## Usage
Connect up to five different latent sources to this node. The node will automatically select the first available latent (checking inputs 1 through 5 in order) and pass it to the output. If no latents are connected, it will generate a default empty latent with dimensions equivalent to a 512x512 image in latent space (64x64x4).

This optimized version maintains the same functionality as the original Star Latent Input node but with a fixed number of inputs for better performance in complex workflows.
