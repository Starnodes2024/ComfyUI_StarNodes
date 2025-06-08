# Star Latent Input (Dynamic)

## Description
This node acts as an automatic switch for latent inputs. It passes the first connected latent to the output, allowing for flexible workflow design where multiple potential latent sources can be connected without manual switching.

## Inputs
- **Latent 1** (optional): First priority latent input
- Additional latent inputs (up to 20) are dynamically supported

## Outputs
- **latent_out**: The first connected latent from the inputs, or a default empty latent if no latents are connected

## Usage
Connect one or more latent sources to this node. The node will automatically select the first available latent and pass it to the output. If no latents are connected, it will generate a default empty latent with dimensions equivalent to a 512x512 image in latent space (64x64x4).

This node is useful for:
- Creating workflows with alternative latent sources
- Building conditional latent processing pipelines
- Simplifying complex workflows by reducing the need for manual switching between latent sources
- Providing a fallback empty latent when no inputs are connected
