# Star 3 LoRAs

## Description
This node allows you to apply up to three LoRA models simultaneously to a base model and CLIP model. Each LoRA can be individually configured with different strength values for both the diffusion model and the CLIP model. The node also maintains a LoRA stack that can be passed to other nodes like FluxStart for internal conditioning.

## Inputs

### Optional
- **model**: The diffusion model to apply the LoRAs to
- **clip**: The CLIP model to apply the LoRAs to
- **LoRA_Stack**: An existing LoRA stack for chaining multiple LoRA nodes

### LoRA 1 Configuration
- **lora1_name**: The first LoRA to apply (select "None" to skip)
- **strength1_model**: Strength of the first LoRA's effect on the diffusion model (default: 1.0, range: -100.0 to 100.0)
- **strength1_clip**: Strength of the first LoRA's effect on the CLIP model (default: 1.0, range: -100.0 to 100.0)

### LoRA 2 Configuration
- **lora2_name**: The second LoRA to apply (default: "None")
- **strength2_model**: Strength of the second LoRA's effect on the diffusion model (default: 1.0, range: -100.0 to 100.0)
- **strength2_clip**: Strength of the second LoRA's effect on the CLIP model (default: 1.0, range: -100.0 to 100.0)

### LoRA 3 Configuration
- **lora3_name**: The third LoRA to apply (default: "None")
- **strength3_model**: Strength of the third LoRA's effect on the diffusion model (default: 1.0, range: -100.0 to 100.0)
- **strength3_clip**: Strength of the third LoRA's effect on the CLIP model (default: 1.0, range: -100.0 to 100.0)

## Outputs
- **model**: The modified diffusion model with all LoRAs applied
- **clip**: The modified CLIP model with all LoRAs applied (if provided)
- **lora_stack**: A stack of all applied LoRAs for use with FluxStart or chaining to another LoRA node

## Usage
1. Connect a model (and optionally a CLIP model) to the node
2. Select up to three LoRAs to apply
3. Adjust the strength values for each LoRA as needed
4. Use the modified model and CLIP outputs in your generation workflow
5. Optionally, pass the lora_stack output to a FluxStart node or another LoRA node

## Advanced Features
- **Negative Strength Values**: You can use negative strength values to invert the effect of a LoRA
- **Sequential Application**: LoRAs are applied in order (1, 2, 3), allowing for controlled layering of effects
- **LoRA Stacking**: The node outputs a LoRA stack that can be passed to other nodes for consistent application across a workflow
- **Model-Only Application**: You can set clip strength to 0 to apply a LoRA only to the diffusion model
- **CLIP-Only Application**: You can set model strength to 0 to apply a LoRA only to the CLIP model

## Notes
- Setting a LoRA name to "None" or its model strength to 0 will skip that LoRA
- The node caches loaded LoRAs for better performance when reusing the same LoRAs
- The LoRA stack output contains information about all applied LoRAs, including their names and strength values
