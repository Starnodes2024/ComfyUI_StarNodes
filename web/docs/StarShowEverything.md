# ⭐ Star Show Everything

## Overview

The **Star Show Everything** node is a universal debugging and inspection utility. It accepts **any input type** (MODEL, IMAGE, LATENT, CONDITIONING, STRING, INT, FLOAT, VAE, CLIP, etc.), previews it when possible, and passes it through unchanged. It also generates a human-readable **info string** describing the connected value — type, shape, dtype, model names, dict keys, and more — which can be routed to other nodes.

## Description

Place this node anywhere in your workflow to inspect what is flowing through a connection. It is especially useful for:

- Identifying **model names** and file paths from loaded checkpoints, LoRAs, VAEs, CLIPs, etc.
- Checking **tensor shapes and dtypes** for images, latents, and conditioning.
- Viewing **dict keys and values** for complex objects.
- Previewing **image tensors** (up to 4 batch items saved as temp previews).
- Reading **int/float/bool/str** values directly.
- Passing the original value through unchanged so the workflow continues normally.

The `info` STRING output contains a multi-line summary of the connected value and can be connected to any node that accepts a STRING (e.g., a text display node, a file saver, or a prompt builder).

## Inputs

### Required

- **anything** (`*` / ANY)
  - Accepts **any data type**: MODEL, IMAGE, LATENT, CONDITIONING, CLIP, VAE, CONTROL_NET, STRING, INT, FLOAT, LIST, DICT, etc.
  - The value is inspected and passed through unchanged.
  - *Tooltip: "Connect any data type — MODEL, IMAGE, LATENT, CONDITIONING, STRING, INT, FLOAT, etc. The value is previewed and passed through unchanged."*

## Outputs

- **anything** (`*` / ANY)
  - The original input value, passed through unchanged.
  - Connect this to the next node in your workflow as if this node wasn't there.

- **info** (`STRING`)
  - A multi-line human-readable summary of the connected value.
  - Includes: type name, detected model/file name (when applicable), tensor shape/dtype/device, min/max/mean for large tensors, dict keys, conditioning metadata, scalar values, and more.
  - Connect to any STRING input on other nodes.

- **name** (`STRING`)
  - Outputs only the detected name of the connected value (e.g., `flux1-dev.safetensors`).
  - Empty string if no name could be detected.
  - Useful for feeding model filenames or paths into other nodes without parsing the full info text.

## Info Text Examples

**For a MODEL:**
```
Type: ModelPatcher
Name: flux1-dev.safetensors
  model name: flux1-dev.safetensors
```

**For an IMAGE tensor:**
```
Type: Tensor
Shape: (1, 1024, 1024, 3)
Dtype: torch.float32
Device: cuda:0
Min: 0.000000
Max: 1.000000
Mean: 0.482316
```

**For an INT:**
```
Type: int
Value: 42
```

**For CONDITIONING:**
```
Type: list
Length: 1
Item[0] type: tuple
Detected: CONDITIONING
  pooled_output: tensor (1, 4096)
  width: 1024
  height: 1024
```

## Notes

- Image previews are saved to the ComfyUI temp folder under `star_show_everything/` and displayed in the node UI.
- Up to 4 batch images are previewed.
- The info text is also shown in the node's UI text area for quick reading without connecting anything.
- The node is an `OUTPUT_NODE`, so it will execute even if nothing is connected to its outputs.
- The ANY wildcard type means ComfyUI will allow any connection type to be wired in.

---

**Category**: `⭐StarNodes/Helpers And Tools`

**Node name**: `StarShowEverything`

**Display name**: `⭐ Star Show Everything`
