# ⭐ Star Stop And Go

## Overview

The **Star Stop And Go** node provides interactive workflow control with preview functionality. It allows you to pause your workflow execution and decide whether to continue or stop processing, making it perfect for reviewing intermediate results before proceeding with expensive operations.

## Star Stop And Go

Interactive workflow control node that allows you to pause, review, and decide whether to continue or stop your ComfyUI workflow execution. **Works with ANY data type** - place it anywhere in your workflow!

## Description

The **Star Stop And Go** node provides manual control over workflow execution with optional preview. It accepts **any input type** (IMAGE, LATENT, MODEL, STRING, etc.) and passes it through unchanged, making it perfect for quality control, debugging, or reviewing intermediate results at any point in your workflow.

## Inputs

All inputs have helpful tooltips when you hover over them in ComfyUI!

### Required

- **mode** (Dropdown)
  - **User Select**: Wait for user to click Stop or Go button
  - **Pause**: Automatically pause for specified seconds then continue
  - **Bypass**: Pass through data without any interaction (preview if image)
  - *Tooltip: "User Select: Wait for GO/STOP button click | Pause: Auto-continue after delay | Bypass: Pass through immediately"*

### Optional

- **any_input** (ANY)
  - Accepts **any data type**: IMAGE, LATENT, MODEL, CONDITIONING, CLIP, VAE, CONTROL_NET, STRING, INT, FLOAT, etc.
  - Node automatically detects the input type
  - If input is an IMAGE, it will be previewed
  - For other types, data passes through without preview
  - Can be left unconnected to use node as a checkpoint only
  - *Tooltip: "Connect any data type (IMAGE, LATENT, MODEL, etc.). Image inputs will show preview if enabled."*

- **pause_seconds** (FLOAT)
  - Default: 5.0
  - Range: 0.1 to 300.0
  - Only used when mode is set to "Pause"
  - Specifies how many seconds to pause before continuing
  - *Tooltip: "How long to pause in Pause mode (0.1 to 300 seconds). Only used when mode is set to Pause."*

- **timeout_seconds** (FLOAT)
  - Default: 300.0 (5 minutes)
  - Range: 10.0 to 3600.0 (10 seconds to 1 hour)
  - Only used when mode is set to "User Select"
  - Specifies how long to wait for user decision before auto-continuing
  - Displayed in user-friendly format (e.g., "5 min", "2 min 30 sec", "45 sec")
  - *Tooltip: "Auto-continue timeout for User Select mode (10 to 3600 seconds). Workflow continues automatically if no decision is made within this time."*

- **preview_image** (BOOLEAN)
  - Default: True
  - When enabled, shows image preview if input is an IMAGE
  - When disabled, skips preview generation even for images
  - Useful for faster processing when preview is not needed
  - Has no effect on non-image inputs
  - *Tooltip: "Show image preview if input is an IMAGE. Disable to skip preview generation for faster processing."*

- **empty_vram_ram** (BOOLEAN)
  - Default: False
  - When enabled, clears VRAM and RAM before executing the mode
  - Unloads all models from memory
  - Clears CUDA cache and Python garbage collection
  - **Shows memory usage before and after cleaning**
  - Displays how much VRAM/RAM was freed
  - Useful for freeing memory before expensive operations or when reviewing intermediate results
  - *Tooltip: "Clear VRAM before executing. Unloads models, clears cache, and shows memory usage before/after. Useful for freeing memory between operations."*

## Outputs

- **output** (ANY)
  - Returns the input data unchanged
  - Automatically matches the type of the input
  - Works with IMAGE, LATENT, MODEL, CONDITIONING, CLIP, VAE, CONTROL_NET, STRING, INT, FLOAT, etc.
  - Can be connected to any node that accepts the same type as the input

## Usage

### Basic Workflow Control

1. Add the **Star Stop And Go** node to your workflow
2. Connect any output to the **any_input** socket
3. Connect the **output** socket to the next node
4. Set mode to "User Select"
5. Run your workflow
6. When the node executes:
   - If input is an IMAGE, **the preview is shown immediately**
   - For other types, you'll see the data type in console
   - **The workflow will PAUSE and wait for your decision**
   - Console shows timeout duration (default: "Auto-continue timeout: 5 min (300 seconds)")
   - Review the preview/output
   - The node polls every 0.5 seconds for your input
   - Click **🛑 STOP Workflow** to halt execution immediately
   - Click **✅ GO Continue** to proceed with the workflow
   - **If no decision is made within the timeout period, the workflow continues automatically**
   - You can customize the timeout using the `timeout_seconds` parameter (10 sec to 1 hour)

### Automatic Pause Mode

1. Set mode to "Pause"
2. Set `pause_seconds` to desired duration (e.g., 5.0)
3. Run workflow
4. The node will:
   - Display the image
   - Pause for the specified seconds
   - Automatically continue

### Bypass Mode

1. Set mode to "Bypass"
2. Run workflow
3. The node will:
   - Display the image
   - Immediately pass it through
   - No user interaction required

## Visual Indicators

The node changes color to indicate its current state:

- **Dark Red (#553333)**: Waiting for user input
- **Red (#AA3333)**: Workflow stopped
- **Green (#33AA33)**: Workflow continuing (Go clicked)
- **Dark Green (#335533)**: Paused
- **Dark Blue (#333355)**: Bypass mode

## Use Cases

### 1. Quality Check Before Upscaling

```
Image Generation → Star Stop And Go → Upscale Node → Save Image
```

Review the generated image before committing to an expensive upscale operation.

### 2. Multi-Stage Processing Review

```
Initial Process → Star Stop And Go → Enhancement → Star Stop And Go → Final Output
```

Review results at multiple stages of your workflow.

### 3. Conditional Workflow Execution

```
Batch Processing → Star Stop And Go → Expensive Operation
```

Stop the workflow if you see an issue in the batch before processing all images.

### 4. Timed Preview

```
Image → Star Stop And Go (Pause mode, 3 seconds) → Next Node
```

Automatically show preview for 3 seconds before continuing.

### 5. Memory Management

```
Large Model Generation → Star Stop And Go (empty_vram_ram=True) → Different Model
```

Clear VRAM between different model operations to prevent out-of-memory errors.

### 6. Review with Memory Cleanup

```
Initial Generation → Star Stop And Go (User Select + empty_vram_ram=True) → Upscale
```

Review the result, clear memory, then decide whether to proceed with upscaling.

### 7. Control Any Data Type

```
Load Model → Star Stop And Go → Apply LoRA
CLIP Text Encode → Star Stop And Go → KSampler
VAE Encode → Star Stop And Go (Latent) → KSampler
```

Use the node anywhere in your workflow with any data type - MODEL, CONDITIONING, LATENT, etc.

## Tips

- **User Select Mode**: Best for manual quality control and decision-making
- **Pause Mode**: Useful for creating timed previews or debugging workflows
- **Bypass Mode**: Use when you want to preview but don't need interaction
- **Memory Clearing**: Enable `empty_vram_ram` when switching between large models or before expensive operations
- The node always re-executes to allow for interactive control
- Images are saved to ComfyUI's temp directory for preview
- Clicking Stop will interrupt the entire workflow execution
- Memory clearing happens before the mode logic (before waiting/pausing)

## Technical Details

- **Category**: ⭐StarNodes/Helpers And Tools
- **Output Node**: Yes (displays UI)
- **Input Type**: ANY (accepts all data types via AlwaysEqualProxy)
- **Return Type**: ANY (matches input type)
- **Function**: process
- **Type Detection**: Automatic via isinstance() checks

## Status Information

The node displays helpful status text:
- **User Select**: "Click STOP to halt or GO to continue"
- **Pause**: "Will pause for X seconds"
- **Bypass**: "Bypassing - image passes through"

## Notes

- The Stop button interrupts the entire ComfyUI execution queue
- The Go button allows the workflow to continue normally
- In User Select mode, the workflow waits indefinitely for user input
- The node supports batch image processing
- Images are temporarily saved for preview purposes

## Keyboard Shortcuts

Currently, no keyboard shortcuts are implemented. Use the on-screen buttons for control.

## Troubleshooting

**Node doesn't stop the workflow**
- Ensure you're in "User Select" mode
- Click the Stop button before the node finishes executing
- Check that ComfyUI's execution hasn't already moved past this node

**Image not displaying**
- Verify the image input is connected
- Check ComfyUI's temp directory permissions
- Ensure the image tensor is valid

**Pause mode not working**
- Verify `pause_seconds` is set to a positive value
- Check that mode is set to "Pause"
- Note that very short pauses (<0.5s) may not be noticeable

## Important Notes

- The Stop and Go buttons are **clickable buttons** that appear in the node interface
- The buttons work similarly to the Dynamic LoRA loader buttons
- Clicking Stop will immediately interrupt the workflow execution
- The node title updates to show the current mode and status

## Console Output

The node provides clean, colored console messages:

**Yellow (Informational):**
- `⭐ StarNodes: Input type: IMAGE - Preview will be shown`
- `⭐ StarNodes: Input type: IMAGE - Preview disabled`
- `⭐ StarNodes: Input type: [type] - No preview available`
- `⭐ StarNodes: Mode: Bypass - Passing through immediately`
- `⭐ StarNodes: Mode: Pause - Workflow will pause for X seconds`
- `⭐ StarNodes: Pause complete - Continuing workflow`
- `⭐ StarNodes: Mode: User Select - Workflow paused, waiting for your decision`
- `⭐ StarNodes: Timeout reached (5 min) - Continuing workflow automatically` (format varies based on timeout setting)
- `⭐ StarNodes: VRAM allocated: X.XX GB, reserved: Y.YY GB / Z.ZZ GB (W.W%)`

**Cyan (Actions):**
- `⭐ StarNodes: Click GO to continue or STOP to halt the workflow`
- `⭐ StarNodes: Auto-continue timeout: 5 min (300 seconds)` (format varies based on timeout setting)
- `⭐ StarNodes: Auto-continue timeout: 2 min 30 sec (150 seconds)` (example with mixed format)
- `⭐ StarNodes: Auto-continue timeout: 45 sec (45 seconds)` (example for seconds only)
- `⭐ StarNodes: Clearing VRAM...`

**Green (Success):**
- `⭐ StarNodes: Preview sent to UI`
- `⭐ StarNodes: User clicked GO - Continuing workflow`
- `⭐ StarNodes: Memory cleared successfully`
- `⭐ StarNodes: VRAM allocated: X.XX GB, reserved: Y.YY GB / Z.ZZ GB (W.W%)`
- `⭐ StarNodes: VRAM freed - allocated: X.XX GB, reserved: Y.YY GB`

**Purple (Stop):**
- `⭐ StarNodes: Workflow stopped by user`

**Red (Errors):**
- `⭐ StarNodes: Error clearing memory: [error details]`

No error tracebacks are shown when stopping - just a clean purple message.

## Technical Implementation

The node uses a polling mechanism for User Select mode:
- When User Select mode is active, the Python backend enters a polling loop
- It checks every 0.5 seconds for a user decision
- Button clicks send HTTP POST requests to `/starnodes/stop_and_go/decision` endpoint
- The backend receives the decision and either continues or stops gracefully
- Stop button triggers `api.interrupt()` from frontend for clean shutdown
- **Timeout is customizable via `timeout_seconds` parameter (default: 300 seconds / 5 minutes)**
- Range: 10 seconds to 3600 seconds (1 hour)
- User is informed about the timeout via console message at the start
- Timeout is displayed in user-friendly format (e.g., "5 min", "2 min 30 sec", "45 sec")

Memory clearing implementation:
- Uses `torch.cuda.memory_allocated()` to measure actively allocated VRAM
- Uses `torch.cuda.memory_reserved()` to measure PyTorch's reserved VRAM cache
- **Reserved memory** shows the total VRAM PyTorch has claimed (includes cache)
- **Allocated memory** shows what's actually in use by tensors
- Displays total VRAM capacity and percentage used (based on reserved)
- Uses `comfy.model_management.unload_all_models()` to unload models from VRAM
- Calls `comfy.model_management.soft_empty_cache()` to clear model cache
- Runs Python garbage collection with `gc.collect()`
- Clears CUDA cache with `torch.cuda.empty_cache()` and `torch.cuda.ipc_collect()`
- Shows exactly how much VRAM was freed for both allocated and reserved (in GB)
- Memory clearing happens before the mode logic executes

Preview control:
- `preview_image` option allows disabling preview generation
- When disabled, saves processing time by skipping image conversion and saving
- Preview is only generated for IMAGE inputs when enabled

## Version History

- **v1.4.1**: Added helpful tooltips to all input parameters for better user experience
- **v1.4.0**: Added customizable timeout_seconds parameter (10 sec to 1 hour, default 300 sec) with user-friendly display format
- **v1.3.5**: Added console message informing user about 300-second auto-continue timeout in User Select mode
- **v1.3.4**: Added preview_image option to control preview generation; removed RAM reporting (only VRAM shown now)
- **v1.3.3**: Enhanced VRAM reporting - now shows both allocated and reserved memory for better accuracy
- **v1.3.2**: Added memory usage reporting - shows VRAM/RAM before and after cleaning with freed amounts
- **v1.3.1**: Fixed preview display - now shows immediately before user decision, not after
- **v1.3.0**: Simplified to single any_input/output using AlwaysEqualProxy - true universal type support
- **v1.2.0**: Changed to ANY type input/output - works with all data types, not just images. Added detailed yellow console messages for all operations
- **v1.1.0**: Added Empty VRAM/RAM option to clear memory before execution
- **v1.0.3**: Added clean colored console output, removed error tracebacks on stop
- **v1.0.2**: Implemented proper User Select mode with API endpoint and polling
- **v1.0.1**: Fixed button visibility and pause mode error
- **v1.0**: Initial release with User Select, Pause, and Bypass modes
