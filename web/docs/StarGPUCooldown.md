# ⭐ Star GPU Cooldown

A flow-control utility node that completely empties the GPU VRAM, waits a configurable cooldown time, runs a second aggressive cleanup and then lets the workflow continue. Useful between heavy generation steps (e.g. after upscaling/refining and before the next model batch) to let the GPU start with a clean, unfragmented allocator.

- **Category**: `⭐StarNodes/Helpers And Tools`
- **Node name**: `StarGPUCooldown`
- **Output**: passthrough (`*`)

## Inputs

- **wait_seconds** (`FLOAT`, default `10.0`, range `0.0` – `600.0`)
  - Cooldown time in seconds between the first full VRAM cleanup and the second cleanup.

- **clean_ram** (`BOOLEAN`, default `off`)
  - `VRAM only`: only GPU memory is cleared.
  - `Clean RAM too`: additionally runs a full Python garbage collection to free unreferenced RAM.

- **any_input** (`*`, optional)
  - Connect any data type (IMAGE, LATENT, MODEL, ...) to pass it through unchanged after the cooldown. Can also be left unconnected — the node then works as a pure cleanup step.

## Behavior

1. **Cleanup 1/2**: unloads all models from VRAM (`unload_all_models`), flushes the cache (`soft_empty_cache`), synchronizes the GPU and empties the CUDA allocator. If **clean_ram** is enabled, a full garbage collection frees RAM as well.
2. **Cooldown**: sleeps for `wait_seconds` to let the GPU and driver settle.
3. **Cleanup 2/2**: runs the same aggressive cleanup a second time to catch memory that was only released during the wait.
4. Continues the workflow, passing `any_input` through unchanged.

VRAM usage (allocated/reserved) is printed to the console before and after each cleanup step.

The node always re-executes on every queue (`IS_CHANGED` returns NaN), so the cooldown runs even if the inputs did not change.
