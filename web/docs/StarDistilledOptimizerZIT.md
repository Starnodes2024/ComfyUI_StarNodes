# ⭐ Distilled Optimizer (QWEN/ZIT)

This node enables a **two-pass “distilled” sampling strategy** inspired by Z-Image-Turbo (ZIT) style workflows.

It is designed to connect into the **`options`** input of **⭐ StarSampler (Unified)**.

---

## Node overview

- **Category**
  - `⭐StarNodes/Sampler`
- **Node name**
  - `StarDistilledOptimizerZIT`
- **Display name**
  - `⭐ Distilled Optimizer (QWEN/ZIT)`
- **Output**
  - `options` (`*`)

---

## Input

- **enable** (`BOOLEAN`)
  - Enables/disables the optimizer.
  - When disabled, StarSampler ignores this options input and behaves normally.

- **start_sampler** (SAMPLER)
  - Sampler used for the initial ZIT pass.

- **refine_sampler** (SAMPLER)
  - Sampler used for the refinement pass. Useful if `res_multistep` is not available in your ComfyUI build.

- **start_steps** (`INT`)
  - Steps used for the initial ZIT pass.

- **refine_steps** (`INT`)
  - Steps used for the refinement pass. Set to `0` to disable the refine pass.

- **start_denoise** (`FLOAT`)
  - Denoise for the initial ZIT pass (typically `1.0`).

- **refine_denoise** (`FLOAT`)
  - Denoise for the refinement pass (typical ZIT preset uses `0.6`).

- **patch_shift** (`FLOAT`)
  - Model sampling patch shift value (ModelSamplingZImage-style patch).

- **patch_multiplier** (`FLOAT`)
  - Model sampling patch multiplier value (ModelSamplingZImage-style patch).

- **noise_multiplier** (`FLOAT`)
  - Multiplier applied to the generated noise latent.

---

## What it does (high level)

When connected to ⭐ StarSampler, StarSampler will:

1. **Pass 1 (base / turbo pass)**
   - Creates a fresh noise latent.
   - Applies a temporary model patch that adjusts the model’s internal sampling schedule.
   - Runs the first part of the steps using your sampler/scheduler selection.

2. **Pass 2 (refine / detail pass)**
   - Uses the latent from pass 1.
   - Runs a short refinement with **lower denoise** to recover fine detail.
   - Uses the configured **refine_sampler** (with fallback if it’s missing).

This mirrors the common “fast base + short refine” approach used in high-detail ZIT workflows.

---

## Notes / Compatibility

- This optimizer is meant to be **model-agnostic**.
- If the current model does not support the internal patching approach, StarSampler will **ignore** the optimizer and fall back to normal sampling.

---

## How to use

1. Add **⭐ StarSampler (Unified)**.
2. Add **⭐ Distilled Optimizer (QWEN/ZIT)**.
3. Connect:
   - `⭐ Distilled Optimizer (QWEN/ZIT) -> options` into `⭐ StarSampler (Unified) -> options`.
4. Toggle **Enable** on/off.

---

## Recommended starting point

- **enable**: `True`
- **start_steps**: `6`
- **refine_steps**: `3`
- **start_sampler**: `euler`
- **refine_sampler**: `res_multistep` (or `euler` if unavailable)
- **start_denoise**: `1.0`
- **refine_denoise**: `0.6`
- **patch_shift**: `2.55`
- **patch_multiplier**: `1.0`
- **noise_multiplier**: `1.0`
