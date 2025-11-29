# ⭐ Star FlowMatch Option

The **⭐ Star FlowMatch Option** node exposes all key parameters of the
**FlowMatch Euler Discrete Scheduler** and outputs a custom `SIGMAS`
noise schedule that can be used by the unified **⭐ StarSampler**.

When connected, this node lets you drive Flux/Aura-style models with a
highly tuned FlowMatch schedule instead of the default scheduler curve.

---

## Node overview

- **Category**
  - `⭐StarNodes/Sampler`
- **Node name**
  - `⭐ Star FlowMatch Option`
- **Output**
  - `options` (`SIGMAS`)

The output is intended to be connected to the **`options`** input of the
`⭐ StarSampler (Unified)` node.

---

## Inputs

- **steps** (`INT`, default `9`)
  - Number of diffusion steps. For Turbo / Flux-like models, 9 steps is a
    good default (8 DiT forwards).

- **base_image_seq_len** (`INT`, default `256`)
  - Base sequence length for dynamic shifting. Typically matches the
    training resolution of the model (e.g. 256 for 512×512 images).

- **base_shift** (`FLOAT`, default `0.5`)
  - Stabilizes generation and helps control overall behavior of the
    schedule. `0.5` works well for many Turbo/Flux-style models.

- **invert_sigmas** (`disable` / `enable`)
  - Reverses the sigma schedule. Usually keep **disabled** unless
    experimenting.

- **max_image_seq_len** (`INT`, default `8192`)
  - Maximum sequence length for dynamic shifting, important for large
    resolutions.

- **max_shift** (`FLOAT`, default `1.15`)
  - Maximum variation allowed in the schedule. `1.15` is tuned for
    Turbo/Flux-style models.

- **num_train_timesteps** (`INT`, default `1000`)
  - Number of training timesteps used by the model (commonly `1000`).

- **shift** (`FLOAT`, default `3.0`)
  - Global shift of the timestep schedule. `3.0` is a sweet-spot for
    Turbo-like models.

- **shift_terminal** (`FLOAT`, default `0.0`)
  - End value for the shifted schedule. `0.0` disables terminal shift
    and is usually fine.

- **stochastic_sampling** (`disable` / `enable`)
  - Adds controlled randomness to each step (similar to ancestral
    samplers). Enable for more variation; keep disabled for maximum
    determinism.

- **time_shift_type** (`exponential` / `linear`)
  - Method for resolution-dependent shifting. `exponential` is
    recommended for most workflows.

- **use_beta_sigmas** (`disable` / `enable`)
  - Use beta-distributed sigmas. Experimental noise schedule.

- **use_dynamic_shifting** (`disable` / `enable`)
  - Automatically adapts timesteps to image resolution. Disabled by
    default for more consistent Turbo/Flux behavior.

- **use_exponential_sigmas** (`disable` / `enable`)
  - Use exponential spacing for sigmas.

- **use_karras_sigmas** (`disable` / `enable`)
  - Use a Karras-style noise schedule (often smoother transitions,
    DPM++-like behavior).

---

## How to use with ⭐ StarSampler (Unified)

1. **Add nodes**
   - Place `⭐ StarSampler (Unified)` in your workflow.
   - Add `⭐ Star FlowMatch Option` nearby.

2. **Connect options**
   - Connect `Star FlowMatch Option` → `options` input on
     `⭐ StarSampler (Unified)`.

3. **Use a Flux/Aura-style model**
   - When a Flux/Aura model is detected and `options` is connected,
     **StarSampler** will use the `SIGMAS` from this node instead of the
     internal scheduler curve.

4. **Tune quality / speed**
   - Adjust **steps**, **shift**, **max_shift**, and the other FlowMatch
     parameters to find your preferred quality / speed trade‑off.

If the `options` input is not connected, StarSampler falls back to its
normal ComfyUI scheduler behavior.

---

## Global scheduler registration

This extension also registers a global
`FlowMatch Euler DiscScheduler` entry in the ComfyUI scheduler list
(using tuned defaults). You can select it from compatible sampler
nodes as an alternative to the standard schedules.

---

## Credits & Thanks

The FlowMatch Euler Discrete Scheduler integration and parameter design
in this extension are strongly inspired by and adapted from the work of
**erosDiffusion**:

- Original repository:
  - https://github.com/erosDiffusion/ComfyUI-EulerDiscreteScheduler

Many thanks to the original author for publishing and sharing the
FlowMatch Euler scheduler implementation for ComfyUI.
