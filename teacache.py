import math
import torch
from torch import Tensor

# --- TeaCache patching for Flux, adapted from teacache nodes.py ---

def poly1d(coefficients, x):
    result = torch.zeros_like(x)
    for i, coeff in enumerate(coefficients):
        result += coeff * (x ** (len(coefficients) - 1 - i))
    return result

SUPPORTED_MODELS_COEFFICIENTS = {
    "flux": [4.98651651e+02, -2.83781631e+02, 5.58554382e+01, -3.82021401e+00, 2.64230861e-01],
}

def teacache_flux_forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor = None,
        control = None,
        transformer_options={},
        attn_mask: Tensor = None,
    ):
    patches_replace = transformer_options.get("patches_replace", {})
    rel_l1_thresh = transformer_options.get("rel_l1_thresh")
    coefficients = transformer_options.get("coefficients")
    max_skip_steps = transformer_options.get("max_skip_steps")

    img = self.img_in(img)
    vec = self.time_in(self.timestep_embedding(timesteps, 256).to(img.dtype))
    if self.params.guidance_embed:
        if guidance is None:
            raise ValueError("Didn't get guidance strength for guidance distilled model.")
        vec = vec + self.guidance_in(self.timestep_embedding(guidance, 256).to(img.dtype))
    vec = vec + self.vector_in(y[:,:self.params.vec_in_dim])
    txt = self.txt_in(txt)
    ids = torch.cat((txt_ids, img_ids), dim=1)
    pe = self.pe_embedder(ids)
    blocks_replace = patches_replace.get("dit", {})

    img_mod1, _ = self.double_blocks[0].img_mod(vec)
    modulated_inp = self.double_blocks[0].img_norm1(img)
    modulated_inp = self.apply_mod(modulated_inp, (1 + img_mod1.scale), img_mod1.shift)

    if not hasattr(self, 'accumulated_rel_l1_distance'):
        should_calc = True
        self.accumulated_rel_l1_distance = 0
        self.skip_steps = 0
    elif self.skip_steps == max_skip_steps:
        should_calc = True
        self.accumulated_rel_l1_distance = 0
        self.skip_steps = 0
    else:
        try:
            self.accumulated_rel_l1_distance += poly1d(coefficients, ((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()))
            if self.accumulated_rel_l1_distance < rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
                self.skip_steps = 0
        except Exception:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
            self.skip_steps = 0

    self.previous_modulated_input = modulated_inp.detach()
    if not should_calc:
        self.skip_steps += 1
        return img
    # ...rest of forward (omitted for brevity, patching logic only)
    return img

# Patch application function for Flux models

def apply_teacache(model, model_type: str = "flux", rel_l1_thresh: float = 0.40, max_skip_steps: int = 1):
    if model_type != "flux":
        raise NotImplementedError("Only flux teacache patching supported here.")
    coefficients = SUPPORTED_MODELS_COEFFICIENTS[model_type]
    def unet_wrapper_function(model_function, kwargs):
        transformer_options = kwargs.get("transformer_options", {})
        transformer_options.update({
            "rel_l1_thresh": rel_l1_thresh,
            "coefficients": coefficients,
            "max_skip_steps": max_skip_steps,
            "enable_teacache": True,
        })
        kwargs["transformer_options"] = transformer_options
        return model_function(**kwargs)
    model.set_model_unet_function_wrapper(unet_wrapper_function)
    return (model,)
