"""Shared sampling helper: run a CFG-guided sample with an explicit sigma schedule."""

import comfy.samplers
import comfy.sample
from comfy_extras.nodes_custom_sampler import Noise_RandomNoise, SamplerCustomAdvanced


def star_sample_with_sigmas(model, positive, negative, cfg, seed, sampler_name, sigmas, latent):
    """Sample a latent with the given sigma schedule instead of the built-in scheduler."""
    noise_gen = Noise_RandomNoise(seed)
    guider = comfy.samplers.CFGGuider(model)
    guider.set_conds(positive, negative)
    guider.set_cfg(cfg)
    sampler_obj = comfy.samplers.sampler_object(sampler_name)
    return SamplerCustomAdvanced().sample(noise_gen, guider, sampler_obj, sigmas, latent)[0]
