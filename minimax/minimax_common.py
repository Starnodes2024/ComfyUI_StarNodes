"""Shared MiniMax H3 sampling / decode helpers for the StarNodes minimax nodes."""

import torch

import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview

STILL_GROUP, STILL_KEEP = 5, 3   # stills: 1 latent frame replicated into a 5-frame group, pixel frame 3 kept


class GuiderBasic(comfy.samplers.CFGGuider):
    """Same as the core BasicGuider."""
    def set_conds(self, positive):
        self.inner_set_conds({"positive": positive})


def run_sample(model, cond, latent, seed, sampler_name, sigmas):
    """RandomNoise + BasicGuider + KSamplerSelect + SamplerCustomAdvanced, in-process."""
    guider = GuiderBasic(model)
    guider.set_conds(cond)
    sampler = comfy.samplers.sampler_object(sampler_name)

    latent = latent.copy()
    latent_image = comfy.sample.fix_empty_latent_channels(
        guider.model_patcher, latent["samples"],
        latent.get("downscale_ratio_spacial", None),
        latent.get("downscale_ratio_temporal", None))
    latent["samples"] = latent_image

    batch_inds = latent["batch_index"] if "batch_index" in latent else None
    noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    x0_output = {}
    callback = latent_preview.prepare_callback(
        guider.model_patcher, sigmas.shape[-1] - 1, x0_output)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = guider.sample(noise, latent_image, sampler, sigmas,
                            denoise_mask=None, callback=callback,
                            disable_pbar=disable_pbar, seed=seed)
    return samples.to(comfy.model_management.intermediate_device())


def decode_video(vae, samples):
    """VAEDecode on the video member."""
    latent = samples.unbind()[0] if samples.is_nested else samples
    images = vae.decode(latent)
    if len(images.shape) == 5:  # combine batches
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return images


def decode_still(vae, samples):
    """Decodes an H3 still the way Fizgig's previews do: H3's ViT decoder was
    trained on 5-latent temporal groups, so a lone latent token (a still) comes
    back banded and dark from the stock VAE Decode. The single latent frame is
    replicated into a full 5-latent group, decoded (spatially tiled as usual)
    and pixel frame 3 - just past the decoder's causal lead-in - is kept."""
    latent = samples.unbind()[0] if samples.is_nested else samples
    fsm = vae.first_stage_model
    if latent.ndim != 5 or latent.shape[2] != 1 or not hasattr(fsm, "_adaptive_decode"):
        return decode_video(vae, samples)   # clips / other VAEs: the stock path
    group_shape = (1, latent.shape[1], STILL_GROUP, latent.shape[3], latent.shape[4])
    mem = vae.memory_used_decode(group_shape, vae.vae_dtype)
    comfy.model_management.load_models_gpu([vae.patcher], memory_required=mem,
                                           force_full_load=getattr(vae, "disable_offload", False))
    out = []
    with torch.no_grad():
        for b in range(latent.shape[0]):
            z = latent[b:b + 1].to(device=vae.device, dtype=vae.vae_dtype)
            lm = fsm.latents_mean.view(1, -1, 1, 1, 1).to(z)
            ls = fsm.latents_std.view(1, -1, 1, 1, 1).to(z)
            zz = (z * ls + lm).repeat(1, 1, STILL_GROUP, 1, 1)
            raw = fsm._adaptive_decode(zz)                            # [1, 3, 20, H, W], tiled as usual
            px = fsm._finalize_pixels(raw[:, :, STILL_KEEP:STILL_KEEP + 1])   # [1, 3, 1, H, W] in [0, 1]
            out.append(px[:, :, 0].movedim(1, -1).to(comfy.model_management.intermediate_device()))
            del raw, zz
    return torch.cat(out)


def decode_audio(audio_vae, samples):
    """VAEDecodeAudio on the audio member, level-normalized like the stock node."""
    latent = samples.unbind()[-1] if samples.is_nested else samples
    audio = audio_vae.decode(latent).movedim(-1, 1)
    std = torch.std(audio, dim=[1, 2], keepdim=True) * 5.0
    std[std < 1.0] = 1.0
    audio = audio / std
    vae_sr = getattr(audio_vae, "audio_sample_rate_output",
                     getattr(audio_vae, "audio_sample_rate", 44100))
    return {"waveform": audio, "sample_rate": vae_sr}
