"""Shared MiniMax H3 sampling / decode helpers for the StarNodes minimax nodes."""

import torch

import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview

IMAGE_MODE_FRAMES = 9    # stills: 9 frames fully rendered, frame index 8 is the output


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


def decode_video(vae, samples, image_mode=False):
    """VAEDecode on the video member (all 9 frames, then frame 8, in image mode)."""
    latent = samples.unbind()[0] if samples.is_nested else samples
    images = vae.decode(latent)
    if len(images.shape) == 5:  # combine batches
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    if image_mode:
        images = images[min(IMAGE_MODE_FRAMES - 1, images.shape[0] - 1):][:1]
    return images


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
