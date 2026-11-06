"""Star Minimax All In One Advanced

Same complete MiniMax H3 reference-to-video pipeline as Star Minimax All In
One, plus three optional injection points so externally built conditioning and
latents (e.g. from the ComfyUI-BFSNodes shot nodes like
BFSShotH3ConditioningSimple, or any other node that outputs CONDITIONING /
LATENT) can be sampled with this node's sampler, sigmas, upscale pass and
decode stages.

- positive  (CONDITIONING, optional): when connected, the internal prompt
  encode and reference conditioning are skipped entirely (prompt, ref_* and
  clip widgets are ignored, the text encoder is not even loaded) and the
  connected conditioning is sampled instead.
- negative  (CONDITIONING, optional): when connected, a real CFGGuider at the
  cfg widget value is used instead of the basic positive-only guider.
- latent    (LATENT, optional): when connected, sampling runs on the connected
  latent; aspect ratio, megapixels and duration are ignored (canvas and frame
  count are read back from the latent itself).

Any combination works: positive without latent still builds the internal empty
AV latent, latent without positive still runs the internal ref2va conditioning.
RefMod and multiref guide option nodes apply on top of whichever conditioning
is active, and the model_override / vae_override connectors behave exactly as
on the base node.
"""

import logging

import torch

from comfy_api.latest import io

from .minimax_common import decode_audio, decode_still, run_sample
from .minimax_common import decode_video as _decode_video
from .star_minimax_aio import (
    StarMinimaxAllInOne,
    OUTPUT_FPS,
    _apply_guides,
    _build_conditioning,
    _build_latent,
    _duration_to_length,
    _resolve_dimensions,
)
from ..ltx_video.star_video_sound_enricher import process_audio as _enrich_sound
from ..misc.star_preview import apply_star_preview


def _latent_frame_count(video_member):
    """Inverse of the 17k+5 frame grid -> video latent t mapping."""
    if video_member.ndim != 5:
        return 1
    t = video_member.shape[2]
    return 5 if t <= 2 else ((t - 2) // 5) * 17 + 5


class StarMinimaxAllInOneAdvanced(StarMinimaxAllInOne):
    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = "StarMinimaxAllInOneAdvanced"
        schema.display_name = "⭐ Star Minimax All In One Advanced"
        schema.description = (
            "The complete Star Minimax All In One pipeline with three extra optional "
            "connectors for externally built inputs (e.g. the BFSNodes shot nodes or "
            "any CONDITIONING / LATENT source). Connect positive to sample connected "
            "conditioning instead of the internal prompt + reference encode - the text "
            "encoder is then not loaded at all and prompt, clip and ref inputs are "
            "ignored. Connect negative to sample with a real CFGGuider at the cfg "
            "value (at cfg 1.0 the negative is ignored by the sampler anyway). Connect "
            "latent to sample a connected latent - aspect ratio, megapixels and "
            "duration are then ignored and canvas plus frame count are read back from "
            "the latent. Any combination works; RefMod, multiref guides, sigmas, the "
            "upscale pass, sound enricher and the model/VAE overrides behave exactly "
            "as on the base node."
        )
        denoise_idx = next(i for i, x in enumerate(schema.inputs) if x.id == "denoise")
        schema.inputs.insert(denoise_idx + 1, io.Float.Input(
            "cfg", default=1.0, min=0.0, max=100.0, step=0.1, advanced=True,
            tooltip="CFG scale for the guider. Only used when a negative conditioning is connected; at 1.0 the negative is ignored by the sampler anyway."))
        vae_idx = next(i for i, x in enumerate(schema.inputs) if x.id == "vae_override")
        schema.inputs[vae_idx + 1:vae_idx + 1] = [
            io.Conditioning.Input("positive", optional=True,
                                  tooltip="Optional external positive conditioning (e.g. the 'positive' output of a BFSNodes shot conditioning node). When connected, the internal prompt + reference encode is skipped - the clip is not loaded and prompt, clip and ref inputs are ignored. Connect the matching latent too if the conditioning was built for a specific canvas/duration."),
            io.Conditioning.Input("negative", optional=True,
                                  tooltip="Optional external negative conditioning. When connected, sampling uses a real CFGGuider at the cfg widget value instead of the positive-only basic guider."),
            io.Latent.Input("latent", optional=True,
                            tooltip="Optional external latent (e.g. the 'latent' output of a BFSNodes shot conditioning node). When connected, aspect ratio, megapixels and duration are ignored and canvas + frame count are read back from the latent."),
        ]
        return schema

    # ------------------------------------------------------------------
    @classmethod
    def execute(cls, mode, prompt, aspect_ratio, megapixels, match_ratio_from_image, duration,
                ref_image_size, seed, steps, sampler_name, scheduler, denoise, cfg,
                diffusion_model, weight_dtype, clip_name, clip_type, clip_device,
                vae_name, audio_vae_name, audio_vae_precision, audio_vae_device, decode_video,
                model_override=None, vae_override=None, positive=None, negative=None,
                latent=None, sound_settings=None, sigmas=None, options=None,
                ref_mod_settings=None, multiref_settings=None, preview=None,
                ref_images=None, ref_videos=None, ref_video_audios=None,
                ref_audios=None) -> io.NodeOutput:

        audio_only = (megapixels == "audio only")
        ext_cond = positive is not None
        ext_latent = latent is not None and latent.get("samples") is not None

        # 1. Canvas - external latent wins and reports its own size/frame count
        if ext_latent:
            member = latent["samples"]
            video_member = member.unbind()[0] if getattr(member, "is_nested", False) else member
            width, height = video_member.shape[-1] * 16, video_member.shape[-2] * 16
            length = _latent_frame_count(video_member)
            logging.info("[Star Minimax AIO+] %s mode | external latent: canvas %dx%d, ~%d frames",
                         mode, width, height, length)
        else:
            if mode == "video" and not audio_only and float(megapixels) > 2.0:
                raise ValueError("[Star Minimax AIO+] megapixels above 2.0 are for image mode stills - "
                                 "drop to 2.0 MP or less, or switch mode to 'image'.")
            width, height, matched = _resolve_dimensions(
                aspect_ratio, megapixels, match_ratio_from_image, ref_images)
            length = 1 if mode == "image" else _duration_to_length(duration)
            logging.info("[Star Minimax AIO+] %s mode | canvas %dx%d%s, %d frames (%.2fs @ 24fps)",
                         mode, width, height, " (ratio matched from image)" if matched else "",
                         length, length / 24.0)

        # 2. Models - external MODEL wins over the internal dropdown; the clip is
        #    only loaded when the internal conditioning encode actually runs
        if model_override is not None:
            model = model_override
            logging.info("[Star Minimax AIO+] using connected MODEL override; "
                         "internal diffusion model '%s' ignored", diffusion_model)
        else:
            model = cls._load_model(diffusion_model, weight_dtype)
        clip = None if ext_cond else cls._load_clip(clip_name, clip_type, clip_device)
        if ext_cond:
            logging.info("[Star Minimax AIO+] external positive conditioning - "
                         "clip not loaded; prompt and ref inputs ignored")
            n_refs = sum(1 for d in (ref_images, ref_videos, ref_video_audios, ref_audios)
                         for v in (d or {}).values() if v is not None)
            if n_refs:
                logging.warning("[Star Minimax AIO+] %d reference input(s) connected but ignored "
                                "- external positive conditioning is active", n_refs)
        if vae_override is not None:
            vae = vae_override
            logging.info("[Star Minimax AIO+] using connected VAE override; "
                         "internal video VAE '%s' ignored", vae_name)
        else:
            vae = cls._load_video_vae(vae_name)
        has_audio_refs = (any(a is not None for a in (ref_video_audios or {}).values())
                          or any(a is not None for a in (ref_audios or {}).values()))
        ext_nested = (ext_latent
                      and getattr(latent["samples"], "is_nested", False))
        audio_vae = (cls._load_audio_vae(audio_vae_name, audio_vae_precision, audio_vae_device)
                     if mode == "video" or has_audio_refs or audio_only or ext_nested else None)

        # 3. Conditioning + latent - each side falls back to the internal build
        if ext_cond:
            cond = positive
            latent_out = latent if ext_latent else _build_latent(
                width, height, length, mode == "image")[0]
        else:
            cond, built_latent = _build_conditioning(
                clip, vae, audio_vae, prompt, width, height, length, ref_image_size,
                ref_images, ref_videos, ref_video_audios, ref_audios,
                image_mode=(mode == "image"))
            latent_out = latent if ext_latent else built_latent

        # 3b. Optional ref-mods from a Star Ref Mod Option node - appended after
        #     the active conditioning, exactly like Apply H3 RefMod
        if ref_mod_settings is not None:
            refmod_blocks = ref_mod_settings.get("blocks") or []
            if refmod_blocks:
                out = []
                for t in cond:
                    d = dict(t[1])
                    d["minimax_refs"] = list(d.get("minimax_refs", [])) + refmod_blocks
                    out.append([t[0], d])
                cond = out
                logging.info("[Star Minimax AIO+] %d ref-mod block(s) injected from Star Ref Mod Option",
                             len(refmod_blocks))
        # 3c. Optional timed guides from a Star Minimax Multiref Option node
        guides = (multiref_settings or {}).get("guides") or []
        if guides:
            cond = _apply_guides(cond, latent_out, vae, guides)
            logging.info("[Star Minimax AIO+] %d guide(s) anchored from Star Minimax Multiref Option",
                         len(guides))

        # 4. Sampling - a connected negative conditioning switches the basic
        #    guider for a real CFGGuider at the cfg widget value
        if negative is None and cfg != 1.0:
            logging.warning("[Star Minimax AIO+] cfg %.2f ignored - no negative conditioning connected",
                            cfg)
        if sigmas is not None:
            if sigmas.numel() < 2:
                logging.warning("[Star Minimax AIO+] custom sigmas ignored (need at least 2 sigma values)")
                sigmas = None
            else:
                logging.info("[Star Minimax AIO+] custom sigmas, %d steps", sigmas.numel() - 1)
        if sigmas is None:
            sigmas = cls._get_sigmas(model, scheduler, steps, denoise)
        sample_model = apply_star_preview(model, preview) if preview is not None else model
        samples = run_sample(sample_model, cond, latent_out, seed, sampler_name, sigmas,
                             negative=negative, cfg=cfg)

        # 4b. Optional second pass: latent upscale + refine (Star Minimax Latent Upscaler Option)
        samples_pass1 = samples
        ran_pass2 = options is not None and not audio_only
        if ran_pass2 and not getattr(samples, "is_nested", False):
            logging.warning("[Star Minimax AIO+] upscale options ignored - "
                            "the sampled latent has no AV members to upscale")
            ran_pass2 = False
        if options is not None and audio_only:
            logging.info("[Star Minimax AIO+] 'audio only' mode - upscale options ignored")
        if ran_pass2:
            samples = cls._run_upscale_pass(sample_model, cond, samples, seed, options,
                                            negative=negative, cfg=cfg)

        # 5. Decode (VAEDecode + VAEDecodeAudio) - a latent without an audio
        #    member yields the same silent placeholder as image mode
        if decode_video:
            images = decode_still(vae, samples) if mode == "image" else _decode_video(vae, samples)
        else:
            images = None
            logging.info("[Star Minimax AIO+] video VAE decode disabled - latent and audio outputs only")
        samples_nested = getattr(samples, "is_nested", False)
        if not samples_nested or (mode == "image" and not audio_only):
            audio = {"waveform": torch.zeros([1, 2, 4410]), "sample_rate": 44100}
        else:
            audio_samples = samples_pass1 if (ran_pass2 and not options["upscale_pass_audio"]) else samples
            audio = decode_audio(audio_vae, audio_samples)
            if sound_settings is not None:
                logging.info("[Star Minimax AIO+] applying sound enricher settings to the audio output")
                audio = _enrich_sound(audio, sound_settings)

        return io.NodeOutput(images, audio, OUTPUT_FPS, {"samples": samples}, model, clip, vae, audio_vae)


NODE_CLASS_MAPPINGS = {
    "StarMinimaxAllInOneAdvanced": StarMinimaxAllInOneAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarMinimaxAllInOneAdvanced": "⭐ Star Minimax All In One Advanced",
}
