"""Star Minimax All In One

A single ComfyUI node that runs the complete MiniMax H3 reference-to-video
pipeline internally — model loading, reference conditioning, sampling and
video/audio VAE decoding — with no sub-graph and no wrapper nodes.

Replicates (in-process) the exact behavior of:
  UNETLoader, CLIPLoader, VAELoader, VAELoaderKJ (audio VAE, FP32 selectable),
  ResolutionSelector, MathExpression (duration -> frame count),
  MiniMaxH3ReferenceToVideo, RandomNoise, BasicGuider, KSamplerSelect,
  BasicScheduler, SamplerCustomAdvanced, VAEDecode and VAEDecodeAudio.

Requires a ComfyUI version with MiniMax H3 support (comfy_extras.nodes_minimax_h3)
and a frontend with Autogrow input support — the same requirement as the stock
MiniMax H3 template workflow.
"""

import logging
import math

import torch
import torchaudio

import folder_paths
import nodes
import node_helpers
import comfy.model_management
import comfy.nested_tensor
import comfy.sample
import comfy.samplers
import comfy.sd
import comfy.utils
import latent_preview
from comfy_api.latest import io

# ---------------------------------------------------------------------------
# MiniMax H3 helpers — imported from ComfyUI core, with local fallbacks so the
# node keeps working if module layout ever shifts.
# ---------------------------------------------------------------------------
try:
    from comfy_extras.nodes_minimax_h3 import (
        AUDIO_LATENT_FPS,
        CANVAS_MULTIPLE,
        FPS,
        REF_IMAGE_SHORT_EDGE,
        _empty_av_latent,
        _resize,
        adapt_canvas,
    )
except Exception:  # pragma: no cover - fallback definitions (identical logic)
    CANVAS_MULTIPLE = 32
    BASE_SHORT_EDGE = 768
    MAX_PIXELS = 768 * 1344
    REF_IMAGE_SHORT_EDGE = 2048
    FPS = 24
    AUDIO_LATENT_FPS = 40

    def _resize(image, width, height, crop):
        samples = image[..., :3].movedim(-1, 1)
        samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
        return samples.movedim(1, -1)

    def _align_frame_count(n):
        while n % 17 != 5:
            n += 1
        return n

    def _video_latent_t(frame_count):
        return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2

    def _empty_av_latent(width, height, length, batch_size=1):
        frame_count = _align_frame_count(max(5, length))
        duration = frame_count / FPS
        video = torch.zeros(
            [batch_size, 24, _video_latent_t(frame_count), height // 16, width // 16],
            device=comfy.model_management.intermediate_device())
        audio = torch.zeros(
            [batch_size, 32, 2, round(duration * AUDIO_LATENT_FPS)],
            device=comfy.model_management.intermediate_device())
        return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}, frame_count

    def adapt_canvas(width, height):
        ratio = width / height
        if ratio >= 1.0:
            nom_w, nom_h = BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE
        else:
            nom_w, nom_h = BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio
        if nom_w * nom_h > MAX_PIXELS:
            s = math.sqrt(MAX_PIXELS / (nom_w * nom_h))
            nom_w, nom_h = nom_w * s, nom_h * s
        return (max(CANVAS_MULTIPLE, round(nom_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
                max(CANVAS_MULTIPLE, round(nom_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE))


# ---------------------------------------------------------------------------
# Resolution tables (identical to the core ResolutionSelector node)
# ---------------------------------------------------------------------------
ASPECT_RATIOS = {
    "1:1 (Square)": (1, 1),
    "2:3 (Portrait Photo)": (2, 3),
    "3:2 (Photo)": (3, 2),
    "3:4 (Portrait Standard)": (3, 4),
    "4:3 (Standard)": (4, 3),
    "9:16 (Portrait Widescreen)": (9, 16),
    "16:9 (Widescreen)": (16, 9),
    "2:1 (Panorama)": (2, 1),
    "21:9 (Ultrawide)": (21, 9),
}

DEFAULT_DIFFUSION_MODEL = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
DEFAULT_CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
DEFAULT_VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
DEFAULT_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"

CLIP_TYPES = ["stable_diffusion", "stable_cascade", "sd3", "stable_audio", "mochi", "ltxv",
              "pixart", "cosmos", "lumina2", "wan", "hidream", "chroma", "ace", "omnigen2",
              "qwen_image", "hunyuan_image", "flux2", "ovis", "longcat_image", "cogvideox",
              "lens", "pixeldit", "ideogram4", "boogu", "krea2", "joyimage", "mage", "minimax"]

MEGAPIXEL_OPTIONS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98, 1.0, 1.2, 1.5, 1.8, 2.0, "audio only"]

OUTPUT_FPS = 24.0

IMAGE_MODE_FRAMES = 9    # stills: 9 frames fully rendered, frame index 8 is the output

WEIGHT_DTYPES = ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"]


def _first_present(options, preferred):
    if preferred in options:
        return preferred
    return options[0] if options else preferred


def _resolve_dimensions(aspect_ratio, megapixels, match_ratio_from_image, ref_images):
    """ResolutionSelector math (multiple = 32), optionally ratio-matched to the
    first connected reference image at the selected pixel size."""
    if megapixels == "audio only":
        return 32, 32, False
    wr, hr = ASPECT_RATIOS.get(aspect_ratio, ASPECT_RATIOS["16:9 (Widescreen)"])
    matched = False
    if match_ratio_from_image and ref_images:
        img = next((v for v in ref_images.values() if v is not None), None)
        if img is not None and img.shape[1] > 0 and img.shape[2] > 0:
            target = img.shape[2] / img.shape[1]
            wr, hr = min(ASPECT_RATIOS.values(),
                         key=lambda r: abs(math.log(r[0] / r[1]) - math.log(target)))
            matched = True
    total_pixels = float(megapixels) * 1024 * 1024
    scale = math.sqrt(total_pixels / (wr * hr))
    width = max(CANVAS_MULTIPLE, round(wr * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    height = max(CANVAS_MULTIPLE, round(hr * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    return width, height, matched


def _duration_to_length(duration_seconds):
    """Identical to the template's Math Expression:
    max(5, round(a*24)) + (5 - (max(5, round(a*24)) % 17)) % 17"""
    n = max(5, round(duration_seconds * FPS))
    return n + (5 - (n % 17)) % 17


def _encode_ref_audio(audio_vae, audio):
    waveform = audio["waveform"]  # [B, C, L]
    sr = audio["sample_rate"]
    vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
    if sr != vae_sr:
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    z = audio_vae.encode(waveform[:1].movedim(1, -1))  # [1, 32, 2, T]
    return z, z.shape[-1]


def _build_conditioning(clip, vae, audio_vae, prompt, width, height, length,
                        ref_image_size, ref_images, ref_videos, ref_video_audios, ref_audios,
                        image_mode=False):
    """In-node replica of MiniMaxH3ReferenceToVideo.execute (ref2va task)."""
    if image_mode:
        # still latent, exactly 9 frames (off the 17k+5 video grid)
        device = comfy.model_management.intermediate_device()
        video = torch.zeros([1, 24, 3, height // 16, width // 16], device=device)
        audio = torch.zeros([1, 32, 2, round(IMAGE_MODE_FRAMES / FPS * AUDIO_LATENT_FPS)], device=device)
        latent = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}
        frame_count = IMAGE_MODE_FRAMES
    else:
        latent, frame_count = _empty_av_latent(width, height, length)

    ref_items = []   # for the tokenizer presentation, in request order
    ref_blocks = []  # for the DiT payload, same order

    for img in (ref_images or {}).values():
        if img is None:
            continue
        h, w = img.shape[1], img.shape[2]
        if ref_image_size == "match":
            scale = min(1.0, math.sqrt((width * height) / (w * h)))
        else:
            scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
        tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        resized = _resize(img[:1], tw, th, "disabled")
        z = vae.encode(resized)
        ref_items.append({"type": "image", "data": resized})
        ref_blocks.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16, "latent": z})

    ref_video_audios = ref_video_audios or {}
    for name, video_frames in (ref_videos or {}).items():
        if video_frames is None:
            continue
        # index-paired soundtrack: ref_video_audio_N belongs to ref_video_N
        soundtrack = ref_video_audios.get("ref_video_audio_" + name.rsplit("_", 1)[-1])
        vh, vw = video_frames.shape[1], video_frames.shape[2]
        cw, ch = adapt_canvas(vw, vh)
        if vw * vh < cw * ch:
            cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        frames = _resize(video_frames, cw, ch, "disabled")
        if frames.shape[0] > frame_count:
            frames = frames[:frame_count]
        n = frames.shape[0]
        if n < 5:
            raise ValueError("MiniMax H3 reference videos need at least 5 frames (~0.2s at 24 fps)")
        while n % 17 != 5:
            n -= 1
        frames = frames[:n]
        z = vae.encode(frames)
        audio_latent, ref_audio_t = (None, 0)
        if soundtrack is not None:
            audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, soundtrack)
            ref_items.append({"type": "audio"})
        # Qwen sees the video at 2 fps with timestamps
        sample_idx = list(range(0, frames.shape[0], FPS // 2))
        qwen_frames = frames[sample_idx]
        ref_items.append({"type": "video", "data": qwen_frames,
                          "timestamps": [i / 2.0 for i in range(len(sample_idx))]})
        ref_blocks.append({"kind": "video_audio" if ref_audio_t else "video",
                           "latent_t": z.shape[2], "latent_h": ch // 16, "latent_w": cw // 16,
                           "ref_audio_t": ref_audio_t, "latent": z, "audio_latent": audio_latent})

    for audio in (ref_audios or {}).values():
        if audio is None:
            continue
        audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, audio)
        ref_items.append({"type": "audio"})
        ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t, "audio_latent": audio_latent})

    tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
    cond = clip.encode_from_tokens_scheduled(tokens)
    if ref_blocks:
        cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": ref_blocks})
    return cond, latent


class _GuiderBasic(comfy.samplers.CFGGuider):
    """Same as the core BasicGuider."""
    def set_conds(self, positive):
        self.inner_set_conds({"positive": positive})


class StarMinimaxAllInOne(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        diffusion_models = folder_paths.get_filename_list("diffusion_models")
        text_encoders = folder_paths.get_filename_list("text_encoders")
        vaes = folder_paths.get_filename_list("vae")
        return io.Schema(
            node_id="StarMinimaxAllInOne",
            display_name="⭐ Star Minimax All In One",
            category="⭐StarNodes/Video",
            description=(
                "Complete MiniMax H3 reference-to-video pipeline in a single node: "
                "loads the diffusion model (or uses the optional MODEL override input), "
                "the minimax text encoder and both VAEs, builds <Picture i> / <Video k> / "
                "<Audio j> reference conditioning, samples with the chosen sampler/scheduler "
                "and decodes video + audio. Reference image/video/audio slots grow "
                "automatically, exactly like the core MiniMax H3 Reference to Video node."
            ),
            inputs=[
                # ---------------- Mode ----------------
                io.Combo.Input("mode", options=["video", "image"], default="video",
                               tooltip="'video' renders the full clip with audio. 'image' renders 9 frames at the selected ratio and size and outputs only frame index 8 as a still image (best-quality frame); duration and audio decoding are skipped. Reference inputs work exactly like in video mode."),
                # ---------------- Prompt & user inputs ----------------
                io.String.Input("prompt", multiline=True, dynamic_prompts=True,
                                tooltip="Reference inputs by tag in connection order, e.g. <Picture 1>, <Video 1>, <Audio 1>, then describe scene, motion and audio."),
                io.Combo.Input("aspect_ratio", options=list(ASPECT_RATIOS.keys()),
                               default="16:9 (Widescreen)",
                               tooltip="Aspect ratio for the output dimensions."),
                io.Combo.Input("megapixels", options=MEGAPIXEL_OPTIONS, default=0.5,
                               tooltip='Target total megapixels (output pixel size). 0.5 MP ~ 960x544 at 16:9; 2.0 MP ~ 1920x1088. Select "audio only" for a fixed 32x32 canvas when you only need audio output.'),
                io.Boolean.Input("match_ratio_from_image", default=False,
                                 label_on="match image ratio", label_off="use selected ratio",
                                 tooltip="If enabled and a reference image is connected, the closest aspect ratio of the first reference image is used at the selected pixel size."),
                io.Float.Input("duration", default=5.0, min=0.2, max=150.0, step=0.1,
                               tooltip="Video duration in seconds at 24 fps. Internally snapped to the model's 17k+5 frame grid (5s -> 124 frames)."),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match",
                               tooltip="Reference image sizing. 'match' scales each ref (down only, keeping aspect) to the generation's pixel area; 'max' uses a 2048px short edge for best identity fidelity but is much slower."),
                # ---------------- Sampling ----------------
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff,
                             control_after_generate=True),
                io.Int.Input("steps", default=20, min=1, max=10000),
                io.Combo.Input("sampler_name", options=comfy.samplers.SAMPLER_NAMES,
                               default="res_multistep"),
                io.Combo.Input("scheduler", options=comfy.samplers.SCHEDULER_NAMES,
                               default="simple",
                               tooltip="'beta' or 'normal' tends to outperform 'simple' for reference-heavy prompts."),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01),
                # ---------------- Models ----------------
                io.Combo.Input("diffusion_model", options=diffusion_models,
                               default=_first_present(diffusion_models, DEFAULT_DIFFUSION_MODEL),
                               tooltip="Ignored when a MODEL is connected to the model_override input."),
                io.Combo.Input("weight_dtype", options=WEIGHT_DTYPES, default="default", advanced=True),
                io.Combo.Input("clip_name", options=text_encoders,
                               default=_first_present(text_encoders, DEFAULT_CLIP)),
                io.Combo.Input("clip_type", options=CLIP_TYPES, default="minimax", advanced=True),
                io.Combo.Input("clip_device", options=["default", "cpu"], default="default", advanced=True),
                io.Combo.Input("vae_name", options=vaes,
                               default=_first_present(vaes, DEFAULT_VIDEO_VAE),
                               tooltip="Video VAE (e.g. minimax_h3_video_vae_fp16)."),
                io.Combo.Input("audio_vae_name", options=vaes,
                               default=_first_present(vaes, DEFAULT_AUDIO_VAE),
                               tooltip="Audio VAE (e.g. minimax_h3_audio_vae_fp32)."),
                io.Combo.Input("audio_vae_precision", options=["fp32", "fp16", "bf16"],
                               default="fp32",
                               tooltip="Precision the audio VAE runs at. fp32 is recommended (same as the KJ loader preset)."),
                io.Combo.Input("audio_vae_device", options=["main_device", "cpu"],
                               default="main_device", advanced=True),
                # ---------------- Connectors ----------------
                io.Model.Input("model_override", optional=True,
                               tooltip="Optional external MODEL (e.g. a sage-attention patched MiniMax H3). When connected, the internal diffusion_model dropdown is ignored."),
                io.Autogrow.Input("ref_images", optional=True,
                                  template=io.Autogrow.TemplatePrefix(
                                      input=io.Image.Input("ref_image",
                                                           tooltip="Reference image (downscaled to 2048 short edge if larger, never upscaled)"),
                                      prefix="ref_image_", min=0, max=9)),
                io.Autogrow.Input("ref_videos", optional=True,
                                  template=io.Autogrow.TemplatePrefix(
                                      input=io.Image.Input("ref_video",
                                                           tooltip="Reference video frames at 24 fps (2-15s)"),
                                      prefix="ref_video_", min=0, max=3)),
                io.Autogrow.Input("ref_video_audios", optional=True,
                                  template=io.Autogrow.TemplatePrefix(
                                      input=io.Audio.Input("ref_video_audio",
                                                           tooltip="Soundtrack of the same-numbered reference video"),
                                      prefix="ref_video_audio_", min=0, max=3)),
                io.Autogrow.Input("ref_audios", optional=True,
                                  template=io.Autogrow.TemplatePrefix(
                                      input=io.Audio.Input("ref_audio",
                                                           tooltip="Standalone reference audio"),
                                      prefix="ref_audio_", min=0, max=3)),
            ],
            outputs=[
                io.Image.Output(display_name="IMAGE"),
                io.Audio.Output(display_name="AUDIO"),
                io.Float.Output(display_name="FPS",
                                tooltip="Fixed frame rate of the generated video (24.0). Connect straight into your video combine/save node."),
                io.Latent.Output(display_name="LATENT",
                                 tooltip="The combined processed latent from the sampler (NestedTensor with video+audio), before VAE decoding."),
                io.Model.Output(display_name="MODEL",
                                tooltip="The diffusion model used for sampling."),
                io.Clip.Output(display_name="CLIP",
                               tooltip="The loaded text encoder."),
                io.Vae.Output(display_name="VAE",
                              tooltip="The video VAE."),
                io.Vae.Output(display_name="AUDIO_VAE",
                              tooltip="The audio VAE (None in image mode without audio references)."),
            ],
        )

    # ------------------------------------------------------------------
    # Internal pipeline stages (mirroring the stock nodes, run in-process)
    # ------------------------------------------------------------------
    @staticmethod
    def _load_model(diffusion_model, weight_dtype):
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2
        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", diffusion_model)
        return comfy.sd.load_diffusion_model(unet_path, model_options=model_options)

    @staticmethod
    def _load_clip(clip_name, clip_type, clip_device):
        clip_type_enum = getattr(comfy.sd.CLIPType, clip_type.upper(),
                                 comfy.sd.CLIPType.STABLE_DIFFUSION)
        model_options = {}
        if clip_device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")
        clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
        return comfy.sd.load_clip(
            ckpt_paths=[clip_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type_enum, model_options=model_options)

    @staticmethod
    def _load_video_vae(vae_name):
        vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
        sd = comfy.utils.load_torch_file(vae_path)
        vae = comfy.sd.VAE(sd=sd)
        vae.throw_exception_if_invalid()
        return vae

    @staticmethod
    def _load_audio_vae(audio_vae_name, precision, device_name):
        """VAELoaderKJ behavior with user-selectable precision (fp32 default)."""
        dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
        device = (comfy.model_management.get_torch_device()
                  if device_name == "main_device" else torch.device("cpu"))
        vae_path = folder_paths.get_full_path_or_raise("vae", audio_vae_name)
        try:
            sd, metadata = comfy.utils.load_torch_file(vae_path, return_metadata=True)
        except TypeError:
            sd, metadata = comfy.utils.load_torch_file(vae_path), None

        is_audio_vae = (
            "vocoder.conv_post.weight" in sd
            or "vocoder.vocoder.conv_post.weight" in sd
            or "vocoder.resblocks.0.convs1.0.weight" in sd
            or "vocoder.vocoder.resblocks.0.convs1.0.weight" in sd
        )
        if is_audio_vae:
            sd = comfy.utils.state_dict_prefix_replace(
                dict(sd), {"audio_vae.": "autoencoder.", "vocoder.": "vocoder."}, filter_keys=True)
        try:
            vae = comfy.sd.VAE(sd=sd, device=device, dtype=dtype, metadata=metadata)
        except TypeError:  # older comfy.sd.VAE without dtype/device kwargs
            vae = comfy.sd.VAE(sd=sd)
        vae.throw_exception_if_invalid()
        return vae

    @staticmethod
    def _get_sigmas(model, scheduler, steps, denoise):
        total_steps = steps
        if denoise < 1.0:
            if denoise <= 0.0:
                raise ValueError("denoise must be greater than 0")
            total_steps = int(steps / denoise)
        sigmas = comfy.samplers.calculate_sigmas(
            model.get_model_object("model_sampling"), scheduler, total_steps).cpu()
        return sigmas[-(steps + 1):]

    @staticmethod
    def _sample(model, cond, latent, seed, sampler_name, sigmas):
        guider = _GuiderBasic(model)
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

    @staticmethod
    def _decode_video(vae, samples, image_mode=False):
        latent = samples.unbind()[0] if samples.is_nested else samples
        images = vae.decode(latent)
        if len(images.shape) == 5:  # combine batches
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        if image_mode:
            images = images[min(IMAGE_MODE_FRAMES - 1, images.shape[0] - 1):][:1]
        return images

    @staticmethod
    def _decode_audio(audio_vae, samples):
        latent = samples.unbind()[-1] if samples.is_nested else samples
        audio = audio_vae.decode(latent).movedim(-1, 1)
        std = torch.std(audio, dim=[1, 2], keepdim=True) * 5.0
        std[std < 1.0] = 1.0
        audio = audio / std
        vae_sr = getattr(audio_vae, "audio_sample_rate_output",
                         getattr(audio_vae, "audio_sample_rate", 44100))
        return {"waveform": audio, "sample_rate": vae_sr}

    # ------------------------------------------------------------------
    @classmethod
    def execute(cls, mode, prompt, aspect_ratio, megapixels, match_ratio_from_image, duration,
                ref_image_size, seed, steps, sampler_name, scheduler, denoise,
                diffusion_model, weight_dtype, clip_name, clip_type, clip_device,
                vae_name, audio_vae_name, audio_vae_precision, audio_vae_device,
                model_override=None, ref_images=None, ref_videos=None,
                ref_video_audios=None, ref_audios=None) -> io.NodeOutput:

        audio_only = (megapixels == "audio only")

        # 1. Resolution (ResolutionSelector, multiple = 32, optional image ratio match)
        #    same size logic in both modes
        width, height, matched = _resolve_dimensions(
            aspect_ratio, megapixels, match_ratio_from_image, ref_images)
        length = IMAGE_MODE_FRAMES if mode == "image" else _duration_to_length(duration)
        logging.info("[Star Minimax AIO] %s mode | canvas %dx%d%s, %d frames (%.2fs @ 24fps)",
                     mode, width, height, " (ratio matched from image)" if matched else "",
                     length, length / FPS)

        # 2. Models — external MODEL wins over the internal dropdown
        if model_override is not None:
            model = model_override
            logging.info("[Star Minimax AIO] using connected MODEL override; "
                         "internal diffusion model '%s' ignored", diffusion_model)
        else:
            model = cls._load_model(diffusion_model, weight_dtype)
        clip = cls._load_clip(clip_name, clip_type, clip_device)
        vae = cls._load_video_vae(vae_name)
        has_audio_refs = (any(a is not None for a in (ref_video_audios or {}).values())
                          or any(a is not None for a in (ref_audios or {}).values()))
        audio_vae = (cls._load_audio_vae(audio_vae_name, audio_vae_precision, audio_vae_device)
                     if mode == "video" or has_audio_refs or audio_only else None)

        # 3. Reference conditioning + empty AV latent (MiniMaxH3ReferenceToVideo)
        cond, latent = _build_conditioning(
            clip, vae, audio_vae, prompt, width, height, length, ref_image_size,
            ref_images, ref_videos, ref_video_audios, ref_audios,
            image_mode=(mode == "image"))
        # 4. Sampling (RandomNoise + BasicGuider + KSamplerSelect +
        #    BasicScheduler + SamplerCustomAdvanced)
        sigmas = cls._get_sigmas(model, scheduler, steps, denoise)
        samples = cls._sample(model, cond, latent, seed, sampler_name, sigmas)

        # 5. Decode (VAEDecode + VAEDecodeAudio)
        #    image mode: decode all 9 frames, return frame index 8 as the still
        images = cls._decode_video(vae, samples, image_mode=(mode == "image"))
        if mode == "image" and not audio_only:
            audio = {"waveform": torch.zeros([1, 2, 4410]), "sample_rate": 44100}
        else:
            audio = cls._decode_audio(audio_vae, samples)

        return io.NodeOutput(images, audio, OUTPUT_FPS, {"samples": samples}, model, clip, vae, audio_vae)


NODE_CLASS_MAPPINGS = {
    "StarMinimaxAllInOne": StarMinimaxAllInOne,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarMinimaxAllInOne": "⭐ Star Minimax All In One",
}
