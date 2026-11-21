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
import comfy.audio
import comfy.model_management
import comfy.nested_tensor
import comfy.patcher_extension
import comfy.samplers
import comfy.sd
import comfy.utils
from comfy_api.latest import io

from ..ltx_video.star_video_sound_enricher import process_audio as _enrich_sound
from ..misc.star_preview import apply_star_preview
from .minimax_common import decode_audio, decode_still, run_sample
from .minimax_common import decode_video as _decode_video
from .star_minimax_latent_upscaler import upscale_minimax_conditioning, upscale_video_latent_3d

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
        MiniMaxH3AddGuide,
        _empty_av_latent,
        _resize,
        adapt_canvas,
    )
except Exception:  # pragma: no cover - fallback definitions (identical logic)
    MiniMaxH3AddGuide = None
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

MEGAPIXEL_OPTIONS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98, 1.0, 1.2, 1.5, 1.8, 2.0,
                     3.0, 4.0, 5.0, 6.0, 7.0, 8.0, "audio only"]   # 3-8 MP: image mode only

OUTPUT_FPS = 24.0
AUDIO_SR = 32000

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


def _build_latent(width, height, length, image_mode):
    """The empty AV latent for the requested canvas. Image mode uses H3's native
    one-frame still convention, decoded with the Fizgig still decode in
    minimax_common.decode_still. Returns (latent, frame_count)."""
    if image_mode:
        device = comfy.model_management.intermediate_device()
        # the DiT patchifies 2x2 on a 16x latent grid, so the latent size must be even
        lh, lw = (height // 16) // 2 * 2, (width // 16) // 2 * 2
        video = torch.zeros([1, 24, 1, lh, lw], device=device)
        audio_t = max(1, round(1 / FPS * AUDIO_LATENT_FPS))       # 2, what the sampler uses for one frame
        audio = torch.zeros([1, 32, 2, audio_t], device=device)
        return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}, 5
    return _empty_av_latent(width, height, length)


def _build_conditioning(clip, vae, audio_vae, prompt, width, height, length,
                        ref_image_size, ref_images, ref_videos, ref_video_audios, ref_audios,
                        image_mode=False):
    """In-node replica of MiniMaxH3ReferenceToVideo.execute (ref2va task)."""
    latent, frame_count = _build_latent(width, height, length, image_mode)

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


def _apply_guides(cond, latent, vae, guides):
    """The core 'Add Guide for MiniMax H3' chain, run in-process: each guide
    image/clip is anchored at round(start_seconds * 24) on the output timeline
    (the seconds -> frames math of the template's Math Expression nodes)."""
    if MiniMaxH3AddGuide is None:
        raise RuntimeError(
            "The 'multiref_settings' input requires a ComfyUI version whose core "
            "includes the MiniMaxH3AddGuide node (MiniMax H3 multiframe template).")
    for guide in guides:
        frame_idx = round(guide["start_seconds"] * FPS)
        cond = MiniMaxH3AddGuide.execute(positive=cond, latent=latent, frame_idx=frame_idx,
                                         vae=vae, image=guide["image"]).result[0]
        logging.info("[Star Minimax AIO] guide anchored at frame %d (%.2fs)",
                     frame_idx, guide["start_seconds"])
    return cond


# ---------------------------------------------------------------------------
# Viggle-Remake mode (a Star Minimax Remake Option connected to
# remake_settings) — in-node replicas of the ComfyUI-Viggle-Remake nodes,
# which reproduce inference/sample.py of Viggle/Viggle-Remake.
# ---------------------------------------------------------------------------
def _remake_stamps(nf):
    """The conditioner's 2 fps block labels for an nf-frame clip: every 12th
    frame, merged in pairs labelled with their mean time ("%.1f" rounds half to
    even, so the first block is <0.2 seconds>)."""
    idx, cursor = [], 0.0
    while round(cursor) < nf:
        if not idx or round(cursor) > idx[-1]:
            idx.append(round(cursor))
        cursor += FPS / 2
    t = [i / 2 for i in range(len(idx))]
    t += t[-1:] * (len(t) % 2)
    return [(t[i] + t[i + 1]) / 2 for i in range(0, len(t), 2)]


def _remake_cover(image, width, height):
    """Scale to cover the canvas, then centre-crop with (src - dst) // 2."""
    h, w = image.shape[1], image.shape[2]
    s = max(width / w, height / h)
    sw, sh = max(width, round(w * s)), max(height, round(h * s))
    left, top = (sw - width) // 2, (sh - height) // 2
    return _resize(image, sw, sh, "disabled")[:, top:top + height, left:left + width]


_REF_SIZE_MP = {"none": None, "0.5 MP": 500_000, "1.0 MP": 1_000_000,
                "1.5 MP": 1_500_000}


def _build_remake_conditioning(clip, vae, audio_vae, prompt, width, height, length,
                               source, refs, pins, source_audio, ref_image_size="none"):
    """Replica of Viggle's RemakeConditioning on the AIO canvas: the source clip
    as <Video 1>, each reference as <Picture N>, pins as keyframes on the output
    timeline and the source soundtrack held as clean target audio. Returns the
    positive conditioning, the AV latent and its noise mask (audio member 0)."""
    want = length
    nf = 5 + 17 * math.ceil((want - 5) / 17)   # render grid, >= want
    if pins and not all(0 <= p < want for p in pins):
        raise ValueError(f"[Star Minimax AIO] remake pins must be output frames in [0, {want})")

    # the source on the output canvas: cover-cropped like Viggle's 16:9-class resize
    source = source[..., :3]
    if (source.shape[2], source.shape[1]) != (width, height):
        source = comfy.utils.common_upscale(source.movedim(-1, 1), width, height,
                                            "bicubic", "center").movedim(1, -1)
    if source.shape[0] < want:
        raise ValueError(f"[Star Minimax AIO] the remake source has {source.shape[0]} frames, "
                         f"{want} needed - feed a longer clip")
    if source.shape[0] < nf:
        # pad to the render grid by repeating the last frames (the same pad
        # Viggle uses for the VAE's causal tail); the padded tail is trimmed
        # off the decoded output again
        source = torch.cat([source, source[-1:].repeat(nf - source.shape[0], 1, 1, 1)])
    else:
        source = source[:nf]

    # text only: one label per 2 fps block of the source, one per reference, then
    # the prompt, each segment tokenized on its own and merged into one sequence
    segments = ["<Video 1>: ", *["<%.1f seconds>" % s for s in _remake_stamps(nf)],
                *["<Picture %d>: " % (i + 1) for i in range(len(refs))], prompt]
    first = clip.tokenize(segments[0])
    tk_key = next(iter(first))
    merged = list(first[tk_key][0])
    for s in segments[1:]:
        merged += clip.tokenize(s)[tk_key][0]
    cond = clip.encode_from_tokens_scheduled({tk_key: [merged]})

    # references: the source, then the images at the canvas short edge (rounded to 32)
    z = vae.encode(source)
    blocks = [{"kind": "video", "latent_t": z.shape[2], "latent_h": height // 16,
               "latent_w": width // 16, "ref_audio_t": 0, "latent": z, "audio_latent": None}]
    se = min(width, height)
    cap_px = _REF_SIZE_MP.get(ref_image_size)
    for r in refs:
        r = r[:1, ..., :3]
        rh, rw = r.shape[1], r.shape[2]
        th = max(CANVAS_MULTIPLE, round(rh * se / min(rh, rw) / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        tw = max(CANVAS_MULTIPLE, round(rw * se / min(rh, rw) / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        if cap_px is not None and th * tw > cap_px:
            # bound the encoded reference area (VRAM + ref tokens)
            s = math.sqrt(cap_px / (th * tw))
            th = max(CANVAS_MULTIPLE, round(th * s / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            tw = max(CANVAS_MULTIPLE, round(tw * s / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        r = r if (rh, rw) == (th, tw) else _resize(r, tw, th, "disabled")
        blocks.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16,
                       "latent": vae.encode(r)})

    # keyframes on the output canvas: the first stretched, the rest cover-cropped
    keyframes = []
    for i, (r, p) in enumerate(zip(refs, pins)):
        r = r[:1, ..., :3]
        if (r.shape[1], r.shape[2]) != (height, width):
            r = _resize(r, width, height, "disabled") if i == 0 else _remake_cover(r, width, height)
        keyframes.append({"resolved_frame_index": p, "latent": vae.encode(r)})
    values = {"minimax_refs": blocks}
    if keyframes:
        values["minimax_keyframes"] = keyframes
    cond = node_helpers.conditioning_set_values(cond, values)

    # the soundtrack as clean target audio: nf/24 s, padded to (nf+1)/24 s, encoded,
    # cut to the target's length
    n = round((nf + 1) / FPS * AUDIO_SR)
    wav = torch.zeros(2, n)
    if source_audio is not None:
        wav, sr = source_audio["waveform"][0].float(), source_audio["sample_rate"]
        wav = wav.repeat(2, 1) if wav.shape[0] == 1 else wav[:2]
        wav = wav[:, :round(nf / FPS * sr)]
        wav = comfy.audio.resample(wav, sr, AUDIO_SR) if sr != AUDIO_SR else wav
    wav = torch.nn.functional.pad(wav, (0, max(0, n - wav.shape[1])))[:, :n]
    # zero-pad to whole 800-sample latent frames ourselves: VAE.encode would
    # centre-crop instead, shifting the track by up to 400 samples
    wav = torch.nn.functional.pad(wav, (0, -n % 800))
    za = audio_vae.encode(wav[None].movedim(1, -1))[..., :round(nf / FPS * AUDIO_LATENT_FPS)]

    # the target: video to generate, audio held clean (mask member 0)
    video = torch.zeros(1, 24, (nf - 5) // 17 * 5 + 2, height // 16, width // 16,
                        device=comfy.model_management.intermediate_device())
    latent = {"samples": comfy.nested_tensor.NestedTensor((video, za))}
    denoise_mask = comfy.nested_tensor.NestedTensor(
        (torch.ones_like(video), torch.zeros_like(za)))
    return cond, latent, denoise_mask


def _remake_noise(latent, cond, seed):
    """Viggle's RemakeNoise draw order: one CPU generator, drawn once per
    reference and keyframe latent before the target's noise, so a seed gives
    the same video as the reference script."""
    c = cond[0][1]
    skip = [b["latent"].shape
            for b in c.get("minimax_refs", []) + c.get("minimax_keyframes", [])
            if "latent" in b]
    g = torch.Generator().manual_seed(seed)
    for shape in skip:
        torch.randn(shape, generator=g)
    return comfy.nested_tensor.NestedTensor(
        [torch.randn(t.shape, generator=g) for t in latent["samples"].unbind()])


def _remake_keyframe_until(model, keyframe_until):
    """Viggle's RemakeKeyframeUntil wrapper: forwards whose sigma is below the
    cutoff drop the pinned keyframes and keep only the references."""
    def release(executor, x, timestep, context, transformer_options, minimax_payload=None, **kwargs):
        p = minimax_payload or {}
        if p.get("keyframes") and float(timestep.flatten()[0]) / 1000 < keyframe_until:
            refs = p.get("refs") or []
            minimax_payload = dict(p, keyframes=None, layout=None,
                                   cond_video_latents=[r["latent"] for r in refs if "latent" in r],
                                   cond_audio_latents=[r["audio_latent"] for r in refs if r.get("audio_latent") is not None])
        return executor(x, timestep, context, transformer_options,
                        minimax_payload=minimax_payload, **kwargs)
    m = model.clone()
    m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                           "star_minimax_remake_keyframe_until", release)
    return m


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
                "the minimax text encoder and both VAEs (or uses the optional VAE override "
                "input for video decoding), builds <Picture i> / <Video k> / "
                "<Audio j> reference conditioning, samples with the chosen sampler/scheduler "
                "and decodes video + audio. Reference image/video/audio slots grow "
                "automatically, exactly like the core MiniMax H3 Reference to Video node. "
                "Connect a Star Video Sound Enricher Option to sound_settings to clean up "
                "and enrich the soundtrack internally, or a Star Minimax Latent Upscaler "
                "Option to options for a second-pass latent upscale + refine. "
                "Connect a Star Ref Mod Option to ref_mod_settings to inject RefMod "
                "reference blocks (Apply H3 RefMod behavior, ComfyUI-MiniMaxH3Mod pack) "
                "into the internal conditioning. "
                "Connect a Star Minimax Multiref Option to multiref_settings to anchor "
                "reference images/clips at their start seconds on the output timeline "
                "(the core 'Add Guide for MiniMax H3' chain, run in-process). "
                "Connect a Star Preview node to preview for a live animated "
                "sampling preview on the Star Preview node. "
                "Switch decode_video off to skip the video VAE decode and "
                "output only the latent and the audio."
            ),
            inputs=[
                # ---------------- Mode ----------------
                io.Combo.Input("mode", options=["video", "image"], default="video",
                               tooltip="'video' renders the full clip with audio. 'image' renders one single frame (H3's native still convention) at the selected ratio and size and decodes it with the single-frame still decode; duration and audio decoding are skipped. Reference inputs work exactly like in video mode."),
                # ---------------- Prompt & user inputs ----------------
                io.String.Input("prompt", multiline=True, dynamic_prompts=True,
                                tooltip="Reference inputs by tag in connection order, e.g. <Picture 1>, <Video 1>, <Audio 1>, then describe scene, motion and audio."),
                io.Combo.Input("aspect_ratio", options=list(ASPECT_RATIOS.keys()),
                               default="16:9 (Widescreen)",
                               tooltip="Aspect ratio for the output dimensions."),
                io.Combo.Input("megapixels", options=MEGAPIXEL_OPTIONS, default=0.5,
                               tooltip='Target total megapixels (output pixel size). 0.5 MP ~ 960x544 at 16:9; 2.0 MP ~ 1920x1088. Select "audio only" for a fixed 32x32 canvas when you only need audio output. The 3-8 MP presets appear in image mode only (single-frame stills hit their quality sweet spot from 3 MP up).'),
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
                io.Boolean.Input("decode_video", default=True,
                                 label_on="decode video", label_off="latent only",
                                 tooltip="Decode the generated video (or still) with the video VAE. Switch off to skip video decoding entirely - the node then only outputs the LATENT and the AUDIO, e.g. to run an external decode or keep memory free for a follow-up pass."),
                # ---------------- Connectors ----------------
                io.Model.Input("model_override", optional=True,
                               tooltip="Optional external MODEL (e.g. a sage-attention patched MiniMax H3). When connected, the internal diffusion_model dropdown is ignored."),
                io.Vae.Input("vae_override", optional=True,
                             tooltip="Optional external VAE for video decoding. When connected, the internal vae_name dropdown is ignored."),
                io.Custom("SOUND_SETTINGS").Input("sound_settings", optional=True,
                                                  tooltip="Optional sound processing from a 'Star Video Sound Enricher Option' node - the generated soundtrack is cleaned up and enriched with these settings (at least 44.1 kHz, never downsampled) before it leaves the node. Ignored in image mode without audio."),
                io.Sigmas.Input("sigmas", optional=True,
                                tooltip="Optional exact sigma schedule for the main sampling pass (e.g. from a '⭐ Star Custom Sigmas' node or any SIGMAS output) - overrides the scheduler, steps and denoise widgets."),
                io.Custom("UPSCALE_SETTINGS").Input("options", optional=True,
                                                    tooltip="Optional second-pass latent upscale from a 'Star Minimax Latent Upscaler Option' node - the pass-1 video latent is upscaled with the selected latent upscaler model and refined in a short second sampling pass with the same conditioning and the same seed (references are resolution-matched). Ignored when megapixels is 'audio only'."),
                io.Custom("REF_MOD_SETTINGS").Input("ref_mod_settings", optional=True,
                                                    tooltip="Optional RefMod injection from a 'Star Ref Mod Option' node - appends Apply-H3-RefMod-style reference blocks (ComfyUI-MiniMaxH3Mod pack) to the internal conditioning, after the native reference blocks. Also carried into the upscale refine pass (resolution-matched)."),
                io.Custom("MULTIREF_SETTINGS").Input("multiref_settings", optional=True,
                                                     tooltip="Optional timed keyframe references from a 'Star Minimax Multiref Option' node - reference images/clips anchored at their start seconds on the output timeline, exactly like a chain of core 'Add Guide for MiniMax H3' nodes. Works in video and image mode; the guides are also carried into the upscale refine pass (resolution-matched)."),
                io.Custom("STAR_MINIMAX_REMAKE").Input("remake_settings", optional=True,
                                                       tooltip="Optional Viggle-Remake mode from a '⭐ Star Minimax Remake Option' node - patches the swap/edit LoRA onto the model and runs the Viggle remake conditioning (source clip as <Video 1>, painted frames as <Picture N>, optional pins, source soundtrack kept). Video mode only; the node's own ref_image/ref_video/ref_audio inputs are then ignored."),
                io.Custom("STAR_PREVIEW").Input("preview", optional=True,
                                                tooltip="Optional live sampling preview from a '⭐ Star Preview' node - while this node is sampling, an animated preview of the video latent is shown on the Star Preview node (fixed: 512 px, quality 80, 8 fps)."),
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
                                 tooltip="The combined processed latent from the sampler (NestedTensor with video+audio), before VAE decoding. When an upscale options node is connected, this is the refined second-pass latent."),
                io.Model.Output(display_name="MODEL",
                                tooltip="The diffusion model used for the main (pass-1) sampling."),
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

    @classmethod
    def _run_upscale_pass(cls, model, cond, samples, seed, options, negative=None, cfg=1.0,
                          keep_audio_clean=False):
        """Second pass from a Star Minimax Latent Upscaler Option bundle:
        upscale the pass-1 video latent, keep the pass-1 audio latent, refine
        with the baked sigmas and the same (resolution-matched) conditioning,
        reusing the pass-1 seed. keep_audio_clean (remake mode) additionally
        masks the audio member so the held source soundtrack stays untouched."""
        model2 = options["model"] if options["model"] is not None else model
        video, audio = samples.unbind()
        video_up, scale = upscale_video_latent_3d(
            video, options["upscale_model"], options["megapixels"], options["align"],
            options["enable_chunking"], options["device"], options["precision"])
        logging.info("[Star Minimax AIO] upscale pass: %.2fx to %dx%d latent | %s | sampler %s%s",
                     scale, video_up.shape[-1], video_up.shape[-2],
                     options["sigmas_preset"], options["sampler_name"],
                     "" if options["model"] is not None else " | pass-1 model reused")
        video_up = video_up.to(audio.device)
        latent2 = {"samples": comfy.nested_tensor.NestedTensor((video_up, audio))}
        denoise_mask = (comfy.nested_tensor.NestedTensor(
            (torch.ones_like(video_up), torch.zeros_like(audio)))
            if keep_audio_clean else None)
        cond2 = upscale_minimax_conditioning(cond, scale)
        return run_sample(model2, cond2, latent2, seed,
                          options["sampler_name"], options["sigmas"],
                          negative=negative, cfg=cfg, denoise_mask=denoise_mask)

    # ------------------------------------------------------------------
    @classmethod
    def execute(cls, mode, prompt, aspect_ratio, megapixels, match_ratio_from_image, duration,
                ref_image_size, seed, steps, sampler_name, scheduler, denoise,
                diffusion_model, weight_dtype, clip_name, clip_type, clip_device,
                vae_name, audio_vae_name, audio_vae_precision, audio_vae_device, decode_video,
                model_override=None, vae_override=None, sound_settings=None, sigmas=None, options=None,
                ref_mod_settings=None, multiref_settings=None, preview=None,
                remake_settings=None, ref_images=None, ref_videos=None, ref_video_audios=None,
                ref_audios=None) -> io.NodeOutput:

        audio_only = (megapixels == "audio only")
        if mode == "video" and not audio_only and float(megapixels) > 2.0:
            raise ValueError("[Star Minimax AIO] megapixels above 2.0 are for image mode stills - "
                             "drop to 2.0 MP or less, or switch mode to 'image'.")
        if remake_settings is not None and mode != "video":
            raise ValueError("[Star Minimax AIO] remake_settings is video-only - "
                             "switch mode to 'video'.")

        # 1. Resolution (ResolutionSelector, multiple = 32, optional image ratio match)
        #    same size logic in both modes
        width, height, matched = _resolve_dimensions(
            aspect_ratio, megapixels, match_ratio_from_image, ref_images)
        length = 1 if mode == "image" else _duration_to_length(duration)
        remake_out = None
        if remake_settings is not None and mode == "video":
            # the render is padded up to the 17n+5 grid and trimmed back to the
            # exact requested/source length at the output, so parts cut from a
            # longer clip still join seamlessly - no frames are lost
            src_frames = int(remake_settings["source"].shape[0])
            if src_frames < 5:
                raise ValueError(
                    f"[Star Minimax AIO] the remake source has only {src_frames} "
                    "frames - too short even for the smallest clip (5 frames).")
            remake_out = min(max(5, round(duration * FPS)), src_frames)
            length = 5 + 17 * math.ceil((remake_out - 5) / 17)
            if length != remake_out:
                logging.info("[Star Minimax AIO] remake render padded %d -> %d "
                             "frames to the 17n+5 grid; output trimmed back to %d",
                             remake_out, length, remake_out)
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
        if vae_override is not None:
            vae = vae_override
            logging.info("[Star Minimax AIO] using connected VAE override; "
                         "internal video VAE '%s' ignored", vae_name)
        else:
            vae = cls._load_video_vae(vae_name)
        has_audio_refs = (any(a is not None for a in (ref_video_audios or {}).values())
                          or any(a is not None for a in (ref_audios or {}).values()))
        audio_vae = (cls._load_audio_vae(audio_vae_name, audio_vae_precision, audio_vae_device)
                     if mode == "video" or has_audio_refs or audio_only else None)

        # 2b. Viggle-Remake mode: patch the swap/edit LoRA onto the model
        #     (bypass-style patches, like LoraLoaderBypassModelOnly) and wrap it
        #     with the keyframe-release when pins are given.
        if remake_settings is not None:
            lora_path = folder_paths.get_full_path_or_raise("loras", remake_settings["lora"])
            lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
            model, _ = comfy.sd.load_lora_for_models(
                model, None, lora_sd, remake_settings["lora_strength"], 0)
            if remake_settings["pins"]:
                model = _remake_keyframe_until(model, remake_settings["keyframe_until"])
            logging.info("[Star Minimax AIO] remake mode: LoRA '%s' @ %.2f | %d ref(s) <= %s | pins %s",
                         remake_settings["lora"], remake_settings["lora_strength"],
                         len(remake_settings["refs"]),
                         remake_settings.get("ref_image_size") or "none",
                         remake_settings["pins"] or "none")

        # 3. Reference conditioning + empty AV latent (MiniMaxH3ReferenceToVideo),
        #    or the Viggle remake conditioning when remake_settings is connected
        remake_mask = remake_noise = None
        if remake_settings is not None:
            cond, latent, remake_mask = _build_remake_conditioning(
                clip, vae, audio_vae, prompt, width, height, remake_out,
                remake_settings["source"], remake_settings["refs"],
                remake_settings["pins"], remake_settings["audio"],
                remake_settings.get("ref_image_size") or "none")
            remake_noise = _remake_noise(latent, cond, seed)
            if (any(v is not None for v in (ref_images or {}).values())
                    or any(v is not None for v in (ref_videos or {}).values())
                    or any(v is not None for v in (ref_video_audios or {}).values())
                    or any(v is not None for v in (ref_audios or {}).values())):
                logging.warning("[Star Minimax AIO] remake_settings connected - the node's own "
                                "ref_image/ref_video/ref_audio inputs are ignored")
        else:
            cond, latent = _build_conditioning(
                clip, vae, audio_vae, prompt, width, height, length, ref_image_size,
                ref_images, ref_videos, ref_video_audios, ref_audios,
                image_mode=(mode == "image"))

        # 3b. Optional ref-mods from a Star Ref Mod Option node - appended after
        #     the native refs, exactly like Apply H3 RefMod on built CONDITIONING
        if ref_mod_settings is not None:
            refmod_blocks = ref_mod_settings.get("blocks") or []
            if refmod_blocks:
                out = []
                for t in cond:
                    d = dict(t[1])
                    d["minimax_refs"] = list(d.get("minimax_refs", [])) + refmod_blocks
                    out.append([t[0], d])
                cond = out
                logging.info("[Star Minimax AIO] %d ref-mod block(s) injected from Star Ref Mod Option",
                             len(refmod_blocks))
        # 3c. Optional timed guides from a Star Minimax Multiref Option node -
        #     images/clips anchored on the output timeline (Add-Guide chain)
        guides = (multiref_settings or {}).get("guides") or []
        if guides:
            cond = _apply_guides(cond, latent, vae, guides)
            logging.info("[Star Minimax AIO] %d guide(s) anchored from Star Minimax Multiref Option",
                         len(guides))
        # 4. Sampling (RandomNoise + BasicGuider + KSamplerSelect +
        #    BasicScheduler + SamplerCustomAdvanced)
        #    A SIGMAS link (e.g. Star Custom Sigmas) replaces the
        #    scheduler/steps/denoise widgets with the exact sigma list.
        if sigmas is not None:
            if sigmas.numel() < 2:
                logging.warning("[Star Minimax AIO] custom sigmas ignored (need at least 2 sigma values)")
                sigmas = None
            else:
                logging.info("[Star Minimax AIO] custom sigmas, %d steps", sigmas.numel() - 1)
        if sigmas is None:
            sigmas = cls._get_sigmas(model, scheduler, steps, denoise)
        sample_model = apply_star_preview(model, preview) if preview is not None else model
        samples = run_sample(sample_model, cond, latent, seed, sampler_name, sigmas,
                             denoise_mask=remake_mask, noise=remake_noise)

        # 4b. Optional second pass: latent upscale + refine (Star Minimax Latent Upscaler Option)
        samples_pass1 = samples
        ran_pass2 = options is not None and not audio_only
        if options is not None and audio_only:
            logging.info("[Star Minimax AIO] 'audio only' mode - upscale options ignored")
        if ran_pass2:
            samples = cls._run_upscale_pass(sample_model, cond, samples, seed, options,
                                            keep_audio_clean=remake_settings is not None)

        # 5. Decode (VAEDecode + VAEDecodeAudio)
        #    image mode: single-frame still decode (5-frame group, keep pixel frame 3)
        #    decode_video off: skip the video VAE decode - latent + audio outputs only
        if decode_video:
            images = decode_still(vae, samples) if mode == "image" else _decode_video(vae, samples)
            if remake_out is not None and images.shape[0] > remake_out:
                images = images[:remake_out]   # drop the padded grid tail
        else:
            images = None
            logging.info("[Star Minimax AIO] video VAE decode disabled - latent and audio outputs only")
        if mode == "image" and not audio_only:
            audio = {"waveform": torch.zeros([1, 2, 4410]), "sample_rate": 44100}
        else:
            audio_samples = samples_pass1 if (ran_pass2 and not options["upscale_pass_audio"]) else samples
            if remake_out is not None:   # drop the padded grid tail
                av, za = audio_samples.unbind()
                audio_samples = comfy.nested_tensor.NestedTensor(
                    (av, za[..., :round(remake_out / FPS * AUDIO_LATENT_FPS)]))
            audio = decode_audio(audio_vae, audio_samples)
            if sound_settings is not None:
                logging.info("[Star Minimax AIO] applying sound enricher settings to the audio output")
                audio = _enrich_sound(audio, sound_settings)

        return io.NodeOutput(images, audio, OUTPUT_FPS, {"samples": samples}, model, clip, vae, audio_vae)


NODE_CLASS_MAPPINGS = {
    "StarMinimaxAllInOne": StarMinimaxAllInOne,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarMinimaxAllInOne": "⭐ Star Minimax All In One",
}
