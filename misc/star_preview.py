"""⭐ Star Preview — live animated sampling preview (options node).

A tiny options node for the ⭐StarNodes video all-in-one nodes
(⭐ Star LTXV All-in-One, ⭐ Star LTXV 2.5 All-in-One, ⭐ Star Minimax
All In One). Connect its STAR_PREVIEW output to the 'preview' input of an
all-in-one node: while the all-in-one node is sampling, an animated preview
of the video latent is streamed straight onto the Star Preview node.

Works like the KJNodes "Model Preview Override", but fully self-contained in
the StarNodes pack and deliberately simple:
  - video preview only (no charts / sigma graphs),
  - fixed settings: preview scaled to 512 px, JPEG/WebP quality 80,
    preview playback at 8 fps, up to 16 frames per step,
  - one optional dropdown: pick a tiny preview VAE from models/vae_approx
    (default 'none' = fast Latent2RGB colors).

Mechanics: the all-in-one node clones its internal model and attaches an
OUTER_SAMPLE wrapper (same mechanism KJNodes uses) that taps the sampler
callback, decodes the current x0 video latent with the model's built-in
latent_rgb_factors (Latent2RGB), encodes the frames as an animated WebP
(or a single JPEG) and pushes them to the frontend over the websocket.
Encoding runs on a background thread that drops frames when busy, so
sampling is never blocked by the preview.
"""

import base64
import io as pyio
import logging
import queue
import threading

import numpy as np
import torch
from PIL import Image

import folder_paths
import latent_preview
import comfy.model_management
import comfy.patcher_extension
import comfy.sd
import comfy.utils

try:
    from server import PromptServer
except ImportError:
    PromptServer = None

# ---------------------------------------------------------------------------
# fixed preview settings (hardcoded on purpose - no widgets)
# ---------------------------------------------------------------------------
MAX_RESOLUTION = 512        # preview size in pixels (longest side, up- or downscaled)
JPEG_QUALITY = 80           # JPEG / WebP quality
PREVIEW_FPS = 8             # playback fps of the animated preview
PREVIEW_MAX_FRAMES = 16     # frames sampled per step for the animation

EVENT_NAME = "star_preview"
WRAPPER_KEY = "star_preview"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _suppressed_preview_image(self_, preview_format, x0):
    """Replacement for LatentPreviewer.decode_latent_to_preview_image while
    our preview is active: no default preview, no wasted decode work."""
    return None


class _AsyncPreviewEncoder:
    """Off-thread WebP/JPEG encoder. Bounded queue (1) drops frames when
    busy so the sampler never blocks on the preview."""

    _STOP = object()

    def __init__(self):
        self.q = queue.Queue(maxsize=1)
        self.thread = threading.Thread(
            target=self._run, name="star_preview_encoder", daemon=True)
        self.thread.start()

    def submit(self, fn):
        try:
            self.q.put_nowait(fn)
            return True
        except queue.Full:
            return False

    def _run(self):
        while True:
            item = self.q.get()
            if item is self._STOP:
                return
            try:
                item()
            except Exception:
                logging.exception("[StarPreview] async encoder error")

    def shutdown(self, drain_timeout=5.0):
        try:
            self.q.put(self._STOP, timeout=drain_timeout)
        except queue.Full:
            pass
        self.thread.join(timeout=drain_timeout)


def _video_member(x0):
    """Nested A/V latent (LTXV Concat AV, MiniMax H3) -> the video member."""
    if getattr(x0, "is_nested", False):
        try:
            parts = x0.unbind()
            if parts:
                return parts[0]
        except Exception:
            return None
    return x0


def _restore_packed_shape(x0, latent_shapes):
    """Some samplers hand the callback a flattened (B, C, N) pack; reshape it
    back to the first (video) latent shape."""
    if latent_shapes and len(latent_shapes) > 0 and x0.ndim == 3:
        target = latent_shapes[0]
        if len(target) >= 3:
            cut = 1
            for d in target[1:]:
                cut *= int(d)
            if x0.shape[-1] >= cut:
                x0 = x0[:, :, :cut].reshape([x0.shape[0]] + list(target)[1:])
    return x0


def _num_ltx_keyframes(guider):
    """LTXV first/last-frame guides append keyframe latents at the end of the
    time axis; count them so they can be trimmed from the preview."""
    try:
        positive = guider.conds.get("positive") if hasattr(guider, "conds") else None
        if positive and len(positive) > 0:
            kf = positive[0].get("keyframe_idxs")
            if kf is not None:
                return int(torch.unique(kf[0, 0, :, 0]).numel())
    except Exception:
        pass
    return 0


def _decode_preview_frames(x0, latent_format, max_frames):
    """Latent2RGB decode of a 5D video latent (or a 4D image latent) into a
    list of PIL frames. Never raises - returns [] when undecodable."""
    factors = getattr(latent_format, "latent_rgb_factors", None)
    if factors is None:
        return []
    try:
        w = torch.tensor(factors, device=x0.device, dtype=x0.dtype).transpose(0, 1)
        bias = getattr(latent_format, "latent_rgb_factors_bias", None)
        b = (torch.tensor(bias, device=x0.device, dtype=x0.dtype)
             if bias is not None else None)
        if x0.ndim == 5:
            x = x0[0]                      # (C, T, H, W) — batch 0
            t_total = x.shape[1]
            if t_total > max_frames:
                idx = np.linspace(0, t_total - 1, max_frames).round().astype(int)
                x = x[:, idx]
            x = x.movedim(0, -1)           # (T, H, W, C)
            rgb = torch.nn.functional.linear(x, w, bias=b)
        elif x0.ndim == 4:
            reshape = getattr(latent_format, "latent_rgb_factors_reshape", None)
            if reshape is not None:
                x0 = reshape(x0)
            rgb = torch.nn.functional.linear(
                x0[0].movedim(0, -1).unsqueeze(0), w, bias=b)
        else:
            return []
        rgb = ((rgb + 1.0) * 127.5).clamp_(0, 255).to(torch.uint8).cpu().numpy()
        return [Image.fromarray(rgb[i]) for i in range(rgb.shape[0])]
    except Exception as e:
        logging.warning(f"[StarPreview] latent decode failed: {e}")
        return []


# ---------------------------------------------------------------------------
# optional tiny preview VAE (models/vae_approx)
# ---------------------------------------------------------------------------
_TINY_VAE_CACHE = {}


def _strip_tae_prefix(sd):
    """Flat TAE checkpoints may carry a 'taesd_decoder.'/'decoder.' prefix —
    strip whatever common prefix there is so positional keys remain."""
    first = next(iter(sd))
    if first.split(".")[0].isdigit():
        return sd
    prefix = first.split(".")[0] + "."
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def _build_flat_tae_decoder(sd):
    """Reconstruct a flat TAESD-style decoder from the checkpoint itself.

    Core's fixed TAESD Decoder hardcodes a 64-wide stack with 3 upsamples,
    so variants like the 2D taeh3 (96 wide, 4 upsamples, 24 latent channels)
    fail to load. The architecture is recoverable: keys are positional
    module indices, 'N.conv.0.weight' is a Block, 'N.weight' a plain conv,
    and gaps are parameterless modules (Clamp / ReLU / Upsample).
    """
    import torch.nn as nn
    from comfy.taesd.taesd import Block, Clamp, conv

    by_index = {}
    for k, v in sd.items():
        head, _, rest = k.partition(".")
        if not head.isdigit():
            raise ValueError(f"not a flat TAE decoder state dict (unexpected key '{k}')")
        by_index.setdefault(int(head), {})[rest] = v

    modules = []
    for i in range(max(by_index) + 1):
        entry = by_index.get(i)
        if entry is None:
            # index 0 is the input Clamp, 2 the ReLU after the input conv,
            # the rest are upsamples
            modules.append(Clamp() if i == 0 else nn.ReLU() if i == 2
                           else nn.Upsample(scale_factor=2))
        elif "conv.0.weight" in entry:
            w = entry["conv.0.weight"]
            # only use the midblock-GN variant when the checkpoint has it
            if "pool.0.weight" in entry:
                modules.append(Block(w.shape[1], w.shape[0], use_midblock_gn=True))
            else:
                modules.append(Block(w.shape[1], w.shape[0]))
        elif "weight" in entry:
            w = entry["weight"]
            modules.append(conv(w.shape[1], w.shape[0], bias="bias" in entry))
        else:
            raise ValueError(f"unrecognized TAE decoder module at index {i}: "
                             f"{sorted(entry)}")
    model = nn.Sequential(*modules)
    model.load_state_dict(sd)
    return model


class _FlatTinyVAEDecoder:
    """Decode-only flat TAE decoder (any width / upsample count).
    Output range is [0, 1], matching the TAE family convention."""

    def __init__(self, sd):
        sd = _strip_tae_prefix(sd)
        self.device = comfy.model_management.vae_device()
        self.dtype = comfy.model_management.vae_dtype(
            self.device, [torch.float16, torch.bfloat16])
        self.model = _build_flat_tae_decoder(sd)
        self.model = self.model.eval().to(device=self.device, dtype=self.dtype)

    def decode(self, latent):
        """[B, C, H, W] -> [B, 3, H, W], float32 in [0, 1]."""
        out = self.model(latent.to(device=self.device, dtype=self.dtype))
        return out.to(device=latent.device, dtype=torch.float32).clamp(0, 1)


class _TinyPreviewVAE:
    """Unified preview decoder: temporal TAEHV-family via comfy.sd.VAE
    (is_video=True) or a flat 2D TAE decoder (is_video=False)."""

    def __init__(self, backend, is_video):
        self.backend = backend
        self.is_video = is_video

    def decode_frames(self, x0, max_frames):
        """Decode a 5D video latent (or 4D image latent) into PIL frames."""
        if self.is_video:
            if x0.ndim != 5:
                return []
            x = x0[:1]
            t_total = x.shape[2]
            if t_total > max_frames:
                idx = np.linspace(0, t_total - 1, max_frames).round().astype(int)
                x = x[:, :, idx]
            out = self.backend.decode(x)       # (B, T, H, W, C), already 0..1
            if out.ndim == 5:
                out = out[0]
            rgb = out.clamp(0, 1)
        elif x0.ndim == 5:
            # flat 2D decoder: decode sampled latent frames one at a time
            x = x0[0]                          # (C, T, H, W)
            t_total = x.shape[1]
            idx = np.arange(t_total)
            if t_total > max_frames:
                idx = np.linspace(0, t_total - 1, max_frames).round().astype(int)
            frames = [self.backend.decode(x[:, t].unsqueeze(0))[0] for t in idx]
            rgb = torch.stack([f.movedim(0, -1) for f in frames], dim=0)
        elif x0.ndim == 4:
            out = self.backend.decode(x0[:1])[0]   # (3, H, W), 0..1
            rgb = out.movedim(0, -1).clamp(0, 1).unsqueeze(0)
        else:
            return []
        if rgb.ndim != 4:
            return []
        u8 = (rgb.float() * 255).to(torch.uint8).cpu().numpy()
        return [Image.fromarray(u8[i]) for i in range(u8.shape[0])]


def _load_tiny_vae_decoder(name):
    """Load a tiny/approximate preview VAE from models/vae_approx (cached).

    Detection is purely content-based, so renamed files keep working:
    the temporal TAEHV family (taehv / taew / lighttae / taeltx_2) carries
    prefixed keys incl. "decoder.22.bias"; everything else is treated as a
    flat 2D TAE decoder whose architecture is rebuilt from the checkpoint
    (this covers the 2D taeh3 variant that core's fixed TAESD Decoder
    cannot load). Returns a _TinyPreviewVAE or None.
    """
    if name in _TINY_VAE_CACHE:
        return _TINY_VAE_CACHE[name]
    result = None
    try:
        path = folder_paths.get_full_path("vae_approx", name)
        if path:
            sd = comfy.utils.load_torch_file(path, safe_load=True)
            if "decoder.1.weight" in sd and "decoder.22.bias" in sd:
                vae = comfy.sd.VAE(sd=sd)
                try:
                    vae.first_stage_model.show_progress_bar = False
                except Exception:
                    pass
                result = _TinyPreviewVAE(vae, True)
            else:
                result = _TinyPreviewVAE(_FlatTinyVAEDecoder(sd), False)
    except Exception as e:
        logging.warning(f"[StarPreview] could not load preview VAE '{name}': {e}")
    _TINY_VAE_CACHE[name] = result
    return result


def _fit_preview(pf, max_res):
    """Scale a PIL frame so its longest side is exactly max_res — small
    latents are upscaled, large ones downscaled."""
    if max_res <= 0:
        return pf
    w, h = pf.size
    longest = max(w, h)
    if longest <= 0 or longest == max_res:
        return pf
    scale = max_res / longest
    return pf.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                     Image.LANCZOS)


def _encode_animated_webp(frames, fps, quality, max_res):
    """Animated WebP, base64. Returns (b64, width, height) or (None, 0, 0)."""
    if not frames:
        return None, 0, 0
    pil_frames = []
    for f in frames:
        pf = f if f.mode == "RGB" else f.convert("RGB")
        pil_frames.append(_fit_preview(pf, max_res))
    duration_ms = max(1, int(round(1000 / max(1, fps))))
    buf = pyio.BytesIO()
    try:
        pil_frames[0].save(
            buf, format="WEBP", save_all=True, append_images=pil_frames[1:],
            duration=duration_ms, loop=0, quality=quality, method=4)
    except Exception as e:
        logging.warning(f"[StarPreview] animated WebP encode failed: {e}")
        return None, 0, 0
    return (base64.b64encode(buf.getvalue()).decode("ascii"),
            pil_frames[0].width, pil_frames[0].height)


def _encode_jpeg(frame, quality, max_res):
    """Single-frame JPEG, base64. Returns (b64, width, height) or (None, 0, 0)."""
    try:
        pf = frame if frame.mode == "RGB" else frame.convert("RGB")
        pf = _fit_preview(pf, max_res)
        buf = pyio.BytesIO()
        pf.save(buf, format="JPEG", quality=quality)
        return (base64.b64encode(buf.getvalue()).decode("ascii"),
                pf.width, pf.height)
    except Exception as e:
        logging.warning(f"[StarPreview] JPEG encode failed: {e}")
        return None, 0, 0


# ---------------------------------------------------------------------------
# the OUTER_SAMPLE wrapper (attached to a model clone by the AIO nodes)
# ---------------------------------------------------------------------------
class _StarPreviewWrapper:
    def __init__(self, node_id, preview_vae="none"):
        self.node_id = str(node_id)
        self.preview_vae = preview_vae or "none"

    def __call__(self, executor, noise, latent_image, sampler, sigmas,
                 denoise_mask, callback, disable_pbar, seed, latent_shapes=None):
        if PromptServer is None:
            return executor(noise, latent_image, sampler, sigmas, denoise_mask,
                            callback, disable_pbar, seed, latent_shapes=latent_shapes)

        guider = executor.class_obj
        model_patcher = guider.model_patcher
        latent_format = model_patcher.model.latent_format
        num_keyframes = _num_ltx_keyframes(guider)
        node_id = self.node_id
        original_callback = callback
        encoder = _AsyncPreviewEncoder()

        # Optional tiny preview VAE (models/vae_approx) for truer colors.
        tiny_decoder = None
        if self.preview_vae != "none":
            tiny_decoder = _load_tiny_vae_decoder(self.preview_vae)

        def new_callback(step, x0, x, total_steps_):
            nonlocal tiny_decoder
            try:
                view = _video_member(x0)
                frames = []
                if view is not None:
                    view = _restore_packed_shape(view, latent_shapes)
                    if (num_keyframes > 0 and view.ndim == 5
                            and view.shape[2] > num_keyframes):
                        view = view[:, :, :-num_keyframes]
                    if tiny_decoder is not None:
                        try:
                            frames = tiny_decoder.decode_frames(
                                view, PREVIEW_MAX_FRAMES)
                        except Exception as e:
                            logging.warning(
                                f"[StarPreview] preview VAE decode failed, "
                                f"falling back to Latent2RGB for this run: {e}")
                            tiny_decoder = None
                    if not frames:
                        frames = _decode_preview_frames(
                            view, latent_format, PREVIEW_MAX_FRAMES)

                if frames:
                    sent_step = int(step) + 1
                    total = int(total_steps_)

                    def _encode_and_send(frames=frames, sent_step=sent_step,
                                         total=total):
                        if len(frames) > 1:
                            b64, w_, h_ = _encode_animated_webp(
                                frames, PREVIEW_FPS, JPEG_QUALITY, MAX_RESOLUTION)
                            mime = "image/webp"
                        else:
                            b64, w_, h_ = _encode_jpeg(
                                frames[0], JPEG_QUALITY, MAX_RESOLUTION)
                            mime = "image/jpeg"
                        if not b64:
                            return
                        PromptServer.instance.send_sync(EVENT_NAME, {
                            "node_id": node_id,
                            "image": b64,
                            "mime": mime,
                            "w": w_,
                            "h": h_,
                            "step": sent_step,
                            "total": total,
                        }, PromptServer.instance.client_id)

                    encoder.submit(_encode_and_send)
            except Exception as e:
                logging.warning(f"[StarPreview] preview callback failed: {e}")
            if original_callback is not None:
                original_callback(step, x0, x, total_steps_)

        # Silence the built-in previewer while our preview is active
        # (otherwise it decodes the same latents again every step).
        patched = []
        stack = [latent_preview.LatentPreviewer]
        stack.extend(latent_preview.LatentPreviewer.__subclasses__())
        seen = set()
        while stack:
            cls = stack.pop()
            if id(cls) in seen:
                continue
            seen.add(id(cls))
            stack.extend(cls.__subclasses__())
            if "decode_latent_to_preview_image" in cls.__dict__:
                patched.append((cls, cls.__dict__["decode_latent_to_preview_image"]))
                cls.decode_latent_to_preview_image = _suppressed_preview_image
        try:
            return executor(noise, latent_image, sampler, sigmas, denoise_mask,
                            new_callback, disable_pbar, seed,
                            latent_shapes=latent_shapes)
        finally:
            encoder.shutdown(drain_timeout=5.0)
            for cls, prev in patched:
                cls.decode_latent_to_preview_image = prev


def apply_star_preview(model, preview_options):
    """Clone *model* and attach the live-preview wrapper.

    Called by the all-in-one nodes with the bundle from the ⭐ Star Preview
    options node. Returns the original model untouched when the bundle is not
    a Star Preview bundle.
    """
    if not isinstance(preview_options, dict) or \
            preview_options.get("kind") != "star_preview":
        return model
    node_id = preview_options.get("node_id")
    if not node_id:
        return model
    m = model.clone()
    m.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
        WRAPPER_KEY,
        _StarPreviewWrapper(node_id, preview_options.get("preview_vae", "none")),
    )
    return m


# ---------------------------------------------------------------------------
# the node
# ---------------------------------------------------------------------------
class StarPreview:
    CATEGORY = "⭐StarNodes/Video"
    FUNCTION = "get_options"
    RETURN_TYPES = ("STAR_PREVIEW",)
    RETURN_NAMES = ("star_preview",)
    DESCRIPTION = (
        "Live animated sampling preview for the ⭐StarNodes video all-in-one "
        "nodes. Connect star_preview to the 'preview' input of a ⭐ Star LTXV "
        "All-in-One / ⭐ Star LTXV 2.5 All-in-One / ⭐ Star Minimax All In One "
        "node - while it samples, an animated preview of the video latent is "
        "shown right here on this node. Fixed settings: 512 px, quality 80, "
        "8 fps playback. Optionally pick a tiny preview VAE from "
        "models/vae_approx for truer preview colors."
    )

    @classmethod
    def INPUT_TYPES(cls):
        vae_approx = ["none"] + folder_paths.get_filename_list("vae_approx")
        return {
            "required": {
                "preview_vae": (vae_approx, {
                    "default": "none",
                    "tooltip": "Optional tiny preview VAE from models/vae_approx "
                               "(e.g. a taehv / taeltx / taeh3 decoder) for truer "
                               "colors in the live preview. 'none' = fast Latent2RGB "
                               "preview, no file needed. Video vs image decoders are "
                               "detected automatically, so renamed files work too.",
                }),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def get_options(self, preview_vae="none", unique_id=None):
        return ({"kind": "star_preview", "node_id": str(unique_id),
                 "preview_vae": preview_vae},)


NODE_CLASS_MAPPINGS = {
    "StarPreview": StarPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarPreview": "⭐ Star Preview",
}
