"""
Star Slideshow Maker - a compact ComfyUI slideshow video node.

The node renders images, timing, motion, and transitions directly into an
FFmpeg encoder. Frames are streamed one by one, so a long slideshow does not
allocate a huge ComfyUI IMAGE batch in RAM.
"""

from __future__ import annotations

import math
import os
import random
import re
import shutil
import subprocess
import tempfile
import time
import uuid
import wave
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps

import folder_paths

try:
    from server import PromptServer
except Exception:  # pragma: no cover - only present inside ComfyUI
    PromptServer = None

try:
    from comfy.model_management import throw_exception_if_processing_interrupted
except Exception:  # pragma: no cover
    def throw_exception_if_processing_interrupted() -> None:
        return None

try:  # Optional: true optical-flow morphing when OpenCV is available.
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


DYNAMIC_IMAGE_RE = re.compile(r"^image_([1-9][0-9]*)$")
SUPPORTED_FOLDER_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

ASPECT_RATIOS = {
    "auto": None,
    "16:9": 16.0 / 9.0,
    "9:16": 9.0 / 16.0,
    "1:1": 1.0,
    "4:3": 4.0 / 3.0,
    "3:4": 3.0 / 4.0,
    "21:9": 21.0 / 9.0,
}

RESOLUTION_LONG_EDGE = {
    "HD": 1280,
    "Full HD": 1920,
}

TIMING_MODES = [
    "seconds_per_image",
    "split_total_duration",
]

TRANSITIONS = [
    "none", "fade", "morph",
    "slide_left", "slide_right", "slide_up", "slide_down",
    "wipe_left", "wipe_right", "wipe_up", "wipe_down",
    "zoom", "pixelate", "random",
]

MOTION_EFFECTS = [
    "none", "zoom_in", "zoom_out",
    "pan_left", "pan_right", "pan_up", "pan_down", "random",
]

PRESETS = [
    "ultrafast", "superfast", "veryfast", "faster", "fast",
    "medium", "slow", "slower", "veryslow",
]

SVTAV1_PRESET_MAP = {
    "ultrafast": "12", "superfast": "10", "veryfast": "9",
    "faster": "8", "fast": "7", "medium": "6", "slow": "5",
    "slower": "4", "veryslow": "3",
}

AUDIO_BITRATE = 160_000
VIDEO_FORMATS = {
    "video/h264-mp4": {
        "extension": ".mp4", "container": "mp4", "vcodec": "libx264",
        "audio_args": ["-c:a", "aac", "-b:a", "160k"],
        "extra_args": ["-movflags", "+faststart"],
        "two_pass": True, "max_crf": 51,
    },
    "video/h265-mp4": {
        "extension": ".mp4", "container": "mp4", "vcodec": "libx265",
        "audio_args": ["-c:a", "aac", "-b:a", "160k"],
        "extra_args": ["-tag:v", "hvc1", "-movflags", "+faststart"],
        "two_pass": True, "max_crf": 51,
    },
    "video/vp9-webm": {
        "extension": ".webm", "container": "webm", "vcodec": "libvpx-vp9",
        "audio_args": ["-c:a", "libopus", "-b:a", "160k"],
        "extra_args": [], "two_pass": True, "max_crf": 63,
    },
    "video/av1-mp4": {
        "extension": ".mp4", "container": "mp4", "vcodec": "libsvtav1",
        "audio_args": ["-c:a", "aac", "-b:a", "160k"],
        "extra_args": ["-movflags", "+faststart"],
        "two_pass": False, "max_crf": 63,
    },
}


# ---------------------------------------------------------------------------
# Dynamic input and progress helpers
# ---------------------------------------------------------------------------

class _DynamicOptionalInputs(dict):
    """Accept frontend-created image_N sockets during prompt validation."""

    def __contains__(self, key: object) -> bool:
        return dict.__contains__(self, key) or bool(
            isinstance(key, str) and DYNAMIC_IMAGE_RE.match(key)
        )

    def __getitem__(self, key: str) -> Any:
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        if isinstance(key, str) and DYNAMIC_IMAGE_RE.match(key):
            return ("IMAGE", {})
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        if dict.__contains__(self, key):
            return dict.get(self, key, default)
        if isinstance(key, str) and DYNAMIC_IMAGE_RE.match(key):
            return ("IMAGE", {})
        return default


class ProgressReporter:
    """Progress bridge used by web/star_slideshow_maker.js."""

    def __init__(self, unique_id: Optional[str]) -> None:
        self.unique_id = unique_id
        self._last_sent = 0.0

    def update(self, fraction: float, sub: str = "",
               force: bool = False) -> None:
        fraction = min(1.0, max(0.0, float(fraction)))
        now = time.time()
        if not force and now - self._last_sent < 0.08:
            return
        self._last_sent = now
        if PromptServer is None or self.unique_id is None:
            return
        try:
            PromptServer.instance.send_sync("star_slideshow.progress", {
                "node": str(self.unique_id),
                "value": fraction,
                "text": f"{fraction * 100:.0f}%",
                "sub": sub,
            })
        except Exception:
            pass

    def finish(self, sub: str = "done") -> None:
        self.update(1.0, sub, force=True)


# ---------------------------------------------------------------------------
# Files, audio, and media helpers
# ---------------------------------------------------------------------------

def _ffmpeg_binary() -> str:
    binary = shutil.which("ffmpeg")
    if not binary:
        raise RuntimeError(
            "Star Slideshow Maker needs FFmpeg for video encoding. Install "
            "FFmpeg and make sure the ffmpeg command is available.")
    return binary


def _ffprobe_binary() -> str:
    binary = shutil.which("ffprobe")
    if not binary:
        raise RuntimeError(
            "Star Slideshow Maker needs ffprobe to inspect the encoded "
            "video. Install the complete FFmpeg tools package.")
    return binary


def _encoder_available(vcodec: str) -> bool:
    result = subprocess.run(
        [_ffmpeg_binary(), "-hide_banner", "-encoders"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        check=False)
    return result.returncode == 0 and vcodec in result.stdout


def _natural_key(value: str) -> List[Any]:
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"([0-9]+)", value)]


def _resolve_image_folder(image_folder: str) -> Optional[str]:
    value = (image_folder or "").strip().strip('"')
    if not value:
        return None
    candidates = [value] if os.path.isabs(value) else [
        os.path.join(folder_paths.get_input_directory(), value), value]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(f"Image folder not found: {image_folder}")


def _load_image_folder(path: str) -> List[np.ndarray]:
    names = [name for name in os.listdir(path)
             if name.lower().endswith(SUPPORTED_FOLDER_EXTENSIONS)
             and os.path.isfile(os.path.join(path, name))]
    names.sort(key=_natural_key)
    frames: List[np.ndarray] = []
    for name in names:
        full_path = os.path.join(path, name)
        try:
            with Image.open(full_path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                frames.append(np.asarray(image, dtype=np.uint8))
        except Exception as exc:
            print(f"[StarSlideshowMaker] skipped unreadable image "
                  f"{full_path}: {exc}")
    return frames


def _to_uint8_frames(image_value: Any) -> np.ndarray:
    arr = image_value.detach().cpu().numpy() if hasattr(image_value, "detach") \
        else image_value.cpu().numpy() if hasattr(image_value, "cpu") \
        else np.asarray(image_value)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[None, ...]
    if arr.ndim != 4:
        raise ValueError("Each IMAGE input must be one image or an IMAGE batch.")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.shape[-1] != 3:
        raise ValueError("IMAGE inputs must have 3 RGB or 4 RGBA channels.")
    if arr.max(initial=0.0) <= 1.5:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _audio_duration(audio: Any) -> Optional[float]:
    if not isinstance(audio, dict) or "waveform" not in audio:
        return None
    waveform = audio["waveform"]
    shape = getattr(waveform, "shape", None)
    if not shape:
        return None
    sample_rate = float(audio.get("sample_rate", 44100) or 44100)
    return float(shape[-1]) / max(sample_rate, 1.0)


def _audio_to_temp_wav(audio: Dict[str, Any]) -> str:
    waveform = audio.get("waveform")
    if waveform is None:
        raise ValueError("The connected AUDIO value has no waveform.")
    arr = waveform.detach().cpu().numpy() if hasattr(waveform, "detach") \
        else waveform.cpu().numpy() if hasattr(waveform, "cpu") \
        else np.asarray(waveform)
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[0] if arr.shape[0] == 1 else arr.mean(axis=0)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError("AUDIO waveform must have shape [batch, channels, samples].")
    if arr.shape[0] > 2:
        arr = arr[:2]
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, -1.0, 1.0)
        pcm = (arr * 32767.0).astype(np.int16)
    else:
        pcm = np.clip(arr, -32768, 32767).astype(np.int16)

    sample_rate = int(audio.get("sample_rate", 44100) or 44100)
    path = os.path.join(
        folder_paths.get_temp_directory(),
        f"star_slideshow_audio_{uuid.uuid4().hex}.wav")
    with wave.open(path, "wb") as handle:
        handle.setnchannels(int(pcm.shape[0]))
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(np.ascontiguousarray(pcm.T).tobytes())
    return path


def _build_output_path(base_dir: str, filename_prefix: str,
                       extension: str) -> Tuple[str, str]:
    prefix = (filename_prefix or "StarSlideshow").strip().strip('"')
    prefix = prefix.replace("\\", "/").strip("/") or "StarSlideshow"
    if prefix.lower().endswith(extension.lower()):
        prefix = prefix[:-len(extension)]
    subfolder, name = os.path.split(prefix)
    safe_parts = [re.sub(r"[^A-Za-z0-9_. -]", "_", part).strip()
                  for part in subfolder.split("/")
                  if part not in ("", ".", "..")]
    subfolder = "/".join(part for part in safe_parts
                         if part not in ("", ".", ".."))
    name = re.sub(r"[^A-Za-z0-9_. -]", "_", name).strip() or "StarSlideshow"

    directory = os.path.join(base_dir, subfolder) if subfolder else base_dir
    os.makedirs(directory, exist_ok=True)
    for counter in range(1, 100000):
        filename = f"{name}_{counter:05d}{extension}"
        path = os.path.join(directory, filename)
        if not os.path.exists(path):
            return path, subfolder
    raise RuntimeError("Could not create a unique output filename.")


def _probe_media(path: str) -> Dict[str, Any]:
    command = [
        _ffprobe_binary(), "-v", "error", "-show_format", "-show_streams",
        "-of", "json", path,
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, check=False)
    if result.returncode != 0:
        return {"size_mb": os.path.getsize(path) / (1024 * 1024)}
    try:
        import json
        data = json.loads(result.stdout)
    except Exception:
        data = {}
    fmt = data.get("format", {})
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio = next((s for s in streams if s.get("codec_type") == "audio"), {})

    fps = 0.0
    rate = video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1"
    try:
        numerator, denominator = rate.split("/", 1)
        fps = float(numerator) / max(float(denominator), 1.0)
    except Exception:
        fps = 0.0

    bitrate = fmt.get("bit_rate")
    return {
        "duration": float(fmt.get("duration", 0.0) or 0.0),
        "size_mb": float(fmt.get("size", 0.0) or 0.0) / (1024 * 1024),
        "bitrate_kbps": int(float(bitrate) / 1000.0) if bitrate else 0,
        "width": int(video.get("width", 0) or 0),
        "height": int(video.get("height", 0) or 0),
        "fps": fps,
        "vcodec": video.get("codec_name", ""),
        "acodec": audio.get("codec_name", ""),
    }


def _media_brief(info: Dict[str, Any]) -> str:
    parts = []
    if info.get("width") and info.get("height"):
        parts.append(f"{info['width']}x{info['height']}")
    if info.get("fps"):
        parts.append(f"{info['fps']:.3g} fps")
    if info.get("duration"):
        parts.append(f"{info['duration']:.2f}s")
    if info.get("size_mb") is not None:
        parts.append(f"{info['size_mb']:.2f} MiB")
    if info.get("vcodec"):
        parts.append(str(info["vcodec"]))
    if info.get("acodec"):
        parts.append(f"audio {info['acodec']}")
    return " | ".join(parts) if parts else "media information unavailable"


# ---------------------------------------------------------------------------
# Image geometry, motion, and transitions
# ---------------------------------------------------------------------------

def _choose_frame_size(first_image: Image.Image, aspect_ratio: str,
                       resolution: str) -> Tuple[int, int, str]:
    if aspect_ratio == "auto":
        source_ratio = first_image.width / max(first_image.height, 1)
        aspect_ratio = min(
            (name for name in ASPECT_RATIOS if name != "auto"),
            key=lambda name: abs(math.log(source_ratio / ASPECT_RATIOS[name]))
        )
    ratio = ASPECT_RATIOS[aspect_ratio]
    long_edge = RESOLUTION_LONG_EDGE[resolution]
    if ratio >= 1.0:
        width, height = long_edge, int(round(long_edge / ratio))
    else:
        height, width = long_edge, int(round(long_edge * ratio))
    width -= width % 2
    height -= height % 2
    return width, height, aspect_ratio


def _fit_to_frame(image: Image.Image, size: Tuple[int, int], fit_mode: str,
                  background: str) -> Image.Image:
    width, height = size
    color = (255, 255, 255) if background == "white" else (0, 0, 0)
    image = image.convert("RGB")
    if fit_mode == "cover":
        scale = max(width / image.width, height / image.height)
        resized = image.resize((max(1, int(math.ceil(image.width * scale))),
                                max(1, int(math.ceil(image.height * scale)))),
                               Image.Resampling.LANCZOS)
        left = max(0, (resized.width - width) // 2)
        top = max(0, (resized.height - height) // 2)
        return resized.crop((left, top, left + width, top + height))

    scale = min(width / image.width, height / image.height)
    resized = image.resize((max(1, int(round(image.width * scale))),
                            max(1, int(round(image.height * scale)))),
                           Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, color)
    canvas.paste(resized, ((width - resized.width) // 2,
                           (height - resized.height) // 2))
    return canvas


def _resize_center(image: Image.Image, scale: float,
                   size: Tuple[int, int]) -> Image.Image:
    width, height = size
    scaled = image.resize((max(1, int(round(width * scale))),
                           max(1, int(round(height * scale)))),
                          Image.Resampling.LANCZOS)
    left = (scaled.width - width) // 2
    top = (scaled.height - height) // 2
    return scaled.crop((left, top, left + width, top + height))


def _apply_motion(image: Image.Image, effect: str, progress: float,
                  size: Tuple[int, int]) -> Image.Image:
    if effect == "none":
        return image
    p = min(1.0, max(0.0, progress))
    width, height = size
    if effect == "zoom_in":
        return _resize_center(image, 1.0 + 0.08 * p, size)
    if effect == "zoom_out":
        return _resize_center(image, 1.08 - 0.08 * p, size)

    scale = 1.08
    scaled = image.resize((int(round(width * scale)),
                           int(round(height * scale))),
                          Image.Resampling.LANCZOS)
    max_x = (scaled.width - width) // 2
    max_y = (scaled.height - height) // 2
    if effect == "pan_left":
        x, y = int(round((1.0 - p) * 2 * max_x)), max_y
    elif effect == "pan_right":
        x, y = int(round(p * 2 * max_x)), max_y
    elif effect == "pan_up":
        x, y = max_x, int(round((1.0 - p) * 2 * max_y))
    elif effect == "pan_down":
        x, y = max_x, int(round(p * 2 * max_y))
    else:
        return image
    return scaled.crop((x, y, x + width, y + height))


def _smoothstep(p: float) -> float:
    p = min(1.0, max(0.0, p))
    return p * p * (3.0 - 2.0 * p)


def _flow_pair(a: Image.Image, b: Image.Image,
               cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]],
               key: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    if key in cache:
        return cache[key]
    if cv2 is None:
        raise RuntimeError("OpenCV is unavailable")
    width, height = a.size
    scale = min(1.0, 640.0 / max(width, height))
    small_size = (max(1, int(round(width * scale))),
                  max(1, int(round(height * scale))))
    a_small = np.asarray(a.resize(small_size, Image.Resampling.BILINEAR))
    b_small = np.asarray(b.resize(small_size, Image.Resampling.BILINEAR))
    gray_a = cv2.cvtColor(a_small, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(b_small, cv2.COLOR_RGB2GRAY)
    flow_ab = cv2.calcOpticalFlowFarneback(gray_a, gray_b, None, 0.5, 5,
                                           25, 5, 7, 1.5, 0)
    flow_ba = cv2.calcOpticalFlowFarneback(gray_b, gray_a, None, 0.5, 5,
                                           25, 5, 7, 1.5, 0)
    if scale < 1.0:
        flow_ab = cv2.resize(flow_ab, (width, height),
                             interpolation=cv2.INTER_LINEAR)
        flow_ba = cv2.resize(flow_ba, (width, height),
                             interpolation=cv2.INTER_LINEAR)
        flow_ab[..., 0] /= scale
        flow_ab[..., 1] /= scale
        flow_ba[..., 0] /= scale
        flow_ba[..., 1] /= scale
    result = (flow_ab.astype(np.float32), flow_ba.astype(np.float32))
    cache[key] = result
    return result


def _optical_flow_morph(a: Image.Image, b: Image.Image, p: float,
                        cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]],
                        key: Tuple[int, int]) -> Image.Image:
    flow_ab, flow_ba = _flow_pair(a, b, cache, key)
    width, height = a.size
    grid_y, grid_x = np.mgrid[0:height, 0:width].astype(np.float32)
    map_a_x = grid_x - flow_ab[..., 0] * p
    map_a_y = grid_y - flow_ab[..., 1] * p
    map_b_x = grid_x - flow_ba[..., 0] * (1.0 - p)
    map_b_y = grid_y - flow_ba[..., 1] * (1.0 - p)
    arr_a = np.asarray(a)
    arr_b = np.asarray(b)
    warped_a = cv2.remap(arr_a, map_a_x, map_a_y,
                         interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)
    warped_b = cv2.remap(arr_b, map_b_x, map_b_y,
                         interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)
    return Image.fromarray(cv2.addWeighted(warped_a, 1.0 - p,
                                           warped_b, p, 0.0)
                           .astype(np.uint8), "RGB")


def _soft_morph(a: Image.Image, b: Image.Image, p: float) -> Image.Image:
    width, height = a.size
    warp = math.sin(math.pi * p)
    old = _resize_center(a, 1.0 + 0.06 * warp * p, (width, height))
    new = _resize_center(b, 1.08 - 0.08 * p + 0.04 * warp * (1.0 - p),
                         (width, height))
    blended = Image.blend(old, new, p)
    radius = 7.0 * warp
    return blended.filter(ImageFilter.GaussianBlur(radius)) \
        if radius > 0.05 else blended


def _transition_frame(a: Image.Image, b: Image.Image, transition: str,
                      progress: float, background: str,
                      flow_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]],
                      flow_key: Tuple[int, int]) -> Image.Image:
    p = _smoothstep(progress)
    width, height = a.size
    color = (255, 255, 255) if background == "white" else (0, 0, 0)
    if transition == "fade":
        return Image.blend(a, b, p)
    if transition == "morph":
        if cv2 is not None:
            try:
                return _optical_flow_morph(a, b, p, flow_cache, flow_key)
            except Exception as exc:
                print(f"[StarSlideshowMaker] morph fallback: {exc}")
        return _soft_morph(a, b, p)
    if transition.startswith("slide_"):
        canvas = Image.new("RGB", (width, height), color)
        direction = transition.split("_", 1)[1]
        if direction == "left":
            a_pos, b_pos = (-int(width * p), 0), (int(width * (1 - p)), 0)
        elif direction == "right":
            a_pos, b_pos = (int(width * p), 0), (-int(width * (1 - p)), 0)
        elif direction == "up":
            a_pos, b_pos = (0, -int(height * p)), (0, int(height * (1 - p)))
        else:
            a_pos, b_pos = (0, int(height * p)), (0, -int(height * (1 - p)))
        canvas.paste(a, a_pos)
        canvas.paste(b, b_pos)
        return canvas
    if transition.startswith("wipe_"):
        direction = transition.split("_", 1)[1]
        canvas = a.copy()
        if direction == "left":
            reveal = b.crop((0, 0, max(1, int(width * p)), height))
            canvas.paste(reveal, (0, 0))
        elif direction == "right":
            x = int(width * (1 - p))
            canvas.paste(b.crop((x, 0, width, height)), (x, 0))
        elif direction == "up":
            reveal = b.crop((0, 0, width, max(1, int(height * p))))
            canvas.paste(reveal, (0, 0))
        else:
            y = int(height * (1 - p))
            canvas.paste(b.crop((0, y, width, height)), (0, y))
        return canvas
    if transition == "zoom":
        old = _resize_center(a, 1.0 + 0.12 * p, (width, height))
        new = _resize_center(b, 1.08 - 0.08 * p, (width, height))
        return Image.blend(old, new, p)
    if transition == "pixelate":
        amount = math.sin(math.pi * p)
        pixel = max(1, int(round(1.0 + 31.0 * amount)))
        small = (max(1, width // pixel), max(1, height // pixel))
        old = a.resize(small, Image.Resampling.BILINEAR).resize(
            (width, height), Image.Resampling.NEAREST)
        new = b.resize(small, Image.Resampling.BILINEAR).resize(
            (width, height), Image.Resampling.NEAREST)
        return Image.blend(old, new, p)
    return b if progress >= 1.0 else a


# ---------------------------------------------------------------------------
# Slideshow node
# ---------------------------------------------------------------------------

class StarSlideshowMaker:
    """Render and encode a slideshow without creating a full IMAGE batch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_folder": ("STRING", {"default": "", "tooltip":
                    "Optional folder of JPG/JPEG, PNG, or WebP images. "
                    "Relative paths resolve from the ComfyUI input folder."}),
                "aspect_ratio": (list(ASPECT_RATIOS.keys()),
                                 {"default": "auto", "tooltip":
                    "Output frame aspect ratio. 'auto' picks the listed "
                    "ratio closest to the first image."}),
                "resolution": (list(RESOLUTION_LONG_EDGE.keys()),
                               {"default": "Full HD", "tooltip":
                    "Long-edge pixel size of the output frame: HD = 1280, "
                    "Full HD = 1920."}),
                "duration_mode": (["fixed", "audio"], {"default": "fixed",
                                   "tooltip":
                    "'fixed' uses fixed_duration. 'audio' matches the "
                    "duration of the connected AUDIO input."}),
                "fixed_duration": ("FLOAT", {"default": 10.0, "min": 0.25,
                                             "max": 86400.0, "step": 0.05,
                                             "tooltip":
                    "Total slideshow length in seconds. Only used when "
                    "duration_mode is 'fixed'."}),
                "timing_mode": (TIMING_MODES,
                                {"default": "seconds_per_image", "tooltip":
                    "'seconds_per_image' repeats the image sequence at a "
                    "fixed length per image until the total duration is "
                    "filled. 'split_total_duration' divides the total "
                    "duration equally among all images."}),
                "seconds_per_image": ("FLOAT", {"default": 3.0, "min": 0.1,
                                                "max": 3600.0,
                                                "step": 0.05, "tooltip":
                    "Seconds each image is shown. Only used when "
                    "timing_mode is 'seconds_per_image'."}),
                "transition": (TRANSITIONS, {"default": "fade", "tooltip":
                    "Effect used to blend from one image to the next. "
                    "'random' picks a different transition for each image "
                    "(see seed)."}),
                "transition_duration": ("FLOAT", {"default": 1.0,
                                                  "min": 0.0, "max": 10.0,
                                                  "step": 0.05, "tooltip":
                    "Length of the transition in seconds, clamped so it "
                    "cannot consume almost the whole image segment."}),
                "motion_effect": (MOTION_EFFECTS, {"default": "none",
                                   "tooltip":
                    "Pan/zoom effect applied while each image is shown. "
                    "'random' picks a different effect for each image "
                    "(see seed)."}),
                "seed": ("INT", {"default": 0, "min": 0,
                                 "max": 0xffffffffffffffff, "tooltip":
                    "Used when transition or motion_effect is 'random' to "
                    "pick a different effect per image. 0 re-rolls a new "
                    "random pick every run; any other value reproduces the "
                    "same random picks."}),
                "fit_mode": (["contain", "cover"], {"default": "contain",
                              "tooltip":
                    "'contain' fits the whole image inside the frame "
                    "(letterboxed with background). 'cover' fills the "
                    "frame and center-crops overflow."}),
                "background": (["black", "white"], {"default": "black",
                                "tooltip":
                    "Letterbox color used around images when fit_mode is "
                    "'contain'."}),
                "frame_rate": ("FLOAT", {"default": 30.0, "min": 1.0,
                                         "max": 120.0, "step": 0.01,
                                         "tooltip":
                    "Output video frame rate in frames per second."}),
                "quality": ("INT", {"default": 60, "min": 0, "max": 100,
                                    "step": 1, "display": "slider",
                                    "tooltip":
                                    "Compression quality. Ignored when "
                                    "target_size_mb is greater than 0."}),
                "format": (list(VIDEO_FORMATS.keys()),
                           {"default": "video/h264-mp4", "tooltip":
                    "Output video codec/container. Available options "
                    "depend on the encoders built into your FFmpeg."}),
                "preset": (PRESETS, {"default": "medium", "tooltip":
                    "Encoder speed vs. efficiency. Slower presets usually "
                    "give a smaller file at the same quality."}),
                "filename_prefix": ("STRING", {"default": "StarSlideshow",
                                    "tooltip":
                    "Output file name, optionally including subfolders "
                    "(e.g. 'subfolder/name')."}),
                "target_size_mb": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1000000.0,
                    "step": 0.1, "tooltip":
                    "Desired output size in MiB. 0 = use the quality slider. "
                    "When greater than 0, target size wins and uses two-pass "
                    "encoding where supported."}),
                "save_audio": ("BOOLEAN", {"default": True, "tooltip":
                    "Mux the connected AUDIO input into the output video."}),
                "save_output": ("BOOLEAN", {"default": True, "tooltip":
                                "True = ComfyUI output folder, False = "
                                "temp folder."}),
            },
            "optional": _DynamicOptionalInputs({
                "image_1": ("IMAGE", {"tooltip":
                    "Optional first image. Connect more image_N sockets; "
                    "they are used before folder files."}),
                "audio": ("AUDIO", {"tooltip":
                    "Optional audio. It can set the slideshow duration and "
                    "is muxed into the output when save_audio is enabled."}),
            }),
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        for name, input_type in (input_types or {}).items():
            if DYNAMIC_IMAGE_RE.match(str(name)) and input_type != "IMAGE":
                return f"{name} must be connected to an IMAGE output."
            if name == "audio" and input_type != "AUDIO":
                return "audio must be connected to an AUDIO output."
        return True

    @classmethod
    def IS_CHANGED(cls, transition="none", motion_effect="none", seed=0,
                   **kwargs):
        # seed 0 re-rolls the random transition/motion picks every run.
        if seed == 0 and "random" in (transition, motion_effect):
            return random.random()
        return seed

    RETURN_TYPES = ("STAR_FILENAMES", "STRING")
    RETURN_NAMES = ("Filenames", "info")
    FUNCTION = "make_slideshow"
    CATEGORY = "⭐StarNodes/Video"
    OUTPUT_NODE = True
    DESCRIPTION = ("Create and encode a slideshow from one or more images. "
                   "Frames stream directly to FFmpeg for low RAM use. "
                   "See web/docs/StarSlideshowMaker.md")

    def make_slideshow(self, image_folder, aspect_ratio, resolution,
                       duration_mode, fixed_duration, timing_mode,
                       seconds_per_image, transition, transition_duration,
                       motion_effect, seed, fit_mode, background, frame_rate,
                       quality, format, preset, filename_prefix,
                       target_size_mb, save_audio, save_output,
                       image_1=None, audio=None, unique_id=None, **kwargs):
        data = self._prepare_render_data(
            image_folder, aspect_ratio, resolution, duration_mode,
            fixed_duration, timing_mode, seconds_per_image, transition,
            transition_duration, motion_effect, seed, fit_mode, background,
            frame_rate, image_1, audio, kwargs)

        if format not in VIDEO_FORMATS:
            raise ValueError(f"Unknown video format '{format}'.")
        fmt = VIDEO_FORMATS[format]
        if not _encoder_available(fmt["vcodec"]):
            raise RuntimeError(
                f"Your FFmpeg build does not include the encoder "
                f"'{fmt['vcodec']}'. Pick another format or install a full "
                "FFmpeg build.")

        base_dir = folder_paths.get_output_directory() if save_output \
            else folder_paths.get_temp_directory()
        out_type = "output" if save_output else "temp"
        out_path, subfolder = _build_output_path(
            base_dir, filename_prefix, fmt["extension"])

        audio_file = None
        if audio is not None and save_audio:
            audio_file = _audio_to_temp_wav(audio)

        reporter = ProgressReporter(unique_id)
        started = time.time()
        try:
            self._encode(data, fmt, int(quality), preset,
                         float(target_size_mb), audio_file, out_path,
                         reporter)
        except BaseException:
            try:
                if os.path.exists(out_path):
                    os.remove(out_path)
            except OSError:
                pass
            raise
        finally:
            if audio_file:
                try:
                    os.remove(audio_file)
                except OSError:
                    pass

        reporter.finish("video encoded")
        elapsed = time.time() - started
        out_info = _probe_media(out_path)
        info = self._build_info(data, format, preset, int(quality),
                                float(target_size_mb), save_audio,
                                out_path, out_info, elapsed)
        print("[StarSlideshowMaker]\n" + info)

        preview = {
            "filename": os.path.basename(out_path),
            "subfolder": subfolder,
            "type": out_type,
            "format": format,
            "fullpath": out_path,
        }
        return {"ui": {"star_videos": [preview]},
                "result": ((save_output, [out_path]), info)}

    # ---------------------------- render planning -------------------------

    @staticmethod
    def _collect_images(image_1: Any, kwargs: Dict[str, Any],
                        image_folder: str
                        ) -> Tuple[List[np.ndarray], int, int, Optional[str]]:
        dynamic_values: List[Tuple[int, Any]] = []
        for name, value in kwargs.items():
            match = DYNAMIC_IMAGE_RE.match(str(name))
            if match and value is not None:
                dynamic_values.append((int(match.group(1)), value))
        dynamic_values.sort(key=lambda item: item[0])

        frames: List[np.ndarray] = []
        connected_values = ([image_1] if image_1 is not None else []) + [
            value for _, value in dynamic_values
        ]
        for value in connected_values:
            frames.extend(_to_uint8_frames(value))
        connected_count = len(frames)

        resolved_folder = _resolve_image_folder(image_folder)
        folder_count = 0
        if resolved_folder:
            folder_frames = _load_image_folder(resolved_folder)
            folder_count = len(folder_frames)
            frames.extend(folder_frames)
        return frames, connected_count, folder_count, resolved_folder

    def _prepare_render_data(self, image_folder, aspect_ratio, resolution,
                             duration_mode, fixed_duration, timing_mode,
                             seconds_per_image, transition,
                             transition_duration, motion_effect, seed,
                             fit_mode, background, frame_rate, image_1, audio,
                             kwargs) -> Dict[str, Any]:
        frames_input, connected_count, folder_count, resolved_folder = \
            self._collect_images(image_1, kwargs, image_folder)
        if not frames_input:
            raise ValueError("Connect at least one IMAGE input or set "
                             "image_folder to a supported image folder.")

        first = Image.fromarray(frames_input[0], "RGB")
        width, height, selected_aspect = _choose_frame_size(
            first, aspect_ratio, resolution)
        base_images = [
            _fit_to_frame(Image.fromarray(frame, "RGB"), (width, height),
                          fit_mode, background)
            for frame in frames_input
        ]

        audio_duration = _audio_duration(audio) if audio is not None else None
        if duration_mode == "audio":
            if audio_duration is None:
                raise ValueError("duration_mode is 'audio', but no AUDIO "
                                 "input is connected.")
            requested_duration = audio_duration
        else:
            requested_duration = float(fixed_duration)

        frame_rate = float(frame_rate)
        total_frames = max(1, int(round(requested_duration * frame_rate)))
        duration = total_frames / frame_rate
        segments = self._build_segments(
            duration, len(base_images), timing_mode, float(seconds_per_image))
        rng = random.Random(seed) if seed else random.Random()
        for segment in segments:
            segment["motion"] = (
                rng.choice([m for m in MOTION_EFFECTS if m != "random"])
                if motion_effect == "random" else motion_effect)
            segment["transition"] = (
                rng.choice([t for t in TRANSITIONS if t != "random"])
                if transition == "random" else transition)
        return {
            "base_images": base_images,
            "connected_count": connected_count,
            "folder_count": folder_count,
            "resolved_folder": resolved_folder,
            "width": width,
            "height": height,
            "selected_aspect": selected_aspect,
            "resolution": resolution,
            "duration_mode": duration_mode,
            "duration": duration,
            "frame_rate": frame_rate,
            "total_frames": total_frames,
            "segments": segments,
            "timing_mode": timing_mode,
            "seconds_per_image": float(seconds_per_image),
            "transition": transition,
            "transition_duration": float(transition_duration),
            "motion_effect": motion_effect,
            "fit_mode": fit_mode,
            "background": background,
        }

    @staticmethod
    def _build_segments(duration: float, image_count: int, timing_mode: str,
                        seconds_per_image: float) -> List[Dict[str, Any]]:
        if timing_mode == "split_total_duration":
            each = duration / image_count
            return [{"image_index": index,
                     "start": index * each,
                     "end": (index + 1) * each,
                     "duration": each}
                    for index in range(image_count)]

        each = max(float(seconds_per_image), 0.001)
        segments: List[Dict[str, Any]] = []
        start = 0.0
        cycle_index = 0
        while start < duration - 1e-9:
            image_index = cycle_index % image_count
            end = min(duration, start + each)
            segments.append({"image_index": image_index,
                             "start": start, "end": end,
                             "duration": end - start})
            start = end
            cycle_index += 1
        return segments

    def _render_one_frame(self, data: Dict[str, Any], frame_index: int,
                          state: Dict[str, Any]) -> Image.Image:
        frame_rate = data["frame_rate"]
        t = frame_index / frame_rate
        segments = data["segments"]
        segment_position = state["segment_position"]
        while segment_position + 1 < len(segments) and \
                t >= segments[segment_position]["end"] - 1e-9:
            segment_position += 1
        state["segment_position"] = segment_position

        segment = segments[segment_position]
        local_time = t - segment["start"]
        image_index = segment["image_index"]
        old_image = data["base_images"][image_index]
        motion = segment["motion"]

        next_segment = (segments[segment_position + 1]
                        if segment_position + 1 < len(segments) else None)
        transition_progress = None
        active_transition = "none"
        transition_seconds = 0.0
        if next_segment is not None and len(data["base_images"]) > 1:
            active_transition = segment["transition"]
            transition_seconds = min(float(data["transition_duration"]),
                                     segment["duration"] * 0.75)
            if transition_seconds <= 0.5 / frame_rate:
                active_transition = "none"
            elif local_time >= segment["duration"] - transition_seconds:
                transition_progress = (
                    (local_time - (segment["duration"] - transition_seconds))
                    / transition_seconds)

        if transition_progress is None or next_segment is None:
            if motion == "none":
                return old_image
            progress = min(1.0, local_time /
                           max(segment["duration"], 1e-9))
            return _apply_motion(old_image, motion, progress,
                                 (data["width"], data["height"]))

        next_image_index = next_segment["image_index"]
        old_progress = min(1.0, local_time /
                           max(segment["duration"], 1e-9))
        next_progress = max(0.0,
            (t - next_segment["start"]) /
            max(next_segment["duration"], 1e-9))
        old_frame = _apply_motion(old_image, motion, old_progress,
                                  (data["width"], data["height"]))
        new_frame = _apply_motion(
            data["base_images"][next_image_index], next_segment["motion"],
            next_progress, (data["width"], data["height"]))
        return _transition_frame(
            old_frame, new_frame, active_transition, transition_progress,
            data["background"], state["flow_cache"],
            (image_index, next_image_index))

    # ------------------------------- encoding -----------------------------

    @staticmethod
    def _quality_to_crf(quality: int, max_crf: int) -> int:
        quality = min(100, max(0, int(quality)))
        return int(round(max_crf * (100 - quality) / 100.0))

    @staticmethod
    def _target_bitrate(target_size_mb: float, duration: float,
                        include_audio: bool) -> int:
        total_bps = target_size_mb * 1024.0 * 1024.0 * 8.0 / duration
        audio_bps = AUDIO_BITRATE if include_audio else 0
        return max(100_000, int(total_bps - audio_bps))

    def _video_args(self, fmt: Dict[str, Any], quality: int, preset: str,
                    target_size_mb: float, duration: float,
                    include_audio: bool) -> List[str]:
        vcodec = fmt["vcodec"]
        preset_value = SVTAV1_PRESET_MAP.get(preset, "6") \
            if vcodec == "libsvtav1" else preset
        if target_size_mb > 0:
            bitrate = self._target_bitrate(target_size_mb, duration,
                                           include_audio)
            return ["-c:v", vcodec, "-preset", preset_value,
                    "-b:v", str(bitrate)]

        crf = self._quality_to_crf(quality, fmt["max_crf"])
        args = ["-c:v", vcodec, "-preset", preset_value,
                "-crf", str(crf)]
        if vcodec == "libvpx-vp9":
            args += ["-b:v", "0"]
        return args

    @staticmethod
    def _pass_args(fmt: Dict[str, Any], pass_number: int,
                   passlog: str) -> List[str]:
        vcodec = fmt["vcodec"]
        if vcodec == "libx265":
            return ["-x265-params", f"pass={pass_number}:stats={passlog}"]
        return ["-pass", str(pass_number), "-passlogfile", passlog]

    def _ffmpeg_command(self, data: Dict[str, Any], fmt: Dict[str, Any],
                        video_args: List[str], audio_file: Optional[str],
                        output_path: str, pass_number: Optional[int] = None,
                        passlog: Optional[str] = None) -> List[str]:
        cmd = [
            _ffmpeg_binary(), "-hide_banner", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{data['width']}x{data['height']}",
            "-r", f"{data['frame_rate']:g}", "-i", "-",
        ]
        include_audio = audio_file is not None and pass_number in (None, 2)
        if include_audio:
            cmd += ["-i", audio_file]
        cmd += ["-frames:v", str(data["total_frames"])]
        cmd += video_args
        if pass_number is not None and passlog is not None:
            cmd += self._pass_args(fmt, pass_number, passlog)
        if include_audio:
            cmd += ["-map", "0:v:0", "-map", "1:a:0", "-shortest"]
            cmd += fmt["audio_args"]
        else:
            cmd += ["-an"]
        cmd += ["-pix_fmt", "yuv420p"]

        if pass_number == 1:
            cmd += ["-f", "null", os.devnull]
        else:
            cmd += fmt["extra_args"] + [output_path]
        return cmd

    def _encode(self, data: Dict[str, Any], fmt: Dict[str, Any],
                quality: int, preset: str, target_size_mb: float,
                audio_file: Optional[str], out_path: str,
                reporter: ProgressReporter) -> None:
        use_target = target_size_mb > 0
        passes = 2 if use_target and fmt["two_pass"] else 1
        video_args = self._video_args(
            fmt, quality, preset, target_size_mb, data["duration"],
            include_audio=audio_file is not None)

        if use_target:
            print(f"[StarSlideshowMaker] target {target_size_mb:g} MiB, "
                  f"duration {data['duration']:.2f}s")
        else:
            crf = self._quality_to_crf(quality, fmt["max_crf"])
            print(f"[StarSlideshowMaker] quality {quality} -> CRF {crf} "
                  f"({fmt['vcodec']})")

        if passes == 2:
            with tempfile.TemporaryDirectory() as temp_dir:
                passlog = os.path.join(temp_dir, "ffpass")
                command = self._ffmpeg_command(
                    data, fmt, video_args, audio_file, out_path,
                    pass_number=1, passlog=passlog)
                self._run_stream(command, data, reporter, 0.0, 0.5,
                                 "pass 1/2")
                command = self._ffmpeg_command(
                    data, fmt, video_args, audio_file, out_path,
                    pass_number=2, passlog=passlog)
                self._run_stream(command, data, reporter, 0.5, 0.5,
                                 "pass 2/2")
        else:
            command = self._ffmpeg_command(
                data, fmt, video_args, audio_file, out_path)
            label = "target size" if use_target else "encoding"
            self._run_stream(command, data, reporter, 0.0, 1.0, label)

        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            raise RuntimeError("FFmpeg finished without creating a video file.")

    def _run_stream(self, command: List[str], data: Dict[str, Any],
                    reporter: ProgressReporter, progress_offset: float,
                    progress_span: float, label: str) -> None:
        log_path = os.path.join(
            folder_paths.get_temp_directory(),
            f"star_slideshow_ffmpeg_{uuid.uuid4().hex}.log")
        proc: Optional[subprocess.Popen] = None
        try:
            with open(log_path, "wb") as err:
                proc = subprocess.Popen(
                    command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                    stderr=err)
                assert proc.stdin is not None
                state = {"segment_position": 0, "flow_cache": {}}
                total = data["total_frames"]
                update_every = max(1, int(data["frame_rate"]))
                for frame_index in range(total):
                    if frame_index % 10 == 0:
                        throw_exception_if_processing_interrupted()
                    frame = self._render_one_frame(data, frame_index, state)
                    proc.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
                    if frame_index % update_every == 0:
                        fraction = progress_offset + progress_span * \
                            ((frame_index + 1) / total)
                        reporter.update(
                            fraction,
                            f"{label}: frame {frame_index + 1}/{total}")
                proc.stdin.close()
                proc.stdin = None
                returncode = proc.wait()
        except BaseException:
            if proc is not None:
                try:
                    if proc.stdin is not None:
                        proc.stdin.close()
                except Exception:
                    pass
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
            raise
        finally:
            try:
                with open(log_path, "r", encoding="utf-8",
                          errors="replace") as handle:
                    log_tail = handle.read()[-5000:]
            except Exception:
                log_tail = ""
            try:
                os.remove(log_path)
            except OSError:
                pass

        if returncode != 0:
            raise RuntimeError(
                f"FFmpeg encoding failed during {label} "
                f"(exit {returncode}).\n\nCommand:\n{' '.join(command)}\n\n"
                f"Last FFmpeg output:\n{log_tail}")

    # --------------------------------- info --------------------------------

    @staticmethod
    def _build_info(data: Dict[str, Any], format_name: str, preset: str,
                    quality: int, target_size_mb: float, save_audio: bool,
                    out_path: str, out_info: Dict[str, Any],
                    elapsed: float) -> str:
        connected = data["connected_count"]
        folder_count = data["folder_count"]
        if connected and folder_count:
            source = f"{connected} connected + {folder_count} folder"
        elif folder_count:
            source = f"{folder_count} folder"
        else:
            source = f"{connected} connected"

        if target_size_mb > 0:
            mode = f"target {target_size_mb:g} MiB (quality ignored)"
        else:
            mode = f"quality {quality}"
        lines = [
            f"slideshow: {data['width']}x{data['height']} "
            f"{data['selected_aspect']} | {data['resolution']} | "
            f"{len(data['base_images'])} image(s) ({source}) | "
            f"{data['duration']:.2f}s ({data['duration_mode']}) | "
            f"{data['total_frames']} frames @ {data['frame_rate']:g} fps",
            f"timing: {data['timing_mode']} | "
            f"{data['seconds_per_image']:g}s/image | transition "
            f"{data['transition']} {data['transition_duration']:g}s | "
            f"motion {data['motion_effect']} | fit {data['fit_mode']}",
            f"encoder: {format_name} | preset {preset} | {mode} | "
            f"audio {'on' if save_audio else 'off'}",
            f"output: {_media_brief(out_info)}",
        ]
        if target_size_mb > 0 and out_info.get("size_mb"):
            deviation = (out_info["size_mb"] - target_size_mb) / \
                target_size_mb * 100.0
            lines.append(f"target result: {out_info['size_mb']:.2f} MiB "
                         f"({deviation:+.1f}%)")
        if data["resolved_folder"]:
            lines.append(f"folder: {data['resolved_folder']}")
        lines.append(f"saved: {out_path}")
        lines.append(f"time: {elapsed:.1f}s | frames streamed directly to "
                     "FFmpeg; no full IMAGE batch was allocated")
        return "\n".join(lines)


NODE_CLASS_MAPPINGS = {
    "StarSlideshowMaker": StarSlideshowMaker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarSlideshowMaker": "⭐ Star Slideshow Maker",
}
