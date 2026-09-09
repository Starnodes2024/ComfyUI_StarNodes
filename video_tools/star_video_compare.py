"""
Star Video Compare - interactive video comparison with a draggable slider.

Accepts IMAGE batches (video frames) and/or native VIDEO inputs. Both videos
are normalized (smaller is upscaled to match the larger via Lanczos), then
shown side-by-side in the node's interactive preview with a draggable wipe
slider. Also produces a stitched comparison video (IMAGE batch) with
optional captions that can be piped into any save/compress node.
"""

import logging
import os
import platform
import uuid

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import folder_paths


def _decode_native_video(video_native):
    """Decode a native ComfyUI VIDEO input to an IMAGE batch via ffmpeg."""
    from .star_nodes_common import probe_media, run_ffmpeg_pipe
    import tempfile

    src = video_native.get_stream_source()
    tmp = None
    if not isinstance(src, str):
        fd, tmp = tempfile.mkstemp(suffix="_star_vcmp.mp4")
        os.close(fd)
        with open(tmp, "wb") as f:
            f.write(src.read())
        src = tmp

    try:
        info = probe_media(src)
        w, h = info.get("width"), info.get("height")
        if not w or not h:
            return None
        raw = run_ffmpeg_pipe(["-i", src, "-an", "-f", "rawvideo",
                               "-pix_fmt", "rgb24", "pipe:1"])
        frame_size = w * h * 3
        n = len(raw) // frame_size
        if n == 0:
            return None
        arr = np.frombuffer(raw[:n * frame_size],
                            dtype=np.uint8).reshape(n, h, w, 3).copy()
        return torch.from_numpy(arr).float() / 255.0
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _save_preview_video(frames_tensor, fps, output_dir, label):
    """Save a frame batch as a small preview mp4. Returns metadata dict or None."""
    from .star_nodes_common import run_ffmpeg

    if frames_tensor is None or frames_tensor.shape[0] == 0:
        return None

    arr = (frames_tensor.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    n, h, w, c = arr.shape
    if w % 2:
        arr = arr[:, :, :-1, :]
        w -= 1
    if h % 2:
        arr = arr[:, :-1, :, :]
        h -= 1

    filename = f"star_vcompare_{label}_{uuid.uuid4().hex}.mp4"
    filepath = os.path.join(output_dir, filename)
    duration = n / max(fps, 0.1)

    payload = arr.tobytes()
    run_ffmpeg(
        ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
         "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-preset", "fast", "-crf", "26",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         filepath],
        duration=duration, input_bytes=payload)

    return {"filename": filename, "type": "temp", "subfolder": ""}


def _lanczos_resize_batch(img_tensor, target_w, target_h):
    """Resize an IMAGE batch (N,H,W,C) to (target_w, target_h) via PIL Lanczos."""
    n = img_tensor.shape[0]
    out = np.zeros((n, target_h, target_w, 3), dtype=np.uint8)
    arr = (img_tensor.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    for i in range(n):
        pil = Image.fromarray(arr[i])
        pil = pil.resize((target_w, target_h), Image.LANCZOS)
        out[i] = np.array(pil)
    return torch.from_numpy(out).float() / 255.0


def _get_font(size=18):
    search_dirs = []
    if platform.system() == "Windows":
        search_dirs.append(os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts"))
    elif platform.system() == "Darwin":
        search_dirs.extend(["/System/Library/Fonts", "/Library/Fonts"])
    else:
        search_dirs.extend(["/usr/share/fonts", "/usr/local/share/fonts"])

    font_names = ["arial.ttf", "Arial.ttf", "segoeui.ttf", "Verdana.ttf",
                  "DejaVuSans.ttf", "FreeSans.ttf"]
    for name in font_names:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    for name in font_names:
        for d in search_dirs:
            path = os.path.join(d, name)
            if os.path.isfile(path):
                try:
                    return ImageFont.truetype(path, size)
                except Exception:
                    pass

    logging.warning("[StarVideoCompare] no truetype font found, "
                    "falling back to default bitmap font")
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def _draw_caption_bar(img, caption):
    """Draw a caption bar below a PIL image. Returns new image."""
    w, h = img.size
    bar_h = max(24, min(140, int(h * 0.08)))
    font_size = max(16, int(bar_h * 0.65))
    out = Image.new("RGB", (w, h + bar_h), (0, 0, 0))
    out.paste(img, (0, 0))
    text = (caption or "").strip()
    if text:
        draw = ImageDraw.Draw(out)
        font = _get_font(font_size)
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            x = (w - tw) / 2 - bbox[0]
            y = h + (bar_h - th) / 2 - bbox[1]
        except Exception:
            try:
                tw, th = draw.textsize(text, font=font)
            except Exception:
                tw, th = w, bar_h
            x = (w - tw) / 2
            y = h + (bar_h - th) / 2
        draw.text((x, y), text, fill=(255, 255, 255), font=font)
    return out


class StarVideoCompare:
    """Compare two videos with an interactive wipe slider."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "layout": (["left/right", "top/bottom"], {
                    "default": "left/right",
                    "tooltip": "Orientation of the stitched comparison video output."}),
                "max_size": ("INT", {
                    "default": 1920, "min": 0, "max": 16384, "step": 8,
                    "tooltip": "Longest side of the stitched output video. "
                               "0 = no resize."}),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01,
                    "tooltip": "Frame rate for the preview playback and "
                               "the stitched output."}),
                "loop": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Loop the preview playback."}),
                "caption_video1": ("STRING", {
                    "default": "", "tooltip": "Caption for video 1 "
                    "(left / top half of the stitched output)."}),
                "caption_video2": ("STRING", {
                    "default": "", "tooltip": "Caption for video 2 "
                    "(right / bottom half of the stitched output)."}),
                "compare_position": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Initial slider position. 0.0 = Video 2, "
                               "1.0 = Video 1."}),
            },
            "optional": {
                "video_1": ("IMAGE", {"forceInput": True, "tooltip":
                            "First video as IMAGE batch (frames)."}),
                "video_2": ("IMAGE", {"forceInput": True, "tooltip":
                            "Second video as IMAGE batch (frames)."}),
                "video_native_1": ("VIDEO", {"tooltip":
                                   "Native ComfyUI VIDEO input for video 1. "
                                   "Used when video_1 is not connected."}),
                "video_native_2": ("VIDEO", {"tooltip":
                                   "Native ComfyUI VIDEO input for video 2. "
                                   "Used when video_2 is not connected."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("comparison_video",)
    FUNCTION = "compare_videos"
    OUTPUT_NODE = True
    CATEGORY = "⭐StarNodes/Video"
    DESCRIPTION = ("Interactive video comparison with a draggable slider. "
                   "Produces a stitched comparison video with optional "
                   "captions that can be piped to a save/compress node.")

    def compare_videos(self, layout, max_size, fps, loop,
                       caption_video1, caption_video2, compare_position=0.5,
                       video_1=None, video_2=None,
                       video_native_1=None, video_native_2=None):

        # Resolve inputs: IMAGE batches take priority, fall back to native VIDEO
        frames_1 = video_1
        if frames_1 is None and video_native_1 is not None:
            frames_1 = _decode_native_video(video_native_1)

        frames_2 = video_2
        if frames_2 is None and video_native_2 is not None:
            frames_2 = _decode_native_video(video_native_2)

        if frames_1 is None and frames_2 is None:
            empty = torch.zeros(1, 64, 64, 3)
            return {"ui": {"star_video_compare": [{"video1": None, "video2": None,
                                                    "compare_position": compare_position,
                                                    "fps": fps, "loop": loop}]},
                    "result": (empty,)}

        # Normalize dimensions: upscale the smaller to match the larger
        if frames_1 is not None and frames_2 is not None:
            h1, w1 = frames_1.shape[1], frames_1.shape[2]
            h2, w2 = frames_2.shape[1], frames_2.shape[2]
            target_w, target_h = max(w1, w2), max(h1, h2)
            if (w1, h1) != (target_w, target_h):
                frames_1 = _lanczos_resize_batch(frames_1, target_w, target_h)
            if (w2, h2) != (target_w, target_h):
                frames_2 = _lanczos_resize_batch(frames_2, target_w, target_h)

        # Preview videos for the frontend
        temp_dir = folder_paths.get_temp_directory()
        meta1 = _save_preview_video(frames_1, fps, temp_dir, "v1")
        meta2 = _save_preview_video(frames_2, fps, temp_dir, "v2")

        # Stitched comparison video
        stitched = self._create_comparison_video(
            frames_1, frames_2, layout, max_size, caption_video1, caption_video2)

        return {
            "ui": {"star_video_compare": [{
                "video1": meta1, "video2": meta2,
                "compare_position": compare_position,
                "fps": fps, "loop": loop}]},
            "result": (stitched,)
        }

    def _create_comparison_video(self, frames_1, frames_2, layout, max_size,
                                 caption1, caption2):
        """Build the stitched comparison video as an IMAGE batch."""
        if frames_1 is None and frames_2 is None:
            return torch.zeros(1, 64, 64, 3)

        # Use whichever we have; pad with itself if only one is connected
        f1 = frames_1 if frames_1 is not None else frames_2
        f2 = frames_2 if frames_2 is not None else frames_1
        cap1 = caption1 or ""
        cap2 = caption2 or ""
        has_cap = bool(cap1.strip()) or bool(cap2.strip())

        # Match frame counts: loop the shorter one
        n1, n2 = f1.shape[0], f2.shape[0]
        n_out = max(n1, n2)
        if n1 < n_out:
            reps = (n_out + n1 - 1) // n1
            f1 = f1.repeat(reps, 1, 1, 1)[:n_out]
        if n2 < n_out:
            reps = (n_out + n2 - 1) // n2
            f2 = f2.repeat(reps, 1, 1, 1)[:n_out]

        h, w = f1.shape[1], f1.shape[2]

        side_by_side = (layout == "left/right")

        # Calculate caption bar height (same as _draw_caption_bar logic)
        bar_h = 0
        if has_cap:
            bar_h = max(24, min(140, int(h * 0.08)))

        # Compose each frame
        arr1 = (f1.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        arr2 = (f2.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

        stitched_frames = []
        for i in range(n_out):
            pil1 = Image.fromarray(arr1[i])
            pil2 = Image.fromarray(arr2[i])

            if has_cap:
                pil1 = _draw_caption_bar(pil1, cap1)
                pil2 = _draw_caption_bar(pil2, cap2)

            if side_by_side:
                total_w = pil1.width + pil2.width
                total_h = max(pil1.height, pil2.height)
                out = Image.new("RGB", (total_w, total_h), (0, 0, 0))
                out.paste(pil1, (0, 0))
                out.paste(pil2, (pil1.width, 0))
            else:
                total_w = max(pil1.width, pil2.width)
                total_h = pil1.height + pil2.height
                out = Image.new("RGB", (total_w, total_h), (0, 0, 0))
                out.paste(pil1, (0, 0))
                out.paste(pil2, (0, pil1.height))

            # Enforce max_size on longest side
            if max_size > 0:
                sw, sh = out.size
                longest = max(sw, sh)
                if longest > max_size:
                    scale = max_size / longest
                    new_w = max(8, int(sw * scale) & ~1)
                    new_h = max(8, int(sh * scale) & ~1)
                    out = out.resize((new_w, new_h), Image.LANCZOS)

            stitched_frames.append(np.array(out))

        batch = np.stack(stitched_frames, axis=0).astype(np.float32) / 255.0
        return torch.from_numpy(batch)


NODE_CLASS_MAPPINGS = {
    "StarVideoCompare": StarVideoCompare,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarVideoCompare": "⭐ Star Video Compare",
}
