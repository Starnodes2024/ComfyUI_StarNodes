"""
Star Video Loader - standalone video loading node.

Loads a video from the ComfyUI input folder (upload supported) and outputs:
  - video  : STAR_FILENAMES (path list for the Star Video Compressor)
  - images : IMAGE batch with the decoded frames
  - audio  : core AUDIO dict (or None if the video has no audio track)
  - fps    : FLOAT, effective frames per second of the returned images
  - frames : INT, number of returned frames
  - info   : STRING report
  - video_native : VIDEO (native ComfyUI VideoInput, cut to the same range)

Frame control widgets (force_rate / skip / every-kth / cap) mirror the
classic load-video behavior so the images output drops straight into any
image workflow - or back into the Star Video Compressor. force_rate is
an original/24/25/30 selector that converts the frame rate by
duplicating or dropping frames evenly across the clip.

The frontend adds a "Load Video" button: it probes the selected file without
running the workflow (/starnodes/video_loader/info), shows an inline preview
and sets the start_frame/end_frame slider ranges - the cut is applied to
frames and audio when the workflow runs.

A "Crop" button opens a dialog with a scrub slider: draw a box over the
frame and Apply stores it in the crop_rect widget as 'x,y,w,h' fractions of
the source frame. The crop runs inside ffmpeg, so the images output is
already cropped - smaller output also means faster decoding and less RAM.
"""

import io
import math
import os
import wave

import numpy as np
import torch

import folder_paths

from comfy_api.latest import VideoFromFile

_RESIZE_MODES = ["original", "0.5 MP", "1.0 MP", "1.5 MP", "2.0 MP", "3.0 MP",
                 "HD (1280x720)", "Full HD (1920x1080)"]


def _resize_dims(width, height, mode):
    """Output resolution for the resize widget: MP presets scale the frame to
    that pixel area, HD/Full HD fit it inside the box - aspect-preserved,
    rounded to even sizes."""
    if mode == "original":
        return width, height
    if "(" in mode:
        bw, bh = (int(v) for v in mode.split("(", 1)[1].rstrip(")").split("x"))
        s = min(bw / width, bh / height)
    else:
        s = math.sqrt(float(mode.split()[0]) * 1e6 / (width * height))
    return max(2, int(width * s) // 2 * 2), max(2, int(height * s) // 2 * 2)

from .star_nodes_common import (
    ProgressReporter,
    fmt_media_brief,
    make_event_cb,
    probe_media,
    run_ffmpeg_pipe,
)

VIDEO_EXTENSIONS = (".mp4", ".webm", ".mkv", ".mov", ".avi", ".m4v",
                    ".mpg", ".mpeg", ".gif", ".wmv", ".flv", ".ts")


def list_input_videos():
    input_dir = folder_paths.get_input_directory()
    files = []
    for root, _, names in os.walk(input_dir):
        for name in names:
            if name.lower().endswith(VIDEO_EXTENSIONS):
                rel = os.path.relpath(os.path.join(root, name), input_dir)
                files.append(rel.replace(os.sep, "/"))
    return sorted(files, key=str.lower)


class StarVideoLoader:
    """Load a video: file reference, frames, audio, fps and frame count."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (list_input_videos(), {"video_upload": True}),
                "force_rate": (["original", "24", "25", "30"], {
                    "default": "original",
                    "tooltip": "Output frame rate. Frames are duplicated "
                               "or dropped evenly across the clip (no "
                               "blending) to hit the target - e.g. "
                               "24 -> 30 adds 6 frames per second, "
                               "30 -> 24 removes 6. 'original' keeps "
                               "the video's own frame rate."}),
                "skip_first_frames": ("INT", {
                    "default": 0, "min": 0, "max": 1000000, "step": 1,
                    "tooltip": "Drop this many frames from the start "
                               "(applied after force_rate / every-kth)."}),
                "select_every_kth": ("INT", {
                    "default": 1, "min": 1, "max": 1000, "step": 1,
                    "tooltip": "Keep only every k-th frame (1 = keep all). "
                               "Also divides the output fps."}),
                "frame_load_cap": ("INT", {
                    "default": 0, "min": 0, "max": 1000000, "step": 1,
                    "tooltip": "Maximum number of frames to load "
                               "(0 = all). Use this to protect your RAM "
                               "with long videos."}),
                "start_frame": ("INT", {
                    "default": 0, "min": 0, "max": 1000000, "step": 1,
                    "tooltip": "First frame to keep (0 = start of the "
                               "video). Click the Load button to probe the "
                               "video and preview the cut point."}),
                "end_frame": ("INT", {
                    "default": 0, "min": 0, "max": 1000000, "step": 1,
                    "tooltip": "End frame (exclusive, 0 = to the end). "
                               "Click the Load button to probe the video "
                               "and preview the cut point."}),
            },
            "optional": {
                "crop_rect": ("STRING", {
                    "default": "",
                    "tooltip": "Crop region as 'x,y,w,h' fractions of the "
                               "source frame (0-1), e.g. 0.1,0.2,0.8,0.6. "
                               "Set it with the Crop button on the node. "
                               "Empty = full frame."}),
                "resize": (_RESIZE_MODES, {
                    "default": "original",
                    "tooltip": "Resize the decoded output frames after "
                               "cropping: a megapixel target (scales to that "
                               "pixel area) or an HD/Full HD box (fits inside "
                               "it) - aspect preserved, sizes rounded to "
                               "even. 'original' keeps the source/crop size."}),
                "minimax_snap": ("BOOLEAN", {
                    "default": False,
                    "label_on": "17n+5 grid", "label_off": "free",
                    "tooltip": "Snap the output to the MiniMax H3 17n+5 frame "
                               "grid (5, 22, 39, 56, ... frames): the trim "
                               "sliders and start/end fields only produce "
                               "lengths a Minimax node renders without "
                               "padding. A non-aligned tail is unusable "
                               "anyway, so nothing is lost. Off = free "
                               "cutting."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STAR_FILENAMES", "IMAGE", "AUDIO", "FLOAT", "INT",
                    "STRING", "VIDEO", "FLOAT")
    RETURN_NAMES = ("video", "images", "audio", "fps", "frames", "info",
                    "video_native", "seconds")
    FUNCTION = "load"
    CATEGORY = "⭐StarNodes/Video"
    OUTPUT_NODE = True
    DESCRIPTION = ("Load a video from the ComfyUI input folder: file "
                   "reference for the Star Video Compressor, decoded "
                   "frames (IMAGE), audio (AUDIO), fps, frame count and "
                   "duration in seconds.")

    # ------------------------------------------------------------------

    def load(self, video, force_rate="original", skip_first_frames=0,
             select_every_kth=1, frame_load_cap=0, start_frame=0,
             end_frame=0, crop_rect="", resize="original",
             minimax_snap=False, unique_id=None):
        path = folder_paths.get_annotated_filepath(video)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Star Video Loader: video not found: {video}")

        info = probe_media(path)
        if not info.get("width") or not info.get("height"):
            raise RuntimeError(
                "Star Video Loader: could not probe the video dimensions - "
                "is this really a video file?")
        w, h = info["width"], info["height"]
        src_fps = info.get("fps") or 30.0
        # combo sends "original"/"24"/"25"/"30"; old workflows may still
        # carry a plain number - both parse through float()
        try:
            fr = float(force_rate)
        except (TypeError, ValueError):
            fr = 0.0
        eff_fps = (fr if fr > 0 else src_fps) / max(1, int(select_every_kth))

        # ---- crop rect (fractions of the source frame -> pixels) --------
        crop = None
        rect = (crop_rect or "").strip()
        if rect:
            try:
                fx, fy, fw, fh = [
                    float(v) for v in rect.split(",")[:4]]
            except (TypeError, ValueError):
                raise ValueError(
                    "Star Video Loader: crop_rect must be 'x,y,w,h' "
                    f"fractions of the frame, got '{rect}'")
            x0 = max(0, min(int(round(fx * w)), w - 1))
            y0 = max(0, min(int(round(fy * h)), h - 1))
            x1 = max(x0 + 1, min(int(round((fx + fw) * w)), w))
            y1 = max(y0 + 1, min(int(round((fy + fh) * h)), h))
            if x1 - x0 < w or y1 - y0 < h:
                crop = (x0, y0, x1 - x0, y1 - y0)
        out_w = crop[2] if crop else w
        out_h = crop[3] if crop else h

        units = 2 if info.get("acodec") else 1
        reporter = ProgressReporter(total_units=units, label="loading",
                                    event_cb=make_event_cb(unique_id))
        import time
        t0 = time.time()

        # ---- decode frames -------------------------------------------
        vf = []
        if fr > 0:
            vf.append(f"fps={fr:g}")
        if select_every_kth and int(select_every_kth) > 1:
            vf.append(f"select='not(mod(n\\,{int(select_every_kth)}))'")
        if crop:
            vf.append(f"crop={crop[2]}:{crop[3]}:{crop[0]}:{crop[1]}")
        tw, th = _resize_dims(out_w, out_h, resize)
        if (tw, th) != (out_w, out_h):
            vf.append(f"scale={tw}:{th}")
            out_w, out_h = tw, th
        args = ["-i", path]
        if vf:
            args += ["-vf", ",".join(vf)]
        args += ["-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]

        raw = run_ffmpeg_pipe(args, duration=info.get("duration"),
                              reporter=reporter, sub="decoding frames")
        reporter.finish_unit()

        frame_size = out_w * out_h * 3
        n = len(raw) // frame_size
        if n == 0:
            raise RuntimeError("Star Video Loader: no frames were decoded "
                               "from this video.")
        arr = np.frombuffer(raw[:n * frame_size],
                            dtype=np.uint8).reshape(n, out_h, out_w, 3)
        decoded = n

        if skip_first_frames:
            arr = arr[int(skip_first_frames):]
        trimmed = ""
        start = 0
        if start_frame or end_frame > 0:
            n0 = arr.shape[0]
            start = min(max(0, int(start_frame)), n0)
            end = int(end_frame) if 0 < int(end_frame) <= n0 else n0
            arr = arr[start:max(start, end)]
            trimmed = f", trimmed {start}-{end}"
        if frame_load_cap and frame_load_cap > 0:
            arr = arr[:int(frame_load_cap)]
        if minimax_snap and arr.shape[0] > 5:
            # keep the output on the MiniMax 17n+5 grid so a Minimax node can
            # render it without padding - a non-aligned tail is unusable anyway
            arr = arr[:5 + 17 * ((arr.shape[0] - 5) // 17)]
        if arr.shape[0] == 0:
            raise RuntimeError(
                "Star Video Loader: skip_first_frames/start_frame/end_frame/"
                "frame_load_cap left 0 frames.")

        if not arr.flags.writeable:
            arr = arr.copy()
        images = torch.from_numpy(arr).float() / 255.0
        loaded = int(images.shape[0])

        # ---- extract audio (cut to the same range as the kept frames) ----
        audio_out = None
        if info.get("acodec"):
            cut = skip_first_frames or start or end_frame > 0 \
                or (frame_load_cap and frame_load_cap > 0)
            if cut:
                ss = (int(skip_first_frames) + start) / eff_fps
                aargs = ["-ss", f"{ss:.6f}", "-i", path, "-vn",
                         "-t", f"{loaded / eff_fps:.6f}", "-f", "wav", "pipe:1"]
            else:
                aargs = ["-i", path, "-vn", "-f", "wav", "pipe:1"]
            wav_bytes = run_ffmpeg_pipe(
                aargs, duration=info.get("duration"), reporter=reporter,
                sub="extracting audio")
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                channels = wf.getnchannels()
                sample_rate = wf.getframerate()
                pcm = wf.readframes(wf.getnframes())
            a = np.frombuffer(pcm, dtype=np.int16) \
                .astype(np.float32) / 32768.0
            a = a.reshape(-1, channels).T.copy()  # (C, N)
            audio_out = {"waveform": torch.from_numpy(a).unsqueeze(0),
                         "sample_rate": sample_rate}
            reporter.finish_unit()

        reporter.finish_all(time.time() - t0)

        # ---- native VIDEO output (VideoFromFile with cut applied) ----
        ss = (int(skip_first_frames) + start) / eff_fps
        native_video = VideoFromFile(
            path, start_time=ss, duration=loaded / eff_fps)

        # ---- info string + preview ------------------------------------
        est_total = int(round((info.get("duration") or 0) * src_fps))
        info_str = (
            f"{os.path.basename(path)} | {fmt_media_brief(info)}\n"
            f"frames: {loaded} loaded"
            + (f" | crop {crop[2]}x{crop[3]}" if crop else "")
            + (f" | resize {out_w}x{out_h}" if resize != "original" else "")
            + (" | 17n+5 snap" if minimax_snap else "")
            + (
                f" (decoded {decoded}{trimmed}, source ~{est_total} @ {src_fps:g} fps)"
                if decoded != loaded or est_total != decoded
                else f" (source @ {src_fps:g} fps)"
            )
            + f"\neffective fps: {eff_fps:g}"
            + (
                f" | audio: yes ({audio_out['sample_rate']} Hz)"
                if audio_out
                else " | audio: none"
            )
            + f"\npath: {path}"
        )     
        print("[StarVideoLoader]\n" + info_str)

        # The inline preview is filled by the frontend via the Load button
        # (see web/js/star_video_compressor.js) - no workflow run needed.
        return {"result": ((True, [path]), images, audio_out,
                           float(eff_fps), loaded, info_str, native_video,
                           loaded / eff_fps)}

    # ------------------------------------------------------------------

    @classmethod
    def IS_CHANGED(cls, video, force_rate="original", skip_first_frames=0,
                   select_every_kth=1, frame_load_cap=0, start_frame=0,
                   end_frame=0, crop_rect="", resize="original",
                   minimax_snap=False, **kwargs):
        try:
            mtime = os.path.getmtime(
                folder_paths.get_annotated_filepath(video))
        except OSError:
            mtime = float("nan")
        return f"{mtime}-{force_rate}-{skip_first_frames}-" \
               f"{select_every_kth}-{frame_load_cap}-{start_frame}-{end_frame}-" \
               f"{crop_rect}-{resize}-{minimax_snap}"

    @classmethod
    def VALIDATE_INPUTS(cls, video, **kwargs):
        if not folder_paths.exists_annotated_filepath(video):
            return f"Invalid video file: {video}"
        return True


NODE_CLASS_MAPPINGS = {
    "StarVideoLoader": StarVideoLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarVideoLoader": "⭐ Star Video Loader",
}


# API endpoint for the node's Load button: probe the video without running
# the workflow so the frontend can set the start/end slider ranges and show
# the preview.
try:
    from server import PromptServer
    from aiohttp import web

    @PromptServer.instance.routes.post("/starnodes/video_loader/info")
    async def star_video_loader_info(request):
        try:
            data = await request.json()
            video = data.get("video", "")
            try:
                force_rate = float(data.get("force_rate") or 0)
            except (TypeError, ValueError):
                force_rate = 0.0   # "original" keeps the source rate
            kth = max(1, int(data.get("select_every_kth") or 1))
            path = folder_paths.get_annotated_filepath(video)
            if not os.path.exists(path):
                return web.json_response(
                    {"status": "error", "message": f"video not found: {video}"},
                    status=404)
            info = probe_media(path)
            src_fps = info.get("fps") or 30.0
            work_fps = force_rate if force_rate > 0 else src_fps
            frames_est = max(1, int(round((info.get("duration") or 0)
                                          * work_fps / kth)))
            return web.json_response({
                "status": "ok",
                "frames_est": frames_est,
                "fps": work_fps / kth,
                "src_fps": src_fps,
                "duration": info.get("duration"),
                "width": info.get("width"),
                "height": info.get("height"),
                "has_audio": bool(info.get("acodec")),
                "brief": fmt_media_brief(info),
            })
        except Exception as e:
            return web.json_response({"status": "error", "message": str(e)},
                                     status=500)
except Exception as e:
    print(f"[StarVideoLoader] could not register info endpoint: {e}")
