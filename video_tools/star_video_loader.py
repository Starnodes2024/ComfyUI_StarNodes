"""
Star Video Loader - standalone video loading node (no VHS required).

Loads a video from the ComfyUI input folder (upload supported) and outputs:
  - video  : STAR_FILENAMES (path list for the Star Video Compressor)
  - images : IMAGE batch with the decoded frames
  - audio  : core AUDIO dict (or None if the video has no audio track)
  - fps    : FLOAT, effective frames per second of the returned images
  - frames : INT, number of returned frames
  - info   : STRING report

Frame control widgets (force_rate / skip / every-kth / cap) mirror the
classic load-video behavior so the images output drops straight into any
image workflow - or back into the Star Video Compressor.
"""

import io
import os
import wave

import numpy as np
import torch

import folder_paths

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
                "force_rate": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 240.0, "step": 0.01,
                    "tooltip": "Resample to this fps. 0 = keep the video's "
                               "original frame rate."}),
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
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STAR_FILENAMES", "IMAGE", "AUDIO", "FLOAT", "INT",
                    "STRING")
    RETURN_NAMES = ("video", "images", "audio", "fps", "frames", "info")
    FUNCTION = "load"
    CATEGORY = "⭐StarNodes/Video"
    OUTPUT_NODE = True
    DESCRIPTION = ("Load a video from the ComfyUI input folder: file "
                   "reference for the Star Video Compressor, decoded "
                   "frames (IMAGE), audio (AUDIO), fps and frame count.")

    # ------------------------------------------------------------------

    def load(self, video, force_rate=0.0, skip_first_frames=0,
             select_every_kth=1, frame_load_cap=0, unique_id=None):
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
        eff_fps = (force_rate if force_rate and force_rate > 0 else src_fps)
        eff_fps = eff_fps / max(1, int(select_every_kth))

        units = 2 if info.get("acodec") else 1
        reporter = ProgressReporter(total_units=units, label="loading",
                                    event_cb=make_event_cb(unique_id))
        import time
        t0 = time.time()

        # ---- decode frames -------------------------------------------
        vf = []
        if force_rate and force_rate > 0:
            vf.append(f"fps={force_rate:g}")
        if select_every_kth and int(select_every_kth) > 1:
            vf.append(f"select='not(mod(n\\,{int(select_every_kth)}))'")
        args = ["-i", path]
        if vf:
            args += ["-vf", ",".join(vf)]
        args += ["-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]

        raw = run_ffmpeg_pipe(args, duration=info.get("duration"),
                              reporter=reporter, sub="decoding frames")
        reporter.finish_unit()

        frame_size = w * h * 3
        n = len(raw) // frame_size
        if n == 0:
            raise RuntimeError("Star Video Loader: no frames were decoded "
                               "from this video.")
        arr = np.frombuffer(raw[:n * frame_size],
                            dtype=np.uint8).reshape(n, h, w, 3)
        decoded = n

        if skip_first_frames:
            arr = arr[int(skip_first_frames):]
        if frame_load_cap and frame_load_cap > 0:
            arr = arr[:int(frame_load_cap)]
        if arr.shape[0] == 0:
            raise RuntimeError(
                "Star Video Loader: skip_first_frames/frame_load_cap left "
                "0 frames.")
        if not arr.flags.writeable:
            arr = arr.copy()
        images = torch.from_numpy(arr).float() / 255.0
        loaded = int(images.shape[0])

        # ---- extract audio --------------------------------------------
        audio_out = None
        if info.get("acodec"):
            wav_bytes = run_ffmpeg_pipe(
                ["-i", path, "-vn", "-f", "wav", "pipe:1"],
                duration=info.get("duration"), reporter=reporter,
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

        # ---- info string + preview ------------------------------------
        est_total = int(round((info.get("duration") or 0) * src_fps))
        info_str = (
            f"{os.path.basename(path)} | {fmt_media_brief(info)}\n"
            f"frames: {loaded} loaded"
            + (f" (decoded {decoded}, source ~{est_total} @ "
               f"{src_fps:g} fps)" if decoded != loaded or
               est_total != decoded else f" (source @ {src_fps:g} fps)")
            + f"\neffective fps: {eff_fps:g}"
            f" | audio: {'yes (' + str(audio_out['sample_rate']) + ' Hz)'
                      if audio_out else 'none'}"
            f"\npath: {path}")
        print("[StarVideoLoader]\n" + info_str)

        # No custom preview here on purpose: the ComfyUI frontend already
        # renders a video preview for the upload/selection widget - a
        # second one would only duplicate it (and overflow the node).
        return {"result": ((True, [path]), images, audio_out,
                           float(eff_fps), loaded, info_str)}

    # ------------------------------------------------------------------

    @classmethod
    def IS_CHANGED(cls, video, force_rate=0.0, skip_first_frames=0,
                   select_every_kth=1, frame_load_cap=0, **kwargs):
        try:
            mtime = os.path.getmtime(
                folder_paths.get_annotated_filepath(video))
        except OSError:
            mtime = float("nan")
        return f"{mtime}-{force_rate}-{skip_first_frames}-" \
               f"{select_every_kth}-{frame_load_cap}"

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
