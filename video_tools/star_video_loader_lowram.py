"""
Star Video Loader (Low RAM) - windowed video loading for large/long files.

Unlike "Star Video Loader" (which decodes the whole video into one big
tensor), this node only ever holds one chunk of frames in memory:

  - start_frame / frame_count pick a window of the output frame sequence
    (after force_rate / select_every_kth), decoded with a fast ffmpeg seek.
  - next_start_frame + total_frames outputs make it easy to chain or loop
    chunks through a workflow.
  - cache_to_disk=True decodes the video ONCE to a raw frame file in the
    ComfyUI temp folder; later runs read the window straight from disk via
    memory-mapping, so re-runs are fast and never decode the file again.

Outputs: images (chunk), audio (matching time window), fps, frames loaded,
total source frames, next_start_frame and an info string.
"""

import hashlib
import io
import json
import os
import subprocess
import time
import wave

import numpy as np
import torch

import folder_paths

from .star_nodes_common import (
    ProgressReporter,
    fmt_media_brief,
    get_ffmpeg,
    make_event_cb,
    probe_media,
    run_ffmpeg_pipe,
)
from .star_video_loader import list_input_videos


def _read_wav_bytes(wav_bytes):
    """Parse wav bytes into (float32 array (C, N), sample_rate)."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    return a.reshape(-1, channels).T.copy(), sample_rate


def _decode_all_to_disk(path, raw_path, vf, duration, reporter):
    """Decode the whole video to a raw rgb24 file on disk, streaming the
    ffmpeg output straight to the file so memory stays flat even for
    very long videos."""
    cmd = [get_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y"]
    use_progress = bool(duration) and duration > 0 and reporter is not None
    if use_progress:
        cmd += ["-nostats", "-progress", "pipe:2"]
    cmd += ["-i", path]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    cmd += ["-an", "-f", "rawvideo", "-pix_fmt", "rgb24", raw_path]

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE)
    err_lines = []
    for raw_line in proc.stderr:
        line = raw_line.decode("utf-8", "ignore").strip()
        if not line:
            continue
        if use_progress and "=" in line:
            key, _, val = line.partition("=")
            if key in ("out_time_ms", "out_time_us"):
                try:
                    cur_s = int(val) / 1_000_000.0
                except ValueError:
                    continue
                reporter.report(min(cur_s / duration, 0.999),
                                cur_s=cur_s, total_s=duration,
                                sub="extracting frames to cache")
            elif key == "progress" and val == "end":
                reporter.report(1.0, cur_s=duration, total_s=duration,
                                sub="extracting frames to cache")
        else:
            err_lines.append(line)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("Star Video Loader (Low RAM): ffmpeg cache "
                           "extraction failed.\n" + "\n".join(err_lines[-30:]))


class StarVideoLoaderLowRAM:
    """Load only a window of a video into memory - built for large files."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (list_input_videos(), {"video_upload": True}),
                "start_frame": ("INT", {
                    "default": 0, "min": 0, "max": 100000000, "step": 1,
                    "tooltip": "First frame of the window, counted in the "
                               "output frame sequence (after force_rate / "
                               "every-kth). Feed next_start_frame back in "
                               "here to walk through the video in chunks."}),
                "frame_count": ("INT", {
                    "default": 81, "min": 1, "max": 1000000, "step": 1,
                    "tooltip": "How many frames to load into memory. This "
                               "is the ONLY part of the video that uses "
                               "RAM - pick what fits (81 frames of 1080p "
                               "is roughly 500 MB)."}),
                "force_rate": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 240.0, "step": 0.01,
                    "tooltip": "Resample to this fps. 0 = keep the video's "
                               "original frame rate."}),
                "select_every_kth": ("INT", {
                    "default": 1, "min": 1, "max": 1000, "step": 1,
                    "tooltip": "Keep only every k-th frame (1 = keep all). "
                               "Also divides the output fps."}),
                "load_audio": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Extract the audio matching the loaded "
                               "frame window."}),
                "cache_to_disk": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Decode the video ONCE into a raw frame "
                               "cache inside the ComfyUI temp folder. Later "
                               "runs (any window) read straight from disk "
                               "without decoding again - great for "
                               "repeated runs on the same long video. The "
                               "cache is invalidated automatically when "
                               "the file or the fps settings change."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("images", "audio", "fps", "frames", "total_frames",
                    "next_start_frame", "info")
    FUNCTION = "load"
    CATEGORY = "⭐StarNodes/Video"
    OUTPUT_NODE = True
    DESCRIPTION = ("Load only a window of a video (start_frame + "
                   "frame_count) so even very long videos fit in RAM. "
                   "Optionally caches decoded frames on disk for fast "
                   "re-runs. Chain windows via the next_start_frame "
                   "output. See web/docs/StarVideoLoaderLowRAM.md")

    # ------------------------------------------------------------------
    # disk cache
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_dir(path, force_rate, select_every_kth):
        st = os.stat(path)
        key = hashlib.sha1(
            f"{os.path.abspath(path)}|{st.st_mtime}|{st.st_size}|"
            f"{force_rate}|{select_every_kth}".encode("utf-8")
        ).hexdigest()[:16]
        return os.path.join(folder_paths.get_temp_directory(),
                            "star_lowram_cache", key)

    @staticmethod
    def _cache_valid(cache_dir):
        meta_path = os.path.join(cache_dir, "meta.json")
        raw_path = os.path.join(cache_dir, "frames.raw")
        if not (os.path.exists(meta_path) and os.path.exists(raw_path)):
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, ValueError):
            return None
        frame_size = meta["w"] * meta["h"] * 3
        if frame_size <= 0:
            return None
        actual = os.path.getsize(raw_path) // frame_size
        if actual < meta["frames"]:
            return None
        return meta

    # ------------------------------------------------------------------

    def load(self, video, start_frame=0, frame_count=81, force_rate=0.0,
             select_every_kth=1, load_audio=True, cache_to_disk=False,
             unique_id=None):
        path = folder_paths.get_annotated_filepath(video)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Star Video Loader (Low RAM): video not found: {video}")

        info = probe_media(path)
        if not info.get("width") or not info.get("height"):
            raise RuntimeError(
                "Star Video Loader (Low RAM): could not probe the video "
                "dimensions - is this really a video file?")
        w, h = info["width"], info["height"]
        src_fps = info.get("fps") or 30.0
        kth = max(1, int(select_every_kth))
        eff_fps = (force_rate if force_rate and force_rate > 0 else src_fps)
        eff_fps = eff_fps / kth

        start_frame = max(0, int(start_frame))
        frame_count = max(1, int(frame_count))
        seek_time = start_frame / eff_fps
        window_s = frame_count / eff_fps

        vf = []
        if force_rate and force_rate > 0:
            vf.append(f"fps={force_rate:g}")
        if kth > 1:
            vf.append(f"select='not(mod(n\\,{kth}))'")

        has_audio = bool(info.get("acodec")) and load_audio
        event_cb = make_event_cb(unique_id)
        t0 = time.time()
        from_cache = False

        # ---- frames --------------------------------------------------
        if cache_to_disk:
            cache_dir = self._cache_dir(path, force_rate, select_every_kth)
            meta = self._cache_valid(cache_dir)
            raw_path = os.path.join(cache_dir, "frames.raw")
            if meta is None:
                os.makedirs(cache_dir, exist_ok=True)
                reporter = ProgressReporter(
                    total_units=2 if has_audio else 1,
                    label="building cache", event_cb=event_cb)
                _decode_all_to_disk(path, raw_path, vf,
                                    info.get("duration"), reporter)
                reporter.finish_unit()
                frame_size = w * h * 3
                total = os.path.getsize(raw_path) // frame_size
                if total == 0:
                    raise RuntimeError(
                        "Star Video Loader (Low RAM): no frames were "
                        "decoded from this video.")
                meta = {"w": w, "h": h, "frames": total, "fps": eff_fps}
                with open(os.path.join(cache_dir, "meta.json"), "w",
                          encoding="utf-8") as f:
                    json.dump(meta, f)
                if has_audio:
                    wav = run_ffmpeg_pipe(
                        ["-i", path, "-vn", "-f", "wav", "pipe:1"],
                        duration=info.get("duration"), reporter=reporter,
                        sub="extracting audio to cache")
                    with open(os.path.join(cache_dir, "audio.wav"),
                              "wb") as f:
                        f.write(wav)
                    reporter.finish_unit()
                print(f"[StarVideoLoaderLowRAM] cache built: {cache_dir} "
                      f"({meta['frames']} frames)")
            else:
                reporter = ProgressReporter(
                    total_units=2 if has_audio else 1,
                    label="reading cache", event_cb=event_cb)
                from_cache = True

            total = meta["frames"]
            if start_frame >= total:
                raise ValueError(
                    f"Star Video Loader (Low RAM): start_frame "
                    f"{start_frame} is beyond the end of the video "
                    f"({total} frames).")
            n = min(frame_count, total - start_frame)
            mm = np.memmap(raw_path, dtype=np.uint8, mode="r",
                           shape=(total, meta["h"], meta["w"], 3))
            arr = np.array(mm[start_frame:start_frame + n])
            del mm
            reporter.report(1.0, sub="reading window from cache")
            reporter.finish_unit()
        else:
            reporter = ProgressReporter(total_units=2 if has_audio else 1,
                                        label="loading", event_cb=event_cb)
            args = ["-ss", f"{seek_time:.6f}", "-i", path]
            if vf:
                args += ["-vf", ",".join(vf)]
            args += ["-frames:v", str(frame_count), "-an", "-f", "rawvideo",
                     "-pix_fmt", "rgb24", "pipe:1"]
            raw = run_ffmpeg_pipe(args, duration=window_s,
                                  reporter=reporter, sub="decoding window")
            reporter.finish_unit()

            frame_size = w * h * 3
            n = len(raw) // frame_size
            if n == 0:
                raise ValueError(
                    f"Star Video Loader (Low RAM): 0 frames decoded - "
                    f"start_frame {start_frame} is probably beyond the end "
                    f"of the video.")
            arr = np.frombuffer(raw[:n * frame_size],
                                dtype=np.uint8).reshape(n, h, w, 3).copy()
            # windowed mode can only estimate the total (no full decode)
            total = int(round((info.get("duration") or 0) * eff_fps))

        images = torch.from_numpy(arr).float() / 255.0
        loaded = int(images.shape[0])

        # ---- audio (matching time window) -----------------------------
        audio_out = None
        if has_audio:
            if from_cache:
                wav_path = os.path.join(cache_dir, "audio.wav")
                if not os.path.exists(wav_path):
                    # cache was built with load_audio off - add the wav now
                    wav = run_ffmpeg_pipe(
                        ["-i", path, "-vn", "-f", "wav", "pipe:1"],
                        duration=info.get("duration"), reporter=reporter,
                        sub="extracting audio to cache")
                    with open(wav_path, "wb") as f:
                        f.write(wav)
                with open(wav_path, "rb") as f:
                    a, sr = _read_wav_bytes(f.read())
                s0 = int(round(seek_time * sr))
                s1 = s0 + int(round((loaded / eff_fps) * sr))
                if s0 < a.shape[1]:
                    audio_out = {"waveform": torch.from_numpy(
                        a[:, s0:s1]).unsqueeze(0),
                        "sample_rate": sr}
            else:
                wav_bytes = run_ffmpeg_pipe(
                    ["-ss", f"{seek_time:.6f}", "-t", f"{window_s:.6f}",
                     "-i", path, "-vn", "-f", "wav", "pipe:1"],
                    duration=window_s, reporter=reporter,
                    sub="extracting audio")
                a, sr = _read_wav_bytes(wav_bytes)
                audio_out = {"waveform": torch.from_numpy(a).unsqueeze(0),
                             "sample_rate": sr}
            reporter.finish_unit()

        reporter.finish_all(time.time() - t0)

        next_start = start_frame + loaded
        info_str = (
            f"{os.path.basename(path)} | {fmt_media_brief(info)}\n"
            f"window: frames {start_frame}..{start_frame + loaded - 1} "
            f"({loaded} loaded, ~{total} total @ {eff_fps:g} fps)\n"
            f"next_start_frame: {next_start}"
            + ("" if next_start < total else " (end of video reached)")
            + f"\nmode: {'disk cache' if from_cache else 'direct seek'}"
            + (f" (built {cache_dir})" if cache_to_disk and not from_cache
               else "")
            + (f" | audio: {sr} Hz" if audio_out else " | audio: none")
            + f"\npath: {path}"
        )
        print("[StarVideoLoaderLowRAM]\n" + info_str)

        return {"result": (images, audio_out, float(eff_fps), loaded,
                           int(total), int(next_start), info_str)}

    # ------------------------------------------------------------------

    @classmethod
    def IS_CHANGED(cls, video, start_frame=0, frame_count=81,
                   force_rate=0.0, select_every_kth=1, load_audio=True,
                   cache_to_disk=False, **kwargs):
        try:
            mtime = os.path.getmtime(
                folder_paths.get_annotated_filepath(video))
        except OSError:
            mtime = float("nan")
        return f"{mtime}-{start_frame}-{frame_count}-{force_rate}-" \
               f"{select_every_kth}-{load_audio}-{cache_to_disk}"

    @classmethod
    def VALIDATE_INPUTS(cls, video, **kwargs):
        if not folder_paths.exists_annotated_filepath(video):
            return f"Invalid video file: {video}"
        return True


NODE_CLASS_MAPPINGS = {
    "StarVideoLoaderLowRAM": StarVideoLoaderLowRAM,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarVideoLoaderLowRAM": "⭐ Star Video Loader (Low RAM)",
}
