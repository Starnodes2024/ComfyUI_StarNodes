"""
Star Audio Loader - standalone audio loading node.

Loads an audio file from the ComfyUI input folder (upload supported) and
outputs:
  - audio       : core AUDIO dict ({'waveform': (1,C,N) tensor, 'sample_rate'})
  - seconds     : INT, duration of the returned (cut) audio in whole seconds
  - seconds_str : STRING, duration of the returned audio as a decimal string
  - info        : STRING report

The frontend adds a "Load Audio" button: it probes the selected file without
running the workflow (/starnodes/audio_loader/info), shows an inline <audio>
preview and sets the start_time/end_time slider ranges - the cut is applied
when the workflow runs.
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

AUDIO_EXTENSIONS = (".wav", ".wave", ".flac", ".mp3", ".m4a", ".aac",
                    ".ogg", ".opus", ".wma", ".aiff", ".aif", ".mp2",
                    ".ac3", ".amr")


def list_input_audios():
    input_dir = folder_paths.get_input_directory()
    files = []
    for root, _, names in os.walk(input_dir):
        for name in names:
            if name.lower().endswith(AUDIO_EXTENSIONS):
                rel = os.path.relpath(os.path.join(root, name), input_dir)
                files.append(rel.replace(os.sep, "/"))
    return sorted(files, key=str.lower)


class StarAudioLoader:
    """Load an audio file: AUDIO dict, duration in seconds and an info report."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": (list_input_audios(), {"audio_upload": True}),
                "start_time": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1000000.0,
                    "step": 0.01,
                    "tooltip": "Start of the cut in seconds. Click the Load "
                               "button to probe the file and preview the cut "
                               "point."}),
                "end_time": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1000000.0,
                    "step": 0.01,
                    "tooltip": "End of the cut in seconds (0 = to the end). "
                               "Click the Load button to probe the file and "
                               "preview the cut point."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("AUDIO", "INT", "STRING", "STRING")
    RETURN_NAMES = ("audio", "seconds", "seconds_str", "info")
    FUNCTION = "load"
    CATEGORY = "⭐StarNodes/Video"
    OUTPUT_NODE = True
    DESCRIPTION = ("Load an audio file (wav/flac/mp3/...) from the ComfyUI "
                   "input folder: AUDIO output plus the cut duration in "
                   "seconds (int and string) and an info report.")

    # ------------------------------------------------------------------

    def load(self, audio, start_time=0.0, end_time=0.0, unique_id=None):
        path = folder_paths.get_annotated_filepath(audio)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Star Audio Loader: audio not found: {audio}")

        info = probe_media(path)
        if not info.get("acodec") and not info.get("duration"):
            raise RuntimeError(
                "Star Audio Loader: could not probe an audio stream - is "
                "this really an audio file?")
        duration = info.get("duration") or 0.0

        reporter = ProgressReporter(total_units=1, label="loading",
                                    event_cb=make_event_cb(unique_id))

        start = max(0.0, float(start_time))
        end = float(end_time) if end_time and end_time > 0 else duration
        if duration and end > duration:
            end = duration
        if end and start >= end:
            start = max(0.0, end - 0.01)

        cut = start > 0.0 or (end > 0.0 and end < duration)
        if cut:
            aargs = ["-ss", f"{start:.6f}", "-i", path, "-vn",
                     "-t", f"{end - start:.6f}", "-f", "wav", "pipe:1"]
        else:
            aargs = ["-i", path, "-vn", "-f", "wav", "pipe:1"]
        wav_bytes = run_ffmpeg_pipe(
            aargs, duration=(end - start) or duration,
            reporter=reporter, sub="decoding audio")

        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            pcm = wf.readframes(wf.getnframes())
        a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        a = a.reshape(-1, channels).T.copy()  # (C, N)
        audio_out = {"waveform": torch.from_numpy(a).unsqueeze(0),
                     "sample_rate": sample_rate}
        reporter.finish_unit()
        reporter.finish_all(0.0)

        kept = a.shape[1] / float(sample_rate) if sample_rate else 0.0
        seconds_int = int(round(kept))
        seconds_str = f"{kept:.3f}"

        info_str = (
            f"{os.path.basename(path)} | {fmt_media_brief(info)}\n"
            f"cut: {start:.3f}s - {end:.3f}s | kept {seconds_str}s"
            f" | {channels} ch @ {sample_rate} Hz"
            + (f"\nfull duration: {duration:.3f}s" if duration else "")
            + f"\npath: {path}"
        )
        print("[StarAudioLoader]\n" + info_str)

        return {"result": (audio_out, seconds_int, seconds_str, info_str)}

    # ------------------------------------------------------------------

    @classmethod
    def IS_CHANGED(cls, audio, start_time=0.0, end_time=0.0, **kwargs):
        try:
            mtime = os.path.getmtime(
                folder_paths.get_annotated_filepath(audio))
        except OSError:
            mtime = float("nan")
        return f"{mtime}-{start_time}-{end_time}"

    @classmethod
    def VALIDATE_INPUTS(cls, audio, **kwargs):
        if not folder_paths.exists_annotated_filepath(audio):
            return f"Invalid audio file: {audio}"
        return True


NODE_CLASS_MAPPINGS = {
    "StarAudioLoader": StarAudioLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarAudioLoader": "⭐ Star Audio Loader",
}


# API endpoint for the node's Load button: probe the audio without running
# the workflow so the frontend can set the start/end slider ranges and show
# the preview.
try:
    from server import PromptServer
    from aiohttp import web

    @PromptServer.instance.routes.post("/starnodes/audio_loader/info")
    async def star_audio_loader_info(request):
        try:
            data = await request.json()
            audio = data.get("audio", "")
            path = folder_paths.get_annotated_filepath(audio)
            if not os.path.exists(path):
                return web.json_response(
                    {"status": "error",
                     "message": f"audio not found: {audio}"},
                    status=404)
            info = probe_media(path)
            duration = info.get("duration") or 0.0
            if not duration:
                return web.json_response(
                    {"status": "error",
                     "message": "could not determine the audio duration"})
            return web.json_response({
                "status": "ok",
                "duration": duration,
                "acodec": info.get("acodec"),
                "bitrate_kbps": info.get("bitrate_kbps"),
                "size_mb": info.get("size_mb"),
                "brief": fmt_media_brief(info),
            })
        except Exception as e:
            return web.json_response({"status": "error", "message": str(e)},
                                     status=500)
except Exception as e:
    print(f"[StarAudioLoader] could not register info endpoint: {e}")
