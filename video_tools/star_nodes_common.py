"""
Shared helpers for the Star Nodes video pack (standalone, no VHS needed).

- ffmpeg discovery (imageio-ffmpeg if present, else system ffmpeg on PATH)
- media probing via `ffmpeg -i` parsing (no ffprobe required)
- ffmpeg runners with real-time progress:
    * ComfyUI UI progress bar (comfy.utils.ProgressBar, like KSampler)
    * console % bar
    * custom DOM progress bar in the node (via websocket events)
- output path building / audio helpers
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave

import numpy as np

try:
    from comfy.utils import ProgressBar
except Exception:  # very old ComfyUI or running outside ComfyUI
    ProgressBar = None

# ---------------------------------------------------------------------------
# ffmpeg discovery
# ---------------------------------------------------------------------------

_FFMPEG = None


def _bundled_ffmpeg():
    """
    Look for an ffmpeg binary shipped inside the node package itself:
        StarVideoCompressor/bin/win/ffmpeg.exe    (Windows)
        StarVideoCompressor/bin/linux/ffmpeg      (Linux)
        StarVideoCompressor/bin/mac/ffmpeg        (macOS)
    (also accepts a flat bin/ffmpeg[.exe]). Returns None if not present.
    """
    base = os.path.dirname(os.path.abspath(__file__))
    if sys.platform.startswith("win"):
        rels = ["bin/win/ffmpeg.exe", "bin/ffmpeg.exe"]
    elif sys.platform == "darwin":
        rels = ["bin/mac/ffmpeg", "bin/ffmpeg"]
    else:
        rels = ["bin/linux/ffmpeg", "bin/ffmpeg"]
    for rel in rels:
        p = os.path.join(base, rel)
        if os.path.exists(p):
            if not sys.platform.startswith("win"):
                try:
                    os.chmod(p, 0o755)
                except OSError:
                    pass
            return p
    return None


def get_ffmpeg():
    """
    ffmpeg resolution order:
      1. binary bundled in the node package (bin/ folder)
      2. imageio-ffmpeg (pip package, in requirements.txt)
      3. system ffmpeg found on PATH
    """
    global _FFMPEG
    if _FFMPEG is None:
        exe = _bundled_ffmpeg()
        if exe is None:
            try:
                import imageio_ffmpeg
                exe = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                exe = None
        if not exe or not os.path.exists(exe):
            exe = shutil.which("ffmpeg")
        if not exe:
            raise RuntimeError(
                "Star Nodes: ffmpeg was not found. Either 'pip install "
                "imageio-ffmpeg', place a binary into the node's bin/ "
                "folder, or install a system ffmpeg on PATH.")
        _FFMPEG = exe
    return _FFMPEG


_encoders_cache = None


def get_encoders():
    global _encoders_cache
    if _encoders_cache is None:
        try:
            proc = subprocess.run(
                [get_ffmpeg(), "-hide_banner", "-encoders"],
                capture_output=True, text=True, timeout=60)
            _encoders_cache = (proc.stdout or "") + (proc.stderr or "")
        except Exception:
            _encoders_cache = ""
    return _encoders_cache


# ---------------------------------------------------------------------------
# supported output formats / codec helpers
# ---------------------------------------------------------------------------

VIDEO_FORMATS = {
    "video/h264-mp4": {
        "extension": "mp4",
        "vcodec": "libx264",
        "max_crf": 51,
        "two_pass": True,
        "container": "mp4",
        "extra_args": ["-movflags", "+faststart", "-pix_fmt", "yuv420p"],
    },
    "video/h265-mp4": {
        "extension": "mp4",
        "vcodec": "libx265",
        "max_crf": 51,
        "two_pass": True,
        "container": "mp4",
        "extra_args": ["-movflags", "+faststart", "-pix_fmt", "yuv420p",
                       "-tag:v", "hvc1"],
    },
    "video/vp9-webm": {
        "extension": "webm",
        "vcodec": "libvpx-vp9",
        "max_crf": 63,
        "two_pass": True,
        "container": "webm",
        "extra_args": ["-pix_fmt", "yuv420p", "-row-mt", "1"],
    },
    "video/av1-mp4": {
        "extension": "mp4",
        "vcodec": "libsvtav1",
        "max_crf": 63,
        "two_pass": False,
        "container": "mp4",
        "extra_args": ["-movflags", "+faststart", "-pix_fmt", "yuv420p"],
    },
}

PRESETS = ["ultrafast", "superfast", "veryfast", "faster", "fast",
           "medium", "slow", "slower", "veryslow"]

# libsvtav1 uses a numeric preset scale (0-13, lower = slower/better)
SVTAV1_PRESET_MAP = {
    "ultrafast": "13", "superfast": "11", "veryfast": "10", "faster": "9",
    "fast": "8", "medium": "6", "slow": "4", "slower": "2", "veryslow": "1",
}

AUDIO_BITRATE = 128000  # bits/s, used for size math and encoding


def quality_to_crf(quality, max_crf):
    """Map the 0-100 quality slider onto the codec's CRF range."""
    q = max(0, min(100, int(quality)))
    return int(round((100 - q) / 100.0 * max_crf))


def audio_codec_args(fmt):
    if fmt["container"] == "webm":
        encoders = get_encoders()
        codec = "libopus" if "libopus" in encoders else "libvorbis"
        return ["-c:a", codec, "-b:a", f"{AUDIO_BITRATE // 1000}k"]
    return ["-c:a", "aac", "-b:a", f"{AUDIO_BITRATE // 1000}k"]


def compute_target_bitrate(target_size_mb, duration, with_audio):
    """Video bitrate (bits/s) needed to land near target_size_mb (MiB)."""
    margin = 0.95  # headroom for container overhead / VBR overshoot
    total_bits = target_size_mb * 1024 * 1024 * 8 * margin
    audio_bits = AUDIO_BITRATE * duration if with_audio else 0
    video_bps = int((total_bits - audio_bits) / duration)
    return max(video_bps, 32000)  # practical floor


# ---------------------------------------------------------------------------
# probing (ffmpeg -i writes its report to stderr; exit code is non-zero,
# which is expected and harmless)
# ---------------------------------------------------------------------------

def probe_media(path):
    """Return dict with duration/width/height/fps/codecs/size (best effort)."""
    info = {"duration": None, "width": None, "height": None, "fps": None,
            "vcodec": None, "acodec": None, "bitrate_kbps": None,
            "size_mb": None}
    try:
        info["size_mb"] = os.path.getsize(path) / (1024 * 1024)
    except OSError:
        pass
    try:
        proc = subprocess.run([get_ffmpeg(), "-hide_banner", "-i", path],
                              capture_output=True, text=True, timeout=60)
        text = proc.stderr or ""
    except Exception:
        return info

    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if m:
        info["duration"] = int(m.group(1)) * 3600 + int(m.group(2)) * 60 \
            + float(m.group(3))
    m = re.search(r"bitrate:\s*(\d+)\s*kb/s", text)
    if m:
        info["bitrate_kbps"] = int(m.group(1))
    m = re.search(r"Stream #.*?Video:\s*([a-zA-Z0-9_]+).*?"
                  r"(\d{2,6})x(\d{2,6})", text)
    if m:
        info["vcodec"] = m.group(1)
        info["width"] = int(m.group(2))
        info["height"] = int(m.group(3))
    m = re.search(r"Stream #.*?Video:.*?(\d+(?:\.\d+)?)\s*fps", text)
    if m:
        info["fps"] = float(m.group(1))
    m = re.search(r"Stream #.*?Audio:\s*([a-zA-Z0-9_]+)", text)
    if m:
        info["acodec"] = m.group(1)
    return info


def fmt_media_brief(info):
    """One-line summary like '640x480 @ 30.00 fps | 3.00 s | h264+aac'."""
    parts = []
    if info.get("width") and info.get("height"):
        parts.append(f"{info['width']}x{info['height']}")
    if info.get("fps"):
        parts.append(f"@ {info['fps']:.2f} fps")
    if info.get("duration"):
        parts.append(f"| {info['duration']:.2f} s")
    codecs = "+".join(c for c in (info.get("vcodec"), info.get("acodec")) if c)
    if codecs:
        parts.append(f"| {codecs}")
    if info.get("size_mb") is not None:
        parts.append(f"| {info['size_mb']:.2f} MiB")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# progress: UI ProgressBar + console % bar + DOM progress events
# ---------------------------------------------------------------------------

def _console_bar(pct, width=28):
    filled = int(round(width * min(pct, 100.0) / 100.0))
    if filled >= width:
        return "=" * width
    return "=" * filled + ">" + "-" * (width - filled - 1)


def make_event_cb(unique_id):
    """
    Build a callback that pushes progress to the node's DOM progress bar
    through the ComfyUI websocket. Returns None outside the server.
    """
    if unique_id is None:
        return None
    try:
        from server import PromptServer
        srv = PromptServer.instance
    except Exception:
        return None

    def cb(frac, text, sub):
        try:
            srv.send_sync("star_nodes.progress", {
                "node": str(unique_id),
                "value": round(min(max(frac, 0.0), 1.0), 4),
                "text": text,
                "sub": sub,
            })
        except Exception:
            pass
    return cb


class ProgressReporter:
    """Feeds ffmpeg progress to console, UI ProgressBar and DOM events."""

    def __init__(self, total_units, label="", event_cb=None):
        self.pbar = ProgressBar(total_units) if ProgressBar else None
        self.done_units = 0.0
        self.total_units = max(total_units, 1)
        self.label = label or "working"
        self.event_cb = event_cb
        self._last_print = 0.0

    def report(self, fraction, fps=None, speed=None, cur_s=None,
               total_s=None, sub=""):
        """fraction: 0..1 progress of the current unit (pass/file/step)."""
        fraction = min(max(fraction, 0.0), 1.0)
        abs_units = self.done_units + fraction
        if self.pbar is not None:
            try:
                self.pbar.update_absolute(min(abs_units, self.total_units))
            except Exception:
                pass
        now = time.time()
        if now - self._last_print >= 0.25 or fraction >= 1.0:
            self._last_print = now
            pct = fraction * 100.0
            line = (f"\r[StarNodes] {self.label} "
                    f"[{_console_bar(pct)}] {pct:5.1f}%")
            if cur_s is not None and total_s:
                line += f" | {cur_s:5.1f}s/{total_s:.1f}s"
            if fps and fps not in ("0", "0.00", "N/A"):
                line += f" | {fps} fps"
            if speed and speed != "N/A":
                line += f" | {speed}"
            sys.stdout.write(line + " " * 4)
            sys.stdout.flush()

            if self.event_cb is not None:
                overall = abs_units / self.total_units
                sub_parts = [p for p in (sub,
                                         f"{cur_s:.1f}/{total_s:.1f}s"
                                         if cur_s is not None and total_s
                                         else "",
                                         f"{fps} fps" if fps and
                                         fps not in ("0", "0.00", "N/A")
                                         else "",
                                         speed if speed and speed != "N/A"
                                         else "") if p]
                try:
                    self.event_cb(overall, f"{pct:.0f}%",
                                  " | ".join(sub_parts))
                except Exception:
                    pass

    def finish_unit(self):
        self.done_units += 1.0
        if self.pbar is not None:
            try:
                self.pbar.update_absolute(min(self.done_units,
                                              self.total_units))
            except Exception:
                pass

    def finish_all(self, elapsed):
        if self.pbar is not None:
            try:
                self.pbar.update_absolute(self.total_units)
            except Exception:
                pass
        sys.stdout.write(f"\r[StarNodes] {self.label} "
                         f"[{_console_bar(100)}] 100.0% | done in "
                         f"{elapsed:.1f}s" + " " * 20 + "\n")
        sys.stdout.flush()
        if self.event_cb is not None:
            try:
                self.event_cb(1.0, "100%", f"done in {elapsed:.1f}s")
            except Exception:
                pass


# ---------------------------------------------------------------------------
# ffmpeg runners
# ---------------------------------------------------------------------------

def run_ffmpeg(args, duration=None, input_bytes=None, reporter=None,
               pass_label=""):
    """
    Run ffmpeg with inputs/outputs as files (or raw frames on stdin).
    Streams `-progress` to the reporter when duration is known.
    """
    cmd = [get_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y"]
    use_progress = bool(duration) and duration > 0 and reporter is not None
    if use_progress:
        cmd += ["-nostats", "-progress", "pipe:1"]
    cmd += args

    err_file = tempfile.TemporaryFile()
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE if input_bytes is not None else None,
        stdout=subprocess.PIPE if use_progress else subprocess.DEVNULL,
        stderr=err_file)

    if input_bytes is not None:
        def _feed():
            try:
                proc.stdin.write(input_bytes)
                proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        threading.Thread(target=_feed, daemon=True).start()

    fps = speed = None
    if use_progress:
        while True:
            raw = proc.stdout.readline()
            if not raw:
                break
            key, _, val = raw.decode("utf-8", "ignore").strip().partition("=")
            if key in ("out_time_ms", "out_time_us"):
                try:
                    cur_s = int(val) / 1_000_000.0
                except ValueError:
                    continue
                reporter.report(min(cur_s / duration, 0.999),
                                fps=fps, speed=speed, cur_s=cur_s,
                                total_s=duration, sub=pass_label)
            elif key == "fps":
                fps = val
            elif key == "speed":
                speed = val
            elif key == "progress" and val == "end":
                reporter.report(1.0, fps=fps, speed=speed, cur_s=duration,
                                total_s=duration, sub=pass_label)
    proc.wait()

    if proc.returncode != 0:
        err_file.seek(0)
        err = err_file.read().decode("utf-8", "ignore")
        err_file.close()
        sys.stdout.write("\n")
        raise RuntimeError(
            "Star Nodes: ffmpeg failed"
            + (f" ({pass_label})" if pass_label else "")
            + ".\nCommand: " + " ".join(cmd) + "\n" + err[-2000:])
    err_file.close()


def run_ffmpeg_pipe(args, duration=None, reporter=None, sub=""):
    """
    Run ffmpeg whose OUTPUT is stdout (pipe:1) and return the bytes.
    Progress is parsed from stderr (`-progress pipe:2`) in a reader thread.
    """
    cmd = [get_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y"]
    use_progress = bool(duration) and duration > 0 and reporter is not None
    if use_progress:
        cmd += ["-nostats", "-progress", "pipe:2"]
    cmd += args

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    err_lines = []

    def _read_stderr():
        for raw in proc.stderr:
            line = raw.decode("utf-8", "ignore").strip()
            err_lines.append(line)
            if use_progress:
                key, _, val = line.partition("=")
                if key in ("out_time_ms", "out_time_us"):
                    try:
                        cur_s = int(val) / 1_000_000.0
                    except ValueError:
                        continue
                    reporter.report(min(cur_s / duration, 0.999),
                                    cur_s=cur_s, total_s=duration, sub=sub)
                elif key == "progress" and val == "end":
                    reporter.report(1.0, cur_s=duration, total_s=duration,
                                    sub=sub)

    t = threading.Thread(target=_read_stderr, daemon=True)
    t.start()
    data = proc.stdout.read()
    proc.wait()
    t.join(timeout=5)

    if proc.returncode != 0:
        sys.stdout.write("\n")
        tail = [l for l in err_lines if "=" not in l][-30:]
        raise RuntimeError("Star Nodes: ffmpeg failed.\nCommand: "
                           + " ".join(cmd) + "\n" + "\n".join(tail))
    return data


# ---------------------------------------------------------------------------
# paths / audio
# ---------------------------------------------------------------------------

def build_output_path(base_dir, prefix, extension, index=0):
    """
    Turn a user prefix (may contain subfolders, e.g. 'discord/my_clip')
    into a collision-free absolute path inside base_dir.
    Returns (absolute_path, subfolder_relative_to_base_dir).
    """
    prefix = (prefix or "").strip().replace("\\", "/")
    parts = [p for p in prefix.split("/") if p not in ("", ".", "..")]
    base = parts[-1] if parts else "StarVideo"
    subfolder = "/".join(parts[:-1]) if len(parts) > 1 else ""

    root = os.path.abspath(base_dir)
    out_dir = os.path.abspath(os.path.join(root, subfolder)) if subfolder \
        else root
    if os.path.commonpath([root, out_dir]) != root:
        raise ValueError(
            "Star Nodes: filename_prefix must stay inside the ComfyUI "
            "output/temp directory.")
    os.makedirs(out_dir, exist_ok=True)

    if index > 0:
        base = f"{base}_{index + 1:05d}"

    candidate = os.path.join(out_dir, f"{base}.{extension}")
    n = 1
    while os.path.exists(candidate):
        candidate = os.path.join(out_dir, f"{base}_{n:05d}.{extension}")
        n += 1
    return candidate, subfolder


def audio_to_temp_wav(audio):
    """
    Convert a ComfyUI core AUDIO dict ({'waveform': (B,C,N) tensor,
    'sample_rate': int}) to a temporary 16-bit PCM wav file. Returns path.
    """
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    arr = waveform[0].cpu().numpy() if hasattr(waveform, "cpu") \
        else np.asarray(waveform)[0]
    arr = np.asarray(arr, dtype=np.float32).T  # (samples, channels)
    pcm = (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16)

    fd, path = tempfile.mkstemp(suffix="_star_audio.wav")
    os.close(fd)
    with wave.open(path, "wb") as w:
        w.setnchannels(pcm.shape[1] if pcm.ndim > 1 else 1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return path
