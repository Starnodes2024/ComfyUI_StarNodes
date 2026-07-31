"""
Star Video Compressor - a standalone ComfyUI custom node (no VHS required).

Compresses videos to a chosen quality (CRF) or to a desired target file
size (bitrate-driven, two-pass where supported). Accepts:
  - STAR_FILENAMES from the bundled "Star Video Loader" node
  - a direct file path via the video_path widget
  - an IMAGE batch (encode + compress in one step, uses frame_rate)

Mode priority: if target_size_mb > 0 it ALWAYS wins and the quality
slider is ignored; quality (CRF) is only used when target_size_mb is 0.
"""

import os
import time

import numpy as np

import folder_paths

from .star_nodes_common import (
    AUDIO_BITRATE,  # noqa: F401  (re-exported for docs/tests)
    ProgressReporter,
    VIDEO_FORMATS,
    PRESETS,
    SVTAV1_PRESET_MAP,
    audio_codec_args,
    audio_to_temp_wav,
    build_output_path,
    compute_target_bitrate,
    fmt_media_brief,
    get_encoders,
    make_event_cb,
    probe_media,
    quality_to_crf,
    run_ffmpeg,
)


class StarVideoCompressor:
    """Compress videos to a friendly file size for Discord & co."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "quality": ("INT", {
                    "default": 60, "min": 0, "max": 100, "step": 1,
                    "display": "slider",
                    "tooltip": "Compression quality (0 = smallest file, 100 "
                               "= best quality). IGNORED when target_size_mb "
                               "is greater than 0."}),
                "format": (list(VIDEO_FORMATS.keys()),
                           {"default": "video/h264-mp4"}),
                "preset": (PRESETS, {"default": "medium", "tooltip":
                           "Encoder speed vs efficiency. Slower = smaller "
                           "file at the same quality, but takes longer."}),
                "filename_prefix": ("STRING", {
                    "default": "StarVideo",
                    "tooltip": "Output filename. May contain subfolders, "
                               "e.g. 'discord/my_clip' - missing folders are "
                               "created inside the ComfyUI output folder."}),
                "target_size_mb": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1000000.0,
                    "step": 0.1,
                    "tooltip": "Desired output size in MiB. 0 = off (quality "
                               "slider is used). When > 0 this WINS over the "
                               "quality slider: the bitrate is calculated "
                               "from the duration (two-pass where "
                               "supported)."}),
                "video_path": ("STRING", {
                    "default": "",
                    "tooltip": "Optional: direct path to a video file "
                               "(absolute, or relative to the ComfyUI input "
                               "folder). Used when no 'video' input is "
                               "connected. Leave empty to use the 'images' "
                               "input."}),
                "frame_rate": ("FLOAT", {
                    "default": 30.0, "min": 1.0, "max": 240.0, "step": 0.01,
                    "tooltip": "Only used when compressing an IMAGE batch. "
                               "Tip: right-click the node to convert this "
                               "widget into an input and connect the "
                               "loader's fps output."}),
                "save_audio": ("BOOLEAN", {"default": True}),
                "save_output": ("BOOLEAN", {"default": True, "tooltip":
                                "True = ComfyUI output folder, False = "
                                "temp folder."}),
                "drop_first_frames": ("INT", {
                    "default": 0, "min": 0, "max": 1000, "step": 1,
                    "tooltip": "Skip this many frames from the start of "
                               "the output (e.g. 10 = the output starts at "
                               "frame 11). 0 = off. Any connected/embedded "
                               "audio is shifted by the same amount to "
                               "stay in sync."}),
                "drop_last_frames": ("INT", {
                    "default": 0, "min": 0, "max": 1000, "step": 1,
                    "tooltip": "Cut this many frames from the end of the "
                               "output (e.g. 10 = the last 10 frames are "
                               "removed). 0 = off. Works together with "
                               "drop_first_frames. Any connected/embedded "
                               "audio is trimmed to match so it stays in "
                               "sync."}),
            },
            "optional": {
                "video": ("STAR_FILENAMES", {"tooltip":
                          "Video file(s) from the 'Star Video Loader' "
                          "node."}),
                "images": ("IMAGE", {"tooltip":
                           "Alternative: encode + compress an image batch "
                           "directly (uses frame_rate)."}),
                "audio": ("AUDIO", {"tooltip":
                          "Optional external audio track (ComfyUI core "
                          "audio, e.g. from the 'LoadAudio' node or the "
                          "Star Video Loader)."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STAR_FILENAMES", "STRING")
    RETURN_NAMES = ("Filenames", "info")
    FUNCTION = "compress"
    CATEGORY = "⭐StarNodes/Video"
    OUTPUT_NODE = True
    DESCRIPTION = ("Compress videos to a chosen quality or a desired file "
                   "size - handy for Discord's 10 MB limit. Standalone, no "
                   "Video Helper Suite required. "
                   "See web/docs/StarVideoCompressor.md")

    # ------------------------------------------------------------------

    def compress(self, quality, format, preset, filename_prefix,
                 target_size_mb, video_path, frame_rate, save_audio,
                 save_output, drop_first_frames=0, drop_last_frames=0,
                 video=None, images=None, audio=None, unique_id=None):

        drop_first_frames = max(0, min(1000, int(drop_first_frames)))
        drop_last_frames = max(0, min(1000, int(drop_last_frames)))

        if format not in VIDEO_FORMATS:
            raise ValueError(f"Star Video Compressor: unknown format "
                             f"'{format}'.")
        fmt = VIDEO_FORMATS[format]

        encoders = get_encoders()
        if fmt["vcodec"] not in encoders:
            raise RuntimeError(
                f"Star Video Compressor: your ffmpeg build does not include "
                f"the encoder '{fmt['vcodec']}'. Pick another format or "
                f"install a full ffmpeg build.")

        base_dir = folder_paths.get_output_directory() if save_output \
            else folder_paths.get_temp_directory()
        out_type = "output" if save_output else "temp"

        audio_file = None
        if audio is not None and save_audio:
            audio_file = audio_to_temp_wav(audio)

        # ---- collect the jobs: (kind, source) -------------------------
        jobs = []
        if video is not None:
            files = video[1] if isinstance(video, (list, tuple)) \
                and len(video) == 2 else video
            if not files:
                raise ValueError("Star Video Compressor: the 'video' input "
                                 "contains no files.")
            jobs = [("file", self._resolve_path(f)) for f in files]
        elif video_path and video_path.strip():
            jobs = [("file", self._resolve_path(video_path))]
        elif images is not None:
            jobs = [("images", self._to_uint8_frames(images))]
        else:
            raise ValueError(
                "Star Video Compressor: provide a 'video' input (Star Video "
                "Loader), fill 'video_path', or connect 'images'.")

        use_target = bool(target_size_mb) and target_size_mb > 0
        passes = 2 if (use_target and fmt["two_pass"]) else 1
        reporter = ProgressReporter(total_units=passes * len(jobs),
                                    label="compressing",
                                    event_cb=make_event_cb(unique_id))

        output_files, previews, info_lines = [], [], []
        t_start = time.time()

        for idx, (kind, src) in enumerate(jobs):
            out_path, subfolder = build_output_path(
                base_dir, filename_prefix, fmt["extension"], idx)
            t0 = time.time()

            if kind == "file":
                in_info = probe_media(src)
                duration = in_info["duration"]
                in_desc = f"{os.path.basename(src)} | " \
                          f"{fmt_media_brief(in_info)}"
                drop_fps = in_info["fps"] or frame_rate
                drop_time = drop_first_frames / max(drop_fps, 0.001) \
                    if drop_first_frames > 0 else 0.0
                drop_time_end = drop_last_frames / max(drop_fps, 0.001) \
                    if drop_last_frames > 0 else 0.0
                if drop_last_frames > 0 and not duration:
                    raise ValueError(
                        "Star Video Compressor: drop_last_frames needs a "
                        "known input duration, but it could not be "
                        f"determined for '{os.path.basename(src)}'. Set "
                        "drop_last_frames to 0 for this file.")
                if duration:
                    duration = max(0.001, duration - drop_time - drop_time_end)
                trim_in_args = (["-ss", f"{drop_time:.6f}"]
                                 if drop_time > 0 else []) + \
                                (["-t", f"{duration:.6f}"]
                                 if drop_time_end > 0 else [])
                source_args = trim_in_args + ["-i", src]
                map_args, filter_args, payload = [], [], None
                if not save_audio:
                    a_args = ["-an"]
                elif audio_file:
                    source_args += trim_in_args + ["-i", audio_file]
                    map_args = ["-map", "0:v:0", "-map", "1:a:0", "-shortest"]
                    a_args = audio_codec_args(fmt)
                elif in_info["acodec"]:
                    a_args = audio_codec_args(fmt)
                else:
                    a_args = []
            else:  # images batch
                frames = src
                total_frames = frames.shape[0]
                if drop_first_frames + drop_last_frames >= total_frames:
                    raise ValueError(
                        "Star Video Compressor: drop_first_frames + "
                        f"drop_last_frames ({drop_first_frames + drop_last_frames}) "
                        "is >= the number of frames in the IMAGE batch "
                        f"({total_frames}).")
                end_idx = total_frames - drop_last_frames \
                    if drop_last_frames > 0 else total_frames
                frames = frames[drop_first_frames:end_idx]
                drop_time = drop_first_frames / max(frame_rate, 0.001)
                n, h, w = frames.shape[0], frames.shape[1], frames.shape[2]
                duration = n / max(frame_rate, 0.001)
                in_desc = (f"IMAGE batch | {n} frames @ {frame_rate:g} fps "
                           f"| {w}x{h}")
                source_args = ["-f", "rawvideo", "-pix_fmt", "rgb24",
                               "-s", f"{w}x{h}", "-r", str(frame_rate),
                               "-i", "-"]
                payload = frames.tobytes()
                map_args = []
                filter_args = ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"] \
                    if (w % 2 or h % 2) else []
                if not save_audio:
                    a_args = ["-an"]
                elif audio_file:
                    source_args += (["-ss", f"{drop_time:.6f}"]
                                     if drop_time > 0 else []) + \
                                     ["-i", audio_file]
                    map_args = ["-map", "0:v:0", "-map", "1:a:0", "-shortest"]
                    a_args = audio_codec_args(fmt)
                else:
                    a_args = []

            self._encode(fmt, quality, preset, target_size_mb, duration,
                         a_args, source_args, out_path, map_args, filter_args,
                         payload, reporter)

            elapsed = time.time() - t0
            out_info = probe_media(out_path)
            output_files.append(out_path)
            previews.append({
                "filename": os.path.basename(out_path),
                "subfolder": subfolder,
                "type": out_type,
                "format": format,
                "fullpath": out_path,
            })
            info_lines.append(self._format_info(
                idx, out_path, in_desc, out_info, fmt, quality, preset,
                target_size_mb, passes, save_audio, elapsed,
                in_size_mb=(probe_media(src)["size_mb"]
                            if kind == "file" else None)))

        if audio_file and os.path.exists(audio_file):
            try:
                os.remove(audio_file)
            except OSError:
                pass

        reporter.finish_all(time.time() - t_start)
        info = self._header(format, preset, quality, target_size_mb,
                            save_audio) + "\n\n" + "\n\n".join(info_lines)
        print("[StarVideoCompressor]\n" + info)
        return {"ui": {"star_videos": previews},
                "result": ((save_output, output_files), info)}

    # ------------------------------------------------------------------

    @staticmethod
    def _header(format, preset, quality, target_size_mb, save_audio):
        if target_size_mb and target_size_mb > 0:
            mode = (f"target size {target_size_mb} MiB "
                    f"(quality slider ignored)")
        else:
            mode = f"quality {quality}"
        return (f"settings: format={format} | preset={preset} | {mode} "
                f"| audio={'on' if save_audio else 'off'}")

    @staticmethod
    def _format_info(idx, out_path, in_desc, out_info, fmt, quality, preset,
                     target_size_mb, passes, save_audio, elapsed,
                     in_size_mb=None):
        lines = [f"[{idx + 1}] {os.path.basename(out_path)}",
                 f"    in : {in_desc}",
                 f"    out: {fmt_media_brief(out_info)}"]
        if out_info.get("bitrate_kbps"):
            lines[-1] += f" | overall {out_info['bitrate_kbps']} kb/s"
        if target_size_mb and target_size_mb > 0:
            dev = (out_info["size_mb"] - target_size_mb) \
                / target_size_mb * 100 if out_info.get("size_mb") else 0.0
            lines.append(f"    mode: target size | {passes}-pass | "
                         f"preset {preset} | result "
                         f"{out_info['size_mb']:.2f} MiB "
                         f"({dev:+.1f}% vs target)")
        else:
            lines.append(f"    mode: quality {quality} "
                         f"(CRF {quality_to_crf(quality, fmt['max_crf'])}) "
                         f"| preset {preset}")
        if in_size_mb is not None and out_info.get("size_mb") is not None \
                and in_size_mb > 0:
            ratio = out_info["size_mb"] / in_size_mb
            change = (ratio - 1.0) * 100
            verb = "smaller" if ratio < 1 else "larger"
            lines.append(f"    size: {in_size_mb:.2f} -> "
                         f"{out_info['size_mb']:.2f} MiB "
                         f"({change:+.1f}%, {abs(change):.0f}% {verb})")
        lines.append(f"    time: {elapsed:.1f} s | saved to: {out_path}")
        return "\n".join(lines)

    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_path(path):
        path = path.strip().strip('"') if isinstance(path, str) else path
        if os.path.isabs(path) and os.path.exists(path):
            return path
        try:
            resolved = folder_paths.get_annotated_filepath(path)
            if os.path.exists(resolved):
                return resolved
        except Exception:
            pass
        candidate = os.path.join(folder_paths.get_input_directory(), path)
        if os.path.exists(candidate):
            return candidate
        raise FileNotFoundError(
            f"Star Video Compressor: input video not found: {path}")

    @staticmethod
    def _to_uint8_frames(images):
        arr = images.cpu().numpy() if hasattr(images, "cpu") \
            else np.asarray(images)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[None, ...]
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.shape[-1] != 3:
            raise ValueError("Star Video Compressor: IMAGE batch must have "
                             "3 (RGB) or 4 (RGBA) channels.")
        return np.clip(arr * 255.0, 0, 255).astype(np.uint8)

    def _encode(self, fmt, quality, preset, target_size_mb, duration,
                a_args, source_args, out_path, map_args, filter_args,
                input_bytes, reporter):
        extra = fmt["extra_args"]
        vcodec = fmt["vcodec"]
        use_target = bool(target_size_mb) and target_size_mb > 0

        if use_target and (not duration or duration <= 0):
            raise RuntimeError(
                "Star Video Compressor: could not determine the input "
                "duration, so a target file size cannot be calculated. "
                "Set target_size_mb to 0 and use the quality slider.")

        if use_target:
            v_bps = compute_target_bitrate(
                target_size_mb, duration,
                bool(a_args) and "-an" not in a_args)
            print(f"[StarVideoCompressor] target {target_size_mb} MiB, "
                  f"duration {duration:.2f}s -> video bitrate "
                  f"{v_bps / 1000:.0f} kbps")
            v_args = ["-c:v", vcodec, "-b:v", str(v_bps)]
            v_args += (["-preset", SVTAV1_PRESET_MAP.get(preset, "6")]
                       if vcodec == "libsvtav1" else ["-preset", preset])

            if fmt["two_pass"]:
                import tempfile
                with tempfile.TemporaryDirectory() as td:
                    passlog = os.path.join(td, "ffpass")
                    # pass 1 must see the SAME pixel format / filters as
                    # pass 2, otherwise x264 aborts with e.g. "different
                    # bitdepth setting than first pass (8 vs 10)" when the
                    # source is a 10-bit video.
                    run_ffmpeg(source_args + v_args + filter_args
                               + ["-pix_fmt", "yuv420p"]
                               + ["-pass", "1", "-passlogfile", passlog,
                                  "-an", "-f", fmt["container"], os.devnull],
                               duration=duration, input_bytes=input_bytes,
                               reporter=reporter, pass_label="pass 1/2")
                    reporter.finish_unit()
                    run_ffmpeg(source_args + v_args
                               + ["-pass", "2", "-passlogfile", passlog]
                               + map_args + filter_args + a_args + extra
                               + [out_path],
                               duration=duration, input_bytes=input_bytes,
                               reporter=reporter, pass_label="pass 2/2")
                    reporter.finish_unit()
            else:  # libsvtav1: single-pass VBR
                run_ffmpeg(source_args + v_args + map_args + filter_args
                           + a_args + extra + [out_path],
                           duration=duration, input_bytes=input_bytes,
                           reporter=reporter, pass_label="av1 vbr")
                reporter.finish_unit()
        else:
            crf = quality_to_crf(quality, fmt["max_crf"])
            if vcodec == "libsvtav1":
                v_args = ["-c:v", vcodec, "-preset",
                          SVTAV1_PRESET_MAP.get(preset, "6"),
                          "-crf", str(crf)]
            else:
                v_args = ["-c:v", vcodec, "-preset", preset,
                          "-crf", str(crf)]
            if vcodec == "libvpx-vp9":
                v_args += ["-b:v", "0"]  # required for VP9 CRF mode
            print(f"[StarVideoCompressor] quality {quality} -> CRF {crf} "
                  f"({vcodec})")
            run_ffmpeg(source_args + v_args + map_args + filter_args
                       + a_args + extra + [out_path],
                       duration=duration, input_bytes=input_bytes,
                       reporter=reporter, pass_label="crf")
            reporter.finish_unit()


NODE_CLASS_MAPPINGS = {
    "StarVideoCompressor": StarVideoCompressor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarVideoCompressor": "⭐ Star Video Compressor",
}
