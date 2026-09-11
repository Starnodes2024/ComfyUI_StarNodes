import re

import numpy as np
import torch
import torch.nn.functional as F

from ..video_tools.star_nodes_common import probe_media, run_ffmpeg_pipe

_IMAGE_RE = re.compile(r"^image_([1-9][0-9]*)$")
_AUDIO_RE = re.compile(r"^audio_([1-9][0-9]*)$")
_VIDEO_RE = re.compile(r"^video_([1-9][0-9]*)$")


def _decode_video_frames(video_ref):
    """Decode a STAR_FILENAMES reference (from Star Video Loader / Star
    Video Compressor) into an IMAGE batch via ffmpeg."""
    if video_ref is None:
        return None
    paths = video_ref[1] if isinstance(video_ref, (list, tuple)) \
        and len(video_ref) == 2 else video_ref
    if not paths:
        return None

    frames_list = []
    for path in paths:
        info = probe_media(path)
        w, h = info.get("width"), info.get("height")
        if not w or not h:
            continue
        raw = run_ffmpeg_pipe(["-i", path, "-an", "-f", "rawvideo",
                               "-pix_fmt", "rgb24", "pipe:1"])
        frame_size = w * h * 3
        n = len(raw) // frame_size
        if n == 0:
            continue
        arr = np.frombuffer(raw[:n * frame_size],
                            dtype=np.uint8).reshape(n, h, w, 3).copy()
        frames_list.append(torch.from_numpy(arr).float() / 255.0)

    if not frames_list:
        return None
    return torch.cat(frames_list, dim=0) if len(frames_list) > 1 else frames_list[0]


class StarVideoJoiner:
    BGCOLOR = "#3d124d"
    COLOR = "#19124d"
    CATEGORY = '⭐StarNodes/Video'
    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "join_video"
    DESCRIPTION = ("Combine multiple image batches, videos and audio inputs into a "
                   "single output. Image, video and audio slots grow automatically as "
                   "you connect them (up to 20 each). Video inputs are decoded to image "
                   "frames. The first connected image/video sets the reference "
                   "dimensions; everything else is resized to match.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "image_1": ("IMAGE", {"tooltip": "Primary image batch — sets the reference dimensions."}),
                "video_1": ("STAR_FILENAMES", {"tooltip": "Primary video input from Star Video Loader — decoded to image frames (optional)."}),
                "audio_1": ("AUDIO", {"tooltip": "Primary audio input (optional)."}),
            },
        }

    def join_video(self, **kwargs):
        images_list = []
        audio_list = []

        image_indices = sorted(
            int(m.group(1)) for key in kwargs
            if (m := _IMAGE_RE.match(key)) and kwargs[key] is not None
        )
        video_indices = sorted(
            int(m.group(1)) for key in kwargs
            if (m := _VIDEO_RE.match(key)) and kwargs[key] is not None
        )
        audio_indices = sorted(
            int(m.group(1)) for key in kwargs
            if (m := _AUDIO_RE.match(key)) and kwargs[key] is not None
        )

        combined_sources = [kwargs[f"image_{idx}"] for idx in image_indices]
        combined_sources += [_decode_video_frames(kwargs[f"video_{idx}"]) for idx in video_indices]
        combined_sources = [img for img in combined_sources if img is not None]

        reference_height = None
        reference_width = None

        for img in combined_sources:
            if reference_height is None:
                reference_height = img.shape[1]
                reference_width = img.shape[2]
                images_list.append(img)
                continue
            if img.shape[1] != reference_height or img.shape[2] != reference_width:
                img = F.interpolate(
                    img.permute(0, 3, 1, 2),
                    size=(reference_height, reference_width),
                    mode='bilinear',
                    align_corners=False
                ).permute(0, 2, 3, 1)
            images_list.append(img)

        combined_images = torch.cat(images_list, dim=0) if images_list else None

        for idx in audio_indices:
            audio_list.append(kwargs[f"audio_{idx}"])
        
        if audio_list:
            if len(audio_list) == 1:
                combined_audio = audio_list[0]
            else:
                waveforms = []
                sample_rate = None
                
                for audio_dict in audio_list:
                    if isinstance(audio_dict, dict):
                        waveform = audio_dict.get('waveform')
                        if sample_rate is None:
                            sample_rate = audio_dict.get('sample_rate', 44100)
                        
                        if waveform is not None:
                            waveforms.append(waveform)
                    else:
                        waveforms.append(audio_dict)
                
                if waveforms:
                    combined_waveform = torch.cat(waveforms, dim=-1)
                    
                    if isinstance(audio_list[0], dict):
                        combined_audio = {
                            'waveform': combined_waveform,
                            'sample_rate': sample_rate
                        }
                    else:
                        combined_audio = combined_waveform
                else:
                    combined_audio = None
        else:
            combined_audio = None
        
        return (combined_images, combined_audio)

NODE_CLASS_MAPPINGS = {
    "StarVideoJoiner": StarVideoJoiner
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarVideoJoiner": "⭐ Star Video Joiner"
}
