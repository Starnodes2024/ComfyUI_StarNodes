import re

import torch

MAX_INPUTS = 12
_AUDIO_RE = re.compile(r"^audio_([1-9][0-9]*)$")
_VOLUME_RE = re.compile(r"^volume_([1-9][0-9]*)$")
_PAUSE_RE = re.compile(r"^start_pause_([1-9][0-9]*)$")


class _DynamicMixerInputs(dict):
    """Autogrow optional inputs for StarSoundMixer.

    The frontend probes whether the next slot (e.g. ``audio_2``) exists;
    ``__contains__`` returns True for any ``audio_N`` / ``volume_N`` within
    the MAX_INPUTS limit, and ``__getitem__`` returns the matching type.
    ``start_pause_N`` is exposed for N >= 2 (the first audio is the master
    track and always starts at 0).
    """

    def __contains__(self, key):
        m = _AUDIO_RE.match(key)
        if m and 1 <= int(m.group(1)) <= MAX_INPUTS:
            return True
        m = _VOLUME_RE.match(key)
        if m and 1 <= int(m.group(1)) <= MAX_INPUTS:
            return True
        m = _PAUSE_RE.match(key)
        if m and 2 <= int(m.group(1)) <= MAX_INPUTS:
            return True
        return super().__contains__(key)

    def __getitem__(self, key):
        m = _AUDIO_RE.match(key)
        if m and 1 <= int(m.group(1)) <= MAX_INPUTS:
            return ("AUDIO",)
        m = _VOLUME_RE.match(key)
        if m and 1 <= int(m.group(1)) <= MAX_INPUTS:
            return ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0,
                              "step": 0.01, "display": "slider",
                              "tooltip": f"Volume for audio_{m.group(1)} (0-100%)"})
        m = _PAUSE_RE.match(key)
        if m and 2 <= int(m.group(1)) <= MAX_INPUTS:
            return ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0,
                              "step": 0.1, "display": "number",
                              "tooltip": f"Pause before audio_{m.group(1)} starts (seconds)"})
        return super().__getitem__(key)


class StarSoundMixer:
    BGCOLOR = "#1a3a2a"
    COLOR = "#0d2a1a"
    CATEGORY = "⭐StarNodes/Video"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "mix_audio"
    DESCRIPTION = ("Mix multiple audio inputs with individual volume controls and "
                   "per-track start pause. Audio slots grow automatically as you "
                   "connect them (up to 12). The first connected audio sets the "
                   "reference sample rate and the total output length. Every other "
                   "audio gets its own volume slider (0-100%, default 100%) and a "
                   "start-pause value in seconds (default 0) so it can begin later "
                   "than the first track. All audio is resampled to the first "
                   "input's sample rate, then summed.")

    @classmethod
    def INPUT_TYPES(cls):
        base = {
            "audio_1": ("AUDIO", {"tooltip": "First audio input — sets the reference sample rate and the output length."}),
            "volume_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0,
                                    "step": 0.01, "display": "slider",
                                    "tooltip": "Volume for audio_1 (0-100%)"}),
        }
        return {
            "required": {},
            "optional": _DynamicMixerInputs(base),
        }

    def mix_audio(self, **kwargs):
        audio_indices = sorted(
            int(m.group(1)) for key in kwargs
            if (m := _AUDIO_RE.match(key)) and kwargs[key] is not None
        )

        if not audio_indices:
            return (None,)

        reference_sample_rate = None
        target_len = None
        mixed_waveform = None

        for idx in audio_indices:
            audio = kwargs[f"audio_{idx}"]
            volume = float(kwargs.get(f"volume_{idx}", 1.0))
            if volume <= 0.0:
                continue

            if isinstance(audio, dict):
                waveform = audio.get("waveform")
                sample_rate = audio.get("sample_rate", 44100)
            else:
                waveform = audio
                sample_rate = 44100

            if waveform is None:
                continue

            if reference_sample_rate is None:
                reference_sample_rate = sample_rate
                target_len = waveform.shape[-1]

            if sample_rate != reference_sample_rate:
                import torchaudio
                waveform = torchaudio.functional.resample(
                    waveform, sample_rate, reference_sample_rate)

            if volume != 1.0:
                waveform = waveform * volume

            # Prepend silence for non-master tracks so they start later.
            if idx >= 2:
                start_pause = float(kwargs.get(f"start_pause_{idx}", 0.0))
                if start_pause > 0.0:
                    pause_samples = int(round(start_pause * reference_sample_rate))
                    if pause_samples > 0:
                        waveform = torch.nn.functional.pad(waveform, (pause_samples, 0))

            if mixed_waveform is None:
                # Master track defines the output length.
                mixed_waveform = waveform
            else:
                # Align every other track to the master length: truncate
                # overflows, pad shortfalls with silence.
                cur_len = waveform.shape[-1]
                if cur_len > target_len:
                    waveform = waveform[..., :target_len]
                elif cur_len < target_len:
                    waveform = torch.nn.functional.pad(waveform, (0, target_len - cur_len))

                max_ch = max(mixed_waveform.shape[-2], waveform.shape[-2])
                if mixed_waveform.shape[-2] < max_ch:
                    pad_ch = max_ch - mixed_waveform.shape[-2]
                    mixed_waveform = torch.nn.functional.pad(
                        mixed_waveform, (0, 0, 0, pad_ch))
                if waveform.shape[-2] < max_ch:
                    pad_ch = max_ch - waveform.shape[-2]
                    waveform = torch.nn.functional.pad(waveform, (0, 0, 0, pad_ch))

                mixed_waveform = mixed_waveform + waveform

        if mixed_waveform is None:
            return (None,)

        max_val = mixed_waveform.abs().max().item()
        if max_val > 1.0:
            mixed_waveform = mixed_waveform / max_val

        result = {
            "waveform": mixed_waveform,
            "sample_rate": reference_sample_rate or 44100,
        }
        return (result,)


NODE_CLASS_MAPPINGS = {
    "StarSoundMixer": StarSoundMixer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarSoundMixer": "⭐ Star Sound Mixer"
}
