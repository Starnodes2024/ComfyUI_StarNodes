"""
Star Video Sound Enricher (+ Option node).

AI video models (LTXV and friends) often generate audio with a harsh,
scratchy top end and a thin low end. This module runs the sound through a
small mastering-style chain:

  highpass (24 Hz rumble removal)
  -> de-harsh bell cut (the 2-6 kHz "scratch" region)
  -> low-mid warmth bell (300 Hz)
  -> low-shelf bass boost
  -> high-shelf cut (tames the fizzy top)
  -> gentle tanh saturation (harmonic richness)
  -> dry/wet intensity blend
  -> resample up to at least 44.1 kHz (polyphase, torchaudio) + peak normalize
     (a 48 kHz input stays 48 kHz - the sound is never downsampled)

Two nodes share the chain:
  - StarVideoSoundEnricher: standalone AUDIO -> AUDIO processor.
  - StarVideoSoundEnricherOption: same widgets, outputs a SOUND_SETTINGS
    bundle for other nodes (e.g. the LTXV 2.5 All-in-One) to process their
    audio output internally.

Four tuned presets cover the common cases; "Custom" exposes every knob.
"""

import torch
import torchaudio

TARGET_SAMPLE_RATE = 44100

# preset -> EQ/saturation parameters (all gains in dB, frequencies in Hz)
PRESETS = {
    "Cinematic Warm": dict(harsh_freq=3500, harsh_cut=5.5, high_cut_freq=9000,
                           high_cut_db=9.0, bass_freq=110, bass_boost=6.0,
                           warmth=2.5, drive=0.25),
    "Smooth & Soft": dict(harsh_freq=3000, harsh_cut=4.5, high_cut_freq=10000,
                          high_cut_db=6.0, bass_freq=120, bass_boost=3.5,
                          warmth=1.5, drive=0.15),
    "Voice Clarity": dict(harsh_freq=4000, harsh_cut=6.0, high_cut_freq=11000,
                          high_cut_db=5.0, bass_freq=150, bass_boost=1.5,
                          warmth=1.0, drive=0.10),
    "Deep Bass Boost": dict(harsh_freq=3500, harsh_cut=4.5, high_cut_freq=9000,
                            high_cut_db=7.0, bass_freq=90, bass_boost=8.0,
                            warmth=3.0, drive=0.30),
}


def _clamp_freq(freq, sample_rate):
    """Keep every EQ corner safely below Nyquist (input may be 16/24 kHz)."""
    return max(20.0, min(float(freq), sample_rate * 0.45))


def _enrich(waveform, sample_rate, p):
    """The EQ + saturation chain, on [B, C, T] float32 at the input rate."""
    w = waveform.float()
    w = torchaudio.functional.highpass_biquad(w, sample_rate, 24.0)
    if p["harsh_cut"] > 0.0:
        w = torchaudio.functional.equalizer_biquad(
            w, sample_rate, _clamp_freq(p["harsh_freq"], sample_rate),
            -p["harsh_cut"], Q=1.2)
    if p["warmth"] != 0.0:
        w = torchaudio.functional.equalizer_biquad(
            w, sample_rate, _clamp_freq(300.0, sample_rate), p["warmth"], Q=0.8)
    if p["bass_boost"] > 0.0:
        w = torchaudio.functional.bass_biquad(
            w, sample_rate, p["bass_boost"],
            central_freq=_clamp_freq(p["bass_freq"], sample_rate), Q=0.7)
    if p["high_cut_db"] > 0.0:
        w = torchaudio.functional.treble_biquad(
            w, sample_rate, -p["high_cut_db"],
            central_freq=_clamp_freq(p["high_cut_freq"], sample_rate), Q=0.707)
    if p["drive"] > 0.0:
        k = 1.0 + p["drive"] * 8.0
        w = torch.tanh(w * k) / torch.tanh(torch.tensor(k, device=w.device, dtype=w.dtype))
    return w


def make_sound_settings(preset, harsh_freq=3500, harsh_cut=5.5,
                        high_cut_freq=9000, high_cut_db=9.0, bass_freq=110,
                        bass_boost=6.0, warmth=2.5, drive=0.25,
                        intensity=1.0, normalize=True):
    """Bundle preset/custom parameters into a SOUND_SETTINGS dict."""
    if preset == "Custom":
        params = dict(harsh_freq=harsh_freq, harsh_cut=harsh_cut,
                      high_cut_freq=high_cut_freq, high_cut_db=high_cut_db,
                      bass_freq=bass_freq, bass_boost=bass_boost,
                      warmth=warmth, drive=drive)
    else:
        params = dict(PRESETS[preset])
    return {"preset": preset, "params": params,
            "intensity": float(intensity), "normalize": bool(normalize)}


def process_audio(audio, settings):
    """Apply a SOUND_SETTINGS bundle to an AUDIO dict; returns AUDIO at
    44.1 kHz or the input rate, whichever is higher (never downsamples)."""
    waveform = audio.get("waveform")
    if waveform is None:
        raise ValueError("[Star Video Sound Enricher] audio input has no waveform")
    sample_rate = int(audio.get("sample_rate", TARGET_SAMPLE_RATE))
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)

    params = settings["params"]
    intensity = settings["intensity"]
    target_sr = max(sample_rate, TARGET_SAMPLE_RATE)
    print(f"[Star Video Sound Enricher] preset={settings['preset']} | "
          f"{sample_rate} Hz -> {target_sr} Hz | {params}")

    wet = _enrich(waveform, sample_rate, params)
    if intensity < 1.0:
        wet = waveform + intensity * (wet - waveform)

    if sample_rate != target_sr:
        wet = torchaudio.functional.resample(wet, sample_rate, target_sr)

    if settings["normalize"]:
        peak = wet.abs().max().item()
        if peak > 1e-6:
            wet = wet * (0.89 / peak)

    return {"waveform": wet, "sample_rate": target_sr}


def _knob_inputs():
    """The tweakable widgets shared by both nodes (used in Custom mode)."""
    return {
        "harsh_freq": ("INT", {"default": 3500, "min": 1000, "max": 10000, "step": 50,
                       "tooltip": "Custom: center of the de-harsh cut (the scratchy region, "
                                  "typically 2.5-5 kHz)."}),
        "harsh_cut": ("FLOAT", {"default": 5.5, "min": 0.0, "max": 12.0, "step": 0.1,
                      "tooltip": "Custom: how much to cut at harsh_freq, in dB."}),
        "high_cut_freq": ("INT", {"default": 9000, "min": 4000, "max": 20000, "step": 100,
                          "tooltip": "Custom: high-shelf corner - everything above gets tamed. "
                                     "Clamped below Nyquist of the input automatically."}),
        "high_cut_db": ("FLOAT", {"default": 9.0, "min": 0.0, "max": 18.0, "step": 0.1,
                        "tooltip": "Custom: high-shelf cut amount, in dB."}),
        "bass_freq": ("INT", {"default": 110, "min": 40, "max": 300, "step": 5,
                      "tooltip": "Custom: low-shelf corner for the deep-bass boost."}),
        "bass_boost": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 12.0, "step": 0.1,
                       "tooltip": "Custom: bass boost amount, in dB."}),
        "warmth": ("FLOAT", {"default": 2.5, "min": -6.0, "max": 6.0, "step": 0.1,
                   "tooltip": "Custom: low-mid bell at 300 Hz, in dB. Positive = warmer, "
                              "negative = thinner."}),
        "drive": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01,
                  "display": "slider",
                  "tooltip": "Custom: gentle tube-style saturation - adds harmonics for a "
                             "richer sound. 0 = off."}),
        "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                      "display": "slider",
                      "tooltip": "Dry/wet mix of the whole effect. 100% = fully processed, "
                                 "0% = original sound. Applies to every preset too."}),
        "normalize": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled",
                      "tooltip": "Peak-normalize the result to -1 dBFS for a consistent, "
                                 "clipping-free output level."}),
    }


def _preset_input():
    return ([*PRESETS.keys(), "Custom"],
            {"default": "Cinematic Warm",
             "tooltip": "Tuned filter chains. 'Custom' uses the knobs below."})


class StarVideoSoundEnricher:
    BGCOLOR = "#3a2a1a"
    COLOR = "#2a1a0d"
    CATEGORY = "⭐StarNodes/Video"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "enrich"
    DESCRIPTION = ("Fix the typical AI-video soundtrack: filters out the high "
                   "scratchy noise (de-harsh bell + high-shelf cut), enriches "
                   "mids and deep bass, and adds gentle analog-style warmth. "
                   "Four tuned presets plus a fully tweakable Custom mode. "
                   "Output is at least 44.1 kHz - a 48 kHz input stays "
                   "48 kHz, the sound is never downsampled.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "The soundtrack to clean up and enrich."}),
                "preset": _preset_input(),
            },
            "optional": _knob_inputs(),
        }

    def enrich(self, audio, preset, **kwargs):
        settings = make_sound_settings(preset, **kwargs)
        return (process_audio(audio, settings),)


class StarVideoSoundEnricherOption:
    BGCOLOR = "#3a2a1a"
    COLOR = "#2a1a0d"
    CATEGORY = "⭐StarNodes/Video"
    RETURN_TYPES = ("SOUND_SETTINGS",)
    RETURN_NAMES = ("sound_settings",)
    FUNCTION = "options"
    DESCRIPTION = ("Same sound-enricher settings as the Star Video Sound "
                   "Enricher, but outputs a sound_settings bundle instead of "
                   "processing audio itself. Connect it to the LTXV 2.5 "
                   "All-in-One node to clean up and enrich its audio output "
                   "internally.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": _preset_input(),
            },
            "optional": _knob_inputs(),
        }

    def options(self, preset, **kwargs):
        return (make_sound_settings(preset, **kwargs),)


NODE_CLASS_MAPPINGS = {
    "StarVideoSoundEnricher": StarVideoSoundEnricher,
    "StarVideoSoundEnricherOption": StarVideoSoundEnricherOption,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarVideoSoundEnricher": "⭐ Star Video Sound Enricher",
    "StarVideoSoundEnricherOption": "⭐ Star Video Sound Enricher Option",
}
