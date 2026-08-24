# ⭐ Star Video Sound Enricher Option — Help

The settings twin of the **⭐ Star Video Sound Enricher**. It has the exact
same presets and knobs, but instead of processing audio itself it outputs a
`sound_settings` bundle (`SOUND_SETTINGS`).

Connect it to the **`sound_settings` input** of the **⭐ Star LTXV 2.5
All-in-One** or the **⭐ Star Minimax All In One** node — the generated
soundtrack is then cleaned up and enriched **inside** the video node, right
before it reaches the `audio` output (at least 44.1 kHz — a 48 kHz source
stays 48 kHz, never downsampled). No extra audio wiring needed.

```
[Star Video Sound Enricher Option] ──sound_settings──> [Star LTXV 2.5 All-in-One] ──> audio (processed)
                                                     └> [Star Minimax All In One]     ──> audio (processed)
```

The processing is identical to the standalone node — same chain, same
presets, bit-identical results:

- de-harsh bell (kills the 2–6 kHz scratch), warmth bell @ 300 Hz,
  low-shelf bass boost, high-shelf cut, gentle tanh saturation,
  dry/wet `intensity`, 44.1 kHz resample, optional −1 dBFS normalize.

See **StarVideoSoundEnricher.md** for the full preset table and a
description of every Custom knob.

## Notes

- The settings apply to whichever audio the LTXV node outputs: generated
  audio (first pass) or a passed-through audio file.
- Nothing connected → the LTXV node outputs its audio untouched, exactly as
  before.
