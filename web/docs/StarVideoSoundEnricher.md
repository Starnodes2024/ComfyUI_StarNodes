# ⭐ Star Video Sound Enricher — Help

AI video models (LTXV & friends) often generate soundtracks with a **harsh,
scratchy top end** and a **thin, lifeless low end**. This node fixes that with
a small mastering-style chain:

1. **Highpass @ 24 Hz** — removes subsonic rumble
2. **De-harsh bell cut** — targets the 2–6 kHz "scratch" region
3. **Warmth bell @ 300 Hz** — low-mid body
4. **Low-shelf bass boost** — deep bass
5. **High-shelf cut** — tames the fizzy top end
6. **Gentle tanh saturation** — analog-style harmonic richness
7. **Resample up to at least 44.1 kHz** (polyphase) + **peak normalize** to −1 dBFS —
   a 48 kHz input stays 48 kHz, the sound is **never downsampled**

**Input:** any `AUDIO` (e.g. the `audio` output of the Star LTXV 2.5 All-in-One).
**Output:** `AUDIO` at **44.1 kHz or the input rate, whichever is higher**,
ready for your video save/compressor nodes.

> Tip: prefer zero extra wiring? The **⭐ Star Video Sound Enricher Option**
> node outputs these same settings as a `sound_settings` bundle that plugs
> straight into the LTXV 2.5 All-in-One node, which then processes its audio
> output internally.

---

## Presets

| Preset | Character |
|---|---|
| **Cinematic Warm** (default) | strong bass body, clearly tamed top, light saturation — the "movie trailer" sound |
| **Smooth & Soft** | gentler version for material that only needs a light cleanup |
| **Voice Clarity** | strongest de-harsh cut, light bass — for dialogue / speech-heavy clips |
| **Deep Bass Boost** | maximum low-end weight, still keeps the scratch under control |
| **Custom** | uses the knobs below |

`intensity` and `normalize` apply to **every** preset, not just Custom.

## Custom knobs

- `harsh_freq` / `harsh_cut` — center (1–10 kHz) and depth (0–12 dB) of the
  de-harsh bell. The scratch usually lives at 2.5–5 kHz.
- `high_cut_freq` / `high_cut_db` — high-shelf corner (4–20 kHz) and cut
  (0–18 dB). Everything above the corner gets tamed; corners are clamped below
  the input's Nyquist automatically (24 kHz source → max ~10.8 kHz).
- `bass_freq` / `bass_boost` — low-shelf corner (40–300 Hz) and boost (0–12 dB).
- `warmth` — low-mid bell at 300 Hz (−6…+6 dB). Positive = warmer, negative = thinner.
- `drive` — tube-style saturation (0–1). Adds harmonics for richness; 0 = off.
- `intensity` — dry/wet mix of the whole chain (0% = original sound).
- `normalize` — peak-normalize the output to −1 dBFS (clipping-free, consistent level).

## Tips

- If the result sounds *too* dark, raise `high_cut_freq` or lower `high_cut_db`
  in Custom mode.
- If it sounds boomy, lower `bass_boost` (or use Voice Clarity as a starting point).
- The node never adds content above the source's own frequency range — a 24 kHz
  source stays band-limited, just beautifully rebalanced and delivered at 44.1 kHz.
  A 48 kHz source keeps its full bandwidth and stays at 48 kHz.
