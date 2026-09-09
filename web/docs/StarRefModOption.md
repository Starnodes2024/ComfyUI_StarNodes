# ⭐ Star Ref Mod Option — Help

RefMod injection options for the **⭐ Star Minimax All In One** node, for use
with the RefMods of the **ComfyUI-MiniMaxH3Mod** pack. The Minimax AIO node
does its conditioning internally, so the *Apply H3 RefMod* node cannot be
chained onto it — this option node closes that gap: same options, same
behavior, applied inside the AIO node.

```
[Load H3 RefMods] ──H3_REF_MODS──> [Star Ref Mod Option] ──ref_mod_settings──> [Star Minimax All In One]
                                                                        mods are injected into the
                                                                        internal conditioning
```

It reuses the actual *Apply H3 RefMod* code from the ComfyUI-MiniMaxH3Mod
pack at runtime, so everything behaves exactly like that node:

1. Each mod's reference latent is scaled by its loader-row **strength ×
   retention** (weakening a ref blends it toward a blurred copy of itself,
   not toward noise).
2. The optional **curve** (direction / shape / value) weights each mod's own
   ref frames — which reference content dominates, not where in the output
   video the concept appears.
3. The resulting ref blocks are **appended after the native reference blocks**
   in the AIO's conditioning, exactly like running *Apply H3 RefMod* on the
   built conditioning. The DiT attends to them through all blocks at a
   fraction of a full reference's token budget.
4. When the **⭐ Star Minimax Latent Upscaler Option** is also connected, the
   ref-mod blocks are carried into the refine pass and resolution-matched
   automatically, just like native references.

**Requires the ComfyUI-MiniMaxH3Mod pack to be installed** (the node lists no
curve options and raises a clear error without it). RefMods are created with
*Extract H3 RefMod* / *Load H3 RefMod(s)* / *Load H3 RefMod Axis* from that
pack — see its own documentation for extraction and tuning.

## Connectors

| Connector | Type | Notes |
|---|---|---|
| `mods` | H3_REF_MODS | bundle from *Load H3 RefMods*, *Load H3 RefMod Axis*, *Extract H3 RefMod* or the folder loader of the ComfyUI-MiniMaxH3Mod pack |
| **ref_mod_settings** out | REF_MOD_SETTINGS | connect to the `ref_mod_settings` input of ⭐ Star Minimax All In One |
| **curve graph** out | IMAGE | 1024×1024 preview of the resolved strength envelope — leave unconnected to skip; the same graph a preset PNG carries |

## Widgets

All widgets match the *Apply H3 RefMod* node one-to-one (its `conditioning`
in/out is replaced by the internal conditioning of the AIO node):

- **override** — ON uses the config fixed into a mod's metadata by *Fix H3
  RefMod Config* (retention + curve from the first mod that carries one)
  instead of the manual widgets; falls back to the manual values with a
  console note when no mod has a saved config.
- **retention** — master reference strength multiplied with each loader
  row's strength: `1.0` fully_preserved (default), `0.7` partially_preserved,
  `0.4` attribute_transfer, `0.15` weak_reference, `0` = no reference.
- **curve_direction** — weighting envelope across a mod's own ref frames:
  `constant` (default, official-ref parity), `concept_at_start` /
  `concept_at_end` / `concept_at_middle` / `concept_at_ends`.
- **curve_shape** — how the envelope travels between its endpoints: `linear`
  (default), `ease`, `sigmoid`, `tanh`, `quadratic`, `cubic`, `exponential`,
  `stair`, `elastic`, `bump`, `dip`. Only matters when the direction is not
  `constant`.
- **curve_value** — the endpoint weight of the curve; on single-image mods it
  acts as a plain strength cap (e.g. `0.4` blends the ref 40 % toward its
  blurred self).
- **scramble_seed** — `-1` (default) = off. With 2+ refs and a seed ≥ 0 the
  ref order is shuffled and a subset kept, so a different ref leads each run;
  switch the widget's control-after-generate to `randomize` for per-run
  variation.
- **graph_preset** (advanced) — load direction/shape/value from a shared
  preset PNG/json in `models/refmods/graph_presets/` instead of the curve
  widgets. Share the preset PNG itself to share a curve.
- **scramble_mode / scramble_keep** (advanced) — `shuffle` (default, keeps
  all refs), `subset` (keeps the first `scramble_keep` shuffled refs),
  `legacy_subset` (random-sized subset, the original behavior).
- **max_total_tokens** (advanced) — total reference token budget after
  copies; `0` (default) disables the limit.
- **save_preset_as** (advanced) — type a name and run once to save the
  resolved curve as a preset PNG into `models/refmods/graph_presets/`.

## Notes

- RefMods are not prompted with `<Picture i>` tags — use the `prompt_hint`
  output of the loader nodes and concat it onto your prompt.
- The mods influence both sampling passes of the AIO node: the main pass and
  the optional latent-upscale refine pass.
- Tiny token budget: a pooled mod injects ~4–16 tokens where a full reference
  video costs thousands — but identity fidelity is correspondingly lower;
  raise `strength`/copies in the loader or `retention` here for a stronger
  hold.
