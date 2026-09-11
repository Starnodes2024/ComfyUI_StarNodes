"""Star Ref Mod Option

RefMod injection options for the ⭐ Star Minimax All In One node: connect an
H3_REF_MODS bundle from the ComfyUI-MiniMaxH3Mod pack (Load H3 RefMods /
Load H3 RefMod Axis / Extract H3 RefMod / Folder Loader) and the output to
the AIO's 'ref_mod_settings' input. The blocks are injected into the AIO's
internal MiniMax H3 conditioning with the exact same options and behavior as
the 'Apply H3 RefMod' node — retention master strength, per-frame curve
(direction/shape/value), graph presets, ref scrambling and the token budget —
by reusing that pack's own functions at runtime (it must stay installed).
"""

import inspect
import logging
import sys

from comfy_api.latest import io

# Fallback copies of the ComfyUI-MiniMaxH3Mod curve lists, used only to build
# the dropdowns when the pack is not installed (execution then errors clearly).
_CURVE_DIRECTIONS = ["constant", "concept_at_start", "concept_at_middle",
                     "concept_at_end", "concept_at_ends"]
_CURVE_SHAPES = ["linear", "ease", "sigmoid", "tanh", "quadratic", "cubic",
                 "exponential", "stair", "elastic", "bump", "dip"]


def _h3mod(required=True):
    """The loaded ComfyUI-MiniMaxH3Mod nodes module, so this node runs the
    exact Apply H3 RefMod code path instead of duplicating it."""
    for module in list(sys.modules.values()):
        try:
            # classes and defs only - modules with a catch-all __getattr__
            # (e.g. torch.ops) answer any name with a fake object
            if isinstance(getattr(module, "MiniMaxH3RefModApply", None), type) \
                    and inspect.isfunction(getattr(module, "_ref_blocks", None)):
                return module
        except Exception:
            continue
    if required:
        raise RuntimeError(
            "The '⭐ Star Ref Mod Option' node requires the ComfyUI-MiniMaxH3Mod "
            "custom node pack (Load/Apply H3 RefMods) - install or update it.")
    return None


class StarRefModOption(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        h3 = _h3mod(required=False)
        directions = list(getattr(h3, "CURVE_DIRECTIONS", None) or _CURVE_DIRECTIONS)
        shapes = list(getattr(h3, "CURVE_SHAPES", None) or _CURVE_SHAPES)
        presets = ["(none)"] + (h3._list_graph_presets() if h3 is not None else [])
        return io.Schema(
            node_id="StarRefModOption",
            display_name="⭐ Star Ref Mod Option",
            category="⭐StarNodes/Video",
            description=(
                "RefMod injection options for the ⭐ Star Minimax All In One node - "
                "connect an H3_REF_MODS bundle from the ComfyUI-MiniMaxH3Mod pack "
                "(Load H3 RefMods / Load H3 RefMod Axis / Extract H3 RefMod) and the "
                "output to the AIO's 'ref_mod_settings' input. The mods ride inside "
                "the AIO's internal conditioning with the exact same options and "
                "behavior as the 'Apply H3 RefMod' node: retention master strength, "
                "per-frame curve (direction/shape/value), shared graph presets, ref "
                "scrambling and the token budget. Requires the ComfyUI-MiniMaxH3Mod "
                "pack."
            ),
            inputs=[
                io.Custom("H3_REF_MODS").Input("mods",
                    tooltip="Bundle from Load H3 RefMods / Load H3 RefMod Axis / Extract H3 RefMod."),
                io.Boolean.Input("override", default=False,
                    tooltip="Use the config fixed into the mods' own metadata (by 'Fix H3 RefMod "
                            "Config') instead of the widgets below: retention + curve come from "
                            "the first mod in the bundle that carries one. Handy for sharing mods "
                            "whose magic settings took real tuning. Off (default) = use the manual "
                            "parameters. If no mod has a saved config it falls back to the manual "
                            "parameters and prints a note."),
                io.Float.Input("retention", default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="Master reference strength, multiplied with each loader row's "
                            "strength. MiniMax retention levels: 1.0 = fully_preserved, "
                            "0.7 = partially_preserved, 0.4 = attribute_transfer (keep "
                            "style/attributes, not identity), 0.15 = weak_reference. "
                            "0 = no reference."),
                io.Combo.Input("curve_direction", options=directions, default="constant",
                    tooltip="Weighting envelope across THIS MOD'S OWN ref frames "
                            "(stacked images / video-ref latent frames) - i.e. WHICH "
                            "reference content dominates, NOT where the concept "
                            "appears in the output video (ref tokens are not bound "
                            "to output time; for output-timing control use the 'H3 "
                            "RefMod Step Curve' node instead, which runs over the "
                            "denoise timeline). 'constant' (default) = every ref "
                            "frame at full strength (official-ref parity). The old "
                            "default 'concept_at_end' fades early stack frames toward "
                            "blur, roughly HALVING average strength on multi-frame "
                            "mods. Old saved workflows keep their saved values."),
                io.Combo.Input("curve_shape", options=shapes, default="linear",
                    tooltip="How the weighting travels between its endpoints: 'linear', "
                            "'ease' (smoothstep), 'sigmoid'/'tanh' (S-curves, tanh with a "
                            "steeper knee), 'quadratic', 'cubic', 'exponential', 'stair' "
                            "(stepped), 'elastic' (overshoots), 'bump'/'dip' (peak/trough "
                            "mid-stack). Only matters when curve_direction != constant."),
                io.Float.Input("curve_value", default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="Endpoint weight ('user input'): both endpoints for 'constant' and "
                            "'concept_at_ends', the start for 'concept_at_start', the end for "
                            "'concept_at_end', the mid peak for 'concept_at_middle'. On "
                            "single-image ('image'-kind) mods this acts as a simple STRENGTH CAP "
                            "(directions are meaningless on one frame): 0.4 = the ref blends "
                            "40% toward its blurred self."),
                io.Int.Input("scramble_seed", default=-1, min=-1, max=2147483647, step=1,
                    control_after_generate=io.ControlAfterGenerate.fixed,
                    tooltip="Ref scrambling seed. -1 (default) = off: all refs in saved order. "
                            "With 2+ refs in the bundle, a seed >= 0 shuffles the ref order and "
                            "keeps a random subset, so a different ref leads each run (a multi-ref "
                            "mod 'pops' a different video/image per seed). Same seed = same "
                            "scramble; set this widget's control-after-generate to 'randomize' "
                            "for per-run variation."),
                io.Combo.Input("graph_preset", options=presets, default="(none)", advanced=True,
                    tooltip="Optional shared graph preset - leave on '(none)' to use the curve "
                            "widgets above. Selecting one loads direction/shape/value from a "
                            "saved debug-grid PNG (graph embedded in its metadata) or a legacy "
                            ".json, in models/refmods/graph_presets/. Share the preset PNG "
                            "itself to share a curve. New presets appear after a restart."),
                io.Combo.Input("scramble_mode", options=["shuffle", "subset", "legacy_subset"],
                               default="shuffle", advanced=True,
                               tooltip="'shuffle' reorders all refs, 'subset' keeps the first "
                                       "'scramble_keep' of the shuffled refs, 'legacy_subset' keeps "
                                       "a random-sized subset (the original behavior)."),
                io.Int.Input("scramble_keep", default=1, min=1, max=80, advanced=True,
                             tooltip="Refs retained in subset mode; shuffle keeps all refs."),
                io.Int.Input("max_total_tokens", default=0, min=0, max=1048576, advanced=True,
                             tooltip="Total reference token budget after copies; 0 disables the limit."),
                io.String.Input("save_preset_as", default="", advanced=True,
                    tooltip="Optional: type a name and run to save the current (resolved) curve "
                            "as a PNG preset - the curve graph itself with the graph embedded in "
                            "its metadata - in models/refmods/graph_presets/. Share that image "
                            "to share the curve. Leave empty to skip."),
            ],
            outputs=[
                io.Custom("REF_MOD_SETTINGS").Output("ref_mod_settings",
                    tooltip="Settings bundle for the 'ref_mod_settings' input of ⭐ Star Minimax All In One."),
                io.Image.Output("curve_graph", display_name="curve graph",
                    tooltip="1024x1024 curve graph: the strength envelope "
                            "(direction/shape/value) with the concept zone shaded. Leave "
                            "unconnected to skip the preview."),
            ],
        )

    @classmethod
    def execute(cls, mods, override=False, retention=1.0,
                curve_direction="constant", curve_shape="linear", curve_value=1.0,
                scramble_seed=-1, graph_preset="(none)", scramble_mode="shuffle",
                scramble_keep=1, max_total_tokens=0, save_preset_as="") -> io.NodeOutput:
        # exact MiniMaxH3RefModApply.execute resolution: widget curve ->
        # graph preset -> mod's fixed config (override)
        h3 = _h3mod()
        curve = (curve_direction, curve_shape, curve_value)
        preset_name = ""
        if graph_preset and graph_preset != "(none)":
            loaded = h3._load_graph_preset(graph_preset)
            if loaded is None:
                logging.warning("[Star Ref Mod Option] WARNING: graph preset '%s' "
                                "not found or invalid - using widget curve", graph_preset)
            else:
                curve = loaded
                preset_name = graph_preset
        if override:
            found = None
            for m, _s in (mods or []):
                saved = h3._saved_curve(getattr(m, "config", None), "curve")
                if saved is not None:
                    found = (m, saved)
                    break
            if found is None:
                logging.info("[Star Ref Mod Option] override=True but no mod in the bundle "
                             "has a saved config - using the manual parameters.")
            else:
                m, saved = found
                cfg = getattr(m, "config", None) or {}
                curve = saved
                if isinstance(cfg.get("retention"), (int, float)):
                    retention = min(1.0, max(0.0, float(cfg["retention"])))
                logging.info("[Star Ref Mod Option] override: using config from '%s' "
                             "(retention=%.2f, curve=%s + %s @ %.2f)",
                             m.name, retention, curve[0], curve[1], float(curve[2]))
        img = h3.render_debug_grid(curve, preset_name)
        if save_preset_as:
            saved = h3._save_graph_preset(save_preset_as, curve, img)
            if saved:
                logging.info("[Star Ref Mod Option] graph preset saved: %s.png "
                             "(%s + %s @ %.2f)", saved, curve[0], curve[1], float(curve[2]))
        blocks = h3._ref_blocks(mods, retention, curve, seed=scramble_seed,
                                scramble_mode=scramble_mode, scramble_keep=scramble_keep,
                                max_total_tokens=max_total_tokens)
        logging.info("[Star Ref Mod Option] retention=%s (%d ref block(s) prepared)",
                     retention, len(blocks))
        return io.NodeOutput({"blocks": blocks}, h3.pil_to_tensor(img))


NODE_CLASS_MAPPINGS = {
    "StarRefModOption": StarRefModOption,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarRefModOption": "⭐ Star Ref Mod Option",
}
