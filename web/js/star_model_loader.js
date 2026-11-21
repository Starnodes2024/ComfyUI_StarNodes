// Star Model Loader - preset dropdown + Save/Update/Delete buttons.
// The output sockets are fixed by the backend definition (one per slot,
// None for unselected slots) - no frontend socket manipulation, so
// workflow links survive saves and tab switches.
import { app } from "../../../../scripts/app.js";

const NONE = "(none)";

// widget name -> preset key (order matches the backend _SLOTS)
const SLOTS = [
    ["Checkpoint", "checkpoint"],
    ["Model 1", "model_1"],
    ["Model 2", "model_2"],
    ["Model 3", "model_3"],
    ["Text-Encoder 1", "text_encoder_1"],
    ["Text-Encoder 2", "text_encoder_2"],
    ["VAE 1", "vae_1"],
    ["VAE 2", "vae_2"],
    ["VAE 3", "vae_3"],
    ["Sampler", "sampler"],
    ["Scheduler", "scheduler"],
    ["Steps", "steps"],
];

// Advanced-only latent widgets stored in the shared presets file
// (the basic loader ignores these keys)
const LATENT_WIDGETS = [
    ["Ratio", "latent_ratio"],
    ["Custom Ratio", "latent_custom_ratio"],
    ["Resolution", "latent_resolution"],
    ["Custom Width", "latent_width"],
    ["Custom Height", "latent_height"],
    ["Latent Channels", "latent_channels"],
    ["Ratio From Image", "latent_from_image"],
    ["Batch Size", "latent_batch"],
];

let _presets = null;   // shared cache: preset name -> {key: value}

// output socket -> its widget (purely visual state; latent/width/height
// are not listed - they are always active)
const OUT_WIDGET = {
    checkpoint_model: "Checkpoint", checkpoint_clip: "Checkpoint", checkpoint_vae: "Checkpoint",
    model_1: "Model 1", model_2: "Model 2", model_3: "Model 3",
    text_encoder_1: "Text-Encoder 1", text_encoder_2: "Text-Encoder 2",
    vae_1: "VAE 1", vae_2: "VAE 2", vae_3: "VAE 3",
    sampler: "Sampler", scheduler: "Scheduler", steps: "Steps",
};
const HOLLOW_CIRCLE = 7;   // litegraph SlotShape.HollowCircle
const DIM = "#3a3a3a";
const PREFIXED = { sampler: 1, scheduler: 1, steps: 1 };

function getWidget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

function short(v) {
    const base = String(v).split(/[\\/]/).pop().replace(/\.[^.]+$/, "");
    return base.length > 24 ? base.slice(0, 23) + "…" : base;
}

// highlight the usable outputs: selected slots get a filled type-colored
// circle and the selection in their label, unused slots become hollow dim
// rings. Only label/color/shape change - names, types and slot order stay
// untouched, so links are unaffected.
function refreshOutputs(node) {
    for (const o of node.outputs || []) {
        const wname = OUT_WIDGET[o.name];
        if (!wname) continue;
        const v = getWidget(node, wname)?.value;
        const active = wname === "Steps" ? Number(v) > 0 : !!(v && v !== NONE);
        // model/clip/vae sockets are identifiable by color - the label
        // shows the selection only; steps/scheduler/sampler keep a prefix
        o.label = active
            ? (o.name in PREFIXED ? `${o.name}: ${short(v)}` : short(v))
            : o.name;
        o.shape = active ? undefined : HOLLOW_CIRCLE;
        o.color_off = active ? undefined : DIM;
        o.color_on = active ? undefined : DIM;
    }
    node.setSize(node.computeSize());
    node.graph?.setDirtyCanvas?.(true, true);
}

function collectValues(node, advanced) {
    const values = {};
    for (const [wname, key] of SLOTS) values[key] = getWidget(node, wname)?.value ?? NONE;
    if (advanced)
        for (const [wname, key] of LATENT_WIDGETS)
            values[key] = getWidget(node, wname)?.value;
    return values;
}

function applyPreset(node, name, advanced) {
    const p = _presets?.[name];
    if (!p) return;
    for (const [wname, key] of SLOTS) {
        const w = getWidget(node, wname);
        if (w && p[key] !== undefined) w.value = p[key];
    }
    if (advanced)
        for (const [wname, key] of LATENT_WIDGETS) {
            const w = getWidget(node, wname);
            if (w && p[key] !== undefined) w.value = p[key];
        }
}

async function refreshPresets(node) {
    try {
        const resp = await app.api.fetchApi("/starnodes/model_loader/presets");
        const d = await resp.json();
        _presets = d.presets || {};
    } catch (e) {
        _presets = _presets || {};
    }
    const w = getWidget(node, "Preset");
    if (w) {
        w.options = w.options || {};
        w.options.values = [NONE, ...Object.keys(_presets).sort()];
        if (!w.options.values.includes(w.value)) w.value = NONE;
    }
}

async function postPreset(action, name, values) {
    const resp = await app.api.fetchApi("/starnodes/model_loader/presets", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action, name, values }),
    });
    const d = await resp.json();
    if (d.status !== "ok") throw new Error(d.message || resp.status);
    _presets = d.presets || _presets;
}

app.registerExtension({
    name: "StarModelLoader",
    beforeRegisterNodeDef(nodeType, nodeData) {
        const advanced = nodeData.name === "StarModelLoaderAdvanced";
        if (nodeData.name !== "StarModelLoader" && !advanced) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            const node = this;
            const presetW = getWidget(node, "Preset");

            for (const [wname] of SLOTS) {
                const w = getWidget(node, wname);
                if (!w) continue;
                const orig = w.callback;
                w.callback = function () {
                    orig?.apply(this, arguments);
                    refreshOutputs(node);
                };
            }

            if (presetW) {
                const orig = presetW.callback;
                presetW.callback = function () {
                    orig?.apply(this, arguments);
                    if (presetW.value !== NONE) applyPreset(node, presetW.value, advanced);
                    refreshOutputs(node);
                };
            }

            node.addWidget("button", "Save Preset", "save", async () => {
                await refreshPresets(node);
                let name = prompt("Preset name:", presetW?.value !== NONE ? presetW?.value : "");
                name = (name || "").trim();
                if (!name) return;
                let action = "save";
                if (_presets?.[name]) {
                    if (!confirm(`Preset "${name}" already exists - overwrite it?`)) return;
                    action = "update";
                }
                try {
                    await postPreset(action, name, collectValues(node, advanced));
                    await refreshPresets(node);
                    if (presetW) presetW.value = name;
                } catch (e) {
                    alert("Save preset failed: " + e);
                }
            });
            node.addWidget("button", "Update Preset", "update", async () => {
                await refreshPresets(node);
                const name = presetW?.value;
                if (!name || name === NONE || !_presets?.[name]) {
                    alert("Pick a preset in the Preset dropdown first.");
                    return;
                }
                if (!confirm(`Overwrite preset "${name}" with the current selection?`)) return;
                try {
                    await postPreset("update", name, collectValues(node, advanced));
                    await refreshPresets(node);
                } catch (e) {
                    alert("Update preset failed: " + e);
                }
            });
            node.addWidget("button", "Delete Preset", "delete", async () => {
                await refreshPresets(node);
                const name = presetW?.value;
                if (!name || name === NONE || !_presets?.[name]) {
                    alert("Pick a preset in the Preset dropdown first.");
                    return;
                }
                if (!confirm(`Delete preset "${name}"?`)) return;
                try {
                    await postPreset("delete", name, {});
                    await refreshPresets(node);
                    if (presetW) presetW.value = NONE;
                } catch (e) {
                    alert("Delete preset failed: " + e);
                }
            });

            refreshPresets(node);
            refreshOutputs(node);
            return r;
        };

        // after a workflow restore: recompute labels/colors from the
        // restored widget values (purely visual, links untouched)
        const onAfterGraphConfigured = nodeType.prototype.onAfterGraphConfigured;
        nodeType.prototype.onAfterGraphConfigured = function () {
            const r = onAfterGraphConfigured?.apply(this, arguments);
            const node = this;
            requestAnimationFrame(() => refreshOutputs(node));
            return r;
        };
    },
});
