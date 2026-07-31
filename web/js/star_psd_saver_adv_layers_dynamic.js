// Dynamic input handler for ⭐ Star PSD Saver Adv. Layers
// Based on working star_psd_saver_dynamic.js pattern
import { app } from "../../../../scripts/app.js";

const BLEND_MODES = [
    "normal", "dissolve", "darken", "multiply", "color_burn", "linear_burn",
    "darker_color", "lighten", "screen", "color_dodge", "linear_dodge",
    "lighter_color", "overlay", "soft_light", "hard_light", "vivid_light",
    "linear_light", "pin_light", "hard_mix", "difference", "exclusion",
    "subtract", "divide", "hue", "saturation", "color", "luminosity"
];

const PLACEMENTS = [
    "center",
    "top_left",
    "top_middle",
    "top_right",
    "middle_left",
    "middle_right",
    "bottom_left",
    "bottom_middle",
    "bottom_right",
];

function updateInputs(node) {
    if (!node || !Array.isArray(node.inputs)) return;
    if (node._updatingInputs) return;
    node._updatingInputs = true;
    
    try {
        // Collect layer/mask pairs by matching names
        const layerInputs = {};
        const maskInputs = {};
        const layerIndices = {};
        const maskIndices = {};

        for (let i = 0; i < node.inputs.length; i++) {
            const inp = node.inputs[i];
            if (!inp || typeof inp.name !== "string") continue;
            if (inp.name.startsWith("layer")) {
                const idx = parseInt(inp.name.replace("layer", ""));
                if (!isNaN(idx)) {
                    layerInputs[idx] = inp;
                    layerIndices[idx] = i;
                }
            } else if (inp.name.startsWith("mask")) {
                const idx = parseInt(inp.name.replace("mask", ""));
                if (!isNaN(idx)) {
                    maskInputs[idx] = inp;
                    maskIndices[idx] = i;
                }
            }
        }

        const indices = new Set([...Object.keys(layerInputs), ...Object.keys(maskInputs)].map(Number));
        const sortedIndices = Array.from(indices).sort((a, b) => a - b);
        
        const pairs = [];
        for (const i of sortedIndices) {
            const layer = layerInputs[i];
            const mask = maskInputs[i];
            if (layer && mask) {
                pairs.push({
                    num: i,
                    layer,
                    mask,
                    layerIndex: layerIndices[i],
                    maskIndex: maskIndices[i]
                });
            }
        }

        // Ensure at least one pair exists
        if (pairs.length === 0) {
            node.addInput("layer1", "IMAGE");
            node.addInput("mask1", "MASK");
            node._updatingInputs = false;
            return;
        }

        // If last layer in the last pair is connected, add a new empty pair for the next index
        const lastPair = pairs[pairs.length - 1];
        if (lastPair && lastPair.layer.link !== null) {
            const nextIdx = lastPair.num + 1;
            if (!layerInputs[nextIdx] && !maskInputs[nextIdx]) {
                node.addInput(`layer${nextIdx}`, "IMAGE");
                node.addInput(`mask${nextIdx}`, "MASK");
            }
        }

        // Ensure blend_modeN / opacityN / placementN widgets exist for every existing layer index > 1
        if (!node.widgets) node.widgets = [];
        for (const idx of sortedIndices) {
            const num = Number(idx);
            if (!num || num <= 1) continue;

            const hasBlend = node.widgets.some(w => w && w.name === `blend_mode${num}`);
            const hasOpacity = node.widgets.some(w => w && w.name === `opacity${num}`);
            const hasPlacement = node.widgets.some(w => w && w.name === `placement${num}`);

            if (!hasBlend) {
                node.addWidget("combo", `blend_mode${num}`, "normal", () => {}, { values: BLEND_MODES });
            }
            if (!hasOpacity) {
                node.addWidget("number", `opacity${num}`, 100.0, () => {}, { min: 0, max: 100, step: 1 });
            }
            if (!hasPlacement) {
                node.addWidget("combo", `placement${num}`, "center", () => {}, { values: PLACEMENTS });
            }
        }

        // Don't remove any layer pairs - users can disconnect/reconnect freely
        // This is more user-friendly than auto-removing slots

        if (node.graph) node.graph.change();
    } catch (e) {
        console.error("[StarPSDSaverAdvLayersDynamic] Error:", e);
    } finally {
        node._updatingInputs = false;
    }
}

app.registerExtension({
    name: "StarPSDSaverAdvLayersDynamic",
    beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "StarPSDSaverAdvLayers") return;
        
        const origOnConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function(type, index, connected, link_info) {
            if (origOnConnectionsChange)
                origOnConnectionsChange.apply(this, arguments);
            // Only update inputs when an INPUT connection changes (type 1)
            // Ignore OUTPUT connection changes (type 2) to avoid removing empty slots
            if (type === 1) {
                updateInputs(this);
            }
        };
        
        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            if (origOnNodeCreated) origOnNodeCreated.apply(this, arguments);
            updateInputs(this);
        };
    }
});
