// Dynamic input handler for ⭐ Star PSD Saver (Dynamic)
// Adds/removes image/mask inputs as needed
import { app } from "../../../../scripts/app.js";

function updateInputs(node) {
    console.log("[StarPSDSaverDynamic] updateInputs called", node?.title || node?.type);
    if (!node || !Array.isArray(node.inputs)) return;
    if (node._updatingInputs) return;
    node._updatingInputs = true;
    try {
        // Collect layer/mask pairs by matching names like layer1/mask1, layer2/mask2, ...
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
            return; // Re-run on next connection change
        }

        // If last layer in the last pair is connected, add a new pair
        const lastPair = pairs[pairs.length - 1];
        if (lastPair && lastPair.layer.link !== null) {
            const nextIdx = lastPair.num + 1;
            if (!layerInputs[nextIdx] && !maskInputs[nextIdx]) {
                node.addInput(`layer${nextIdx}`, "IMAGE");
                node.addInput(`mask${nextIdx}`, "MASK");
            }
        }

        // Remove trailing unconnected pairs, keeping the first
        // Collect indices to remove
        let toRemove = [];
        for (let i = pairs.length - 1; i > 0; i--) {
            const pair = pairs[i];
            if (pair.layer.link === null && pair.mask.link === null) {
                toRemove.push(pair.layerIndex, pair.maskIndex);
            } else {
                break;
            }
        }
        
        // Remove in reverse order to maintain indices
        toRemove.sort((a, b) => b - a);
        for (const idx of toRemove) {
            node.removeInput(idx);
        }

        // Mark filename_prefix/output_dir as required, pairs optional except first layer
        for (const inp of node.inputs) {
            if (!inp) continue;
            if (inp.name === "filename_prefix" || inp.name === "output_dir") {
                inp.optional = false;
            } else if (inp.name === "layer1") {
                inp.optional = false;
            } else if (inp.name.startsWith("layer") || inp.name.startsWith("mask")) {
                inp.optional = true;
            }
        }

        if (node.graph) node.graph.change();
    } catch (e) {
        console.error("[StarPSDSaverDynamic] Error in updateInputs:", e);
    } finally {
        node._updatingInputs = false;
    }
}

app.registerExtension({
    name: "StarPSDSaverDynamic",
    beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "StarPSDSaver") return;
        console.log("[StarPSDSaverDynamic] beforeRegisterNodeDef for", nodeData.name);
        const origOnConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function(type, index, connected, link_info) {
            if (origOnConnectionsChange)
                origOnConnectionsChange.apply(this, arguments);
            updateInputs(this);
        };
        // Also update on node creation
        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            if (origOnNodeCreated) origOnNodeCreated.apply(this, arguments);
            updateInputs(this);
        };
    }
});
