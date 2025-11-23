// Dynamic input handler for Star Grid Captions Batcher
import { app } from "../../../../scripts/app.js";

function updateInputs(node) {
    console.log("[StarGridCaptionsBatcherDynamic] updateInputs called", node?.title || node?.type);
    if (!node || !Array.isArray(node.inputs)) return;
    if (node._updatingInputs) return;
    node._updatingInputs = true;
    try {
        // Collect all numbered caption inputs: caption 1, caption 2, ...
        let captionInputs = [];
        for (let i = 0; i < node.inputs.length; i++) {
            const inp = node.inputs[i];
            if (inp && /^caption \d+$/.test(inp.name)) {
                const num = parseInt(inp.name.split(" ")[1]);
                if (!isNaN(num)) {
                    captionInputs.push({ input: inp, num, index: i });
                }
            }
        }

        // Sort by number
        captionInputs.sort((a, b) => a.num - b.num);

        // Ensure at least one caption exists
        if (captionInputs.length === 0) {
            node.addInput("caption 1", "STRING");
            return; // Re-run on next connection change
        }

        // If last caption is connected, add a new one
        const last = captionInputs[captionInputs.length - 1];
        if (last && last.input.link !== null) {
            const nextIdx = last.num + 1;
            if (!node.inputs.some(inp => inp && inp.name === `caption ${nextIdx}`)) {
                node.addInput(`caption ${nextIdx}`, "STRING");
            }
        }

        // Remove trailing unconnected captions, keeping at least caption 1
        // Work backwards to avoid index issues
        let toRemove = [];
        for (let i = captionInputs.length - 1; i > 0; i--) {
            const capData = captionInputs[i];
            if (capData.input.link === null) {
                toRemove.push(capData.index);
            } else {
                break;
            }
        }
        
        // Remove in reverse order to maintain indices
        toRemove.sort((a, b) => b - a);
        for (const idx of toRemove) {
            node.removeInput(idx);
        }

        // First caption required, others optional
        for (const inp of node.inputs) {
            if (!inp || !/^caption \d+$/.test(inp.name)) continue;
            if (inp.name === "caption 1") {
                inp.optional = false;
            } else {
                inp.optional = true;
            }
        }

        if (node.graph) node.graph.change();
    } catch (e) {
        console.error("[StarGridCaptionsBatcherDynamic] Error in updateInputs:", e);
    } finally {
        node._updatingInputs = false;
    }
}

app.registerExtension({
    name: "StarGridCaptionsBatcherDynamic",
    beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!nodeData || nodeData.name !== "StarGridCaptionsBatcher") return;
        console.log("[StarGridCaptionsBatcherDynamic] beforeRegisterNodeDef for", nodeData.name);
        const origOnConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function(type, index, connected, link_info) {
            if (origOnConnectionsChange) {
                try { origOnConnectionsChange.apply(this, arguments); } catch(e) { console.error("[StarGridCaptionsBatcherDynamic] Error in origOnConnectionsChange:", e); }
            }
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
