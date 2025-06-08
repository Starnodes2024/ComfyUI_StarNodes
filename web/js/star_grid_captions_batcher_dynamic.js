// Dynamic input handler for Star Grid Captions Batcher
import { app } from "../../../scripts/app.js";

function updateInputs(node) {
    if (!node || !Array.isArray(node.inputs)) return;
    if (node._updatingInputs) return;
    node._updatingInputs = true;
    try {
        // Gather all caption N inputs (with space)
        let captionInputs = node.inputs.filter(inp => /^caption \d+$/.test(inp.name));
        // If no caption inputs, add 'caption 1'
        if (captionInputs.length === 0) {
            node.addInput("caption 1", "STRING");
            captionInputs = node.inputs.filter(inp => /^caption \d+$/.test(inp.name));
        }
        // Sort inputs by number
        captionInputs.sort((a, b) => parseInt(a.name.split(' ')[1]) - parseInt(b.name.split(' ')[1]));
        // If the last caption input is connected, add a new one
        const last = captionInputs[captionInputs.length - 1];
        if (last && last.link !== null) {
            const idx = captionInputs.length + 1;
            // Only add if it doesn't already exist
            if (!node.inputs.some(inp => inp.name === `caption ${idx}`)) {
                node.addInput(`caption ${idx}`, "STRING");
            }
        }
        // Remove trailing unconnected caption N inputs (except the first)
        for (let i = node.inputs.length - 1; i > 0; i--) {
            const inp = node.inputs[i];
            if (/^caption \d+$/.test(inp.name) && inp.link === null) {
                const idx = parseInt(inp.name.split("_")[1]);
                if (idx > 1 && i === node.inputs.length - 1) {
                    node.removeInput(i);
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        // Make first caption required
        if (node.inputs[0]) node.inputs[0].optional = false;
        // All other captions are optional
        for (let i = 1; i < node.inputs.length; i++) {
            node.inputs[i].optional = true;
        }
    } catch(e) {
        console.error("[StarGridCaptionsBatcherDynamic] Error in updateInputs:", e);
    } finally {
        node._updatingInputs = false;
    }
}

app.registerExtension({
    name: "StarGridCaptionsBatcherDynamic",
    beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!nodeData || nodeData.name !== "StarGridCaptionsBatcher") return;
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
