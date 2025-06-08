// Dynamic input handler for Star Grid Image Batcher
import { app } from "../../../scripts/app.js";

function updateInputs(node) {
    if (!node || !Array.isArray(node.inputs)) return;
    if (node._updatingInputs) return;
    node._updatingInputs = true;
    try {
        // Ensure 'image_batch' is present as the first input (optional)
        let batchIdx = node.inputs.findIndex(inp => inp.name === "image_batch");
        if (batchIdx === -1) {
            node.addInput("image_batch", "IMAGE");
        }
        // Gather all image N inputs (ignore 'image batch')
        let imageInputs = node.inputs.filter(inp => /^image \d+$/.test(inp.name) && inp.type === "IMAGE");
        // If no image N inputs, add image 1 (optional)
        if (imageInputs.length === 0) {
            node.addInput("image 1", "IMAGE");
            imageInputs = node.inputs.filter(inp => inp.name.startsWith("image ") && inp.type === "IMAGE");
        }
        // Sort imageInputs by number
        imageInputs.sort((a, b) => parseInt(a.name.split('_')[1]) - parseInt(b.name.split('_')[1]));
        // If the last image N input is connected, add a new one
        const last = imageInputs[imageInputs.length - 1];
        if (last && last.link !== null) {
            const idx = imageInputs.length + 1;
            // Only add if it doesn't already exist
            if (!node.inputs.some(inp => inp.name === `image ${idx}`)) {
                node.addInput(`image ${idx}`, "IMAGE");
            }
        }
        // Remove trailing unconnected image N inputs (except the first)
        for (let i = node.inputs.length - 1; i > 0; i--) {
            const inp = node.inputs[i];
            if (/^image \d+$/.test(inp.name) && inp.link === null && inp.type === "IMAGE") {
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
        // Make first image N input required, others optional
        let imageInputsSorted = node.inputs.filter(inp => /^image \d+$/.test(inp.name) && inp.type === "IMAGE");
        imageInputsSorted.sort((a, b) => parseInt(a.name.split(' ')[1]) - parseInt(b.name.split(' ')[1]));
        let firstSet = false;
        for (let i = 0; i < node.inputs.length; i++) {
            if ((/^image \d+$/.test(node.inputs[i].name) && node.inputs[i].type === "IMAGE") || node.inputs[i].name === "image_batch") {
                if (!firstSet) {
                    node.inputs[i].optional = false;
                    firstSet = true;
                } else {
                    node.inputs[i].optional = true;
                }
            }
        }
    } catch(e) {
        console.error("[StarGridImageBatcherDynamic] Error in updateInputs:", e);
    } finally {
        node._updatingInputs = false;
    }
}

app.registerExtension({
    name: "StarGridImageBatcherDynamic",
    beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "StarGridImageBatcher") return;
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
