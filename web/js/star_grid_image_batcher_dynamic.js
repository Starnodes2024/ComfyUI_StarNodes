// Dynamic input handler for Star Grid Image Batcher
import { app } from "../../../../scripts/app.js";

function updateInputs(node) {
    console.log("[StarGridImageBatcherDynamic] updateInputs called", node?.title || node?.type);
    if (!node || !Array.isArray(node.inputs)) return;
    if (node._updatingInputs) return;
    node._updatingInputs = true;
    try {
        // Ensure a fixed image_batch input exists (optional, not managed dynamically)
        let batchIdx = node.inputs.findIndex(inp => inp.name === "image_batch");
        if (batchIdx === -1) {
            node.addInput("image_batch", "IMAGE");
        }

        // Collect all numbered image inputs: image 1, image 2, ...
        let imageInputs = [];
        for (let i = 0; i < node.inputs.length; i++) {
            const inp = node.inputs[i];
            if (inp && inp.type === "IMAGE" && /^image \d+$/.test(inp.name)) {
                const num = parseInt(inp.name.split(" ")[1]);
                if (!isNaN(num)) {
                    imageInputs.push({ input: inp, num, index: i });
                }
            }
        }

        // Sort by number
        imageInputs.sort((a, b) => a.num - b.num);

        // Ensure at least one numbered image input exists
        if (imageInputs.length === 0) {
            node.addInput("image 1", "IMAGE");
            return; // Re-run on next connection change
        }

        // If last numbered image input is connected, add a new one
        const last = imageInputs[imageInputs.length - 1];
        if (last && last.input.link !== null) {
            const nextIdx = last.num + 1;
            if (!node.inputs.some(inp => inp.name === `image ${nextIdx}`)) {
                node.addInput(`image ${nextIdx}`, "IMAGE");
            }
        }

        // Remove trailing unconnected numbered image inputs, keeping at least image 1
        // Work backwards to avoid index issues
        let toRemove = [];
        for (let i = imageInputs.length - 1; i > 0; i--) {
            const imgData = imageInputs[i];
            if (imgData.input.link === null) {
                toRemove.push(imgData.index);
            } else {
                break;
            }
        }
        
        // Remove in reverse order to maintain indices
        toRemove.sort((a, b) => b - a);
        for (const idx of toRemove) {
            node.removeInput(idx);
        }

        // Make first image input required, all other images and image_batch optional
        for (const inp of node.inputs) {
            if (!inp) continue;
            if (inp.name === "image 1") {
                inp.optional = false;
            } else if (inp.type === "IMAGE" && /^image \d+$/.test(inp.name)) {
                inp.optional = true;
            } else if (inp.name === "image_batch") {
                inp.optional = true;
            }
        }

        if (node.graph) node.graph.change();
    } catch (e) {
        console.error("[StarGridImageBatcherDynamic] Error in updateInputs:", e);
    } finally {
        node._updatingInputs = false;
    }
}

app.registerExtension({
    name: "StarGridImageBatcherDynamic",
    beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "StarGridImageBatcher") return;
        console.log("[StarGridImageBatcherDynamic] beforeRegisterNodeDef for", nodeData.name);
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
