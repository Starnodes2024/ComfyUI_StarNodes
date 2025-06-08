// Dynamic input handler for Star Image Input (Dynamic)
// Adds/removes image/mask inputs as needed
import { app } from "../../../scripts/app.js";

function updateInputs(node) {
    // Find all image/mask input pairs
    let pairs = [];
    for (let i = 0; i < node.inputs.length; i += 2) {
        const image = node.inputs[i];
        const mask = node.inputs[i + 1];
        if (image && mask && image.name.startsWith("image") && mask.name.startsWith("mask")) {
            pairs.push([image, mask]);
        }
    }
    // Ensure at least one image and mask exist
    if (pairs.length === 0) {
        node.addInput("image1", "IMAGE");
        node.addInput("mask1", "MASK");
        pairs.push([node.inputs[node.inputs.length - 2], node.inputs[node.inputs.length - 1]]);
    }
    // If last image input is connected, add new image/mask input
    if (pairs[pairs.length - 1][0].link !== null) {
        const idx = pairs.length + 1;
        node.addInput(`image${idx}`, "IMAGE");
        node.addInput(`mask${idx}`, "MASK");
    }
    // Remove trailing unconnected pairs (except the first)
    for (let i = pairs.length - 1; i > 0; i--) {
        if (pairs[i][0].link === null && pairs[i][1].link === null) {
            node.removeInput(node.inputs.indexOf(pairs[i][0]));
            node.removeInput(node.inputs.indexOf(pairs[i][1]));
        } else {
            break;
        }
    }
    // Make first image required (if possible)
    if (node.inputs[0]) node.inputs[0].optional = false;
    // All other images and all masks are optional
    for (let i = 1; i < node.inputs.length; i++) {
        node.inputs[i].optional = true;
    }
}

app.registerExtension({
    name: "StarImageInputDynamic",
    beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "StarImageInput") return;
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
