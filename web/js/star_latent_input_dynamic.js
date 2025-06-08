// Dynamic input handler for Star Latent Input (Dynamic)
// Adds/removes latent/mask inputs as needed
import { app } from "../../../scripts/app.js";

function updateInputs(node) {
    // Find all latent/mask input pairs
    let pairs = [];
    for (let i = 0; i < node.inputs.length; i += 2) {
        const latent = node.inputs[i];
        const mask = node.inputs[i + 1];
        if (latent && mask && latent.name.startsWith("latent") && mask.name.startsWith("mask")) {
            pairs.push([latent, mask]);
        }
    }
    // Ensure at least one latent and mask exist
    if (pairs.length === 0) {
        node.addInput("latent1", "LATENT");
        node.addInput("mask1", "MASK");
        pairs.push([node.inputs[node.inputs.length - 2], node.inputs[node.inputs.length - 1]]);
    }
    // If last latent input is connected, add new latent/mask input
    if (pairs[pairs.length - 1][0].link !== null) {
        const idx = pairs.length + 1;
        node.addInput(`latent${idx}`, "LATENT");
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
    // Make first latent required (if possible)
    if (node.inputs[0]) node.inputs[0].optional = false;
    // All other latents and all masks are optional
    for (let i = 1; i < node.inputs.length; i++) {
        node.inputs[i].optional = true;
    }
}

app.registerExtension({
    name: "StarLatentInputDynamic",
    beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "StarLatentInput") return;
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
