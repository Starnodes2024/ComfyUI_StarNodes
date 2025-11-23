// Dynamic input handler for Star Latent Input (Dynamic)
// Adds/removes latent/mask inputs as needed
import { app } from "../../../../scripts/app.js";

function updateInputs(node) {
    // Find all Latent inputs (with capital L and space)
    let latentInputs = [];
    for (let i = 0; i < node.inputs.length; i++) {
        const input = node.inputs[i];
        if (input && input.name.startsWith("Latent ")) {
            const num = parseInt(input.name.split(" ")[1]);
            if (!isNaN(num)) {
                latentInputs.push({ input, num, index: i });
            }
        }
    }
    
    // Sort by number
    latentInputs.sort((a, b) => a.num - b.num);
    
    // Ensure at least "Latent 1" exists
    if (latentInputs.length === 0) {
        node.addInput("Latent 1", "LATENT");
        latentInputs = [{ input: node.inputs[node.inputs.length - 1], num: 1, index: node.inputs.length - 1 }];
    }
    
    // If last latent input is connected, add new latent input
    const lastLatent = latentInputs[latentInputs.length - 1];
    if (lastLatent && lastLatent.input.link !== null) {
        const nextNum = lastLatent.num + 1;
        if (!node.inputs.some(inp => inp.name === `Latent ${nextNum}`)) {
            node.addInput(`Latent ${nextNum}`, "LATENT");
        }
    }
    
    // Remove trailing unconnected inputs (keep at least Latent 1)
    for (let i = latentInputs.length - 1; i > 0; i--) {
        const latentData = latentInputs[i];
        if (latentData.input.link === null) {
            node.removeInput(latentData.index);
        } else {
            break;
        }
    }
    
    // Make first latent required, others optional
    for (let i = 0; i < node.inputs.length; i++) {
        if (node.inputs[i].name === "Latent 1") {
            node.inputs[i].optional = false;
        } else if (node.inputs[i].name.startsWith("Latent ")) {
            node.inputs[i].optional = true;
        }
    }
}

app.registerExtension({
    name: "StarLatentInputDynamic",
    beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "StarLatentSwitch") return;
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
