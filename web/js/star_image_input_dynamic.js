// Dynamic input handler for Star Image Input (Dynamic)
// Adds/removes image/mask inputs as needed
import { app } from "../../../../scripts/app.js";

function updateInputs(node) {
    // Find all Image inputs (with capital I and space)
    let imageInputs = [];
    for (let i = 0; i < node.inputs.length; i++) {
        const input = node.inputs[i];
        if (input && input.name.startsWith("Image ")) {
            const num = parseInt(input.name.split(" ")[1]);
            if (!isNaN(num)) {
                imageInputs.push({ input, num, index: i });
            }
        }
    }
    
    // Sort by number
    imageInputs.sort((a, b) => a.num - b.num);
    
    // Ensure at least "Image 1" exists
    if (imageInputs.length === 0) {
        node.addInput("Image 1", "IMAGE");
        imageInputs = [{ input: node.inputs[node.inputs.length - 1], num: 1, index: node.inputs.length - 1 }];
    }
    
    // If last image input is connected, add new image input
    const lastImage = imageInputs[imageInputs.length - 1];
    if (lastImage && lastImage.input.link !== null) {
        const nextNum = lastImage.num + 1;
        if (!node.inputs.some(inp => inp.name === `Image ${nextNum}`)) {
            node.addInput(`Image ${nextNum}`, "IMAGE");
        }
    }
    
    // Remove trailing unconnected inputs (keep at least Image 1)
    for (let i = imageInputs.length - 1; i > 0; i--) {
        const imgData = imageInputs[i];
        if (imgData.input.link === null) {
            node.removeInput(imgData.index);
        } else {
            break;
        }
    }
    
    // Make first image required, others optional
    for (let i = 0; i < node.inputs.length; i++) {
        if (node.inputs[i].name === "Image 1") {
            node.inputs[i].optional = false;
        } else if (node.inputs[i].name.startsWith("Image ")) {
            node.inputs[i].optional = true;
        }
    }
}

app.registerExtension({
    name: "StarImageInputDynamic",
    beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "StarImageSwitch") return;
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
