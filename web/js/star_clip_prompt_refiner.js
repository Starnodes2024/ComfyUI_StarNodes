// Dynamic image slots for Star CLIP Prompt Refiner:
// connecting the last image slot adds a new one (max 4),
// disconnecting removes trailing empty slots again.
import { app } from "../../../../scripts/app.js";

const MAX_IMAGES = 4;

function updateInputs(node) {
    if (!node || !Array.isArray(node.inputs)) return;
    if (node._updatingStarClipImages) return;
    node._updatingStarClipImages = true;
    try {
        // Collect numbered image inputs: image1..image4
        let imageInputs = [];
        for (let i = 0; i < node.inputs.length; i++) {
            const inp = node.inputs[i];
            if (inp && inp.type === "IMAGE" && /^image\d+$/.test(inp.name)) {
                const num = parseInt(inp.name.replace("image", ""));
                if (!isNaN(num)) imageInputs.push({ input: inp, num, index: i });
            }
        }
        imageInputs.sort((a, b) => a.num - b.num);

        // Ensure at least image1 exists
        if (imageInputs.length === 0) {
            node.addInput("image1", "IMAGE");
            return;
        }

        // If the last image slot is connected and we are below the max, add a new slot
        const last = imageInputs[imageInputs.length - 1];
        if (last.input.link !== null && last.num < MAX_IMAGES) {
            const nextIdx = last.num + 1;
            if (!node.inputs.some(inp => inp.name === `image${nextIdx}`)) {
                node.addInput(`image${nextIdx}`, "IMAGE");
            }
        }

        // Remove trailing unconnected slots, always keeping at least image1
        let toRemove = [];
        for (let i = imageInputs.length - 1; i > 0; i--) {
            const imgData = imageInputs[i];
            if (imgData.input.link === null) {
                toRemove.push(imgData.index);
            } else {
                break;
            }
        }
        toRemove.sort((a, b) => b - a);
        for (const idx of toRemove) {
            node.removeInput(idx);
        }

        if (node.graph) node.graph.change();
    } finally {
        node._updatingStarClipImages = false;
    }
}

app.registerExtension({
    name: "StarClipPromptRefinerDynamic",
    beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!nodeData || nodeData.name !== "StarClipPromptRefiner") return;

        const origOnConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
            if (origOnConnectionsChange) {
                origOnConnectionsChange.apply(this, arguments);
            }
            updateInputs(this);
        };

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (origOnNodeCreated) origOnNodeCreated.apply(this, arguments);
            updateInputs(this);
        };
    },
});
