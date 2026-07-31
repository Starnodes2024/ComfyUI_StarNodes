// Dynamic input handler for Star Video Loop
// Adds/removes video inputs as needed for joining multiple videos
import { app } from "../../../../scripts/app.js";

function updateVideoInputs(node) {
    console.log("[StarVideoLoopDynamic] updateInputs called", node?.title || node?.type);
    if (!node || !Array.isArray(node.inputs)) return;
    if (node._updatingInputs) return;
    node._updatingInputs = true;
    
    try {
        // Collect all numbered video inputs: video 1, video 2, ...
        let videoInputs = [];
        for (let i = 0; i < node.inputs.length; i++) {
            const inp = node.inputs[i];
            if (inp && inp.type === "IMAGE" && /^video \d+$/.test(inp.name)) {
                const num = parseInt(inp.name.split(" ")[1]);
                if (!isNaN(num)) {
                    videoInputs.push({ input: inp, num, index: i });
                }
            }
        }

        // Sort by number
        videoInputs.sort((a, b) => a.num - b.num);

        // Ensure at least one numbered video input exists
        if (videoInputs.length === 0) {
            node.addInput("video 1", "IMAGE");
            return; // Re-run on next connection change
        }

        // If last numbered video input is connected, add a new one
        const last = videoInputs[videoInputs.length - 1];
        if (last && last.input.link !== null) {
            const nextIdx = last.num + 1;
            if (!node.inputs.some(inp => inp.name === `video ${nextIdx}`)) {
                node.addInput(`video ${nextIdx}`, "IMAGE");
            }
        }

        // Remove trailing unconnected numbered video inputs, keeping at least video 1
        // Work backwards to avoid index issues
        let toRemove = [];
        for (let i = videoInputs.length - 1; i > 0; i--) {
            const vidData = videoInputs[i];
            if (vidData.input.link === null) {
                toRemove.push(vidData.index);
            } else {
                break;
            }
        }
        
        // Remove in reverse order to maintain indices
        toRemove.sort((a, b) => b - a);
        for (const idx of toRemove) {
            node.removeInput(idx);
        }

        if (node.graph) node.graph.change();
    } catch (e) {
        console.error("[StarVideoLoopDynamic] Error in updateInputs:", e);
    } finally {
        node._updatingInputs = false;
    }
}

app.registerExtension({
    name: "StarVideoLoopDynamic",
    beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "StarVideoLoop") return;
        console.log("[StarVideoLoopDynamic] beforeRegisterNodeDef for", nodeData.name);
        
        const origOnConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function(type, index, connected, link_info) {
            if (origOnConnectionsChange)
                origOnConnectionsChange.apply(this, arguments);
            updateVideoInputs(this);
        };
        
        // Also update on node creation
        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            if (origOnNodeCreated) origOnNodeCreated.apply(this, arguments);
            updateVideoInputs(this);
        };
    }
});
