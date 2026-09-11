// Dynamic input handler for ⭐ Star Video Joiner
// Grows/prunes image_N, video_N and audio_N input sockets independently,
// up to MAX_PER_GROUP each. Same pattern as star_psd_saver_dynamic.js.
import { app } from "../../../../scripts/app.js";

const MAX_PER_GROUP = 20;
const GROUPS = [
    { prefix: "image_", type: "IMAGE" },
    { prefix: "video_", type: "STAR_FILENAMES" },
    { prefix: "audio_", type: "AUDIO" },
];

// Each of image_N / video_N / audio_N is grown and pruned purely from its
// own connection state (max connected index + 1 empty slot). This keeps the
// three groups fully independent: connecting/disconnecting a slot in one
// group must never touch the slots of another group.
function updateGroup(node, prefix, type) {
    const entries = [];
    for (let i = 0; i < node.inputs.length; i++) {
        const inp = node.inputs[i];
        if (!inp || typeof inp.name !== "string") continue;
        if (!inp.name.startsWith(prefix)) continue;
        const idx = parseInt(inp.name.slice(prefix.length));
        if (!isNaN(idx)) entries.push({ idx, inp });
    }

    let maxConnected = 0;
    for (const entry of entries) {
        if (entry.inp.link !== null && entry.idx > maxConnected) {
            maxConnected = entry.idx;
        }
    }
    const desiredLast = Math.min(maxConnected + 1, MAX_PER_GROUP);

    const existingIdx = new Set(entries.map(e => e.idx));
    for (let i = 1; i <= desiredLast; i++) {
        if (!existingIdx.has(i)) {
            node.addInput(`${prefix}${i}`, type);
        }
    }

    for (let i = node.inputs.length - 1; i >= 0; i--) {
        const inp = node.inputs[i];
        if (!inp || typeof inp.name !== "string") continue;
        if (!inp.name.startsWith(prefix)) continue;
        const idx = parseInt(inp.name.slice(prefix.length));
        if (!isNaN(idx) && idx > desiredLast) {
            node.removeInput(i);
        }
    }
}

function updateInputs(node) {
    if (!node || !Array.isArray(node.inputs)) return;
    if (node._starVideoJoinerUpdating) return;
    node._starVideoJoinerUpdating = true;
    try {
        for (const { prefix, type } of GROUPS) {
            updateGroup(node, prefix, type);
        }
        if (node.graph) node.graph.change();
    } finally {
        node._starVideoJoinerUpdating = false;
    }
}

app.registerExtension({
    name: "StarNodes.StarVideoJoinerDynamic",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "StarVideoJoiner") return;

        const origOnConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (type, index, connected, linkInfo) {
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
