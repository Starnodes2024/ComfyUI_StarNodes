import { app } from "../../../../scripts/app.js";

// ---------------------------------------------------------------------------
// Drag-link suggestions for the StarNodes sampler "options" connector.
//
// When the user drags the options output of a Star option node (type
// STARNODES_OPTIONS) onto the empty canvas, the node search suggests both
// option nodes - same mechanism StarNodes already uses for the IMAGE type
// (LiteGraph.slot_types_default_out / slot_types_default_in).
// ---------------------------------------------------------------------------

const SPLIT_CLASS_KEY = "⭐ Star Split Sampler Option";
const ZIT_CLASS_KEY = "⭐ Star Distilled Optimizer (QWEN/ZIT)";

function promoteSlotType(slotType, nodeClass, direction) {
    const lg = globalThis.LiteGraph;
    if (!lg) return;

    const mapName = direction === "out"
        ? "slot_types_default_out"
        : "slot_types_default_in";

    if (!lg[mapName]) lg[mapName] = {};
    if (!lg[mapName][slotType]) lg[mapName][slotType] = [];

    const arr = lg[mapName][slotType];

    for (let i = arr.length - 1; i >= 0; i--) {
        const item = arr[i];
        const val = typeof item === "string" ? item : (item?.value || item?.content);
        if (val === nodeClass) {
            arr.splice(i, 1);
        }
    }

    arr.unshift(nodeClass);
}

app.registerExtension({
    name: "StarNodesV2.SamplerOptionSuggestions",

    setup() {
        setTimeout(() => {
            // Dragging an options OUTPUT to the canvas suggests both option nodes.
            promoteSlotType("STARNODES_OPTIONS", SPLIT_CLASS_KEY, "out");
            promoteSlotType("STARNODES_OPTIONS", ZIT_CLASS_KEY, "out");

            // Dragging FROM an options INPUT (wildcard type "*" on ⭐ StarSampler
            // and ⭐ Star SD Upscale Refiner) suggests both option nodes as well.
            promoteSlotType("*", SPLIT_CLASS_KEY, "in");
            promoteSlotType("*", ZIT_CLASS_KEY, "in");
        }, 100);
    },
});
