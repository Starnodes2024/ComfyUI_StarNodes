import { app } from "../../../../scripts/app.js";

// ---------------------------------------------------------------------------
// ⭐ Star Custom Sigmas - preset <-> text field sync.
//
// Picking a preset (8/12/16 Steps) writes its values into the sigmas text
// widget so the user sees exactly what runs. Editing the text by hand flips
// the dropdown to "Custom" so the typed values win.
// ---------------------------------------------------------------------------

const SIGMA_PRESETS = {
    "8 Steps": "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0",
    "10 Steps": "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.870114, 0.845148, 0.704534, 0.559551, 0.414568, 0.0",
    "12 Steps": "1.0, 0.985635, 0.970362, 0.95418, 0.935811, 0.916111, 0.89508, 0.861792, 0.798277, 0.704534, 0.511223, 0.276379, 0.0",
    "16 Steps": "1.0, 0.989227, 0.978453, 0.966317, 0.954180, 0.940403, 0.926626, 0.910853, 0.895080, 0.870114, 0.845148, 0.774841, 0.704534, 0.559551, 0.414568, 0.207284, 0.0",
};

app.registerExtension({
    name: "StarNodesV2.CustomSigmas",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "StarCustomSigmas") return;

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (onCreated) onCreated.apply(this, arguments);

            const node = this;
            const presetWidget = node.widgets?.find((w) => w.name === "preset");
            const sigmasWidget = node.widgets?.find((w) => w.name === "sigmas");
            if (!presetWidget || !sigmasWidget) return;

            let filling = false;

            const origPresetCallback = presetWidget.callback;
            presetWidget.callback = function () {
                if (origPresetCallback) origPresetCallback.apply(this, arguments);
                const values = SIGMA_PRESETS[presetWidget.value];
                if (values !== undefined) {
                    filling = true;
                    sigmasWidget.value = values;
                    filling = false;
                    node.setDirtyCanvas(true, true);
                }
            };

            const origSigmasCallback = sigmasWidget.callback;
            sigmasWidget.callback = function () {
                if (origSigmasCallback) origSigmasCallback.apply(this, arguments);
                if (!filling && presetWidget.value !== "Custom") {
                    presetWidget.value = "Custom";
                    node.setDirtyCanvas(true, true);
                }
            };
        };
    },
});
