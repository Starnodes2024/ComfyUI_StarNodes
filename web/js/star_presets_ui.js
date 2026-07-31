// StarNodes preset UI helper
// Syncs preset dropdowns with slider values for StarBlackWhite and StarHDREffects.

(function () {
    const BLACK_WHITE_NODE = "StarBlackWhite";
    const HDR_NODE = "StarHDREffects";

    function applyBWPreset(node, presetName) {
        const presets = {
            "Neutral Luminance": { red_weight: 0.3, green_weight: 0.59, blue_weight: 0.11 },
            "Skin High Red": { red_weight: 0.7, green_weight: 0.4, blue_weight: 0.1 },
            "Sky High Blue": { red_weight: 0.2, green_weight: 0.4, blue_weight: 0.8 },
            "Green Filter": { red_weight: 0.2, green_weight: 0.8, blue_weight: 0.1 },
            "High Contrast": { red_weight: 0.5, green_weight: 0.5, blue_weight: 0.2 },
            "Soft Matte": { red_weight: 0.35, green_weight: 0.55, blue_weight: 0.1 },
            "Film Grain Subtle": { red_weight: 0.3, green_weight: 0.59, blue_weight: 0.11, grain_strength: 0.15, grain_density: 0.6, grain_size: 3 },
            "Film Grain Strong": { red_weight: 0.3, green_weight: 0.55, blue_weight: 0.15, grain_strength: 0.35, grain_density: 0.9, grain_size: 6 },
        };
        const p = presets[presetName];
        if (!p) return;

        function setWidget(name, value) {
            const w = node.widgets?.find((w) => w.name === name);
            if (!w) return;
            w.value = value;
            if (typeof w.callback === "function") {
                try { w.callback(w.value, node, w); } catch (e) { /* ignore */ }
            }
        }

        setWidget("red_weight", p.red_weight);
        setWidget("green_weight", p.green_weight);
        setWidget("blue_weight", p.blue_weight);
        if (p.grain_strength !== undefined) setWidget("grain_strength", p.grain_strength);
        if (p.grain_density !== undefined) setWidget("grain_density", p.grain_density);
        if (p.grain_size !== undefined) setWidget("grain_size", p.grain_size);
    }

    function applyHDRPreset(node, presetName) {
        const presets = {
            "Natural HDR": { strength: 0.6, radius: 10, local_contrast: 1.0, global_contrast: 1.0, saturation: 1.0, highlight_protection: 0.5 },
            "Strong HDR": { strength: 1.0, radius: 12, local_contrast: 1.8, global_contrast: 1.1, saturation: 1.1, highlight_protection: 0.6 },
            "Soft HDR Matte": { strength: 0.5, radius: 8, local_contrast: 0.9, global_contrast: 0.8, saturation: 0.9, highlight_protection: 0.7 },
            "Detail Boost": { strength: 0.8, radius: 6, local_contrast: 2.0, global_contrast: 1.0, saturation: 1.0, highlight_protection: 0.4 },
            "Sky Protect Highlights": { strength: 0.7, radius: 10, local_contrast: 1.2, global_contrast: 0.95, saturation: 0.95, highlight_protection: 0.9 },
        };
        const p = presets[presetName];
        if (!p) return;

        function setWidget(name, value) {
            const w = node.widgets?.find((w) => w.name === name);
            if (!w) return;
            w.value = value;
            if (typeof w.callback === "function") {
                try { w.callback(w.value, node, w); } catch (e) { /* ignore */ }
            }
        }

        setWidget("strength", p.strength);
        setWidget("radius", p.radius);
        setWidget("local_contrast", p.local_contrast);
        setWidget("global_contrast", p.global_contrast);
        setWidget("saturation", p.saturation);
        setWidget("highlight_protection", p.highlight_protection);
    }

    function enhanceNode(node) {
        if (!node || !node.widgets) return;
        if (node.comfy_preset_enhanced) return;

        if (node.type === BLACK_WHITE_NODE || node.type === HDR_NODE) {
            const presetWidget = node.widgets.find((w) => w.name === "preset");
            if (!presetWidget) return;

            const originalCallback = presetWidget.callback;
            presetWidget.callback = function (value) {
                if (typeof originalCallback === "function") {
                    try { originalCallback(value); } catch (e) { /* ignore */ }
                }

                if (node.type === BLACK_WHITE_NODE) {
                    applyBWPreset(node, value);
                } else if (node.type === HDR_NODE) {
                    applyHDRPreset(node, value);
                }
            };

            node.comfy_preset_enhanced = true;
        }
    }

    // Hook into node creation when ComfyUI is ready
    const origAddNode = LiteGraph?.LGraph?.prototype?.add; // fallback
    function patchGraph() {
        if (!window.app || !window.app.graph) return;
        const graph = window.app.graph;
        if (graph._starnodes_preset_patched) return;
        graph._starnodes_preset_patched = true;

        const oldAdd = graph.add.bind(graph);
        graph.add = function (node) {
            const res = oldAdd(node);
            try { enhanceNode(node); } catch (e) { /* ignore */ }
            return res;
        };

        // Also enhance already existing nodes when script loads
        if (graph._nodes) {
            for (const n of graph._nodes) {
                try { enhanceNode(n); } catch (e) { /* ignore */ }
            }
        }
    }

    function waitForApp() {
        if (window.app && window.app.graph && window.LiteGraph) {
            patchGraph();
        } else {
            setTimeout(waitForApp, 500);
        }
    }

    waitForApp();
})();
