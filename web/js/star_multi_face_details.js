import { app } from "../../../scripts/app.js";

// Live search filter for the LoRA dropdowns of Star Multi Face Details.
app.registerExtension({
    name: "starnodes.multiFaceDetails",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "StarMultiFaceDetails") return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            const search = this.widgets.find((w) => w.name === "lora_search");
            const loraWidgets = this.widgets.filter((w) => /^lora_\d+$/.test(w.name));
            if (!search || !loraWidgets.length) return;
            const allValues = loraWidgets[0].options.values.slice();
            const applyFilter = () => {
                const query = String(search.value || "").toLowerCase();
                for (const widget of loraWidgets) {
                    // Keep the current selection visible even when it does not match.
                    widget.options.values = allValues.filter(
                        (v) => v === "none" || v === widget.value || v.toLowerCase().includes(query)
                    );
                }
                this.setDirtyCanvas(true, true);
            };
            const original = search.callback;
            search.callback = function (...args) {
                original?.apply(this, args);
                applyFilter();
            };
        };
    },
});
