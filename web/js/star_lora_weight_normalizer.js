import { app } from "../../../../scripts/app.js";

function getMaxLoraIndex(node) {
    let maxIdx = 0;
    if (!node.widgets) return maxIdx;
    for (const w of node.widgets) {
        if (!w || typeof w.name !== "string") continue;
        if (w.name.startsWith("lora") && w.name.endsWith("_name")) {
            const idx = parseInt(w.name.replace("lora", "").replace("_name", ""));
            if (!isNaN(idx)) maxIdx = Math.max(maxIdx, idx);
        }
    }
    return maxIdx;
}

function findLoraNameWidget(node, idx) {
    if (!node.widgets) return null;
    const target = `lora${idx}_name`;
    return node.widgets.find(w => w && w.name === target) || null;
}

function addLoraGroup(node) {
    if (!node.widgets) node.widgets = [];

    // Temporarily remove control buttons so new widgets stay above them
    const controlNames = new Set(["__star_lora_add__", "__star_lora_remove__"]);
    const controlWidgets = [];
    for (let i = node.widgets.length - 1; i >= 0; i--) {
        const w = node.widgets[i];
        const key = w && w.options && w.options.name;
        if (key && controlNames.has(key)) {
            controlWidgets.unshift(w);
            node.widgets.splice(i, 1);
        }
    }

    // Determine new index
    const maxIdx = getMaxLoraIndex(node);
    const newIdx = maxIdx > 0 ? maxIdx + 1 : 1;

    // Get reference values list from first lora widget, if present
    let values = [];
    const first = findLoraNameWidget(node, 1);
    if (first && first.options && Array.isArray(first.options.values)) {
        values = [...first.options.values];
    }

    // Add widgets for the new group (model only - no CLIP)
    const nameWidget = node.addWidget("combo", `lora${newIdx}_name`, values.length > 0 ? values[0] : "None", () => {}, { values });
    const smWidget = node.addWidget("number", `strength${newIdx}_model`, 1.0, () => {}, {
        min: -100.0,
        max: 100.0,
        step: 0.01,
    });

    // Re-attach control buttons at the bottom
    for (const cw of controlWidgets) {
        node.widgets.push(cw);
    }

    if (node.graph) node.graph.change();
    return nameWidget;
}

function removeLastLoraGroup(node) {
    if (!node.widgets) return;

    const maxIdx = getMaxLoraIndex(node);
    if (maxIdx <= 1) return; // keep first fixed

    const toRemove = new Set([
        `lora${maxIdx}_name`,
        `strength${maxIdx}_model`,
    ]);

    for (let i = node.widgets.length - 1; i >= 0; i--) {
        const w = node.widgets[i];
        if (w && toRemove.has(w.name)) {
            node.widgets.splice(i, 1);
        }
    }

    if (node.graph) node.graph.change();
}

function ensureFirstGroup(node) {
    if (!node.widgets) node.widgets = [];
    const first = findLoraNameWidget(node, 1);
    if (!first) {
        addLoraGroup(node);
    }
}

function autoExpandOnChange(node) {
    const maxIdx = getMaxLoraIndex(node);
    if (maxIdx <= 0) return;

    const last = findLoraNameWidget(node, maxIdx);
    if (!last) return;

    // When last slot is changed away from "None", add a new one
    if (last.value && last.value !== "None") {
        const next = findLoraNameWidget(node, maxIdx + 1);
        if (!next) {
            addLoraGroup(node);
        }
    }
}

app.registerExtension({
    name: "StarNodes.LoraWeightNormalizer",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "StarLoraWeightNormalizer") return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (origOnNodeCreated) origOnNodeCreated.apply(this, arguments);
            ensureFirstGroup(this);

            // Add inline + / - buttons in the node UI
            if (!this.widgets) this.widgets = [];

            const hasAddBtn = this.widgets.some(
                (w) => w && w.options && w.options.name === "__star_lora_add__"
            );
            const hasRemoveBtn = this.widgets.some(
                (w) => w && w.options && w.options.name === "__star_lora_remove__"
            );

            if (!hasAddBtn) {
                this.addWidget(
                    "button",
                    "＋ LoRA",
                    null,
                    () => {
                        ensureFirstGroup(this);
                        addLoraGroup(this);
                    },
                    { name: "__star_lora_add__" }
                );
            }

            if (!hasRemoveBtn) {
                this.addWidget(
                    "button",
                    "－ LoRA",
                    null,
                    () => {
                        removeLastLoraGroup(this);
                    },
                    { name: "__star_lora_remove__" }
                );
            }
        };

        const origOnWidgetChanged = nodeType.prototype.onWidgetChanged;
        nodeType.prototype.onWidgetChanged = function (widget, value, prev) {
            if (origOnWidgetChanged) origOnWidgetChanged.apply(this, arguments);
            if (!widget || typeof widget.name !== "string") return;
            if (widget.name.startsWith("lora") && widget.name.endsWith("_name")) {
                autoExpandOnChange(this);
            }
        };

        const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (_, options) {
            if (origGetExtraMenuOptions) origGetExtraMenuOptions.apply(this, arguments);

            options.push({
                content: "Add LoRA",
                callback: () => {
                    ensureFirstGroup(this);
                    addLoraGroup(this);
                },
            });

            options.push({
                content: "Remove Last LoRA",
                callback: () => {
                    removeLastLoraGroup(this);
                },
            });
        };
    },
});
