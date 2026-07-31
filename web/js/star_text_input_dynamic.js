import { app } from "../../../../scripts/app.js";

function ensureTextField(node, name) {
    if (!node) return;

    // Ensure STRING widget exists for this text field
    if (!node.widgets) node.widgets = [];
    let widget = node.widgets.find((w) => w && w.name === name);
    if (!widget) {
        widget = node.addWidget("STRING", name, "", () => {
            updateInputs(node);
        });
    }

    // Ensure a STRING input socket exists for this text field so other nodes can connect
    if (!node.inputs) node.inputs = [];
    let inputIndex = node.inputs.findIndex((i) => i && i.name === name);
    if (inputIndex === -1) {
        node.addInput(name, "STRING");
        inputIndex = node.inputs.length - 1;
    }

    // Link the widget to the corresponding input slot so the connection dot appears next to it
    if (widget && typeof inputIndex === "number" && inputIndex >= 0) {
        widget.linkedSlot = inputIndex;
    }

    return widget;
}

function getTextEntries(node) {
    const entries = [];
    if (!node) return entries;

    const widgets = node.widgets || [];

    for (let i = 0; i < widgets.length; i++) {
        const w = widgets[i];
        if (!w || typeof w.name !== "string") continue;
        if (!/^text\d+$/.test(w.name)) continue;

        const num = parseInt(w.name.replace("text", ""));
        if (isNaN(num)) continue;

        entries.push({
            name: w.name,
            num,
            widget: w,
            input: null,
        });
    }

    // Sort by numeric suffix
    entries.sort((a, b) => a.num - b.num);
    return entries;
}

function getNumStrings(node) {
    if (!node || !node.widgets) return 1;
    const w = node.widgets.find((w) => w && w.name === "num_strings");
    if (!w) return 1;

    const raw = w.value;
    let v = parseInt(raw);
    if (isNaN(v) || v < 1) v = 1;
    if (v > 64) v = 64;
    return v;
}

function updateInputs(node) {
    if (!node) return;
    if (node._updatingInputs) return;
    node._updatingInputs = true;

    try {
        console.log("[StarTextInputDynamic] updateInputs for", node.title || node.type);
        const count = getNumStrings(node);
        console.log("[StarTextInputDynamic] num_strings =", count);

        // Ensure text1..textN exist
        for (let i = 1; i <= count; i++) {
            const name = `text${i}`;
            ensureTextField(node, name);
        }

        // Remove any textM where M > count
        const entries = getTextEntries(node);
        const toRemoveNames = [];
        for (const e of entries) {
            if (e.num > count) {
                toRemoveNames.push(e.name);
            }
        }

        // Remove widgets for textM where M > count
        if (toRemoveNames.length > 0 && node.widgets) {
            for (let i = node.widgets.length - 1; i >= 0; i--) {
                const w = node.widgets[i];
                if (w && toRemoveNames.includes(w.name)) {
                    node.widgets.splice(i, 1);
                }
            }
        }

        // Remove input sockets for textM where M > count
        if (toRemoveNames.length > 0 && node.inputs) {
            for (let i = node.inputs.length - 1; i >= 0; i--) {
                const inp = node.inputs[i];
                if (inp && toRemoveNames.includes(inp.name)) {
                    node.inputs.splice(i, 1);
                }
            }
        }
    } finally {
        node._updatingInputs = false;
    }
}

app.registerExtension({
    name: "StarTextInputDynamic",
    beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!nodeData || nodeData.name !== "StarTextInput") return;

        const origOnConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
            if (origOnConnectionsChange) {
                try {
                    origOnConnectionsChange.apply(this, arguments);
                } catch (e) {
                    console.error("[StarTextInputDynamic] Error in origOnConnectionsChange", e);
                }
            }
            updateInputs(this);
        };

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (origOnNodeCreated) origOnNodeCreated.apply(this, arguments);
            ensureTextField(this, "text1");
            updateInputs(this);
        };

        // Fallback: periodically sync all StarTextInput nodes to num_strings
        if (!app._starTextInputDynamicPollerStarted) {
            app._starTextInputDynamicPollerStarted = true;
            setInterval(() => {
                const graph = app.graph;
                if (!graph || !graph._nodes) return;
                for (const n of graph._nodes) {
                    if (!n || n.type !== "StarTextInput") continue;
                    updateInputs(n);
                }
            }, 500);
        }
    },
});
