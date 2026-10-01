import { app } from "../../../../scripts/app.js";

// ===========================================================================
//  ⭐ Star JSON Preview — collapsible JSON structure tree inside the node
// ===========================================================================

const KEY = "StarJsonPreview";
const MAX_CHILDREN = 1000;   // per object/array level
const MAX_LEAF = 400;        // chars shown for a scalar before truncating

const CSS = `
.sjp-container { display:flex; flex-direction:column; width:100%; min-height:60px;
    background:#1a1a2e; border:1px solid #3d124d; border-radius:4px;
    box-sizing:border-box; font-family:ui-monospace, monospace; font-size:11px;
    color:#c8c8c8; line-height:1.5; overflow:hidden; }
.sjp-toolbar { display:flex; gap:4px; align-items:center; padding:4px 6px;
    border-bottom:1px solid #3d124d; flex:0 0 auto; }
.sjp-btn { background:#19124d; color:#c8c8c8; border:1px solid #3d124d;
    border-radius:3px; font:inherit; padding:1px 8px; cursor:pointer; }
.sjp-btn:hover { background:#2a1a6e; }
.sjp-status { padding:4px 8px; white-space:pre-wrap; word-break:break-word;
    border-bottom:1px solid #3d124d; flex:0 0 auto; color:#9ecbff; }
.sjp-status[data-kind="error"] { color:#ff8a80; }
.sjp-tree { overflow:auto; max-height:460px; padding:4px 8px; flex:1 1 auto;
    white-space:pre-wrap; word-break:break-word; }
.sjp-tree details { margin-left:14px; }
.sjp-tree > details { margin-left:0; }
.sjp-tree summary { cursor:pointer; list-style:none; user-select:none; }
.sjp-tree summary::before { content:"▸ "; color:#7a6aaf; }
.sjp-tree details[open] > summary::before { content:"▾ "; }
.sjp-tree summary:hover { color:#fff; }
.sjp-row { margin-left:14px; }
.sjp-key { color:#9ecbff; }
.sjp-punct { color:#7a6aaf; }
.sjp-str { color:#a5d6a7; }
.sjp-num { color:#ffb74d; }
.sjp-bool { color:#f48fb1; }
.sjp-null { color:#f48fb1; font-style:italic; }
.sjp-count { color:#7a6aaf; font-style:italic; }
.sjp-empty { color:#666; font-style:italic; }
.sjp-more { color:#7a6aaf; font-style:italic; margin-left:14px; }
`;

function el(tag, className, text) {
    const e = document.createElement(tag);
    if (className) e.className = className;
    if (text !== undefined) e.textContent = text;
    return e;
}

function leafText(value) {
    let text = JSON.stringify(value);
    if (text === undefined) text = String(value); // e.g. undefined → "undefined"
    if (text.length > MAX_LEAF) text = text.slice(0, MAX_LEAF) + `… (${text.length - MAX_LEAF} more chars)`;
    return text;
}

function leafClass(value) {
    if (value === null) return "sjp-null";
    if (typeof value === "boolean") return "sjp-bool";
    if (typeof value === "number") return "sjp-num";
    return "sjp-str";
}

function containerSummary(value) {
    if (Array.isArray(value)) return `[…] ${value.length} item${value.length === 1 ? "" : "s"}`;
    return `{…} ${Object.keys(value).length} key${Object.keys(value).length === 1 ? "" : "s"}`;
}

function appendLeaf(parent, key, value) {
    const row = el("div", "sjp-row");
    if (key !== null) row.append(el("span", "sjp-key", key), el("span", "sjp-punct", ": "));
    row.append(el("span", leafClass(value), leafText(value)));
    parent.appendChild(row);
}

function appendValue(parent, key, value) {
    if (value === null || typeof value !== "object") { appendLeaf(parent, key, value); return; }

    const details = el("details");
    details._sjpValue = value;
    const summary = el("summary");
    if (key !== null) summary.append(el("span", "sjp-key", key), el("span", "sjp-punct", ": "));
    summary.append(el("span", "sjp-count", containerSummary(value)));
    details.appendChild(summary);

    // Children are rendered lazily on first expand so huge documents stay cheap.
    details.addEventListener("toggle", () => populate(details, value));
    parent.appendChild(details);
}

function populate(details, value) {
    if (details.dataset.populated) return;
    details.dataset.populated = "1";
    const entries = Array.isArray(value)
        ? value.map((v, i) => [String(i), v])
        : Object.keys(value).map((k) => [k, value[k]]);
    for (const [key, item] of entries.slice(0, MAX_CHILDREN)) appendValue(details, key, item);
    if (entries.length > MAX_CHILDREN)
        details.appendChild(el("div", "sjp-more", `… and ${entries.length - MAX_CHILDREN} more`));
}

function setAllOpen(root, open) {
    let changed = true;
    while (open && changed) {
        changed = false;
        for (const d of root.querySelectorAll("details:not([open])")) {
            populate(d, d._sjpValue);
            d.open = true;
            changed = true;
        }
    }
    if (!open) for (const d of root.querySelectorAll("details[open]")) d.open = false;
}

function render(container, status, raw, info) {
    status.dataset.kind = "info";
    status.textContent = info || "";
    container.replaceChildren();

    let value;
    try {
        value = JSON.parse(raw);
    } catch (error) {
        status.dataset.kind = "error";
        status.textContent = `Invalid JSON: ${error.message}`;
        container.appendChild(el("div", "sjp-empty", "The connected string is not valid JSON — fix the source and run again."));
        return;
    }

    const top = el("details");
    top.open = true;
    const summary = el("summary", "", "root ");
    summary.appendChild(el("span", "sjp-count", typeof value === "object" && value !== null ? containerSummary(value) : leafText(value)));
    top.appendChild(summary);
    container.appendChild(top);

    if (typeof value === "object" && value !== null) {
        top._sjpValue = value;
        populate(top, value);
    } else {
        appendLeaf(top, null, value);
    }
}

function ensureStyle() {
    if (!document.getElementById("sjp-style")) {
        const style = el("style");
        style.id = "sjp-style";
        style.textContent = CSS;
        document.head.appendChild(style);
    }
}

app.registerExtension({
    name: "StarNodes.StarJsonPreview",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== KEY) return;
        ensureStyle();

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            const container = el("div", "sjp-container");
            container.dataset.captureWheel = "true";
            for (const type of ["pointerdown", "mousedown", "dblclick", "wheel", "keydown", "keyup"])
                container.addEventListener(type, (e) => e.stopPropagation());

            const toolbar = el("div", "sjp-toolbar");
            const status = el("div", "sjp-status", "Connect a JSON string and run to preview its structure…");
            const tree = el("div", "sjp-tree");
            tree.appendChild(el("div", "sjp-empty", "Waiting for input…"));

            const state = { raw: "", info: "" };
            const btn = (label, fn, title) => {
                const b = el("button", "sjp-btn", label);
                b.type = "button";
                b.title = title || label;
                b.addEventListener("click", fn);
                toolbar.appendChild(b);
                return b;
            };
            btn("Expand all", () => setAllOpen(tree, true));
            btn("Collapse all", () => setAllOpen(tree, false));
            btn("Copy", () => {
                if (!state.raw) return;
                navigator.clipboard?.writeText(state.raw).catch(() => {});
            }, "Copy the raw JSON to the clipboard");

            container.append(toolbar, status, tree);

            const widget = this.addDOMWidget("star_json_preview", "starJsonPreview", container, {
                serialize: false, hideOnZoom: false,
                getValue() { return ""; }, setValue() {},
            });
            widget.container = tree;
            widget.status = status;
            widget.state = state;
            this._starJsonPreviewWidget = widget;

            const w = this.size ? this.size[0] : 0, h = this.size ? this.size[1] : 0;
            this.setSize([Math.max(w, 340), Math.max(h, 220)]);
            return result;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            if (onExecuted) onExecuted.apply(this, arguments);
            const widget = this._starJsonPreviewWidget;
            if (!widget) return;

            const rawArr = message?.star_json || message?.ui?.star_json;
            const infoArr = message?.text || message?.ui?.text;
            const raw = Array.isArray(rawArr) ? rawArr.join("") : (typeof rawArr === "string" ? rawArr : "");
            const info = Array.isArray(infoArr) ? infoArr.join("\n") : (typeof infoArr === "string" ? infoArr : "");

            widget.state.raw = raw;
            widget.state.info = info;
            render(widget.container, widget.status, raw, info);
            widget.container.scrollTop = 0;
        };
    },
});
