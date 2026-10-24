import { app } from "../../../../scripts/app.js";

// ===========================================================================
//  ⭐ Star Prompt Preview — shows a connected string and lets the user edit
//  it while the workflow pauses (bypass off / Edit) or just previews it
//  (bypass on, the default)
// ===========================================================================

const KEY = "StarPromptPreviewAndEdit";
const WAIT_COLOR = "#553366";
const DONE_COLOR = "#335533";

const CSS = `
.sppv-container { display:flex; flex-direction:column; width:100%; box-sizing:border-box;
    background:#1a1a2e; border:1px solid #3d124d; border-radius:4px; overflow:hidden;
    font-family:ui-monospace, monospace; font-size:11px; color:#c8c8c8; }
.sppv-toolbar { display:flex; gap:4px; align-items:center; padding:4px 6px;
    border-bottom:1px solid #3d124d; flex:0 0 auto; }
.sppv-status { flex:1 1 auto; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
    color:#9ecbff; }
.sppv-status[data-kind="waiting"] { color:#e1bee7; }
.sppv-status[data-kind="done"] { color:#a5d6a7; }
.sppv-btn { background:#19124d; color:#c8c8c8; border:1px solid #3d124d;
    border-radius:3px; font:inherit; padding:1px 8px; cursor:pointer; }
.sppv-btn:hover { background:#2a1a6e; }
.sppv-text { width:100%; min-height:240px; padding:6px 8px; box-sizing:border-box;
    background:#12121f; color:#e8e8e8; border:none; outline:none; resize:vertical;
    font-family:ui-monospace, monospace; font-size:12px; line-height:1.5;
    white-space:pre-wrap; overflow:auto; }
.sppv-text[readonly] { color:#a8a8b8; }
.sppv-continue { padding:6px 10px; border:none; border-top:1px solid #3d124d;
    background:#2e7d32; color:#fff; font:inherit; font-weight:bold; cursor:pointer; }
.sppv-continue:hover { background:#388e3c; }
.sppv-continue:disabled { background:#2a2a3a; color:#666; cursor:default; }
`;

function el(tag, className, text) {
    const e = document.createElement(tag);
    if (className) e.className = className;
    if (text !== undefined) e.textContent = text;
    return e;
}

function ensureStyle() {
    if (!document.getElementById("sppv-style")) {
        const style = el("style");
        style.id = "sppv-style";
        style.textContent = CSS;
        document.head.appendChild(style);
    }
}

function captureInitialAppearance(node) {
    if (!node || node._sppvInitial) return;
    node._sppvInitial = { bgcolor: node.bgcolor, color: node.color, title: node.title };
}

function restoreInitialAppearance(node) {
    const initial = node?._sppvInitial;
    if (!initial) return;
    node.bgcolor = initial.bgcolor ?? node.constructor.nodeData?.bgcolor;
    node.color = initial.color ?? node.constructor.nodeData?.color;
}

app.registerExtension({
    name: "StarNodes.StarPromptPreviewAndEdit",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== KEY) return;
        ensureStyle();

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            const node = this;
            setTimeout(() => captureInitialAppearance(node), 0);

            const container = el("div", "sppv-container");
            container.dataset.captureWheel = "true";
            for (const type of ["pointerdown", "mousedown", "dblclick", "wheel", "keydown", "keyup"])
                container.addEventListener(type, (e) => e.stopPropagation());

            const toolbar = el("div", "sppv-toolbar");
            const status = el("div", "sppv-status", "Waiting for input…");
            const copyBtn = el("button", "sppv-btn", "Copy");
            copyBtn.type = "button";
            copyBtn.title = "Copy the current text to the clipboard";
            toolbar.append(status, copyBtn);

            const textarea = el("textarea", "sppv-text");
            textarea.placeholder = "The connected prompt will appear here…";

            const continueBtn = el("button", "sppv-continue", "▶ Continue");
            continueBtn.type = "button";
            continueBtn.disabled = true;

            const state = { waiting: false, edited: false };

            const updateCount = () => {
                if (!state.waiting) return;
                const chars = textarea.value.length;
                const lines = chars ? textarea.value.split("\n").length : 0;
                status.textContent = `${state.edited ? "Editing… " : "Edit the prompt… "}(${lines} lines, ${chars} chars)`;
            };

            const setWaiting = (waiting) => {
                state.waiting = waiting;
                if (waiting) {
                    captureInitialAppearance(node);
                    node.bgcolor = WAIT_COLOR;
                    status.dataset.kind = "waiting";
                    status.textContent = "Edit the prompt and click Continue…";
                    continueBtn.disabled = false;
                } else {
                    restoreInitialAppearance(node);
                    node.bgcolor = DONE_COLOR;
                    status.dataset.kind = "done";
                    status.textContent = "Continued — edited text passed on.";
                    continueBtn.disabled = true;
                    setTimeout(() => {
                        if (!state.waiting) {
                            restoreInitialAppearance(node);
                            status.dataset.kind = "";
                            node.setDirtyCanvas(true, true);
                        }
                    }, 3000);
                }
                node.setDirtyCanvas(true, true);
            };

            const applyMode = () => {
                const bypass = !!node.widgets?.find(w => w.name === "bypass")?.value;
                textarea.readOnly = bypass;
                continueBtn.style.display = bypass ? "none" : "";
                if (bypass && !state.waiting && status.dataset.kind !== "done")
                    status.textContent = "Preview only (Bypass) — run to show the text.";
            };

            textarea.addEventListener("input", () => {
                if (!state.waiting) return;
                state.edited = true;
                updateCount();
            });

            copyBtn.addEventListener("click", () => {
                navigator.clipboard?.writeText(textarea.value).catch(() => {});
            });

            continueBtn.addEventListener("click", async () => {
                continueBtn.disabled = true;
                status.dataset.kind = "done";
                status.textContent = "Continuing…";
                try {
                    await fetch("/starnodes/prompt_preview/continue", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ node_id: String(node.id), text: textarea.value })
                    });
                } catch (error) {
                    console.error("[StarPromptPreviewAndEdit] Error sending text:", error);
                    status.dataset.kind = "";
                    status.textContent = "Could not reach the server — check the console.";
                    continueBtn.disabled = false;
                }
            });

            // Reflect bypass toggle changes (textarea readonly / button visibility)
            const modeWidget = node.widgets?.find(w => w.name === "bypass");
            if (modeWidget) {
                const originalCallback = modeWidget.callback;
                modeWidget.callback = function () {
                    if (originalCallback) originalCallback.apply(this, arguments);
                    applyMode();
                };
            }
            applyMode();

            container.append(toolbar, textarea, continueBtn);

            const widget = node.addDOMWidget("star_prompt_preview", "starPromptPreview", container, {
                serialize: false, hideOnZoom: false,
                getValue() { return ""; }, setValue() {},
            });
            widget.status = status;
            widget.textarea = textarea;
            widget.state = state;
            node._starPromptPreview = { widget, textarea, status, setWaiting, applyMode, state };

            // Grow the textarea with the node so long prompts stay readable
            const onResize = nodeType.prototype.onResize;
            nodeType.prototype.onResize = function (size) {
                if (onResize) onResize.apply(this, arguments);
                const h = Math.max(240, size[1] - 190);
                if (textarea) textarea.style.height = `${size[1] > 450 ? h : 240}px`;
            };

            const w = node.size ? node.size[0] : 0, h = node.size ? node.size[1] : 0;
            node.setSize([Math.max(w, 360), Math.max(h, 560)]);
            return result;
        };

        // Switching workflow tabs can restore the widget with an empty value,
        // and old workflows stored the dropdown strings — convert anything
        // unknown to a valid boolean, defaulting to Bypass (on).
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            const bypassWidget = this.widgets?.find(w => w.name === "bypass");
            if (bypassWidget) {
                const v = bypassWidget.value;
                if (typeof v === "string")
                    bypassWidget.value = v === "true" || v === "Bypass Mode" || v === "";
                else if (v === undefined || v === null)
                    bypassWidget.value = true;
            }
            this._starPromptPreview?.applyMode();
            return result;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            if (onExecuted) onExecuted.apply(this, arguments);
            const ui = this._starPromptPreview;
            if (!ui) return;

            const textArr = message?.star_prompt_preview || message?.ui?.star_prompt_preview;
            const waitArr = message?.star_prompt_preview_waiting ?? message?.ui?.star_prompt_preview_waiting;

            if (textArr !== undefined) {
                const text = Array.isArray(textArr) ? textArr.join("") : String(textArr);
                ui.textarea.value = text;
            }

            const bypass = !!this.widgets?.find(w => w.name === "bypass")?.value;
            if (bypass) {
                ui.status.dataset.kind = "";
                ui.status.textContent = "Preview only (Bypass).";
            } else {
                ui.setWaiting(Array.isArray(waitArr) ? !!waitArr[0] : !!waitArr);
            }
            ui.state.edited = false;
        };
    },
});
