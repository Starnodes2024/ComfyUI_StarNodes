import { app } from "../../../../scripts/app.js";

// ===========================================================================
//  ⭐ Star Show Everything — drag‑link suggestions
// ===========================================================================

const KEY     = "StarShowEverything";
const DISPLAY = "⭐ Star Show Everything";

const UNIVERSE = [
    "*",
    "STRING", "INT", "FLOAT", "BOOLEAN",
    "IMAGE", "LATENT", "CONDITIONING", "MASK",
    "MODEL", "CLIP", "VAE", "CLIP_VISION", "CLIP_VISION_OUTPUT",
    "CONTROL_NET", "STYLE_MODEL", "GLIGEN",
    "SAMPLER", "SIGMAS", "NOISE", "GUIDER", "TIMESTEPS_RANGE",
    "UPSCALE_MODEL", "LATENT_OPERATION", "HOOKS", "WEIGHT_ADAPTER",
    "AUDIO", "VIDEO", "MESH",
    "SEGS", "BBOX", "MASKS", "INTS", "FLOATS", "STRINGS",
    "COMBO", "WILDCARD", "SEED", "PHOTOMAKER", "FACE_ANALYSIS",
    "PIPE_LINE", "SDXL_TUPLE", "SV3D_TUPLE", "SVD_TUPLE",
];
const TYPE_STR = UNIVERSE.join(",");

const LG = () => globalThis.LiteGraph || window.LiteGraph;

console.log("%c[StarShowEverything] extension module loaded ✔", "color:#ff7ad9;font-weight:bold");

let defaultsLogDone = false, compatDone = false, menuHookDone = false;

// ---------------------------------------------------------------------------
//  L2 — keep the defaults map seeded
// ---------------------------------------------------------------------------
function seedDefaults() {
    const lg = LG();
    if (!lg) return false;
    lg.slot_types_default_in  ||= {};
    lg.slot_types_default_out ||= {};
    for (const type of UNIVERSE) {
        const arr = (lg.slot_types_default_in[type] ||= []);
        const i = arr.indexOf(KEY);
        if (i > 0) arr.splice(i, 1);
        if (i !== 0) arr.unshift(KEY);
    }
    if (!defaultsLogDone) {
        defaultsLogDone = true;
        console.log("[StarShowEverything] L2 defaults seeded for", UNIVERSE.length, "types");
    }
    return true;
}

// ---------------------------------------------------------------------------
//  L3 — compatibility patch
// ---------------------------------------------------------------------------
function patchCompat() {
    const lg = LG();
    if (!lg || compatDone) return;
    const wrap = (orig) => {
        const f = function (a, b) {
            if (!a || !b || a === "*" || b === "*" || a === TYPE_STR || b === TYPE_STR) return true;
            if (typeof orig === "function") { try { return orig.apply(this, arguments); } catch { return a === b; } }
            return a === b;
        };
        f.__starPatched = true;
        return f;
    };
    if (typeof lg.isValidConnection === "function" && !lg.isValidConnection.__starPatched)
        lg.isValidConnection = wrap(lg.isValidConnection);
    else if (!lg.isValidConnection) lg.isValidConnection = wrap(null);
    try {
        const p = lg.LGraphNode && lg.LGraphNode.prototype;
        if (p && typeof p.isValidConnection === "function" && !p.isValidConnection.__starPatched)
            p.isValidConnection = wrap(p.isValidConnection);
    } catch { /* best effort */ }
    compatDone = true;
    console.log("[StarShowEverything] L3 compatibility patched");
}

// ---------------------------------------------------------------------------
//  L2 + L4 — hook the connection menu
// ---------------------------------------------------------------------------
let pendingLink = null;

function buildEntry() {
    return {
        content: DISPLAY,
        value: KEY,
        className: "star-show-everything-suggestion",
        callback: () => {
            const lg = LG();
            const link = pendingLink;
            const canvas = link && link.canvas;
            if (!lg || !canvas) { console.warn("[StarShowEverything] L4: no canvas for connect"); return; }
            
            const node = lg.createNode(KEY);
            if (!node) { console.warn("[StarShowEverything] L4: createNode failed for", KEY); return; }
            
            canvas.graph.add(node);
            
            try {
                const pt = link.e && canvas.convertEventToCanvasOffset
                    ? canvas.convertEventToCanvasOffset(link.e) : [120, 120];
                node.pos = [pt[0], pt[1]];
            } catch { /* leave default pos */ }
            
            const { nodeFrom, slotFrom, nodeTo, slotTo } = link;
            
            // 🐛 BRUTE-FORCE FIX: Timeout + Index Extraction + Validation Bypass
            setTimeout(() => {
                try {
                    // 1. Stelle sicher, dass der Slot zu 100% ein reiner Wildcard ist
                    if (!node.inputs || node.inputs.length === 0) {
                        node.addInput("anything", "*");
                    } else if (node.inputs[0].type !== "*") {
                        node.inputs[0].type = "*";
                    }

                    // 2. Extrahiere den Output-Index als echte Zahl (schützt vor Objekt-Crashes)
                    let outIdx = -1;
                    if (typeof slotFrom === "number") outIdx = slotFrom;
                    else if (typeof slotFrom === "string") outIdx = nodeFrom?.findOutputSlot ? nodeFrom.findOutputSlot(slotFrom) : -1;
                    else if (typeof slotFrom === "object" && slotFrom !== null) {
                        if (slotFrom.index !== undefined) outIdx = slotFrom.index;
                        else if (slotFrom.slot !== undefined) outIdx = slotFrom.slot;
                        else if (nodeFrom?.outputs) outIdx = nodeFrom.outputs.indexOf(slotFrom);
                    }
                    if (outIdx === -1 && nodeFrom?.outputs?.length > 0) outIdx = 0; // Letzter Ausweg

                    // Extrahiere den Input-Index (falls wir rückwärts ziehen)
                    let inIdx = -1;
                    if (typeof slotTo === "number") inIdx = slotTo;
                    else if (typeof slotTo === "string") inIdx = nodeTo?.findInputSlot ? nodeTo.findInputSlot(slotTo) : -1;
                    else if (typeof slotTo === "object" && slotTo !== null) {
                        if (slotTo.index !== undefined) inIdx = slotTo.index;
                        else if (slotTo.slot !== undefined) inIdx = slotTo.slot;
                        else if (nodeTo?.inputs) inIdx = nodeTo.inputs.indexOf(slotTo);
                    }
                    if (inIdx === -1 && nodeTo?.inputs?.length > 0) inIdx = 0; // Letzter Ausweg

                    // 3. LiteGraph Validierung kurzzeitig komplett abschalten
                    const oldLGValid = lg.isValidConnection;
                    const oldNodeValid = lg.LGraphNode.prototype.isValidConnection;
                    lg.isValidConnection = () => true;
                    lg.LGraphNode.prototype.isValidConnection = () => true;

                    try {
                        if (nodeFrom && outIdx !== -1) {
                            const success = nodeFrom.connect(outIdx, node, 0);
                            if (!success) console.warn("[StarShowEverything] L4: Force-Connect vom Original-Knoten schlug fehl. Output-Index war:", outIdx);
                        } else if (nodeTo && inIdx !== -1) {
                            const success = node.connect(0, nodeTo, inIdx);
                            if (!success) console.warn("[StarShowEverything] L4: Force-Connect zum Ziel-Knoten schlug fehl. Input-Index war:", inIdx);
                        }
                    } finally {
                        // Validierung sofort wiederherstellen, damit ComfyUI normal weiterläuft
                        lg.isValidConnection = oldLGValid;
                        lg.LGraphNode.prototype.isValidConnection = oldNodeValid;
                    }
                } catch (e) { 
                    console.warn("[StarShowEverything] L4: Verzögerter Connect fehlgeschlagen", e); 
                }
            }, 50);
        },
    };
}

function injectIfConnectionMenu(values) {
    if (!Array.isArray(values) || !pendingLink || (Date.now() - pendingLink.t) > 1000) return values;
    const present = values.some(v => {
        const val = typeof v === "string" ? v : (v?.value || v?.content);
        return val === KEY || val === DISPLAY;
    });
    if (present) return values;
    const entry = buildEntry();
    let idx = values.findIndex(v => v?.content === "Search");
    if (idx >= 0) {
        const after = values[idx + 1];
        idx = (after === null || after?.content === undefined) ? idx + 2 : idx + 1;
        values.splice(idx, 0, entry);
    } else {
        values.push(entry);
    }
    return values;
}

function patchConnectionMenu() {
    const lg = LG();
    if (!lg || menuHookDone) return;

    const cp = lg.LGraphCanvas && lg.LGraphCanvas.prototype;
    if (cp && typeof cp.showConnectionMenu === "function" && !cp.showConnectionMenu.__starPatched) {
        const orig = cp.showConnectionMenu;
        cp.showConnectionMenu = function (optPass) {
            try {
                const o = optPass || {};
                pendingLink = { t: Date.now(), canvas: this, nodeFrom: o.nodeFrom, slotFrom: o.slotFrom, nodeTo: o.nodeTo, slotTo: o.slotTo, e: o.e };
                seedDefaults();
            } catch { /* never block the menu */ }
            return orig.apply(this, arguments);
        };
        cp.showConnectionMenu.__starPatched = true;
    }

    if (typeof lg.ContextMenu === "function" && !lg.ContextMenu.__starPatched) {
        const Orig = lg.ContextMenu;
        const Wrapped = function (values, options) {
            try { values = injectIfConnectionMenu(values); } catch (e) { console.warn("[StarShowEverything] inject error", e); }
            return Orig.apply(this, [values, options]);
        };
        Wrapped.prototype = Orig.prototype;
        Wrapped.__starPatched = true;
        lg.ContextMenu = Wrapped;
    }
    menuHookDone = true;
}

// ===========================================================================
app.registerExtension({
    name: "StarNodes.StarShowEverything",

    async setup() {
        seedDefaults();
        patchCompat();
        patchConnectionMenu();
        let n = 0;
        const iv = setInterval(() => { seedDefaults(); if (++n >= 8) clearInterval(iv); }, 1500);
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== KEY) return;

        try {
            const inp = nodeData?.input?.required?.anything;
            if (Array.isArray(inp)) {
                inp[0] = TYPE_STR;
            }
        } catch (e) {}

        seedDefaults();
        patchCompat();
        patchConnectionMenu();

        // ---- DOM info widget ----
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            
            // 1. Fange zukünftige / dynamische Inputs ab
            const origAddInput = this.addInput;
            this.addInput = function(name, type, extra_info) {
                if (type === TYPE_STR) type = "*";
                return origAddInput.apply(this, arguments);
            };

            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            
            // 2. Korrigiere bereits existierende Inputs
            if (this.inputs) {
                for (let i = 0; i < this.inputs.length; i++) {
                    if (this.inputs[i].type === TYPE_STR) {
                        this.inputs[i].type = "*";
                    }
                }
            }

            this.serialize_widgets = false;

            const container = document.createElement("div");
            container.className = "star-show-everything-container";
            container.style.cssText = [
                "width:100%", "min-height:40px", "max-height:400px", "overflow-y:auto",
                "background:#1a1a2e", "border:1px solid #3d124d", "border-radius:4px",
                "padding:8px", "box-sizing:border-box", "font-family:monospace",
                "font-size:11px", "white-space:pre-wrap", "word-break:break-all",
                "color:#c8c8c8", "line-height:1.5",
            ].join(";");

            const placeholder = document.createElement("div");
            placeholder.textContent = "Connect any input and run to see info…";
            placeholder.style.cssText = "color:#666;font-style:italic;";
            container.appendChild(placeholder);

            const widget = this.addDOMWidget("star_show_everything_info", "starShowEverythingInfo", container, {
                serialize: false, hideOnZoom: false,
                getValue() { return ""; }, setValue() {},
            });
            widget.container = container;
            widget.placeholder = placeholder;
            this._starShowEverythingWidget = widget;

            const w = this.size ? this.size[0] : 0, h = this.size ? this.size[1] : 0;
            this.setSize([Math.max(w, 320), Math.max(h, 180)]);
            return result;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            if (onExecuted) onExecuted.apply(this, arguments);
            const widget = this._starShowEverythingWidget;
            if (!widget) return;
            const textArr = message?.text || message?.ui?.text;
            const text = Array.isArray(textArr) ? textArr.join("\n") : (typeof textArr === "string" ? textArr : "");
            if (text) {
                widget.container.innerHTML = "";
                const pre = document.createElement("pre");
                pre.style.cssText = "margin:0;padding:0;white-space:pre-wrap;word-break:break-all;font-family:inherit;font-size:inherit;color:inherit;line-height:inherit;";
                pre.textContent = text;
                widget.container.appendChild(pre);
                widget.container.scrollTop = 0;
            }
        };
    },
});