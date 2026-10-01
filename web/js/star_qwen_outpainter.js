import { app } from "../../../../scripts/app.js";

// ⭐ Star Qwen2 Outpainter — draws a preview of the selected aspect ratio
// inside the node. The red box is the canvas the node builds; the inner box is
// the source image. Placement presets position it automatically — with
// "custom" you can drag the image box over the canvas to define which part of
// the new image gets outpainted.
//
// Layout: the preview is a fixed-size DOM widget appended after the last
// settings widget. The node's minimum size always includes it.

const NODE_CLASS = "StarQwenOutpainter";
const STYLE_ID = "star-qwen-outpainter-style";
const STAGE = 300;   // fixed square preview size (px)
const LABEL_H = 22;  // label row incl. margins
const PAD_V = 8;     // .sqop top+bottom padding
const PREVIEW_H = STAGE + LABEL_H + PAD_V;
const MIN_NODE_W = 340;
const MIN_NODE_PAD = 12; // breathing room under the preview
const HANDLE = 16; // px corner grip for resizing the image box (custom mode)
const MIN_SCALE = 0.1;

const PLACE_FRAC = {
    "top left": [0.0, 0.0], "top center": [0.5, 0.0], "top right": [1.0, 0.0],
    "center left": [0.0, 0.5], "center": [0.5, 0.5], "center right": [1.0, 0.5],
    "bottom left": [0.0, 1.0], "bottom center": [0.5, 1.0], "bottom right": [1.0, 1.0],
};

function ensureStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const st = document.createElement("style");
    st.id = STYLE_ID;
    st.textContent = `
.sqop { width: 100%; padding: 4px 8px 4px 8px; font-family: sans-serif;
        user-select: none; box-sizing: border-box; }
.sqop-stage { width: ${STAGE}px; height: ${STAGE}px; margin: 0 auto;
        position: relative; background: #101016;
        border: 1px solid #3a3a4c; border-radius: 6px; overflow: hidden; }
.sqop-box { position: absolute; background: rgba(255, 32, 32, 0.35);
        outline: 1px solid #d0342c; }
.sqop-inner { position: absolute; background: #2a2a38;
        background-size: 100% 100%; background-repeat: no-repeat;
        border: 1px dashed #8a8a9e; border-radius: 3px; }
.sqop.drag .sqop-inner { cursor: grab; }
.sqop.dragging .sqop-inner { cursor: grabbing; border-style: solid; }
.sqop.drag .sqop-inner::after { content: ""; position: absolute; right: 1px;
        bottom: 1px; width: 10px; height: 10px;
        background: rgba(0, 0, 0, 0.55); border: 1px solid #fff;
        border-radius: 3px; pointer-events: none; }
.sqop-label { margin-top: 4px; font-size: 10px; line-height: 14px;
        color: #8a8a9e; text-align: center; white-space: nowrap;
        overflow: hidden; text-overflow: ellipsis; }
`;
    document.head.appendChild(st);
}

// Extract base width/height from labels like "16:9 [1344x768 landscape]"
function baseDims(label) {
    const m = /\[(\d+)\s*x\s*(\d+)/.exec(label || "");
    if (m) return [parseInt(m[1], 10), parseInt(m[2], 10)];
    const r = /(\d+)\s*:\s*(\d+)/.exec(label || "");
    if (r) return [parseInt(r[1], 10), parseInt(r[2], 10)];
    return null;
}

// Mirror of the Python canvas math (ratio at N megapixels, snapped to /8)
function canvasDims(label, mp) {
    const b = baseDims(label);
    if (!b) return null;
    let [w, h] = b;
    const mpv = parseFloat(mp);
    if (!isNaN(mpv) && mpv !== 1.0) {
        const aspect = w / h;
        w = Math.floor(Math.sqrt(mpv * 1000000 * aspect));
        h = Math.floor(w / aspect);
        w -= w % 8;
        h -= h % 8;
    }
    return [w, h];
}

function getWidget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

// The node must always fit: all settings widgets + the fixed preview
function enforceSize(node) {
    const minH = (node._sqopBaseH || 0) + PREVIEW_H + MIN_NODE_PAD;
    let dirty = false;
    if (node.size[0] < MIN_NODE_W) { node.size[0] = MIN_NODE_W; dirty = true; }
    if (node.size[1] < minH) { node.size[1] = minH; dirty = true; }
    if (dirty) node.graph?.setDirtyCanvas?.(true, true);
}

function update(node) {
    const pv = node._sqop;
    if (!pv) return;
    const get = (n) => getWidget(node, n)?.value;
    const dims = canvasDims(get("aspect_ratio"), get("megapixel"));
    const place = get("image_placement") || "center";
    const isCustom = place === "custom";
    pv.wrap.classList.toggle("drag", isCustom);

    if (!dims) {
        pv.box.style.display = "none";
        pv.label.textContent = "";
        return;
    }
    let [w, h] = dims;
    const scale = isCustom ? (get("custom_scale") ?? 1.0) : 1.0;

    // Mirror of build_canvas: cap the image at the target MP, apply the
    // custom scale, grow the canvas (same ratio) until the image fits.
    let iw, ih, imgNote = "";
    if (pv.imgW) {
        const target = parseFloat(get("megapixel")) * 1e6;
        iw = pv.imgW;
        ih = pv.imgH;
        if (iw * ih > target) {
            const sc = Math.sqrt(target / (iw * ih));
            iw *= sc;
            ih *= sc;
        }
        if (isCustom) {
            iw *= scale;
            ih *= scale;
        }
        const fit = Math.max(iw / w, ih / h);
        if (fit > 1) {
            w = Math.ceil(w * fit / 8) * 8;
            h = Math.ceil(h * fit / 8) * 8;
        }
        imgNote = ` · image ${pv.imgW} × ${pv.imgH}`;
    } else {
        iw = ih = Math.min(w, h) * 0.82 * scale; // no image picked yet: square placeholder
    }
    const s = Math.min(STAGE / w, STAGE / h);
    const bw = Math.max(8, w * s);
    const bh = Math.max(8, h * s);
    pv.box.style.display = "";
    pv.box.style.left = (STAGE - bw) / 2 + "px";
    pv.box.style.top = (STAGE - bh) / 2 + "px";
    pv.box.style.width = bw + "px";
    pv.box.style.height = bh + "px";

    const iwPx = iw * s;
    const ihPx = ih * s;
    const frac = isCustom
        ? [get("custom_x") ?? 0.5, get("custom_y") ?? 0.5]
        : (PLACE_FRAC[place] || PLACE_FRAC["center"]);
    pv.inner.style.width = iwPx + "px";
    pv.inner.style.height = ihPx + "px";
    pv.inner.style.left = frac[0] * (bw - iwPx) + "px";
    pv.inner.style.top = frac[1] * (bh - ihPx) + "px";
    pv.label.textContent =
        `canvas ${Math.round(w)} × ${Math.round(h)}${imgNote} — ${isCustom ? "drag to move · corner grip to resize" : "placement: " + place}`;
}

// Fetch the picked image through /view so the preview shows the real
// picture at its true aspect ratio (same endpoint the frontend previews use)
function loadPreviewImage(node) {
    const pv = node._sqop;
    const name = getWidget(node, "image")?.value;
    if (!pv || !name) return;
    if (name === pv.lastImage && (pv.img || pv.loading)) return;
    pv.lastImage = name;
    pv.loading = true;
    let filename = name, subfolder = "", type = "input";
    if (filename.endsWith(" [output]")) {
        type = "output";
        filename = filename.slice(0, -" [output]".length);
    }
    const sep = Math.max(filename.lastIndexOf("/"), filename.lastIndexOf("\\"));
    if (sep >= 0) {
        subfolder = filename.slice(0, sep);
        filename = filename.slice(sep + 1);
    }
    const img = new Image();
    img.onload = () => {
        if (pv.lastImage !== name) return;
        pv.loading = false;
        pv.img = img;
        pv.imgW = img.naturalWidth;
        pv.imgH = img.naturalHeight;
        pv.inner.style.backgroundImage = `url("${img.src}")`;
        update(node);
    };
    img.onerror = () => {
        if (pv.lastImage !== name) return;
        pv.loading = false;
        pv.img = null;
        pv.imgW = 0;
        pv.inner.style.backgroundImage = "";
        update(node);
    };
    img.src = "/view?" + new URLSearchParams({ filename, subfolder, type });
}

function hideCustomWidgets(node) {
    for (const name of ["custom_x", "custom_y", "custom_scale"]) {
        const w = getWidget(node, name);
        if (w) {
            w.hidden = true;
            w.type = "hidden";
        }
    }
}

app.registerExtension({
    name: "StarNodes.StarQwenOutpainter",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_CLASS) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            const node = this;

            // remember the height of everything ABOVE the preview so we can
            // enforce a minimum node size that includes the fixed preview
            node._sqopBaseH = node.computeSize?.()[1] || 0;

            ensureStyle();
            const wrap = document.createElement("div");
            wrap.className = "sqop";
            wrap.innerHTML =
                `<div class="sqop-stage"><div class="sqop-box"><div class="sqop-inner"></div></div></div>` +
                `<div class="sqop-label"></div>`;
            const widget = node.addDOMWidget("star_ratio_preview", "sqop", wrap, {
                serializeValue: () => undefined,
                hideOnZoom: false,
            });
            widget.computedHeight = PREVIEW_H;
            widget.computeLayoutSize = () => ({
                minHeight: PREVIEW_H + MIN_NODE_PAD,
                minWidth: STAGE + 40,
            });

            const pv = {
                widget,
                wrap,
                stage: wrap.querySelector(".sqop-stage"),
                box: wrap.querySelector(".sqop-box"),
                inner: wrap.querySelector(".sqop-inner"),
                label: wrap.querySelector(".sqop-label"),
                dragging: false,
            };
            node._sqop = pv;

            // Drag the image box over the canvas; corner grip resizes it
            // (custom placement only). Drag state is captured once on
            // pointerdown and move/up are tracked on window, so live layout
            // updates of the preview can't disturb the gesture.
            const isCustom = () =>
                getWidget(node, "image_placement")?.value === "custom";
            const overGrip = (e) => {
                const r = pv.inner.getBoundingClientRect();
                return e.clientX >= r.right - HANDLE && e.clientX <= r.right + 4 &&
                       e.clientY >= r.bottom - HANDLE && e.clientY <= r.bottom + 4;
            };
            let drag = null;

            const move = (e) => {
                const r2 = pv.box.getBoundingClientRect();
                const ir = pv.inner.getBoundingClientRect();
                const slackW = r2.width - ir.width;
                const slackH = r2.height - ir.height;
                let fx = slackW > 0 ? (e.clientX - r2.left - ir.width / 2) / slackW : 0.5;
                let fy = slackH > 0 ? (e.clientY - r2.top - ir.height / 2) / slackH : 0.5;
                fx = Math.min(1, Math.max(0, fx));
                fy = Math.min(1, Math.max(0, fy));
                const wx = getWidget(node, "custom_x");
                const wy = getWidget(node, "custom_y");
                if (wx) { wx.value = fx; wx.callback?.(fx); }
                if (wy) { wy.value = fy; wy.callback?.(fy); }
                update(node);
            };
            const resize = (e) => {
                // square resize: side = 2x pointer distance from the box's
                // start center, so every drag direction resizes predictably
                const d = Math.max(Math.abs(e.clientX - drag.cx), Math.abs(e.clientY - drag.cy));
                const sc = Math.min(1, Math.max(MIN_SCALE, (2 * d) / drag.base));
                const ws = getWidget(node, "custom_scale");
                if (ws) { ws.value = sc; ws.callback?.(sc); }
                update(node);
            };
            const onMove = (e) => {
                if (!drag) return;
                e.preventDefault();
                e.stopPropagation();
                if (drag.mode === "resize") resize(e); else move(e);
            };
            const onUp = () => {
                drag = null;
                pv.wrap.classList.remove("dragging");
                window.removeEventListener("pointermove", onMove, true);
                window.removeEventListener("pointerup", onUp, true);
            };
            pv.box.addEventListener("pointerdown", (e) => {
                if (!isCustom() || e.button !== 0) return;
                e.preventDefault();
                e.stopPropagation();
                const ir = pv.inner.getBoundingClientRect();
                const curScale = getWidget(node, "custom_scale")?.value || 1;
                drag = {
                    mode: overGrip(e) ? "resize" : "move",
                    cx: ir.left + ir.width / 2,
                    cy: ir.top + ir.height / 2,
                    // larger inner dimension at scale 1.0, in screen px
                    base: Math.max(ir.width, ir.height) / curScale,
                };
                pv.wrap.classList.add("dragging");
                window.addEventListener("pointermove", onMove, true);
                window.addEventListener("pointerup", onUp, true);
                onMove(e);
            });
            pv.box.addEventListener("pointermove", (e) => {
                if (!drag && isCustom()) {
                    pv.inner.style.cursor = overGrip(e) ? "nwse-resize" : "grab";
                }
            });

            const imgW = getWidget(node, "image");
            if (imgW) {
                const origImgCb = imgW.callback;
                imgW.callback = function () {
                    loadPreviewImage(node);
                    update(node);
                    return origImgCb?.apply(this, arguments);
                };
            }

            hideCustomWidgets(node);
            for (const w of node.widgets || []) {
                if (["aspect_ratio", "megapixel", "image_placement", "custom_x", "custom_y", "custom_scale"].includes(w.name)) {
                    const orig = w.callback;
                    w.callback = function () {
                        update(node);
                        return orig?.apply(this, arguments);
                    };
                }
            }

            // never let the node shrink below settings + fixed preview
            const onResize = node.onResize;
            node.onResize = function () {
                enforceSize(node);
                return onResize?.apply(this, arguments);
            };

            update(node);
            loadPreviewImage(node);
            enforceSize(node);
            // re-fit once the frontend has settled (covers stale saved sizes)
            setTimeout(() => node._sqop && enforceSize(node), 300);
            return r;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);
            const node = this;
            hideCustomWidgets(node);
            loadPreviewImage(node);
            if (node._sqopBaseH == null) {
                // base height without the preview: computeSize minus our widget
                const sz = node.computeSize?.()[1] || 0;
                node._sqopBaseH = sz - PREVIEW_H > 0 ? sz - PREVIEW_H : sz;
            }
            update(node);
            enforceSize(node);
            setTimeout(() => node._sqop && enforceSize(node), 300);
            return r;
        };
    },
});
