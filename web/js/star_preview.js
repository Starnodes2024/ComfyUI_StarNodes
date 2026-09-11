import { app } from "../../../../scripts/app.js";

// ⭐ Star Preview — live animated sampling preview for the StarNodes video
// all-in-one nodes (LTXV / LTXV 2.5 / Minimax All In One).
//
// The Python side (misc/star_preview.py) taps the sampler callback of the
// all-in-one node's internal model and pushes "star_preview" websocket events
// with a base64 animated WebP (video) or JPEG (single frame). This extension
// renders them in a DOM widget on the ⭐ Star Preview node — video preview
// only, no charts, no settings.
//
// Sizing: fixed layout. The preview box is always 512x512 px and the image
// is letterboxed inside it (object-fit: contain), so any aspect ratio fits.
// The widget height is therefore constant and the node is kept at a minimum
// of 550x550 px — big enough for the VAE dropdown plus the whole box.

const STYLE_ID = "star-preview-style";
const BOX = 256;
const MIN_NODE =350;

function ensureStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const st = document.createElement("style");
    st.id = STYLE_ID;
    st.textContent = `
.star-preview { padding: 4px 8px 2px 8px; font-family: sans-serif;
                user-select: none; box-sizing: border-box; }
.star-preview-imgwrap { width: ${BOX}px; height: ${BOX}px; margin: 0 auto;
                border-radius: 6px; overflow: hidden;
                background: #17171f; border: 1px solid #3a3a4c;
                display: flex; align-items: center; justify-content: center; }
.star-preview-imgwrap.has-image { background: transparent;
                border-color: transparent; }
.star-preview-imgwrap img { display: block; max-width: 100%; max-height: 100%;
                width: auto; height: auto; object-fit: contain; }
.star-preview-status { margin-top: 3px; font-size: 10px; color: #8a8a9e;
                min-height: 12px; white-space: nowrap; overflow: hidden;
                text-overflow: ellipsis; }
`;
    document.head.appendChild(st);
}

// Walks subgraph chains for ids like "12:7:5" (same approach as KJNodes).
function findNodeByQualifiedId(rootGraph, qid) {
    if (!rootGraph || qid == null) return null;
    const parts = String(qid).split(":");
    let graph = rootGraph;
    for (let i = 0; i < parts.length - 1; i++) {
        const parent = graph?.getNodeById?.(parseInt(parts[i], 10));
        if (!parent?.subgraph) return null;
        graph = parent.subgraph;
    }
    return graph?.getNodeById?.(parseInt(parts[parts.length - 1], 10)) || null;
}

// Never shrink the node below 550x550 or below the size needed to fit
// the fixed preview box; keep the user's size when larger.
function relayout(node) {
    const sz = node.computeSize();
    node.setSize([
        Math.max(node.size[0], sz[0], MIN_NODE),
        Math.max(node.size[1], sz[1], MIN_NODE),
    ]);
    node.graph?.setDirtyCanvas?.(true, true);
}

// Measure the actually rendered widget height and make the node fit it.
function updateHeight(node, pv) {
    const h = pv.wrap.offsetHeight;
    if (!h) return;
    const target = Math.round(h) + 2;
    if (pv.widget.computedHeight !== target) {
        pv.widget.computedHeight = target;
        relayout(node);
    }
}

function getPreviewWidget(node) {
    // Reuse the widget while it is still registered on the node.
    if (node._starPreview && (node.widgets || []).includes(node._starPreview.widget)) {
        return node._starPreview;
    }
    ensureStyle();
    // Drop stale widgets from previous runs.
    for (const w of [...(node.widgets || [])]) {
        if (w.name === "star_preview") {
            w.onRemove?.();
            w.element?.remove?.();
            node.widgets.splice(node.widgets.indexOf(w), 1);
        }
    }
    const wrap = document.createElement("div");
    wrap.className = "star-preview";
    wrap.innerHTML =
        `<div class="star-preview-imgwrap"><img alt="preview" style="display:none"></div>` +
        `<div class="star-preview-status">waiting for sampling…</div>`;

    const widget = node.addDOMWidget("star_preview", "starPreview", wrap,
        { serialize: false, hideOnZoom: false });

    const pv = {
        widget,
        wrap,
        img: wrap.querySelector("img"),
        imgwrap: wrap.querySelector(".star-preview-imgwrap"),
        status: wrap.querySelector(".star-preview-status"),
        ro: null,
    };
    node._starPreview = pv;

    // Track user resizes / layout changes of the widget itself.
    if (typeof ResizeObserver !== "undefined") {
        pv.ro = new ResizeObserver(() => updateHeight(node, pv));
        pv.ro.observe(wrap);
    }
    const origRemove = widget.onRemove;
    widget.onRemove = function () {
        pv.ro?.disconnect();
        origRemove?.apply(this, arguments);
    };

    updateHeight(node, pv);
    return pv;
}

app.registerExtension({
    name: "StarNodes.StarPreview",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "StarPreview") return;
        const orig = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = orig?.apply(this, arguments);
            // Fresh node: start at the fixed minimum that fits the 512 px
            // box (loaded workflows restore their saved size via configure()).
            this.setSize([
                Math.max(this.size?.[0] || 0, MIN_NODE),
                Math.max(this.size?.[1] || 0, MIN_NODE),
            ]);
            getPreviewWidget(this);
            return r;
        };
        // Loaded workflows may restore a saved size smaller than the fixed
        // box — re-apply the minimum after configure().
        const origConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = origConfigure?.apply(this, arguments);
            const pv = getPreviewWidget(this);
            updateHeight(this, pv);
            relayout(this);
            return r;
        };
    },

    setup() {
        app.api.addEventListener("star_preview", (ev) => {
            const d = ev.detail || {};
            if (d.node_id == null || !d.image) return;
            const node = findNodeByQualifiedId(app.graph, d.node_id);
            if (!node) return;

            const pv = getPreviewWidget(node);
            const mime = d.mime || "image/jpeg";

            const shown = pv.img.style.display === "block";
            pv.img.src = `data:${mime};base64,${d.image}`;
            pv.img.style.display = "block";
            pv.imgwrap.classList.add("has-image");
            pv.status.textContent =
                `step ${d.step ?? "?"}/${d.total ?? "?"} · ${d.w}×${d.h}` +
                (mime === "image/webp" ? " · animated" : "");

            // First frame of a run: wait for the image to load so the
            // measured height is correct; later frames same size anyway.
            if (!shown) {
                pv.img.onload = () => updateHeight(node, pv);
            }
            updateHeight(node, pv);
            node.graph?.setDirtyCanvas?.(true, false);
        });
    },
});
