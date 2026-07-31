import { app } from "../../../../scripts/app.js";

// Star Nodes video pack - frontend companion:
//  1. compressor: inline <video> preview inside a placeholder box that
//     reserves space from the start and snaps to the video's aspect ratio.
//  2. both nodes: fancy DOM progress bar fed by "star_nodes.progress" events.
//
// Layout note: ComfyUI positions DOM widgets using widget.computedHeight,
// NOT the live DOM height. Every time the content height changes we must
// update widget.computedHeight and re-run node.setSize(node.computeSize()),
// otherwise later widgets are drawn on top of the video (overlap bug).
const PREVIEW_NODES = ["StarVideoCompressor"];
const PROGRESS_NODES = ["StarVideoCompressor", "StarVideoLoader"];

const PREVIEW_MIN_H = 80;
const PREVIEW_MAX_H = 480;
const PREVIEW_PLACEHOLDER_H = 200;
const PREVIEW_PAD = 14;      // padding + border around the video box
const PROGRESS_H = 58;       // fixed height of the progress bar widget

const STYLE_ID = "star-nodes-style";

function ensureStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const st = document.createElement("style");
    st.id = STYLE_ID;
    st.textContent = `
.star-pb { padding: 6px 8px 4px 8px; font-family: sans-serif; user-select: none; }
.star-pb-top { display: flex; justify-content: space-between; align-items: baseline;
               font-size: 11px; color: #cfcfe8; margin-bottom: 4px; }
.star-pb-pct { font-weight: 700; font-size: 13px; color: #ffffff;
               font-variant-numeric: tabular-nums; }
.star-pb-track { height: 12px; border-radius: 6px; background: #17171f;
                 border: 1px solid #3a3a4c; overflow: hidden;
                 box-shadow: inset 0 1px 2px rgba(0,0,0,.6); }
.star-pb-fill { height: 100%; width: 0%; border-radius: 6px; position: relative;
                background: linear-gradient(90deg, #6a5cff 0%, #00c8ff 100%);
                box-shadow: 0 0 8px rgba(0,200,255,.55);
                transition: width .15s ease-out; }
.star-pb-fill::after { content: ""; position: absolute; inset: 0;
                background: repeating-linear-gradient(45deg,
                    rgba(255,255,255,.22) 0 10px, transparent 10px 20px);
                animation: starPbStripes .8s linear infinite; }
.star-pb-sub { margin-top: 3px; font-size: 10px; color: #8a8a9e; min-height: 12px;
               white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.star-pb.done .star-pb-fill { background: linear-gradient(90deg, #2fbf71, #8ce99a);
                box-shadow: 0 0 8px rgba(80,220,130,.5); }
.star-pb.done .star-pb-fill::after { animation: none; background: none; }
@keyframes starPbStripes { from { background-position: 0 0; }
                           to   { background-position: 28px 0; } }

.star-vp { width: 100%; padding: 2px 6px 4px 6px; box-sizing: border-box; }
.star-vp-box { width: 100%; border-radius: 6px; overflow: hidden;
               background: #101018; border: 1px solid #2c2c3a; }
.star-vp-empty { height: ${PREVIEW_PLACEHOLDER_H}px; display: flex;
                 align-items: center; justify-content: center; color: #5a5a6e;
                 font-size: 11px; font-family: sans-serif;
                 border: 1px dashed #333344; border-radius: 6px; }
.star-vp video { width: 100%; display: block; background: #000; }
.star-vp-more { font-size: 10px; color: #8a8a9e; font-family: sans-serif;
                padding: 3px 2px 0 2px; }
`;
    document.head.appendChild(st);
}

function relayout(node) {
    node.setSize(node.computeSize());
    node.graph?.setDirtyCanvas?.(true, true);
}

// ---------------------------------------------------------------- progress

function getProgressBar(node) {
    if (node.starProgress && node.starProgress.wrap.isConnected) {
        return node.starProgress;
    }
    ensureStyle();
    const wrap = document.createElement("div");
    wrap.className = "star-pb";
    wrap.innerHTML =
        `<div class="star-pb-top"><span class="star-pb-title">Working…</span>` +
        `<span class="star-pb-pct">0%</span></div>` +
        `<div class="star-pb-track"><div class="star-pb-fill"></div></div>` +
        `<div class="star-pb-sub"></div>`;
    const widget = node.addDOMWidget("star_progress", "starProgress", wrap,
        { serialize: false, hideOnZoom: false });
    widget.computedHeight = PROGRESS_H;
    node.starProgress = {
        widget, wrap,
        title: wrap.querySelector(".star-pb-title"),
        pct: wrap.querySelector(".star-pb-pct"),
        fill: wrap.querySelector(".star-pb-fill"),
        sub: wrap.querySelector(".star-pb-sub"),
    };
    relayout(node);
    return node.starProgress;
}

// ---------------------------------------------------------------- preview

function setPreviewHeight(node, contentH, extraH = 0) {
    const pb = node.starPreview;
    if (!pb) return;
    pb.widget.computedHeight = contentH + PREVIEW_PAD + extraH;
    relayout(node);
}

function createPreviewWidget(node) {
    ensureStyle();
    const wrap = document.createElement("div");
    wrap.className = "star-vp";
    wrap.innerHTML = `<div class="star-vp-empty">Video preview — run the ` +
        `node to see the result here.</div>`;
    const widget = node.addDOMWidget("star_video_preview", "starVideoPreview",
        wrap, { serialize: false, hideOnZoom: false });
    node.starPreview = { widget, wrap };
    setPreviewHeight(node, PREVIEW_PLACEHOLDER_H);
    return node.starPreview;
}

function fillPreview(node, videos) {
    const pb = (node.starPreview && node.starPreview.wrap.isConnected)
        ? node.starPreview : createPreviewWidget(node);
    const wrap = pb.wrap;
    wrap.innerHTML = "";

    if (!videos || !videos.length) {
        wrap.innerHTML = `<div class="star-vp-empty">No video produced.</div>`;
        setPreviewHeight(node, PREVIEW_PLACEHOLDER_H);
        return;
    }

    const v = videos[0];
    const box = document.createElement("div");
    box.className = "star-vp-box";
    const el = document.createElement("video");
    el.controls = true;
    el.loop = true;
    el.muted = true;
    el.preload = "metadata";
    el.style.height = PREVIEW_PLACEHOLDER_H + "px"; // until aspect is known
    const params = new URLSearchParams({
        filename: v.filename,
        subfolder: v.subfolder ?? "",
        type: v.type ?? "output",
    });
    el.src = "/view?" + params.toString();
    box.appendChild(el);
    wrap.appendChild(box);

    let extraH = 0;
    if (videos.length > 1) {
        const more = document.createElement("div");
        more.className = "star-vp-more";
        more.textContent = `+${videos.length - 1} more file(s) — ` +
            `see the info output for details.`;
        wrap.appendChild(more);
        extraH = 18;
    }
    
    pb.extraH = extraH;
    setPreviewHeight(node, PREVIEW_PLACEHOLDER_H, extraH);

    el.addEventListener("loadedmetadata", () => {
        const avail = Math.max(100, (node.size?.[0] ?? 320) - 32);
        let h = avail * (el.videoHeight || 9) / (el.videoWidth || 16);
        h = Math.min(Math.max(h, PREVIEW_MIN_H), PREVIEW_MAX_H);
        el.style.height = h + "px";
        setPreviewHeight(node, h, extraH);
    });
}

// ---------------------------------------------------------------- extension

app.registerExtension({
    name: "StarNodes.VideoPack",

    setup() {
        app.api.addEventListener("star_nodes.progress", (ev) => {
            const d = ev.detail || {};
            const node = app.graph.getNodeById(Number(d.node));
            if (!node) return;
            const pb = getProgressBar(node);
            const frac = Math.min(Math.max(d.value ?? 0, 0), 1);
            pb.fill.style.width = (frac * 100).toFixed(1) + "%";
            pb.pct.textContent = d.text ?? Math.round(frac * 100) + "%";
            pb.sub.textContent = d.sub ?? "";
            if (frac >= 1) {
                pb.wrap.classList.add("done");
                pb.title.textContent = "Done";
            } else {
                pb.wrap.classList.remove("done");
                pb.title.textContent = "Working…";
            }
        });
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!PROGRESS_NODES.includes(nodeData?.name)) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            
            // 1. Zuerst die Progress Bar laden (reserviert den festen Platz)
            getProgressBar(this);

            // 2. Danach das Video Preview anhängen
            if (PREVIEW_NODES.includes(nodeData.name)) {
                createPreviewWidget(this);
            }
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            if (PREVIEW_NODES.includes(nodeData.name)) {
                fillPreview(this, message?.star_videos);
            }
        };

        // Node-Größe dynamisch an das Video anpassen
        const onResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function (size) {
            onResize?.apply(this, arguments);
            const pb = this.starPreview;
            if (pb && pb.wrap) {
                const el = pb.wrap.querySelector("video");
                if (el && el.videoWidth) {
                    // Verfügbare Breite berechnen und Höhe ableiten
                    const avail = Math.max(100, size[0] - 32);
                    let h = avail * (el.videoHeight / el.videoWidth);
                    h = Math.min(Math.max(h, PREVIEW_MIN_H), PREVIEW_MAX_H);
                    
                    // Styles und berechnete Höhe aktualisieren
                    el.style.height = h + "px";
                    pb.widget.computedHeight = h + PREVIEW_PAD + (pb.extraH || 0);
                }
            }
        };
    },
});