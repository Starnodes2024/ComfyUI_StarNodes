import { app } from "../../../../scripts/app.js";

// Star Nodes — Fancy DOM progress bar for sampler / refiner nodes.
//
// Listens to "star_nodes.progress" websocket events (same event name as
// the video pack) and renders an animated gradient progress bar inside
// the node.  The ComfyUI built-in top progress bar continues to work
// as a fallback (it is driven by comfy.utils.ProgressBar on the Python
// side).
//
// The CSS is shared with star_video_compressor.js via a common <style>
// element ID ("star-nodes-style").  Whichever extension loads first
// injects the styles; the other finds them already present and skips.

const PROGRESS_NODES = [
    "StarSampler",
    "StarSDUpscaleRefiner",
    "StarSDUpscaleRefinerAdvanced",
    "LTXVSulphurAllInOne",
    "StarTiledSeedVRUpscaler",
    "StarPanoramaViewerPro",
];

const PROGRESS_H = 58;
const STYLE_ID = "star-nodes-style";

// ─── CSS (shared with star_video_compressor.js) ───────────────────────────
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
`;
    document.head.appendChild(st);
}

// ─── Layout helper ─────────────────────────────────────────────────────────
function relayout(node) {
    node.setSize(node.computeSize());
    node.graph?.setDirtyCanvas?.(true, true);
}

// ─── Progress bar widget ───────────────────────────────────────────────────
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

// ─── ComfyUI extension registration ────────────────────────────────────────
app.registerExtension({
    name: "StarNodes.SamplerProgress",

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
            getProgressBar(this);
        };
    },
});
