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
const PROGRESS_NODES = ["StarVideoCompressor", "StarVideoLoader", "StarVideoLoaderLowRAM"];
const LOADER_NODES = ["StarVideoLoader"];

const PREVIEW_MIN_H = 80;
const PREVIEW_MAX_H = 480;
const PREVIEW_PLACEHOLDER_H = 200;
const PREVIEW_PAD = 14;      // padding + border around the video box
const PROGRESS_H = 58;       // fixed height of the progress bar widget
const TRIM_ROW_H = 26;       // the trim range slider row in the loader
const LOADER_STAGE_MIN_H = 450;   // preview stage minimum (grows on resize)
const LOADER_CONTENT_MIN_H = LOADER_STAGE_MIN_H + 18 + TRIM_ROW_H + 20
    + PREVIEW_PAD;           // stage + info + range slider + values + padding
const LOADER_MIN_WIDTH = 500;

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

.star-vp { width: 100%; height: 100%; padding: 2px 6px 4px 6px;
           box-sizing: border-box; overflow: hidden; position: relative;
           display: flex; flex-direction: column; }
.star-vp-media { flex: 1 1 auto; min-height: 0; display: flex; }
.star-vp-box { width: 100%; border-radius: 6px; overflow: hidden;
               background: #101018; border: 1px solid #2c2c3a; }
.star-vp-empty { height: ${PREVIEW_PLACEHOLDER_H}px; display: flex;
                 align-items: center; justify-content: center; color: #5a5a6e;
                 font-size: 11px; font-family: sans-serif; text-align: center;
                 padding: 0 8px;
                 border: 1px dashed #333344; border-radius: 6px; }
.star-vp video { width: 100%; display: block; background: #000;
                 object-fit: contain; }
.star-vp-more { font-size: 10px; color: #8a8a9e; font-family: sans-serif;
                padding: 3px 2px 0 2px; white-space: nowrap; overflow: hidden;
                text-overflow: ellipsis; flex: none; }
.star-trim { font-family: sans-serif; padding: 2px 2px 0 2px; flex: none; }
.star-vp-fixed { width: min(450px, 100%); height: 100%;
                 max-width: 1024px; max-height: 1024px;
                 margin: auto;
                 display: flex; align-items: center; justify-content: center; }
.star-vp-fixed video { width: 100%; height: 100%;
                       max-width: 1024px; max-height: 1024px; }
.star-vp-fixed .star-vp-empty { width: 100%; height: 100%; border: none; }
.star-range { position: relative; height: ${TRIM_ROW_H}px; margin: 0 2px; }
.star-range-track { position: absolute; left: 0; right: 0; top: 50%;
                    transform: translateY(-50%); height: 8px;
                    border-radius: 4px; background: #3a3a4c; }
.star-range input[type=range] { position: absolute; inset: 0; width: 100%;
    height: 100%; margin: 0; -webkit-appearance: none; appearance: none;
    background: transparent; pointer-events: none; }
.star-range input[type=range]::-webkit-slider-runnable-track {
    background: transparent; }
.star-range input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none; appearance: none; pointer-events: auto;
    width: 14px; height: 14px; border-radius: 50%; background: #e8e8f2;
    border: 2px solid #6a5cff; cursor: ew-resize; }
.star-range input[type=range]::-moz-range-track { background: transparent; }
.star-range input[type=range]::-moz-range-thumb {
    pointer-events: auto; width: 12px; height: 12px; border-radius: 50%;
    background: #e8e8f2; border: 2px solid #6a5cff; cursor: ew-resize; }
.star-range.disabled { opacity: .35; }
.star-range.disabled input[type=range]::-webkit-slider-thumb { cursor: default; }
.star-range.disabled input[type=range]::-moz-range-thumb { cursor: default; }
.star-range-vals { display: flex; justify-content: space-between;
                   font-size: 10px; color: #cfcfe8; padding: 0 2px;
                   font-variant-numeric: tabular-nums; }

.star-crop-overlay { position: fixed; inset: 0; z-index: 10000;
    background: rgba(6,6,12,.75); display: flex; align-items: center;
    justify-content: center; }
.star-crop-panel { background: #16161f; border: 1px solid #3a3a4c;
    border-radius: 10px; padding: 12px; display: flex;
    flex-direction: column; gap: 8px; font-family: sans-serif;
    box-shadow: 0 12px 40px rgba(0,0,0,.6); }
.star-crop-title { font-size: 12px; color: #cfcfe8; }
.star-crop-stage { position: relative; line-height: 0; background: #000;
                   border-radius: 6px; overflow: hidden; }
.star-crop-stage video { width: 100%; height: 100%; display: block; }
.star-crop-draw { position: absolute; inset: 0; cursor: crosshair;
                  touch-action: none; }
.star-crop-rect { position: absolute; border: 2px solid #ff5252;
                  background: rgba(255,82,82,.18); box-sizing: border-box;
                  pointer-events: none; }
.star-crop-rect span { position: absolute; right: 0; bottom: 0;
    background: #ff5252; color: #fff; font-size: 9px; padding: 1px 4px;
    border-radius: 3px 0 0 0; line-height: 1.4; white-space: nowrap; }
.star-crop-controls { display: flex; align-items: center; gap: 8px; }
.star-crop-slider { flex: 1; accent-color: #6a5cff; }
.star-crop-time { font-size: 10px; color: #8a8a9e; min-width: 84px;
    text-align: right; font-variant-numeric: tabular-nums; }
.star-crop-btns { display: flex; justify-content: flex-end; gap: 8px; }
.star-crop-btns button { font-size: 12px; padding: 6px 16px;
    border-radius: 6px; border: 1px solid #3a3a4c; background: #24243a;
    color: #e8e8f2; cursor: pointer; }
.star-crop-btns button.apply { background: #2f7d4f;
    border-color: #2f7d4f; }
.star-crop-btns button.delete { background: #7d2f3a;
    border-color: #7d2f3a; margin-right: auto; }
.star-crop-btns button.apply:disabled { opacity: .4; cursor: default; }
.star-crop-badge { position: absolute; top: 6px; right: 6px; z-index: 7;
    background: rgba(16,16,26,.9); border: 1px solid #ff5252;
    border-radius: 4px; color: #ffd7db; font-size: 10px;
    font-family: sans-serif; padding: 2px 6px; display: none;
    white-space: nowrap; }
.star-crop-badge.show { display: block; }
.star-crop-badge .star-crop-x { cursor: pointer; margin-left: 6px;
    color: #ff8a95; font-weight: 700; }
`;
    document.head.appendChild(st);
}

// Preserve the user's width — only adjust height to fit content.
function relayout(node) {
    const sz = node.computeSize();
    node.setSize([Math.max(node.size[0], sz[0]), sz[1]]);
    node.graph?.setDirtyCanvas?.(true, true);
}

// ---------------------------------------------------------------- progress

function getProgressBar(node) {
    // Reuse the bar while its widget is still registered on the node.
    // The frontend detaches DOM widget elements between runs (e.g. between
    // batch items), so wrap.isConnected is not a reliable signal — trusting
    // it appended a new bar for every batch item instead of resetting.
    if (node.starProgress && (node.widgets || []).includes(node.starProgress.widget)) {
        return node.starProgress;
    }
    ensureStyle();
    // Remove stale bars left over from previous runs before adding a new one.
    for (const w of [...(node.widgets || [])]) {
        if (w.name === "star_progress") {
            w.onRemove?.();
            w.element?.remove?.();
            node.widgets.splice(node.widgets.indexOf(w), 1);
        }
    }
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

// ---------------------------------------------------------------- loader UI
// StarVideoLoader: "Load Video" probes the file via
// /starnodes/video_loader/info (no workflow run), fills an inline preview and
// enables two custom trim sliders. The sliders mirror into the
// start_frame/end_frame widgets (so the cut persists in the workflow) and
// seek the preview to the cut point. The frontend's own upload preview is
// hidden - ours replaces it, so the video shows only after Load is clicked.

function getWidget(node, name) {
    return (node.widgets || []).find((w) => w.name === name);
}

// The frontend adds its own "video-preview" DOM widget (canvasOnly) for
// video_upload inputs as soon as a file is selected. Intercept its creation
// and hide it completely: type "hidden" excludes it from the node layout
// (no leftover spacer), our preview replaces it - so the video only shows
// after the Load button is clicked.
function interceptNativePreview(node) {
    const orig = node.addDOMWidget;
    node.addDOMWidget = function (name, type, element, options) {
        const w = orig.call(node, name, type, element, options);
        if (name === "video-preview") {
            element.style.display = "none";
            w.hidden = true;   // excluded from layout + hit testing
            w.type = "hidden";
            w.computedHeight = 0;
            w.computeLayoutSize = () => ({ minHeight: 0, maxHeight: 0, minWidth: 0 });
        }
        return w;
    };
}

function seekLoaderPreview(node, isStart, v) {
    const st = node.starLoader;
    if (!st?.info || !st.videoEl?.duration) return;
    const fps = st.info.fps || 30;
    const frame = isStart ? v : Math.max(0, v - 1);
    st.videoEl.currentTime = Math.min(st.videoEl.duration, frame / fps);
}

// paint the range track: kept frames green between the handles, cut frames
// red outside
const TRIM_CUT_COLOR = "#d6455b";
const TRIM_KEEP_COLOR = "#2fbf71";

// MiniMax H3 renders frame counts of 17n+5 exactly - the loader's
// minimax_snap toggle snaps cut lengths to that grid (never grows the cut).
function snapMinimax(n) {
    return n > 5 ? 5 + 17 * Math.floor((n - 5) / 17) : n;
}

function minimaxSnapOn(node) {
    return getWidget(node, "minimax_snap")?.value === true;
}

// snap the current end slider value to the grid (no-op when off or when the
// slider range isn't initialised yet)
function applyMinimaxSnap(node) {
    const r = node.starLoader?.range;
    if (!r || !minimaxSnapOn(node)) return;
    const s = parseInt(r.startInput.value, 10) || 0;
    const e = parseInt(r.endInput.value, 10) || 0;
    const max = parseInt(r.endInput.max, 10) || e;
    const nv = Math.min(s + snapMinimax(e - s), max);
    if (nv !== e && nv > s) {
        r.endInput.value = nv;
        const endW = getWidget(node, "end_frame");
        if (endW) endW.value = nv;
    }
}

function paintTrimRange(node) {
    const st = node.starLoader;
    if (!st?.range) return;
    const { track, startInput, endInput, startVal, endVal } = st.range;
    if (startInput.disabled) {
        track.style.background = "#3a3a4c";
        startVal.textContent = "";
        endVal.textContent = "";
        return;
    }
    const max = parseFloat(startInput.max) || 1;
    const s = parseFloat(startInput.value) || 0;
    const e = parseFloat(endInput.value) || 0;
    const sp = Math.min(100, Math.max(0, s / max * 100));
    const ep = Math.min(100, Math.max(0, e / max * 100));
    track.style.background =
        `linear-gradient(90deg, ${TRIM_CUT_COLOR} 0%, ${TRIM_CUT_COLOR} ${sp}%, ` +
        `${TRIM_KEEP_COLOR} ${sp}%, ${TRIM_KEEP_COLOR} ${ep}%, ` +
        `${TRIM_CUT_COLOR} ${ep}%, ${TRIM_CUT_COLOR} 100%)`;
    startVal.textContent = `start ${s}`;
    endVal.textContent = `end ${e}`;
}

// Dual-handle range slider: one bar, left handle = start_frame, right
// handle = end_frame. Mirrors into the start_frame/end_frame widgets so the
// cut persists in the workflow.
function makeTrimRange(node) {
    const box = document.createElement("div");
    const range = document.createElement("div");
    range.className = "star-range disabled";
    const track = document.createElement("div");
    track.className = "star-range-track";
    const startInput = document.createElement("input");
    const endInput = document.createElement("input");
    for (const input of [startInput, endInput]) {
        input.type = "range";
        input.min = "0";
        input.max = "1";
        input.step = "1";
        input.value = input.min;
        input.disabled = true;
    }
    endInput.value = "1";
    range.appendChild(track);
    range.appendChild(startInput);
    range.appendChild(endInput);
    box.appendChild(range);
    const vals = document.createElement("div");
    vals.className = "star-range-vals";
    const startVal = document.createElement("span");
    const endVal = document.createElement("span");
    vals.appendChild(startVal);
    vals.appendChild(endVal);
    box.appendChild(vals);

    const startW = getWidget(node, "start_frame");
    const endW = getWidget(node, "end_frame");

    startInput.addEventListener("input", () => {
        let v = parseInt(startInput.value, 10) || 0;
        const e = parseInt(endInput.value, 10) || 0;
        if (v >= e) {                       // start stays below end
            v = Math.max(0, e - 1);
            startInput.value = v;
        }
        if (startW) startW.value = v;
        applyMinimaxSnap(node);
        paintTrimRange(node);
        seekLoaderPreview(node, true, v);
        node.graph?.setDirtyCanvas?.(true, true);
    });
    endInput.addEventListener("input", () => {
        let v = parseInt(endInput.value, 10) || 0;
        const s = parseInt(startInput.value, 10) || 0;
        if (v <= s) {                       // end stays above start
            v = s + 1;
            endInput.value = v;
        }
        if (minimaxSnapOn(node) && v - s > 5) {
            v = Math.min(s + snapMinimax(v - s),
                         parseInt(endInput.max, 10) || v);
            endInput.value = v;
        }
        if (endW) endW.value = v;
        paintTrimRange(node);
        seekLoaderPreview(node, false, v);
        node.graph?.setDirtyCanvas?.(true, true);
    });

    // number fields -> range handles (two-way sync)
    const hookWidget = (w, input, isStart) => {
        if (!w) return;
        const orig = w.callback;
        w.callback = function () {
            orig?.apply(this, arguments);
            const max = parseInt(input.max, 10) || 0;
            const v = Math.min(parseInt(w.value, 10) || 0, max);
            input.value = v;
            paintTrimRange(node);
            seekLoaderPreview(node, isStart, v);
        };
    };
    hookWidget(startW, startInput, true);
    hookWidget(endW, endInput, false);

    // toggling minimax_snap on snaps the current cut to the 17n+5 grid
    const snapW = getWidget(node, "minimax_snap");
    if (snapW) {
        const orig = snapW.callback;
        snapW.callback = function () {
            orig?.apply(this, arguments);
            applyMinimaxSnap(node);
            paintTrimRange(node);
            node.graph?.setDirtyCanvas?.(true, true);
        };
    }

    return { box, range, track, startInput, endInput, startVal, endVal };
}

function resetLoaderPreview(node) {
    const st = node.starLoader;
    if (!st) return;
    st.info = null;
    st.videoEl = null;
    st.mediaWrap.innerHTML =
        `<div class="star-vp-box star-vp-fixed">` +
        `<div class="star-vp-empty">Click "Load Video" to probe and ` +
        `preview the selected video, then cut it with the sliders ` +
        `below.</div></div>`;
    st.infoLine.textContent = "";
    const r = st.range;
    r.range.classList.add("disabled");
    for (const input of [r.startInput, r.endInput]) {
        input.disabled = true;
        input.max = "1";
    }
    r.startInput.value = "0";
    r.endInput.value = "1";
    paintTrimRange(node);
    relayout(node);
}

function fillLoaderPreview(node, videoName) {
    const st = node.starLoader;
    st.mediaWrap.innerHTML = "";
    const box = document.createElement("div");
    box.className = "star-vp-box star-vp-fixed";
    const el = document.createElement("video");
    el.controls = true;
    el.loop = true;
    el.preload = "auto";
    const params = new URLSearchParams({
        filename: videoName, subfolder: "", type: "input",
    });
    el.src = "/view?" + params.toString();
    box.appendChild(el);
    st.mediaWrap.appendChild(box);
    st.videoEl = el;

    // playback follows the cut: play starts at start_frame and loops back
    // when end_frame is reached
    const trimTimes = () => {
        const fps = st.info?.fps || 30;
        const start = (getWidget(node, "start_frame")?.value || 0) / fps;
        const endW = getWidget(node, "end_frame")?.value || 0;
        return { start, end: endW > 0 ? endW / fps : null };
    };
    el.addEventListener("play", () => {
        const t = trimTimes();
        if (el.currentTime < t.start - 0.05 || (t.end !== null && el.currentTime >= t.end)) {
            el.currentTime = t.start;
        }
    });
    el.addEventListener("timeupdate", () => {
        const t = trimTimes();
        if (t.end === null || el.paused) return;
        if (el.currentTime >= t.end) el.currentTime = t.start;
    });
    // show only the crop area if one is set (needs the video's real dims)
    el.addEventListener("loadedmetadata", () => applyCropToPreview(node));
    relayout(node);
}

async function loadVideoInfo(node) {
    const st = node.starLoader;
    const vw = getWidget(node, "video");
    const video = vw?.value;
    if (!video || !st) return;
    const forceRate = getWidget(node, "force_rate")?.value || 0;
    const kth = getWidget(node, "select_every_kth")?.value || 1;
    st.infoLine.textContent = "Probing video…";
    try {
        const resp = await app.api.fetchApi("/starnodes/video_loader/info", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ video, force_rate: forceRate, select_every_kth: kth }),
        });
        const d = await resp.json();
        if (d.status !== "ok") {
            st.infoLine.textContent = "Probe failed: " + (d.message || resp.status);
            return;
        }
        st.info = d;
        const n = d.frames_est;
        const startW = getWidget(node, "start_frame");
        const endW = getWidget(node, "end_frame");
        for (const w of [startW, endW]) {
            if (!w) continue;
            w.options = w.options || {};
            w.options.min = 0;
            w.options.max = n;
            w.options.step = 1;
        }
        if (startW) startW.value = Math.min(Math.max(0, startW.value || 0), n - 1);
        if (endW && (!(endW.value > 0) || endW.value > n)) endW.value = n;
        const r = st.range;
        r.range.classList.remove("disabled");
        for (const input of [r.startInput, r.endInput]) {
            input.disabled = false;
            input.max = String(n);
        }
        r.startInput.value = startW ? startW.value : 0;
        r.endInput.value = endW ? endW.value : n;
        applyMinimaxSnap(node);
        paintTrimRange(node);
        fillLoaderPreview(node, video);
        st.infoLine.textContent =
            `${d.brief} | ~${n} frames @ ${Number(d.fps).toFixed(2)} fps` +
            (d.has_audio ? " | audio ✓" : " | no audio");
    } catch (e) {
        st.infoLine.textContent = "Probe failed: " + e;
    }
    node.graph?.setDirtyCanvas?.(true, true);
}

// ---------------------------------------------------------------- crop dialog
// The "✂ Crop" button opens a modal with the video at max 1024px (aspect
// kept, fitted to the screen): a slider scrubs to the frame to draw on, the
// user drags a box, Apply stores 'x,y,w,h' fractions in the crop_rect
// widget - the backend runs the crop inside ffmpeg.

const CROP_MAX_SIDE = 1024;

function videoViewUrl(videoName) {
    const parts = String(videoName).split("/");
    const filename = parts.pop();
    return "/view?" + new URLSearchParams({
        filename, subfolder: parts.join("/"), type: "input",
    }).toString();
}

function parseCropRect(node) {
    const v = String(getWidget(node, "crop_rect")?.value ?? "").split(",");
    if (v.length < 4) return null;
    const [x, y, w, h] = v.map(Number);
    if (![x, y, w, h].every(Number.isFinite) || w <= 0 || h <= 0) return null;
    return { x, y, w, h };
}

// Show the crop in the node's inline preview: the box takes the crop's
// pixel aspect and the video is scaled + shifted so the crop rect fills it.
function applyCropToPreview(node) {
    const st = node.starLoader;
    const el = st?.videoEl;
    const box = st?.mediaWrap?.querySelector(".star-vp-fixed");
    if (!el || !box) return;
    const r = parseCropRect(node);
    const vw = st?.info?.width || el.videoWidth;
    const vh = st?.info?.height || el.videoHeight;
    if (!vw || !vh) return;
    // box = video (or crop) aspect fitted into min(stage, 1024) - the preview
    // never exceeds 1024 px in either dimension
    const aspect = r ? (r.w * vw) / (r.h * vh) : vw / vh;
    const maxW = Math.min(st.mediaWrap.clientWidth || 1024, 1024);
    const maxH = Math.min(st.mediaWrap.clientHeight || 1024, 1024);
    let bw = maxW, bh = bw / aspect;
    if (bh > maxH) { bh = maxH; bw = bh * aspect; }
    box.style.width = Math.round(bw) + "px";
    box.style.height = Math.round(bh) + "px";
    box.style.position = "relative";
    box.style.overflow = "hidden";
    if (!r) {
        el.style.cssText = "";
        return;
    }
    el.style.position = "absolute";
    el.style.left = "0";
    el.style.top = "0";
    el.style.objectFit = "fill";
    el.style.width = (100 / r.w) + "%";
    el.style.height = (100 / r.h) + "%";
    el.style.maxWidth = "none";
    el.style.maxHeight = "none";
    el.style.transformOrigin = "0 0";
    el.style.transform = `translate(${-r.x * 100}%, ${-r.y * 100}%)`;
}

function updateCropBadge(node) {
    const st = node.starLoader;
    if (!st?.cropBadge) return;
    const r = parseCropRect(node);
    if (!r) {
        st.cropBadge.classList.remove("show");
        applyCropToPreview(node);
        return;
    }
    st.cropBadge.innerHTML =
        `crop ${Math.round(r.w * 100)}%×${Math.round(r.h * 100)}%` +
        `<span class="star-crop-x">✕</span>`;
    st.cropBadge.classList.add("show");
    st.cropBadge.querySelector(".star-crop-x").onclick = (e) => {
        e.stopPropagation();
        const cw = getWidget(node, "crop_rect");
        if (cw) cw.value = "";
        updateCropBadge(node);
        node.graph?.setDirtyCanvas?.(true, true);
    };
    applyCropToPreview(node);
}

function openCropDialog(node) {
    const st = node.starLoader;
    const videoName = getWidget(node, "video")?.value;
    if (!videoName) {
        if (st) st.infoLine.textContent = "Select a video first.";
        return;
    }
    ensureStyle();

    const overlay = document.createElement("div");
    overlay.className = "star-crop-overlay";
    const panel = document.createElement("div");
    panel.className = "star-crop-panel";
    const title = document.createElement("div");
    title.className = "star-crop-title";
    title.textContent =
        "Crop video — drag on the frame to draw the crop area, " +
        "use the slider to pick a frame";
    const stage = document.createElement("div");
    stage.className = "star-crop-stage";
    const vid = document.createElement("video");
    vid.muted = true;
    vid.preload = "auto";
    vid.src = videoViewUrl(videoName);
    stage.appendChild(vid);
    const draw = document.createElement("div");
    draw.className = "star-crop-draw";
    stage.appendChild(draw);

    const controls = document.createElement("div");
    controls.className = "star-crop-controls";
    const slider = document.createElement("input");
    slider.type = "range";
    slider.className = "star-crop-slider";
    slider.min = "0";
    slider.max = "1";
    slider.step = "0.01";
    slider.value = "0";
    const timeEl = document.createElement("div");
    timeEl.className = "star-crop-time";
    controls.appendChild(slider);
    controls.appendChild(timeEl);

    const btns = document.createElement("div");
    btns.className = "star-crop-btns";
    const del = document.createElement("button");
    del.className = "delete";
    del.textContent = "Delete";
    del.disabled = !parseCropRect(node);
    const cancel = document.createElement("button");
    cancel.textContent = "Cancel";
    const apply = document.createElement("button");
    apply.className = "apply";
    apply.textContent = "Apply";
    btns.appendChild(del);
    btns.appendChild(cancel);
    btns.appendChild(apply);

    panel.appendChild(title);
    panel.appendChild(stage);
    panel.appendChild(controls);
    panel.appendChild(btns);
    overlay.appendChild(panel);
    document.body.appendChild(overlay);

    let rect = parseCropRect(node);   // pre-fill from a stored crop
    let startPt = null;
    const fmt = (t) => {
        const m = Math.floor(t / 60);
        const s = t - m * 60;
        return `${m}:${s.toFixed(1).padStart(4, "0")}`;
    };
    const renderRect = () => {
        draw.innerHTML = "";
        apply.disabled = !rect;
        if (!rect) return;
        const d = document.createElement("div");
        d.className = "star-crop-rect";
        d.style.left = (rect.x * 100) + "%";
        d.style.top = (rect.y * 100) + "%";
        d.style.width = (rect.w * 100) + "%";
        d.style.height = (rect.h * 100) + "%";
        const vw0 = st?.info?.width || vid.videoWidth;
        const vh0 = st?.info?.height || vid.videoHeight;
        if (vw0 && vh0) {
            const lbl = document.createElement("span");
            lbl.textContent =
                `${Math.round(rect.w * vw0)}×` +
                `${Math.round(rect.h * vh0)} px`;
            d.appendChild(lbl);
        }
        draw.appendChild(d);
    };
    const frac = (e) => {
        const b = draw.getBoundingClientRect();
        return {
            x: Math.min(1, Math.max(0, (e.clientX - b.left) / b.width)),
            y: Math.min(1, Math.max(0, (e.clientY - b.top) / b.height)),
        };
    };
    draw.addEventListener("pointerdown", (e) => {
        startPt = frac(e);
        draw.setPointerCapture(e.pointerId);
        e.preventDefault();
    });
    draw.addEventListener("pointermove", (e) => {
        if (!startPt) return;
        const p = frac(e);
        rect = {
            x: Math.min(startPt.x, p.x),
            y: Math.min(startPt.y, p.y),
            w: Math.abs(p.x - startPt.x),
            h: Math.abs(p.y - startPt.y),
        };
        renderRect();
    });
    draw.addEventListener("pointerup", () => {
        startPt = null;
        if (rect && (rect.w < 0.01 || rect.h < 0.01)) rect = null;
        renderRect();
    });
    draw.addEventListener("pointercancel", () => { startPt = null; });

    slider.addEventListener("input", () => {
        if (isFinite(vid.duration)) vid.currentTime = parseFloat(slider.value);
    });
    vid.addEventListener("timeupdate", () => {
        slider.value = vid.currentTime;
        timeEl.textContent =
            `${fmt(vid.currentTime)} / ${fmt(vid.duration || 0)}`;
    });
    vid.addEventListener("loadedmetadata", () => {
        const vw = vid.videoWidth || 16, vh = vid.videoHeight || 9;
        const scale = Math.min(
            CROP_MAX_SIDE / vw, CROP_MAX_SIDE / vh,
            (window.innerWidth * 0.9) / vw,
            (window.innerHeight * 0.65) / vh);
        stage.style.width = Math.round(vw * scale) + "px";
        stage.style.height = Math.round(vh * scale) + "px";
        slider.max = String(vid.duration || 1);
        slider.step = String(Math.max(0.01, (vid.duration || 1) / 500));
        renderRect();
    });
    vid.addEventListener("error", () => {
        title.textContent = "Could not load the video preview.";
    });

    const close = () => {
        document.removeEventListener("keydown", onKey);
        vid.pause();
        overlay.remove();
    };
    const onKey = (e) => { if (e.key === "Escape") close(); };
    document.addEventListener("keydown", onKey);
    overlay.addEventListener("pointerdown", (e) => {
        if (e.target === overlay) close();
    });
    cancel.addEventListener("click", close);
    del.addEventListener("click", () => {
        const cw = getWidget(node, "crop_rect");
        if (cw) {
            cw.value = "";
            cw.callback?.();
        }
        updateCropBadge(node);
        node.graph?.setDirtyCanvas?.(true, true);
        close();
    });
    apply.addEventListener("click", () => {
        if (!rect) return;
        const cw = getWidget(node, "crop_rect");
        if (cw) {
            cw.value = [rect.x, rect.y, rect.w, rect.h]
                .map((v) => v.toFixed(4)).join(",");
            cw.callback?.();
        }
        updateCropBadge(node);
        node.graph?.setDirtyCanvas?.(true, true);
        close();
    });
    renderRect();
}

function setupLoaderNode(node) {
    ensureStyle();
    interceptNativePreview(node);

    const wrap = document.createElement("div");
    wrap.className = "star-vp";
    const mediaWrap = document.createElement("div");
    mediaWrap.className = "star-vp-media";
    const infoLine = document.createElement("div");
    infoLine.className = "star-vp-more";
    const trimBox = document.createElement("div");
    trimBox.className = "star-trim";
    wrap.appendChild(mediaWrap);
    wrap.appendChild(infoLine);
    wrap.appendChild(trimBox);

    node.starLoader = { wrap, mediaWrap, infoLine, trimBox,
                        widget: null, videoEl: null, info: null };

    node.starLoader.range = makeTrimRange(node);
    trimBox.appendChild(node.starLoader.range.box);

    const cropBadge = document.createElement("div");
    cropBadge.className = "star-crop-badge";
    wrap.appendChild(cropBadge);
    node.starLoader.cropBadge = cropBadge;
    const cropW = getWidget(node, "crop_rect");
    if (cropW) {
        const origCropCb = cropW.callback;
        cropW.callback = function () {
            origCropCb?.apply(this, arguments);
            updateCropBadge(node);
        };
    }

    node.addWidget("button", "📼 Load Video", null, () => loadVideoInfo(node));
    node.addWidget("button", "✂ Crop", null, () => openCropDialog(node));
    const widget = node.addDOMWidget("star_loader_preview", "starLoaderPreview",
        wrap, { serialize: false, hideOnZoom: false });
    node.starLoader.widget = widget;
    // computeSize() ignores computedHeight for DOM widgets (defaults to
    // 50px) - computeLayoutSize is what sizes the node at creation.
    // minHeight fits stage+sliders exactly; no maxHeight cap, so the widget
    // absorbs extra space when the user expands the node (the stage grows
    // instead of leaving dead space below the progress bar).
    widget.computeLayoutSize = () => ({
        minHeight: LOADER_CONTENT_MIN_H, minWidth: 220,
    });
    widget.computedHeight = LOADER_CONTENT_MIN_H;  // initial; the layout
    // system's free-space distribution keeps it in sync afterwards
    resetLoaderPreview(node);

    // keep the run progress bar at the very bottom, below button + preview
    const pw = getWidget(node, "star_progress");
    if (pw) {
        node.widgets = node.widgets.filter((w) => w !== pw);
        node.widgets.push(pw);
    }

    // switching to another file invalidates the probe/preview and any crop
    const vw = getWidget(node, "video");
    if (vw) {
        const origCb = vw.callback;
        vw.callback = function () {
            origCb?.apply(this, arguments);
            const cw = getWidget(node, "crop_rect");
            if (cw?.value) {
                cw.value = "";
                updateCropBadge(node);
            }
            resetLoaderPreview(node);
        };
    }
    updateCropBadge(node);
    relayout(node);
    // make sure the 450px stage + range slider + progress bar all fit right away
    if ((node.size?.[0] ?? 0) < LOADER_MIN_WIDTH) {
        node.setSize([LOADER_MIN_WIDTH, node.computeSize()[1]]);
        node.graph?.setDirtyCanvas?.(true, true);
    }
    // workflows saved while the native preview still had layout space keep a
    // stale tall node size - re-fit once the frontend settles
    setTimeout(() => node.starLoader && relayout(node), 300);
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

            // 3. Loader: Load-Button + Probe-Preview + Slider-Hooks
            if (LOADER_NODES.includes(nodeData.name)) {
                setupLoaderNode(this);
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
            // refit the loader's cropped preview to the new stage size
            if (this.starLoader) applyCropToPreview(this);
        };
    },
});