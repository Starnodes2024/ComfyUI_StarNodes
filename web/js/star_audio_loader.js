import { app } from "../../../../scripts/app.js";

// Star Audio Loader — frontend companion.
//  - "Load Audio" button probes the file via /starnodes/audio_loader/info
//    (no workflow run), fills an inline <audio> preview and enables a
//    dual-handle range slider that sets start_time/end_time in seconds.
//  - The slider mirrors into the start_time/end_time widgets so the cut
//    persists in the workflow, and seeks the preview to the dragged handle.
//  - Reuses the same star-range styling as the video loader.
const LOADER_NODES = ["StarAudioLoader"];
const PROGRESS_NODES = ["StarAudioLoader"];

const STYLE_ID = "star-audio-loader-style";
const PREVIEW_PLACEHOLDER_H = 70;
const PREVIEW_PAD = 14;
const TRIM_ROW_H = 26;
const CONTENT_MIN_H = PREVIEW_PLACEHOLDER_H + 18 + TRIM_ROW_H + 20
    + PREVIEW_PAD;
const MIN_WIDTH = 420;
// Slider step in seconds — fine enough for precise cuts, coarse enough to
// not produce a million ticks on long files.
const SLIDER_STEP = 0.01;

const TRIM_CUT_COLOR = "#d6455b";
const TRIM_KEEP_COLOR = "#2fbf71";

function ensureStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const st = document.createElement("style");
    st.id = STYLE_ID;
    st.textContent = `
.star-ap { width: 100%; padding: 2px 6px 4px 6px; box-sizing: border-box;
           overflow: hidden; display: flex; flex-direction: column;
           font-family: sans-serif; }
.star-ap-media { flex: none; display: flex; }
.star-ap-box { width: 100%; border-radius: 6px; overflow: hidden;
               background: #101018; border: 1px solid #2c2c3a; }
.star-ap-empty { height: ${PREVIEW_PLACEHOLDER_H}px; display: flex;
                 align-items: center; justify-content: center; color: #5a5a6e;
                 font-size: 11px; text-align: center; padding: 0 8px;
                 border: 1px dashed #333344; border-radius: 6px; }
.star-ap audio { width: 100%; display: block; }
.star-ap-info { font-size: 10px; color: #8a8a9e; padding: 3px 2px 0 2px;
                white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
                flex: none; }
.star-ap-trim { padding: 2px 2px 0 2px; flex: none; }
.star-ap-range { position: relative; height: ${TRIM_ROW_H}px; margin: 0 2px; }
.star-ap-range-track { position: absolute; left: 0; right: 0; top: 50%;
                       transform: translateY(-50%); height: 8px;
                       border-radius: 4px; background: #3a3a4c; }
.star-ap-range input[type=range] { position: absolute; inset: 0; width: 100%;
    height: 100%; margin: 0; -webkit-appearance: none; appearance: none;
    background: transparent; pointer-events: none; }
.star-ap-range input[type=range]::-webkit-slider-runnable-track {
    background: transparent; }
.star-ap-range input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none; appearance: none; pointer-events: auto;
    width: 14px; height: 14px; border-radius: 50%; background: #e8e8f2;
    border: 2px solid #6a5cff; cursor: ew-resize; }
.star-ap-range input[type=range]::-moz-range-track { background: transparent; }
.star-ap-range input[type=range]::-moz-range-thumb {
    pointer-events: auto; width: 12px; height: 12px; border-radius: 50%;
    background: #e8e8f2; border: 2px solid #6a5cff; cursor: ew-resize; }
.star-ap-range.disabled { opacity: .35; }
.star-ap-range.disabled input[type=range]::-webkit-slider-thumb { cursor: default; }
.star-ap-range.disabled input[type=range]::-moz-range-thumb { cursor: default; }
.star-ap-range-vals { display: flex; justify-content: space-between;
                      font-size: 10px; color: #cfcfe8; padding: 0 2px;
                      font-variant-numeric: tabular-nums; }
`;
    document.head.appendChild(st);
}

function relayout(node) {
    const sz = node.computeSize();
    node.setSize([Math.max(node.size[0], sz[0]), sz[1]]);
    node.graph?.setDirtyCanvas?.(true, true);
}

function getWidget(node, name) {
    return (node.widgets || []).find((w) => w.name === name);
}

// The frontend may add its own "audio-preview" DOM widget for audio_upload
// inputs. Hide it completely — our preview replaces it.
function interceptNativePreview(node) {
    const orig = node.addDOMWidget.bind(node);
    node.addDOMWidget = function (name, type, element, options) {
        const w = orig(name, type, element, options);
        if (name === "audio-preview" || name === "video-preview") {
            element.style.display = "none";
            w.hidden = true;
            w.type = "hidden";
            w.computedHeight = 0;
            w.computeLayoutSize = () => ({ minHeight: 0, maxHeight: 0, minWidth: 0 });
        }
        return w;
    };
}

function fmtTime(s) {
    if (s == null || isNaN(s)) return "0.00";
    return s.toFixed(2);
}

function paintTrimRange(node) {
    const st = node.starAudio;
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
    startVal.textContent = `start ${fmtTime(s)}s`;
    endVal.textContent = `end ${fmtTime(e)}s`;
}

function seekAudioPreview(node, isStart, v) {
    const st = node.starAudio;
    if (!st?.audioEl?.duration) return;
    const dur = st.audioEl.duration;
    const t = isStart ? v : Math.max(0, v - SLIDER_STEP);
    st.audioEl.currentTime = Math.min(dur, Math.max(0, t));
}

// Dual-handle range slider: left = start_time, right = end_time (seconds).
// Mirrors into the start_time/end_time widgets so the cut persists.
function makeTrimRange(node) {
    const box = document.createElement("div");
    const range = document.createElement("div");
    range.className = "star-ap-range disabled";
    const track = document.createElement("div");
    track.className = "star-ap-range-track";
    const startInput = document.createElement("input");
    const endInput = document.createElement("input");
    for (const input of [startInput, endInput]) {
        input.type = "range";
        input.min = "0";
        input.max = "1";
        input.step = String(SLIDER_STEP);
        input.value = input.min;
        input.disabled = true;
    }
    endInput.value = "1";
    range.appendChild(track);
    range.appendChild(startInput);
    range.appendChild(endInput);
    box.appendChild(range);
    const vals = document.createElement("div");
    vals.className = "star-ap-range-vals";
    const startVal = document.createElement("span");
    const endVal = document.createElement("span");
    vals.appendChild(startVal);
    vals.appendChild(endVal);
    box.appendChild(vals);

    const startW = getWidget(node, "start_time");
    const endW = getWidget(node, "end_time");

    startInput.addEventListener("input", () => {
        let v = parseFloat(startInput.value) || 0;
        const e = parseFloat(endInput.value) || 0;
        if (v >= e) {                       // start stays below end
            v = Math.max(0, e - SLIDER_STEP);
            startInput.value = v;
        }
        if (startW) startW.value = v;
        paintTrimRange(node);
        seekAudioPreview(node, true, v);
        node.graph?.setDirtyCanvas?.(true, true);
    });
    endInput.addEventListener("input", () => {
        let v = parseFloat(endInput.value) || 0;
        const s = parseFloat(startInput.value) || 0;
        if (v <= s) {                       // end stays above start
            v = s + SLIDER_STEP;
            endInput.value = v;
        }
        if (endW) endW.value = v;
        paintTrimRange(node);
        seekAudioPreview(node, false, v);
        node.graph?.setDirtyCanvas?.(true, true);
    });

    // number fields -> range handles (two-way sync)
    const hookWidget = (w, input, isStart) => {
        if (!w) return;
        const orig = w.callback;
        w.callback = function () {
            orig?.apply(this, arguments);
            const max = parseFloat(input.max) || 0;
            const v = Math.min(parseFloat(w.value) || 0, max);
            input.value = v;
            paintTrimRange(node);
            seekAudioPreview(node, isStart, v);
        };
    };
    hookWidget(startW, startInput, true);
    hookWidget(endW, endInput, false);

    return { box, range, track, startInput, endInput, startVal, endVal };
}

function resetAudioPreview(node) {
    const st = node.starAudio;
    if (!st) return;
    st.info = null;
    st.audioEl = null;
    st.mediaWrap.innerHTML =
        `<div class="star-ap-box"><div class="star-ap-empty">` +
        `Click "Load Audio" to probe and preview the selected audio, ` +
        `then cut it with the sliders below.</div></div>`;
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

function fillAudioPreview(node, audioName) {
    const st = node.starAudio;
    st.mediaWrap.innerHTML = "";
    const box = document.createElement("div");
    box.className = "star-ap-box";
    const el = document.createElement("audio");
    el.controls = true;
    el.loop = true;
    el.preload = "auto";
    const params = new URLSearchParams({
        filename: audioName, subfolder: "", type: "input",
    });
    el.src = "/view?" + params.toString();
    box.appendChild(el);
    st.mediaWrap.appendChild(box);
    st.audioEl = el;

    // playback follows the cut: play starts at start_time and loops back
    // when end_time is reached
    const trimTimes = () => {
        const start = parseFloat(getWidget(node, "start_time")?.value) || 0;
        const endW = parseFloat(getWidget(node, "end_time")?.value) || 0;
        return { start, end: endW > 0 ? endW : null };
    };
    el.addEventListener("play", () => {
        const t = trimTimes();
        if (el.currentTime < t.start - 0.05 ||
            (t.end !== null && el.currentTime >= t.end)) {
            el.currentTime = t.start;
        }
    });
    el.addEventListener("timeupdate", () => {
        const t = trimTimes();
        if (t.end === null || el.paused) return;
        if (el.currentTime >= t.end) el.currentTime = t.start;
    });
    relayout(node);
}

async function loadAudioInfo(node) {
    const st = node.starAudio;
    const aw = getWidget(node, "audio");
    const audio = aw?.value;
    if (!audio || !st) return;
    st.infoLine.textContent = "Probing audio…";
    try {
        const resp = await app.api.fetchApi("/starnodes/audio_loader/info", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ audio }),
        });
        const d = await resp.json();
        if (d.status !== "ok") {
            st.infoLine.textContent = "Probe failed: " + (d.message || resp.status);
            return;
        }
        st.info = d;
        const dur = d.duration;
        const startW = getWidget(node, "start_time");
        const endW = getWidget(node, "end_time");
        for (const w of [startW, endW]) {
            if (!w) continue;
            w.options = w.options || {};
            w.options.min = 0;
            w.options.max = dur;
            w.options.step = SLIDER_STEP;
        }
        if (startW) startW.value = Math.min(Math.max(0, startW.value || 0), dur - SLIDER_STEP);
        if (endW && (!(endW.value > 0) || endW.value > dur)) endW.value = dur;
        const r = st.range;
        r.range.classList.remove("disabled");
        for (const input of [r.startInput, r.endInput]) {
            input.disabled = false;
            input.max = String(dur);
        }
        r.startInput.value = startW ? startW.value : 0;
        r.endInput.value = endW ? endW.value : dur;
        paintTrimRange(node);
        fillAudioPreview(node, audio);
        st.infoLine.textContent =
            `${d.brief} | ${dur.toFixed(3)}s` +
            (d.acodec ? ` | ${d.acodec}` : "");
    } catch (e) {
        st.infoLine.textContent = "Probe failed: " + e;
    }
    node.graph?.setDirtyCanvas?.(true, true);
}

// Hide a DOM widget completely (excluded from layout + hit testing).
function hideDomWidget(w) {
    if (!w) return;
    if (w.element) w.element.style.display = "none";
    w.hidden = true;
    w.type = "hidden";
    w.computedHeight = 0;
    w.computeLayoutSize = () => ({ minHeight: 0, maxHeight: 0, minWidth: 0 });
}

function setupAudioLoaderNode(node) {
    ensureStyle();
    interceptNativePreview(node);

    // Hide the native audioUI preview widget that Comfy.AudioWidget created
    // for us — we render our own <audio> preview with cut-range looping.
    hideDomWidget(getWidget(node, "audioUI"));

    const wrap = document.createElement("div");
    wrap.className = "star-ap";
    const mediaWrap = document.createElement("div");
    mediaWrap.className = "star-ap-media";
    const infoLine = document.createElement("div");
    infoLine.className = "star-ap-info";
    const trimBox = document.createElement("div");
    trimBox.className = "star-ap-trim";
    wrap.appendChild(mediaWrap);
    wrap.appendChild(infoLine);
    wrap.appendChild(trimBox);

    node.starAudio = { wrap, mediaWrap, infoLine, trimBox,
                      widget: null, audioEl: null, info: null };

    node.starAudio.range = makeTrimRange(node);
    trimBox.appendChild(node.starAudio.range.box);

    node.addWidget("button", "🔊 Load Audio", null, () => loadAudioInfo(node));
    const widget = node.addDOMWidget("star_audio_preview", "starAudioPreview",
        wrap, { serialize: false, hideOnZoom: false });
    node.starAudio.widget = widget;
    widget.computeLayoutSize = () => ({
        minHeight: CONTENT_MIN_H, minWidth: 220,
    });
    widget.computedHeight = CONTENT_MIN_H;
    resetAudioPreview(node);

    // switching to another file invalidates the probe/preview
    const aw = getWidget(node, "audio");
    if (aw) {
        const origCb = aw.callback;
        aw.callback = function () {
            origCb?.apply(this, arguments);
            resetAudioPreview(node);
        };
    }
    relayout(node);
    if ((node.size?.[0] ?? 0) < MIN_WIDTH) {
        node.setSize([MIN_WIDTH, node.computeSize()[1]]);
        node.graph?.setDirtyCanvas?.(true, true);
    }
    setTimeout(() => node.starAudio && relayout(node), 300);
}

app.registerExtension({
    name: "StarNodes.AudioLoader",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!LOADER_NODES.includes(nodeData?.name)) return;

        // Comfy.UploadAudio (core) sees audio_upload:true and appends an
        // `upload` input whose AUDIOUPLOAD widget creator immediately calls
        // updateUIWidget(widgets.find(name==='audioUI'), ...). The audioUI
        // widget is only injected by Comfy.AudioWidget for hardcoded core
        // class names, so for our node it would be undefined and crash node
        // creation. Inject audioUI ourselves — BEFORE the upload entry so the
        // AUDIO_UI widget is created first and AUDIOUPLOAD finds it. The core
        // AUDIO_UI creator (Comfy.AudioWidget) builds the widget for us.
        const req = nodeData.input?.required;
        if (req && req.upload && !req.audioUI) {
            const rebuilt = {};
            for (const [k, v] of Object.entries(req)) {
                if (k === "upload") rebuilt.audioUI = ["AUDIO_UI", {}];
                rebuilt[k] = v;
            }
            nodeData.input.required = rebuilt;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            setupAudioLoaderNode(this);
        };
    },
});
