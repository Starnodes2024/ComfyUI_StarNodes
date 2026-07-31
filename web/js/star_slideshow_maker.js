import { app } from "../../../../scripts/app.js";

// Star Slideshow Maker frontend companion:
//  - grows image_1, image_2, image_3, ... sockets automatically
//  - shows encoding progress
//  - displays the encoded video inline after execution
const SLIDESHOW_NODES = ["StarSlideshowMaker"];
const IMAGE_INPUT_RE = /^image_([1-9][0-9]*)$/;

const PREVIEW_MIN_H = 80;
const PREVIEW_MAX_H = 480;
const PREVIEW_PLACEHOLDER_H = 200;
const PREVIEW_PAD = 14;
const PROGRESS_H = 58;
const STYLE_ID = "star-slideshow-maker-style";

function ensureStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const st = document.createElement("style");
    st.id = STYLE_ID;
    st.textContent = `
.star-ss-pb { padding: 6px 8px 4px 8px; font-family: sans-serif;
              user-select: none; }
.star-ss-pb-top { display: flex; justify-content: space-between;
                  align-items: baseline; font-size: 11px; color: #cfcfe8;
                  margin-bottom: 4px; }
.star-ss-pb-pct { font-weight: 700; font-size: 13px; color: #fff;
                  font-variant-numeric: tabular-nums; }
.star-ss-pb-track { height: 12px; border-radius: 6px; background: #17171f;
                    border: 1px solid #3a3a4c; overflow: hidden;
                    box-shadow: inset 0 1px 2px rgba(0,0,0,.6); }
.star-ss-pb-fill { height: 100%; width: 0%; border-radius: 6px;
                   position: relative;
                   background: linear-gradient(90deg,#6a5cff 0%,#00c8ff 100%);
                   box-shadow: 0 0 8px rgba(0,200,255,.55);
                   transition: width .15s ease-out; }
.star-ss-pb-fill::after { content: ""; position: absolute; inset: 0;
                   background: repeating-linear-gradient(45deg,
                       rgba(255,255,255,.22) 0 10px, transparent 10px 20px);
                   animation: starSlideshowStripes .8s linear infinite; }
.star-ss-pb-sub { margin-top: 3px; font-size: 10px; color: #8a8a9e;
                  min-height: 12px; white-space: nowrap; overflow: hidden;
                  text-overflow: ellipsis; }
.star-ss-pb.done .star-ss-pb-fill {
                   background: linear-gradient(90deg,#2fbf71,#8ce99a);
                   box-shadow: 0 0 8px rgba(80,220,130,.5); }
.star-ss-pb.done .star-ss-pb-fill::after { animation: none; background: none; }
@keyframes starSlideshowStripes { from { background-position: 0 0; }
                                  to { background-position: 28px 0; } }

.star-ss-preview { width: 100%; padding: 2px 6px 4px 6px;
                   box-sizing: border-box; }
.star-ss-preview-box { width: 100%; border-radius: 6px; overflow: hidden;
                       background: #101018; border: 1px solid #2c2c3a; }
.star-ss-preview-empty { height: ${PREVIEW_PLACEHOLDER_H}px; display: flex;
                         align-items: center; justify-content: center;
                         color: #5a5a6e; font-size: 11px;
                         font-family: sans-serif; text-align: center;
                         border: 1px dashed #333344; border-radius: 6px;
                         padding: 0 12px; box-sizing: border-box; }
.star-ss-preview video { width: 100%; display: block; background: #000; }
`;
    document.head.appendChild(st);
}

function relayout(node) {
    node.setSize(node.computeSize());
    node.graph?.setDirtyCanvas?.(true, true);
}

// ---------------------------------------------------------- dynamic inputs

function imageInputIndex(input) {
    const match = IMAGE_INPUT_RE.exec(input?.name || "");
    return match ? Number(match[1]) : null;
}

function updateDynamicInputs(node) {
    if (!node || !Array.isArray(node.inputs)) return;
    let changed = false;

    if (!node.inputs.some((input) => input.name === "image_1")) {
        node.addInput("image_1", "IMAGE");
        changed = true;
    }

    const slots = () => node.inputs
        .filter((input) => imageInputIndex(input) !== null)
        .sort((a, b) => imageInputIndex(a) - imageInputIndex(b));

    // Always keep one trailing empty connector. Middle gaps are preserved so
    // existing links are not shifted silently.
    for (;;) {
        const list = slots();
        if (list.length <= 1) break;
        const last = list[list.length - 1];
        const previous = list[list.length - 2];
        if (last.link === null && previous.link === null &&
                imageInputIndex(last) > 1) {
            node.removeInput(node.inputs.indexOf(last));
            changed = true;
        } else {
            break;
        }
    }

    const list = slots();
    const last = list[list.length - 1];
    if (last.link !== null) {
        node.addInput(`image_${imageInputIndex(last) + 1}`, "IMAGE");
        changed = true;
    }

    if (changed) relayout(node);
}

function installDynamicInputs(nodeType) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        onNodeCreated?.apply(this, arguments);
        updateDynamicInputs(this);
    };

    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function () {
        onConnectionsChange?.apply(this, arguments);
        updateDynamicInputs(this);
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
        onConfigure?.apply(this, arguments);
        setTimeout(() => updateDynamicInputs(this), 0);
    };
}

// ---------------------------------------------------------------- progress

function getProgressBar(node) {
    if (node.starSlideshowProgress &&
            node.starSlideshowProgress.wrap.isConnected) {
        return node.starSlideshowProgress;
    }
    ensureStyle();
    const wrap = document.createElement("div");
    wrap.className = "star-ss-pb";
    wrap.innerHTML =
        `<div class="star-ss-pb-top"><span class="star-ss-pb-title">` +
        `Working…</span><span class="star-ss-pb-pct">0%</span></div>` +
        `<div class="star-ss-pb-track"><div class="star-ss-pb-fill"></div>` +
        `</div><div class="star-ss-pb-sub"></div>`;
    const widget = node.addDOMWidget(
        "star_slideshow_progress", "starSlideshowProgress", wrap,
        { serialize: false, hideOnZoom: false });
    widget.computedHeight = PROGRESS_H;
    node.starSlideshowProgress = {
        widget,
        wrap,
        title: wrap.querySelector(".star-ss-pb-title"),
        pct: wrap.querySelector(".star-ss-pb-pct"),
        fill: wrap.querySelector(".star-ss-pb-fill"),
        sub: wrap.querySelector(".star-ss-pb-sub"),
    };
    relayout(node);
    return node.starSlideshowProgress;
}

// ----------------------------------------------------------------- preview

function setPreviewHeight(node, contentH) {
    const preview = node.starSlideshowPreview;
    if (!preview) return;
    preview.widget.computedHeight = contentH + PREVIEW_PAD;
    relayout(node);
}

function createPreviewWidget(node) {
    ensureStyle();
    const wrap = document.createElement("div");
    wrap.className = "star-ss-preview";
    wrap.innerHTML = `<div class="star-ss-preview-empty">Video preview — ` +
        `run the node to see the encoded slideshow here.</div>`;
    const widget = node.addDOMWidget(
        "star_slideshow_preview", "starSlideshowPreview", wrap,
        { serialize: false, hideOnZoom: false });
    node.starSlideshowPreview = { widget, wrap };
    setPreviewHeight(node, PREVIEW_PLACEHOLDER_H);
    return node.starSlideshowPreview;
}

function fillPreview(node, videos) {
    const preview = (node.starSlideshowPreview &&
            node.starSlideshowPreview.wrap.isConnected)
        ? node.starSlideshowPreview : createPreviewWidget(node);
    const wrap = preview.wrap;
    wrap.innerHTML = "";

    if (!videos || !videos.length) {
        wrap.innerHTML = `<div class="star-ss-preview-empty">No video ` +
            `produced.</div>`;
        setPreviewHeight(node, PREVIEW_PLACEHOLDER_H);
        return;
    }

    const info = videos[0];
    const box = document.createElement("div");
    box.className = "star-ss-preview-box";
    const video = document.createElement("video");
    video.controls = true;
    video.loop = true;
    video.muted = true;
    video.preload = "metadata";
    video.style.height = `${PREVIEW_PLACEHOLDER_H}px`;
    const params = new URLSearchParams({
        filename: info.filename,
        subfolder: info.subfolder ?? "",
        type: info.type ?? "output",
    });
    video.src = `/view?${params.toString()}`;
    box.appendChild(video);
    wrap.appendChild(box);
    setPreviewHeight(node, PREVIEW_PLACEHOLDER_H);

    video.addEventListener("loadedmetadata", () => {
        const available = Math.max(100, (node.size?.[0] ?? 320) - 32);
        let height = available * (video.videoHeight || 9) /
            (video.videoWidth || 16);
        height = Math.min(Math.max(height, PREVIEW_MIN_H), PREVIEW_MAX_H);
        video.style.height = `${height}px`;
        setPreviewHeight(node, height);
    });
}

// ---------------------------------------------------------------- extension

app.registerExtension({
    name: "StarNodes.StarSlideshowMaker",

    setup() {
        app.api.addEventListener("star_slideshow.progress", (event) => {
            const data = event.detail || {};
            const node = app.graph.getNodeById(Number(data.node));
            if (!node || !SLIDESHOW_NODES.includes(node.type)) return;
            const bar = getProgressBar(node);
            const fraction = Math.min(Math.max(data.value ?? 0, 0), 1);
            bar.fill.style.width = `${(fraction * 100).toFixed(1)}%`;
            bar.pct.textContent = data.text ??
                `${Math.round(fraction * 100)}%`;
            bar.sub.textContent = data.sub ?? "";
            if (fraction >= 1) {
                bar.wrap.classList.add("done");
                bar.title.textContent = "Done";
            } else {
                bar.wrap.classList.remove("done");
                bar.title.textContent = "Working…";
            }
        });
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!SLIDESHOW_NODES.includes(nodeData?.name)) return;
        installDynamicInputs(nodeType);

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            getProgressBar(this);
            createPreviewWidget(this);
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            fillPreview(this, message?.star_videos);
        };
    },
});
