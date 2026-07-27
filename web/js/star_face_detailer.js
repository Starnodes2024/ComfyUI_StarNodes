import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const EVENT = "starface.detailer.progress";
const NODE_NAME = "StarFaceDetailerPlus";

// Tooltip fallback for frontends that ignore the python-side "tooltip" option.
const TOOLTIPS = {
    bbox_model: "Ultralytics BBOX detector used to find faces (models/ultralytics/bbox).",
    segm_model: "Optional SEGM detector for a precise face mask (models/ultralytics/segm).",
    max_faces: "Maximum number of faces that will be detailed. Extra faces stay untouched.",
    face_order: "Which faces get priority when Max Faces limits the count.",
    positive: "Positive prompt used for every face refinement.",
    negative: "Negative prompt used for every face refinement.",
    guide_size: "Minimum resolution each face crop is upscaled to before detailing.",
    max_size: "Maximum resolution of the face crop sent to the sampler.",
    crop_factor: "Crop size = face size x crop_factor (context around the face).",
    bbox_threshold: "Minimum detector confidence for a face to be accepted.",
    bbox_dilation: "Grow (or shrink) the detected face box in pixels per side.",
    drop_size: "Faces smaller than this (pixels) are ignored.",
    seed: "Noise seed. The same seed is reused for every face in one run.",
    steps: "Sampler steps per face.",
    cfg: "Classifier-free guidance scale.",
    sampler_name: "Sampler used for the face refinement.",
    scheduler: "Scheduler used for the face refinement.",
    denoise: "Inpaint strength. Lower = closer to original, higher = more regeneration.",
    feather: "Softness of the mask edge when blending the refined face back.",
};
for (let i = 1; i <= 5; i++) {
    TOOLTIPS[`lora_${i}`] = `LoRA applied only while detailing face #${i}. 'none' = disabled.`;
    TOOLTIPS[`lora_strength_${i}`] = `Strength of LoRA ${i} (model + CLIP).`;
}

function injectCss() {
    const css = `
    .sfd-wrap { box-sizing: border-box; width: 100%; padding: 6px 10px 8px;
        display: flex; flex-direction: column; gap: 6px;
        border-top: 1px solid rgba(255,255,255,0.08); }
    .sfd-hidden { display: none !important; }
    .sfd-status { font-size: 11px; letter-spacing: 0.4px; opacity: 0.85;
        font-family: sans-serif; display: flex; justify-content: space-between; }
    .sfd-row { display: flex; gap: 8px; align-items: center; }
    .sfd-img { width: 96px; height: 96px; object-fit: cover; border-radius: 6px;
        background: #111; border: 1px solid rgba(255,255,255,0.12); image-rendering: auto; }
    .sfd-barbox { flex: 1; display: flex; flex-direction: column; gap: 6px; }
    .sfd-bar { height: 10px; border-radius: 5px; overflow: hidden;
        background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.08); }
    .sfd-fill { height: 100%; width: 0%;
        background: linear-gradient(90deg, #f6c453, #f76b8a, #a06bff);
        transition: width 0.35s ease; border-radius: 5px; }
    .sfd-fill.sfd-done { background: linear-gradient(90deg, #3ddc84, #2bb673); }
    .sfd-pct { font-size: 11px; opacity: 0.7; text-align: right; font-family: sans-serif; }
    `;
    const style = document.createElement("style");
    style.textContent = css;
    document.head.appendChild(style);
}

function buildWidget(node) {
    const wrap = document.createElement("div");
    wrap.className = "sfd-wrap sfd-hidden";
    wrap.innerHTML = `
        <div class="sfd-status"><span class="sfd-label">Idle</span><span class="sfd-face"></span></div>
        <div class="sfd-row">
            <img class="sfd-img" alt="current face" draggable="false" />
            <div class="sfd-barbox">
                <div class="sfd-bar"><div class="sfd-fill"></div></div>
                <div class="sfd-pct">0%</div>
            </div>
        </div>`;

    const widget = node.addDOMWidget("sfd_live_preview", "sfd_live", wrap, {
        serialize: false,
        hideOnZoom: false,
    });
    widget.computeSize = () => [0, wrap.classList.contains("sfd-hidden") ? 0 : 150];

    node.__sfd = {
        wrap,
        label: wrap.querySelector(".sfd-label"),
        face: wrap.querySelector(".sfd-face"),
        img: wrap.querySelector(".sfd-img"),
        fill: wrap.querySelector(".sfd-fill"),
        pct: wrap.querySelector(".sfd-pct"),
        show() {
            wrap.classList.remove("sfd-hidden");
            node.setSize([node.size[0], node.computeSize()[1]]);
        },
        reset() {
            wrap.classList.add("sfd-hidden");
            node.setSize([node.size[0], node.computeSize()[1]]);
        },
    };
}

function findNode(id) {
    return app.graph.getNodeById(Number(id)) || app.graph.getNodeById(id);
}

app.registerExtension({
    name: "StarFaceDetailerPlus.LivePreview",

    setup() {
        injectCss();

        api.addEventListener(EVENT, (e) => {
            const d = e.detail || {};
            const node = findNode(d.node);
            if (!node || !node.__sfd) return;
            const ui = node.__sfd;

            if (d.status === "processing") {
                ui.show();
                ui.label.textContent = "Detailing face";
                ui.face.textContent = `${d.face} / ${d.total}`;
                if (d.preview) ui.img.src = d.preview;
                const pct = Math.round(((d.face - 1) / Math.max(d.total, 1)) * 100);
                ui.fill.classList.remove("sfd-done");
                ui.fill.style.width = `${pct}%`;
                ui.pct.textContent = `${pct}%`;
            } else if (d.status === "done") {
                if (d.total > 0) {
                    ui.show();
                    ui.label.textContent = "Finished";
                    ui.face.textContent = `${d.total} face${d.total === 1 ? "" : "s"}`;
                    ui.fill.classList.add("sfd-done");
                    ui.fill.style.width = "100%";
                    ui.pct.textContent = "100%";
                } else {
                    ui.label.textContent = "No faces detected";
                    ui.face.textContent = "";
                    ui.fill.style.width = "0%";
                    ui.pct.textContent = "0%";
                    ui.show();
                }
            }
        });

        // Reset the panel when this node starts executing again.
        api.addEventListener("executing", (e) => {
            if (e.detail == null) return;
            const node = findNode(e.detail);
            if (node && node.__sfd) node.__sfd.reset();
        });
    },

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            buildWidget(this);
            for (const w of this.widgets || []) {
                if (TOOLTIPS[w.name] && !w.tooltip) w.tooltip = TOOLTIPS[w.name];
            }
            return r;
        };
    },
});
