import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";

// Star Minimax All In One — frontend companion.
//  - live resolution readout (ratio + MP, mirrors the Python math)
//  - animated DOM progress bar for the sampling steps
// Reference image/video/audio slots auto-expand through the same native
// Autogrow mechanism the core MiniMax H3 Reference to Video node uses.

const NODE_CLASS = "StarMinimaxAllInOne";

const RATIOS = {
    "1:1 (Square)": [1, 1],
    "2:3 (Portrait Photo)": [2, 3],
    "3:2 (Photo)": [3, 2],
    "3:4 (Portrait Standard)": [3, 4],
    "4:3 (Standard)": [4, 3],
    "9:16 (Portrait Widescreen)": [9, 16],
    "16:9 (Widescreen)": [16, 9],
    "21:9 (Ultrawide)": [21, 9],
};

function computeSize(ratioLabel, mp) {
    const [wr, hr] = RATIOS[ratioLabel] || [16, 9];
    const scale = Math.sqrt((mp * 1024 * 1024) / (wr * hr));
    const w = Math.max(32, Math.round((wr * scale) / 32) * 32);
    const h = Math.max(32, Math.round((hr * scale) / 32) * 32);
    return [w, h];
}

function durationToLength(seconds) {
    const n = Math.max(5, Math.round(seconds * 24));
    return n + ((5 - (n % 17)) % 17);
}

function el(tag, styles, text) {
    const e = document.createElement(tag);
    Object.assign(e.style, styles);
    if (text != null) e.textContent = text;
    return e;
}

function buildInfoRow() {
    const row = el("div", {
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        gap: "6px",
        padding: "4px 8px",
        margin: "2px 6px",
        borderRadius: "6px",
        fontSize: "11px",
        fontFamily: "monospace",
        letterSpacing: "0.4px",
        color: "#8fd3ff",
        background: "rgba(24, 42, 74, 0.55)",
        border: "1px solid rgba(90, 140, 220, 0.35)",
        boxSizing: "border-box",
        width: "calc(100% - 12px)",
    });
    return row;
}

function buildProgressBar() {
    const wrap = el("div", {
        display: "none",
        flexDirection: "column",
        gap: "3px",
        padding: "5px 8px 6px",
        margin: "2px 6px 4px",
        borderRadius: "7px",
        background: "rgba(14, 22, 40, 0.85)",
        border: "1px solid rgba(90, 140, 220, 0.35)",
        boxSizing: "border-box",
        width: "calc(100% - 12px)",
    });

    const track = el("div", {
        position: "relative",
        height: "10px",
        borderRadius: "5px",
        overflow: "hidden",
        background: "rgba(255, 255, 255, 0.08)",
    });

    const fill = el("div", {
        height: "100%",
        width: "0%",
        borderRadius: "5px",
        background: "linear-gradient(90deg, #2bd2ff 0%, #7a5cff 55%, #ff5ca8 100%)",
        boxShadow: "0 0 8px rgba(122, 92, 255, 0.8)",
        transition: "width 120ms linear",
    });

    const shimmer = el("div", {
        position: "absolute",
        top: "0",
        left: "0",
        height: "100%",
        width: "40%",
        borderRadius: "5px",
        background: "linear-gradient(90deg, transparent, rgba(255,255,255,0.35), transparent)",
        animation: "starMinimaxShimmer 1.2s infinite",
        display: "none",
    });

    const label = el("div", {
        fontSize: "10px",
        fontFamily: "monospace",
        textAlign: "center",
        color: "#9fb8dd",
    }, "waiting…");

    track.appendChild(fill);
    track.appendChild(shimmer);
    wrap.appendChild(track);
    wrap.appendChild(label);

    if (!document.getElementById("star-minimax-aio-style")) {
        const style = document.createElement("style");
        style.id = "star-minimax-aio-style";
        style.textContent = `
@keyframes starMinimaxShimmer {
    0%   { transform: translateX(-100%); }
    100% { transform: translateX(350%); }
}`;
        document.head.appendChild(style);
    }

    return { wrap, fill, shimmer, label };
}

app.registerExtension({
    name: "Star.MinimaxAllInOne",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_CLASS) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            const node = this;

            // ---------- live resolution / length readout ----------
            const info = buildInfoRow();
            node.addDOMWidget("star_res_info", "starinfo", info, {
                serializeValue: () => undefined,
                hideOnZoom: false,
            });

            // ---------- progress bar ----------
            const bar = buildProgressBar();
            node.addDOMWidget("star_progress", "starprogress", bar.wrap, {
                serializeValue: () => undefined,
                hideOnZoom: false,
            });

            node._starUpdateInfo = () => {
                const get = (name) => node.widgets?.find((w) => w.name === name)?.value;
                const ratio = get("aspect_ratio") ?? "16:9 (Widescreen)";
                const mp = Number(get("megapixels") ?? 0.5);
                const match = Boolean(get("match_ratio_from_image"));
                const dur = Number(get("duration") ?? 5);
                const [w, h] = computeSize(ratio, mp);
                const len = durationToLength(dur);
                info.textContent = match
                    ? `auto-ratio @ ${mp} MP  •  ~${w}x${h} if 16:9  •  ${len} frames`
                    : `${w} x ${h}  •  ${mp} MP  •  ${len} frames`;
            };

            node._starProgress = {
                start() {
                    bar.wrap.style.display = "flex";
                    bar.fill.style.width = "0%";
                    bar.fill.style.opacity = "0.35";
                    bar.shimmer.style.display = "block";
                    bar.label.textContent = "preparing (load / encode)…";
                },
                step(value, max) {
                    bar.wrap.style.display = "flex";
                    bar.shimmer.style.display = "none";
                    bar.fill.style.opacity = "1";
                    const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0;
                    bar.fill.style.width = pct.toFixed(1) + "%";
                    bar.label.textContent = `step ${value} / ${max}  —  ${pct.toFixed(0)}%`;
                },
                done() {
                    bar.shimmer.style.display = "none";
                    bar.fill.style.opacity = "1";
                    bar.fill.style.width = "100%";
                    bar.label.textContent = "decoding video + audio…";
                    setTimeout(() => {
                        bar.wrap.style.display = "none";
                        bar.fill.style.width = "0%";
                    }, 1800);
                },
                reset() {
                    bar.wrap.style.display = "none";
                    bar.shimmer.style.display = "none";
                    bar.fill.style.width = "0%";
                },
            };

            // update readout whenever a relevant widget changes
            for (const w of node.widgets || []) {
                if (["aspect_ratio", "megapixels", "duration", "match_ratio_from_image"].includes(w.name)) {
                    const orig = w.callback;
                    w.callback = function () {
                        node._starUpdateInfo();
                        return orig?.apply(this, arguments);
                    };
                }
            }
            node._starUpdateInfo();
            return r;
        };
    },

    setup() {
        // per-node step progress from the sampling ProgressBar on the server
        api.addEventListener("progress", (e) => {
            const d = e.detail || {};
            const id = d.node ?? app.runningNodeId;
            if (id == null) return;
            const node = app.graph?.getNodeById?.(id);
            if (node?.comfyClass === NODE_CLASS) {
                node._starProgress?.step(d.value, d.max);
            }
        });

        api.addEventListener("executing", (e) => {
            const id = e.detail;
            if (id == null) {
                // execution finished — complete bars on all instances
                for (const n of app.graph?._nodes || []) {
                    if (n.comfyClass === NODE_CLASS) n._starProgress?.done();
                }
                return;
            }
            const node = app.graph?.getNodeById?.(id);
            if (node?.comfyClass === NODE_CLASS) node._starProgress?.start();
        });

        api.addEventListener("execution_interrupted", () => {
            for (const n of app.graph?._nodes || []) {
                if (n.comfyClass === NODE_CLASS) n._starProgress?.reset();
            }
        });
        api.addEventListener("execution_error", () => {
            for (const n of app.graph?._nodes || []) {
                if (n.comfyClass === NODE_CLASS) n._starProgress?.reset();
            }
        });
    },
});
