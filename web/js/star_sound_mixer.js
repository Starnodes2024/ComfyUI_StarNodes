import { app } from "../../../../scripts/app.js";

// ─── Star Sound Mixer — DOM slider widgets for volume + start pause ─────────
//
// When an audio_N input is connected, a styled slider widget appears in the
// node body showing "Audio N" with a 0-100% volume slider.  For N >= 2 a
// "Pause" number input (seconds) is shown next to it so that track can start
// later than the first (master) track.  When the input is disconnected the
// widgets are removed.  A new audio input slot is added automatically whenever
// the last slot is connected (up to 12).

const MAX_INPUTS = 12;
const STYLE_ID = "star-sound-mixer-style";

function ensureStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const st = document.createElement("style");
    st.id = STYLE_ID;
    st.textContent = `
.star-mix-row {
    display: flex; align-items: center; gap: 8px;
    padding: 4px 8px; font-family: sans-serif; user-select: none;
    font-size: 11px; color: #cfcfe8;
}
.star-mix-label {
    min-width: 52px; font-weight: 600; color: #9ad9b8;
    white-space: nowrap;
}
.star-mix-slider {
    flex: 1; -webkit-appearance: none; appearance: none;
    height: 8px; border-radius: 4px; background: #1a2a22;
    border: 1px solid #2a4a3a; outline: none; cursor: pointer;
    box-shadow: inset 0 1px 2px rgba(0,0,0,.5);
}
.star-mix-slider::-webkit-slider-thumb {
    -webkit-appearance: none; appearance: none;
    width: 16px; height: 16px; border-radius: 50%;
    background: linear-gradient(135deg, #3fbf6f, #1a8f4f);
    border: 1px solid #5fe99f; cursor: grab;
    box-shadow: 0 0 6px rgba(60,220,120,.5);
}
.star-mix-slider::-moz-range-thumb {
    width: 16px; height: 16px; border-radius: 50%;
    background: linear-gradient(135deg, #3fbf6f, #1a8f4f);
    border: 1px solid #5fe99f; cursor: grab;
    box-shadow: 0 0 6px rgba(60,220,120,.5);
}
.star-mix-pct {
    min-width: 38px; text-align: right; font-weight: 700;
    color: #ffffff; font-variant-numeric: tabular-nums;
}
.star-mix-pause {
    width: 52px; -moz-appearance: textfield; appearance: textfield;
    background: #1a2a22; border: 1px solid #2a4a3a; border-radius: 4px;
    color: #ffffff; font-size: 11px; text-align: right;
    padding: 2px 4px; outline: none; font-variant-numeric: tabular-nums;
}
.star-mix-pause::-webkit-outer-spin-button,
.star-mix-pause::-webkit-inner-spin-button {
    -webkit-appearance: none; margin: 0;
}
.star-mix-pause-label {
    color: #9ad9b8; font-weight: 600; white-space: nowrap;
}
`;
    document.head.appendChild(st);
}

function getAudioIndexFromInput(input) {
    if (!input || !input.name) return -1;
    const m = input.name.match(/^audio_(\d+)$/);
    return m ? parseInt(m[1]) : -1;
}

function getMaxAudioIndex(node) {
    let maxIdx = 0;
    for (const input of node.inputs || []) {
        const idx = getAudioIndexFromInput(input);
        if (idx > maxIdx) maxIdx = idx;
    }
    return maxIdx;
}

function findWidget(node, name) {
    if (!node.widgets) return null;
    return node.widgets.find(w => w && w.name === name) || null;
}

function findVolumeWidget(node, idx) {
    return findWidget(node, `volume_${idx}`);
}

function findPauseWidget(node, idx) {
    return findWidget(node, `start_pause_${idx}`);
}

// Preserve the user's width — only adjust height to fit content.
function relayout(node) {
    const sz = node.computeSize();
    node.setSize([Math.max(node.size[0], sz[0]), sz[1]]);
    node.graph?.setDirtyCanvas?.(true, true);
}

// Create or refresh the DOM slider widget for a given audio index.
function ensureSliderWidget(node, idx) {
    const volKey = `volume_${idx}`;
    let volWidget = findVolumeWidget(node, idx);
    const hasPause = idx >= 2;
    let pauseWidget = hasPause ? findPauseWidget(node, idx) : null;

    // If the DOM widget already exists, keep it.
    if (node._starMixSliders && node._starMixSliders[idx]) {
        return node._starMixSliders[idx];
    }

    ensureStyle();

    const wrap = document.createElement("div");
    wrap.className = "star-mix-row";
    let html =
        `<span class="star-mix-label">Audio ${idx}</span>` +
        `<input type="range" class="star-mix-slider" min="0" max="100" step="1" value="100">` +
        `<span class="star-mix-pct">100%</span>`;
    if (hasPause) {
        html +=
            `<span class="star-mix-pause-label">Pause</span>` +
            `<input type="number" class="star-mix-pause" min="0" step="0.1" value="0">` +
            `<span class="star-mix-pause-label">s</span>`;
    }
    wrap.innerHTML = html;

    const sliderEl = wrap.querySelector(".star-mix-slider");
    const pctEl = wrap.querySelector(".star-mix-pct");
    const pauseEl = hasPause ? wrap.querySelector(".star-mix-pause") : null;

    // Try to get the current value from the existing widget, default 100%.
    let currentVal = 100;
    if (volWidget && typeof volWidget.value === "number") {
        currentVal = Math.round(volWidget.value * 100);
    }
    sliderEl.value = String(currentVal);
    pctEl.textContent = currentVal + "%";

    if (hasPause && pauseEl) {
        let currentPause = 0;
        if (pauseWidget && typeof pauseWidget.value === "number") {
            currentPause = pauseWidget.value;
        }
        pauseEl.value = String(currentPause);
    }

    const domWidget = node.addDOMWidget(volKey, "starMixSlider", wrap, {
        serialize: false,
        hideOnZoom: false,
        getValue: () => {
            return parseFloat(sliderEl.value) / 100.0;
        },
        setValue: (v) => {
            const pct = Math.round(parseFloat(v) * 100);
            sliderEl.value = String(pct);
            pctEl.textContent = pct + "%";
        },
    });
    domWidget.computedHeight = 30;

    // Keep the backend widget value in sync.
    sliderEl.addEventListener("input", () => {
        const pct = parseInt(sliderEl.value);
        pctEl.textContent = pct + "%";
        if (volWidget) {
            volWidget.value = pct / 100.0;
        }
        if (node.graph) node.graph.setDirtyCanvas(true);
    });

    if (hasPause && pauseEl) {
        pauseEl.addEventListener("input", () => {
            const v = parseFloat(pauseEl.value);
            const safe = isFinite(v) && v > 0 ? v : 0.0;
            if (pauseWidget) {
                pauseWidget.value = safe;
            }
            if (node.graph) node.graph.setDirtyCanvas(true);
        });
    }

    if (!node._starMixSliders) node._starMixSliders = {};
    node._starMixSliders[idx] = {
        widget: domWidget, wrap, sliderEl, pctEl,
        pauseEl: hasPause ? pauseEl : null,
    };

    relayout(node);
    return node._starMixSliders[idx];
}

// Remove the DOM slider widget for a given audio index.
function removeSliderWidget(node, idx) {
    if (!node._starMixSliders || !node._starMixSliders[idx]) return;
    const entry = node._starMixSliders[idx];
    // Remove the volume DOM widget from node.widgets (pause has no DOM widget).
    if (node.widgets) {
        const pos = node.widgets.indexOf(entry.widget);
        if (pos >= 0) node.widgets.splice(pos, 1);
    }
    if (entry.wrap.parentNode) {
        entry.wrap.parentNode.removeChild(entry.wrap);
    }
    delete node._starMixSliders[idx];
    relayout(node);
}

// Ensure the volume_N backend widget exists for a given index.
function ensureVolumeWidget(node, idx) {
    if (!node.widgets) node.widgets = [];
    if (findVolumeWidget(node, idx)) return;
    node.addWidget("number", `volume_${idx}`, 1.0, () => {}, {
        min: 0.0, max: 1.0, step: 0.01,
    });
}

// Ensure the start_pause_N backend widget exists for a given index (N >= 2).
function ensurePauseWidget(node, idx) {
    if (idx < 2) return;
    if (!node.widgets) node.widgets = [];
    if (findPauseWidget(node, idx)) return;
    node.addWidget("number", `start_pause_${idx}`, 0.0, () => {}, {
        min: 0.0, max: 3600.0, step: 0.1,
    });
}

// Remove the volume_N backend widget for a given index.
function removeVolumeWidget(node, idx) {
    if (!node.widgets) return;
    for (let i = node.widgets.length - 1; i >= 0; i--) {
        if (node.widgets[i] && node.widgets[i].name === `volume_${idx}`) {
            node.widgets.splice(i, 1);
        }
    }
}

// Remove the start_pause_N backend widget for a given index.
function removePauseWidget(node, idx) {
    if (!node.widgets) return;
    for (let i = node.widgets.length - 1; i >= 0; i--) {
        if (node.widgets[i] && node.widgets[i].name === `start_pause_${idx}`) {
            node.widgets.splice(i, 1);
        }
    }
}

// Add a new audio_N input slot.
function addAudioInput(node) {
    const maxIdx = getMaxAudioIndex(node);
    if (maxIdx >= MAX_INPUTS) return;
    const newIdx = maxIdx + 1;
    // Avoid duplicates.
    if (node.inputs.some(inp => inp.name === `audio_${newIdx}`)) return;
    node.addInput(`audio_${newIdx}`, "AUDIO");
    if (node.graph) node.graph.change();
}

// Remove trailing unconnected audio inputs, keeping one empty slot.
function pruneAudioInputs(node) {
    let lastConnected = 0;
    for (const input of node.inputs || []) {
        const idx = getAudioIndexFromInput(input);
        if (idx > 0 && input.link) {
            lastConnected = Math.max(lastConnected, idx);
        }
    }
    for (let i = (node.inputs || []).length - 1; i >= 0; i--) {
        const input = node.inputs[i];
        const idx = getAudioIndexFromInput(input);
        if (idx <= 0) continue;
        if (!input.link && idx > lastConnected + 1) {
            // Remove input and its associated slider + volume/pause widgets.
            removeSliderWidget(node, idx);
            removeVolumeWidget(node, idx);
            removePauseWidget(node, idx);
            node.removeInput(i);
        }
    }
}

// Sync DOM sliders to match connected audio inputs.
function syncSliders(node) {
    const connectedIndices = new Set();
    for (const input of node.inputs || []) {
        const idx = getAudioIndexFromInput(input);
        if (idx > 0 && input.link) {
            connectedIndices.add(idx);
            ensureVolumeWidget(node, idx);
            ensurePauseWidget(node, idx);
            ensureSliderWidget(node, idx);
        }
    }
    // Remove sliders for disconnected inputs.
    if (node._starMixSliders) {
        for (const idx of Object.keys(node._starMixSliders)) {
            const i = parseInt(idx);
            if (!connectedIndices.has(i)) {
                removeSliderWidget(node, i);
                removeVolumeWidget(node, i);
                removePauseWidget(node, i);
            }
        }
    }
}

app.registerExtension({
    name: "StarNodes.StarSoundMixer",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "StarSoundMixer") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (onNodeCreated) onNodeCreated.apply(this, arguments);

            // Start with one audio input.
            this._starMixSliders = {};
            // Remove any auto-created inputs from the dict pattern, then add our own.
            for (let i = this.inputs.length - 1; i >= 0; i--) {
                if (getAudioIndexFromInput(this.inputs[i]) > 0) {
                    this.removeInput(i);
                }
            }
            this.addInput("audio_1", "AUDIO");
            ensureVolumeWidget(this, 1);
            relayout(this);
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (type, index, connected, linkInfo) {
            if (onConnectionsChange) {
                onConnectionsChange.apply(this, arguments);
            }
            if (type !== LiteGraph.INPUT) return;

            // Sync sliders to current connection state.
            syncSliders(this);

            // Auto-grow: if the last audio input is connected, add a new one.
            if (connected) {
                const input = this.inputs[index];
                const idx = getAudioIndexFromInput(input);
                if (idx > 0 && idx === getMaxAudioIndex(this) && idx < MAX_INPUTS) {
                    addAudioInput(this);
                }
            }

            // Prune trailing unconnected inputs (keep one extra).
            pruneAudioInputs(this);
            syncSliders(this);

            if (this.graph) this.graph.change();
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            if (onRemoved) onRemoved.apply(this, arguments);
            this._starMixSliders = {};
        };
    },
});
