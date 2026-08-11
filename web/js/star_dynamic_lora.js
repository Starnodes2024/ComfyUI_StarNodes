import { app } from "../../../../scripts/app.js";

// ---------------------------------------------------------------------------
// Load the StarNodes V2 stylesheet (shared with the other DOM-panel nodes).
// ---------------------------------------------------------------------------
(() => {
    const cssUrl = new URL("../css/star_nodes_v2.css", import.meta.url).href;
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = cssUrl;
    document.head.appendChild(link);
})();

// ---------------------------------------------------------------------------
// Theme helpers — read the active StarNodes color template so the DOM panel
// matches the node's title bar / background / frame.
// ---------------------------------------------------------------------------
const FALLBACK_THEMES = {
    default: { bg: null, title: null, frame: null },
    starnodes_purple: { bg: "#3d124d", title: "#19124d", frame: "#ffffff" },
    midnight: { bg: "#0b1220", title: "#0f2a5f", frame: "#4cc9f0" },
    emerald: { bg: "#0e2a1f", title: "#145a32", frame: "#34d399" },
    sunset: { bg: "#2b0b10", title: "#7c2d12", frame: "#fb923c" },
    ocean: { bg: "#062a3a", title: "#075985", frame: "#22d3ee" },
    rose: { bg: "#2a0b1e", title: "#9f1239", frame: "#fb7185" },
    lavender: { bg: "#1c1630", title: "#5b21b6", frame: "#c4b5fd" },
    amber: { bg: "#1f1407", title: "#92400e", frame: "#fbbf24" },
    forest: { bg: "#0f1f12", title: "#14532d", frame: "#86efac" },
    ice: { bg: "#0b1d26", title: "#164e63", frame: "#a5f3fc" },
    mono: { bg: "#1a1a1a", title: "#333333", frame: "#bdbdbd" },
    coffee: { bg: "#1a0f0a", title: "#7c3f1d", frame: "#e7c6a5" },
};

function getActiveTheme() {
    const themes = globalThis.StarNodesThemeMap || FALLBACK_THEMES;
    let themeId = "starnodes_purple";
    try {
        themeId = app.extensionManager?.setting?.get?.("StarNodes.Theme") ?? themeId;
    } catch (_) { /* ignore */ }
    const theme = themes[themeId] || themes.starnodes_purple;
    return { themeId, theme };
}

// ---------------------------------------------------------------------------
// Shared layout helpers (mirrors star_nodes_v2.js patterns).
// ---------------------------------------------------------------------------

function setDirty(node) {
    node.graph?.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}

// Width is sticky — only grows to minWidth on first creation, never
// auto-grows from content.  Height tracks content so rows are not clipped.
function fitNodeToContent(node, el, minWidth = 340) {
    const apply = () => {
        try {
            const neededW = Math.max(minWidth, node.size[0]);
            const sz = node.computeSize();
            const neededH = Math.max(sz[1] + 6, node.size[1]);
            if (neededW > node.size[0] || neededH > node.size[1]) {
                node.setSize([neededW, neededH]);
                setDirty(node);
            }
        } catch (e) { /* ignore */ }
    };
    requestAnimationFrame(() => {
        apply();
        requestAnimationFrame(apply);
    });
    setTimeout(apply, 60);
    setTimeout(apply, 180);
}

// ---------------------------------------------------------------------------
// Widget helpers
// ---------------------------------------------------------------------------

function getWidget(node, name) {
    return node.widgets?.find((w) => w.name === name) || null;
}

function hideWidget(node, name) {
    const w = getWidget(node, name);
    if (!w) return null;
    const originalSerialize = w.serializeValue ? w.serializeValue.bind(w) : null;
    w.serializeValue = () => (originalSerialize ? originalSerialize() : w.value);
    w.type = "hidden";
    w.hidden = true;
    w.computeSize = () => [0, -4];
    return w;
}

function getMaxLoraIndex(node) {
    let maxIdx = 0;
    if (!node.widgets) return maxIdx;
    for (const w of node.widgets) {
        if (!w || typeof w.name !== "string") continue;
        if (w.name.startsWith("lora") && w.name.endsWith("_name")) {
            const idx = parseInt(w.name.replace("lora", "").replace("_name", ""));
            if (!isNaN(idx)) maxIdx = Math.max(maxIdx, idx);
        }
    }
    return maxIdx;
}

function getLoraValues(node) {
    const first = getWidget(node, "lora1_name");
    if (first && first.options && Array.isArray(first.options.values)) {
        return [...first.options.values];
    }
    return ["None"];
}

// Get the panel widget if it exists.
function getPanelWidget(node) {
    return node.widgets?.find((w) => w && w.name === "star_lora_panel") || null;
}

// Remove the panel widget from the array, call fn, then re-append it so the
// panel always stays at the end of the widget list (critical for correct
// widget_values index alignment during save/load).
function withPanelAtEnd(node, fn) {
    const panel = getPanelWidget(node);
    let panelIdx = -1;
    if (panel && node.widgets) {
        panelIdx = node.widgets.indexOf(panel);
        if (panelIdx >= 0) node.widgets.splice(panelIdx, 1);
    }
    try {
        fn();
    } finally {
        if (panel && node.widgets) {
            node.widgets.push(panel);
        }
    }
}

// Create the three hidden backend widgets for a new LoRA slot.
function addLoraSlot(node, idx) {
    if (!node.widgets) node.widgets = [];
    if (getWidget(node, `lora${idx}_name`)) return;

    const values = getLoraValues(node);

    withPanelAtEnd(node, () => {
        node.addWidget("combo", `lora${idx}_name`, "None", () => {}, { values });
        node.addWidget("number", `strength${idx}`, 1.0, () => {}, {
            min: -100.0, max: 100.0, step: 0.01,
        });
        node.addWidget("boolean", `enabled${idx}`, true, () => {}, {});
    });

    hideWidget(node, `lora${idx}_name`);
    hideWidget(node, `strength${idx}`);
    hideWidget(node, `enabled${idx}`);
}

// Remove all three widgets for a slot.
function removeLoraSlot(node, idx) {
    if (!node.widgets) return;
    const toRemove = new Set([`lora${idx}_name`, `strength${idx}`, `enabled${idx}`]);
    for (let i = node.widgets.length - 1; i >= 0; i--) {
        if (node.widgets[i] && toRemove.has(node.widgets[i].name)) {
            node.widgets.splice(i, 1);
        }
    }
}

// ---------------------------------------------------------------------------
// DOM panel
// ---------------------------------------------------------------------------

function applyThemeToPanel(panel) {
    const { theme, themeId } = getActiveTheme();
    if (themeId === "default" || !theme.bg) return;
    panel.style.setProperty("--star-theme-bg", theme.bg);
    panel.style.setProperty("--star-theme-title", theme.title || "");
    panel.style.setProperty("--star-theme-frame", theme.frame || "");
}

function makeHeader(title, subtitle) {
    const header = document.createElement("div");
    header.className = "star-header";
    const t = document.createElement("div");
    t.className = "star-header-title";
    t.textContent = title;
    header.appendChild(t);
    if (subtitle) {
        const s = document.createElement("div");
        s.className = "star-header-sub";
        s.textContent = subtitle;
        header.appendChild(s);
    }
    return header;
}

function buildPanel(node, hasClip) {
    if (node._starLoraPanel) return node._starLoraPanel;

    const panel = document.createElement("div");
    panel.className = "star-panel star-lora-panel";
    applyThemeToPanel(panel);

    const subtitle = hasClip
        ? "toggle · auto-expand · drag to reorder"
        : "toggle · auto-expand · drag to reorder";
    panel.appendChild(makeHeader("⭐ STAR DYNAMIC LORA", subtitle));

    const rows = document.createElement("div");
    rows.className = "star-lora-rows";
    panel.appendChild(rows);

    const hint = document.createElement("div");
    hint.className = "star-lora-hint";
    hint.textContent = "Pick a LoRA to auto-add the next slot. Use the dot to toggle on/off.";
    panel.appendChild(hint);

    const widget = node.addDOMWidget("star_lora_panel", "starLoraPanel", panel, {
        hideOnZoom: false,
    });
    // The panel is purely visual — exclude it from workflow widget_values so
    // it never shifts the index alignment of the lora slot widgets.
    widget.serialize = false;
    widget.serializeValue = () => undefined;
    widget.computeSize = (width) => [width || 340, (panel.scrollHeight || 80) + 6];

    node._starLoraPanel = panel;
    node._starLoraRows = rows;
    node._starLoraWidget = widget;
    node._starLoraHasClip = hasClip;

    return panel;
}

// ---------------------------------------------------------------------------
// Searchable dropdown — a button-like trigger that opens a floating panel with
// a filter input at the top and a scrollable option list.  Designed for lists
// too long to scroll comfortably (e.g. hundreds of LoRA files).
// ---------------------------------------------------------------------------

function createSearchableSelect(values, currentValue, onSelect) {
    const trigger = document.createElement("div");
    trigger.className = "star-lora-select";
    trigger.tabIndex = 0;
    trigger.textContent = currentValue || "None";

    let panel = null;
    let searchInput = null;
    let listEl = null;
    let activeIndex = -1;

    function closePanel() {
        if (!panel) return;
        panel.remove();
        panel = null;
        searchInput = null;
        listEl = null;
        activeIndex = -1;
        trigger.classList.remove("open");
    }

    function highlightRow(idx) {
        if (!listEl) return;
        const rows = listEl.querySelectorAll(".star-lora-dd-item");
        rows.forEach((r, i) => r.classList.toggle("active", i === idx));
        const target = rows[idx];
        if (target) target.scrollIntoView({ block: "nearest" });
        activeIndex = idx;
    }

    function filterAndRender(query) {
        if (!listEl) return;
        listEl.innerHTML = "";
        const q = (query || "").toLowerCase().trim();
        const filtered = q
            ? values.filter((v) => v.toLowerCase().includes(q))
            : values.slice();

        if (filtered.length === 0) {
            const empty = document.createElement("div");
            empty.className = "star-lora-dd-empty";
            empty.textContent = "No matches";
            listEl.appendChild(empty);
            activeIndex = -1;
            return;
        }

        filtered.forEach((v, i) => {
            const item = document.createElement("div");
            item.className = "star-lora-dd-item";
            item.textContent = v;
            if (v === trigger.textContent) item.classList.add("selected");
            item.addEventListener("mousedown", (e) => {
                e.preventDefault();
                trigger.textContent = v;
                onSelect(v);
                closePanel();
            });
            listEl.appendChild(item);
        });

        highlightRow(0);
    }

    function openPanel() {
        if (panel) return;
        panel = document.createElement("div");
        panel.className = "star-lora-dropdown";

        searchInput = document.createElement("input");
        searchInput.type = "text";
        searchInput.className = "star-lora-dd-search";
        searchInput.placeholder = "Search LoRAs…";
        searchInput.spellcheck = false;
        searchInput.addEventListener("input", () => filterAndRender(searchInput.value));
        searchInput.addEventListener("keydown", (e) => {
            if (e.key === "Escape") {
                e.preventDefault();
                closePanel();
                trigger.focus();
            } else if (e.key === "ArrowDown") {
                e.preventDefault();
                const items = listEl?.querySelectorAll(".star-lora-dd-item");
                if (items && items.length) {
                    highlightRow(Math.min(activeIndex + 1, items.length - 1));
                }
            } else if (e.key === "ArrowUp") {
                e.preventDefault();
                highlightRow(Math.max(activeIndex - 1, 0));
            } else if (e.key === "Enter") {
                e.preventDefault();
                const items = listEl?.querySelectorAll(".star-lora-dd-item");
                if (items && items[activeIndex]) {
                    const val = items[activeIndex].textContent;
                    trigger.textContent = val;
                    onSelect(val);
                    closePanel();
                }
            }
        });
        panel.appendChild(searchInput);

        listEl = document.createElement("div");
        listEl.className = "star-lora-dd-list";
        panel.appendChild(listEl);

        // Position the panel below the trigger, fixed to the viewport.
        const rect = trigger.getBoundingClientRect();
        panel.style.position = "fixed";
        panel.style.left = `${rect.left}px`;
        panel.style.top = `${rect.bottom + 2}px`;
        panel.style.minWidth = `${Math.max(rect.width, 220)}px`;
        document.body.appendChild(panel);
        trigger.classList.add("open");

        filterAndRender("");
        searchInput.focus();

        // Close when clicking outside.
        const onDown = (e) => {
            if (panel && !panel.contains(e.target) && e.target !== trigger) {
                closePanel();
                document.removeEventListener("mousedown", onDown, true);
            }
        };
        setTimeout(() => document.addEventListener("mousedown", onDown, true), 0);

        // Close when scrolling or resizing the canvas.
        const onScrollOrResize = () => {
            closePanel();
            window.removeEventListener("scroll", onScrollOrResize, true);
            window.removeEventListener("resize", onScrollOrResize);
        };
        window.addEventListener("scroll", onScrollOrResize, true);
        window.addEventListener("resize", onScrollOrResize);
    }

    trigger.addEventListener("click", openPanel);
    trigger.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            openPanel();
        }
    });

    return { trigger, closePanel };
}

// Swap the values of two LoRA slots (name, strength, enabled).  The widget
// names stay index-based; only their values move so the backend ordering
// changes while the widget array layout stays stable.
function swapLoraSlots(node, idxA, idxB) {
    const aName = getWidget(node, `lora${idxA}_name`);
    const aStr = getWidget(node, `strength${idxA}`);
    const aEn = getWidget(node, `enabled${idxA}`);
    const bName = getWidget(node, `lora${idxB}_name`);
    const bStr = getWidget(node, `strength${idxB}`);
    const bEn = getWidget(node, `enabled${idxB}`);
    if (!aName || !bName) return;

    const tmpName = aName.value;
    const tmpStr = aStr ? aStr.value : 1.0;
    const tmpEn = aEn ? aEn.value : true;

    aName.value = bName.value;
    if (aStr) aStr.value = bStr ? bStr.value : 1.0;
    if (aEn) aEn.value = bEn ? bEn.value : true;

    bName.value = tmpName;
    if (bStr) bStr.value = tmpStr;
    if (bEn) bEn.value = tmpEn;
}

// Build a single DOM row for slot *idx* and wire it to the hidden widgets.
function buildRow(node, idx) {
    const rows = node._starLoraRows;
    if (!rows) return;

    const nameW = getWidget(node, `lora${idx}_name`);
    const strW = getWidget(node, `strength${idx}`);
    const enW = getWidget(node, `enabled${idx}`);
    if (!nameW) return;

    const row = document.createElement("div");
    row.className = "star-lora-row";
    row.dataset.idx = String(idx);

    // --- drag handle ---
    const handle = document.createElement("div");
    handle.className = "star-lora-handle";
    handle.title = "Drag to reorder";
    handle.textContent = "⠿";
    row.appendChild(handle);

    // --- toggle button ---
    const toggle = document.createElement("button");
    toggle.className = "star-lora-toggle";
    toggle.title = "Enable / disable this LoRA";
    const isOn = () => enW ? !!enW.value : true;
    const syncToggle = () => {
        const on = isOn();
        toggle.classList.toggle("on", on);
        toggle.textContent = on ? "●" : "○";
        row.classList.toggle("disabled", !on);
    };
    toggle.addEventListener("click", (e) => {
        e.stopPropagation();
        if (enW) enW.value = !isOn();
        syncToggle();
        if (node.graph) node.graph.change();
        setDirty(node);
    });
    row.appendChild(toggle);

    // --- lora name searchable dropdown ---
    const values = (nameW.options && Array.isArray(nameW.options.values)) ? nameW.options.values : ["None"];
    const dropdown = createSearchableSelect(values, nameW.value || "None", (val) => {
        nameW.value = val;
        if (node.graph) node.graph.change();
        setDirty(node);
        autoExpand(node);
    });
    row.appendChild(dropdown.trigger);

    // --- strength slider + value ---
    const sliderWrap = document.createElement("div");
    sliderWrap.className = "star-lora-str-wrap";

    const slider = document.createElement("input");
    slider.type = "range";
    slider.className = "star-lora-slider";
    slider.min = "-1";
    slider.max = "2";
    slider.step = "0.01";

    const valBox = document.createElement("input");
    valBox.type = "number";
    valBox.className = "star-lora-val";
    valBox.step = "0.01";

    const syncStrength = () => {
        const v = strW ? Number(strW.value) : 1.0;
        valBox.value = String(v.toFixed(2));
        slider.value = String(Math.max(-1, Math.min(2, v)));
    };

    slider.addEventListener("input", () => {
        const v = parseFloat(slider.value);
        if (strW) strW.value = v;
        valBox.value = v.toFixed(2);
        if (node.graph) node.graph.change();
        setDirty(node);
    });

    valBox.addEventListener("change", () => {
        let v = parseFloat(valBox.value);
        if (isNaN(v)) v = 1.0;
        v = Math.max(-100, Math.min(100, v));
        if (strW) strW.value = v;
        syncStrength();
        if (node.graph) node.graph.change();
        setDirty(node);
    });

    sliderWrap.appendChild(slider);
    sliderWrap.appendChild(valBox);
    row.appendChild(sliderWrap);

    // --- drag-and-drop reordering ---
    row.draggable = false; // only the handle initiates drag
    handle.addEventListener("mousedown", () => { row.draggable = true; });
    row.addEventListener("dragend", () => { row.draggable = false; });

    row.addEventListener("dragstart", (e) => {
        if (!row.draggable) { e.preventDefault(); return; }
        node._starLoraDragIdx = idx;
        e.dataTransfer.effectAllowed = "move";
        e.dataTransfer.setData("text/plain", String(idx));
        requestAnimationFrame(() => row.classList.add("dragging"));
    });

    row.addEventListener("dragover", (e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = "move";
        const dragIdx = node._starLoraDragIdx;
        if (!dragIdx || dragIdx === idx) return;
        row.classList.add("drag-over");
    });

    row.addEventListener("dragleave", () => {
        row.classList.remove("drag-over");
    });

    row.addEventListener("drop", (e) => {
        e.preventDefault();
        e.stopPropagation();
        row.classList.remove("drag-over");
        const fromIdx = node._starLoraDragIdx;
        node._starLoraDragIdx = null;
        if (!fromIdx || fromIdx === idx) return;
        swapLoraSlots(node, fromIdx, idx);
        rebuildRows(node);
        if (node.graph) node.graph.change();
        setDirty(node);
    });

    // initial sync
    syncToggle();
    syncStrength();

    rows.appendChild(row);
    return row;
}

// Rebuild every DOM row from the current hidden widgets.
function rebuildRows(node) {
    const rows = node._starLoraRows;
    if (!rows) return;
    rows.innerHTML = "";
    const maxIdx = getMaxLoraIndex(node);
    for (let idx = 1; idx <= maxIdx; idx++) {
        if (getWidget(node, `lora${idx}_name`)) {
            buildRow(node, idx);
        }
    }
    fitNodeToContent(node, node._starLoraPanel, 340);
}

// Ensure slot 1 exists, then auto-expand: add a new empty slot when the last
// one is filled, and prune trailing "None" slots (keep at least one).
function autoExpand(node) {
    if (!getWidget(node, "lora1_name")) {
        addLoraSlot(node, 1);
    }

    let maxIdx = getMaxLoraIndex(node);

    // Prune trailing None slots — keep at least slot 1.
    while (maxIdx > 1) {
        const lastW = getWidget(node, `lora${maxIdx}_name`);
        if (lastW && (lastW.value === "None" || !lastW.value)) {
            const prevW = getWidget(node, `lora${maxIdx - 1}_name`);
            if (prevW && (prevW.value === "None" || !prevW.value)) {
                removeLoraSlot(node, maxIdx);
                maxIdx--;
                continue;
            }
        }
        break;
    }

    // Add a new empty slot when the last one is filled.
    maxIdx = getMaxLoraIndex(node);
    const lastW = getWidget(node, `lora${maxIdx}_name`);
    if (lastW && lastW.value && lastW.value !== "None") {
        addLoraSlot(node, maxIdx + 1);
    }

    rebuildRows(node);
}

// ---------------------------------------------------------------------------
// Extension registration
// ---------------------------------------------------------------------------

app.registerExtension({
    name: "StarNodes.DynamicLoRA",

    beforeRegisterNodeDef(nodeType, nodeData) {
        const isFull = nodeData.name === "StarDynamicLora";
        const isModelOnly = nodeData.name === "StarDynamicLoraModelOnly";
        if (!isFull && !isModelOnly) return;

        const hasClip = isFull;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (origOnNodeCreated) origOnNodeCreated.apply(this, arguments);
            const node = this;

            // Ensure slot 1 exists (it should from INPUT_TYPES, but be safe).
            if (!getWidget(node, "lora1_name")) {
                addLoraSlot(node, 1);
            } else {
                hideWidget(node, "lora1_name");
                hideWidget(node, "strength1");
                if (getWidget(node, "enabled1")) hideWidget(node, "enabled1");
            }

            buildPanel(node, hasClip);

            // Defer auto-expand — if configure() runs (workflow load) it will
            // handle slot creation; otherwise this fires for fresh nodes.
            node._starLoraPendingNew = true;
            setTimeout(() => {
                if (!node._starLoraPendingNew) return;
                node._starLoraPendingNew = false;
                autoExpand(node);
            }, 0);
        };

        // Override configure so we can pre-create lora slot widgets BEFORE
        // the original configure assigns widget_values by index.  Without
        // this, slots 2+ would be lost on workflow load.
        const origConfigure = nodeType.prototype.configure;
        if (origConfigure) {
            nodeType.prototype.configure = function (data) {
                const node = this;
                node._starLoraPendingNew = false;

                // Pre-create lora slots from the saved widget_values.
                // Each slot = 3 widgets (name, strength, enabled).  The panel
                // widget has serialize=false so it is not counted.
                if (data && Array.isArray(data.widgets_values)) {
                    const wv = data.widgets_values;
                    const numSlots = Math.ceil(wv.length / 3);
                    for (let idx = 1; idx <= numSlots; idx++) {
                        if (!getWidget(node, `lora${idx}_name`)) {
                            addLoraSlot(node, idx);
                        }
                    }
                }

                // Call original configure — assigns values by index.
                origConfigure.apply(this, arguments);

                // Hide all lora slot widgets and rebuild the DOM.
                const maxIdx = getMaxLoraIndex(node);
                for (let idx = 1; idx <= maxIdx; idx++) {
                    hideWidget(node, `lora${idx}_name`);
                    hideWidget(node, `strength${idx}`);
                    hideWidget(node, `enabled${idx}`);
                }

                buildPanel(node, hasClip);
                autoExpand(node);
            };
        }
    },
});
