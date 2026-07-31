import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";

// ---------------------------------------------------------------------------
// Load the StarNodes V2 stylesheet (path derived from this module's URL,
// so it works no matter how the pack folder is named).
// ---------------------------------------------------------------------------
(() => {
    const cssUrl = new URL("../css/star_nodes_v2.css", import.meta.url).href;
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = cssUrl;
    document.head.appendChild(link);
})();

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

function getWidget(node, name) {
    return node.widgets?.find((w) => w.name === name) || null;
}

function hideWidget(node, name) {
    const w = getWidget(node, name);
    if (!w) return null;
    // Guarantee the value is serialized into the workflow / prompt no matter
    // how the frontend treats hidden widgets.
    const originalSerialize = w.serializeValue ? w.serializeValue.bind(w) : null;
    w.serializeValue = () => (originalSerialize ? originalSerialize() : w.value);
    w.type = "hidden";
    w.hidden = true;
    w.computeSize = () => [0, -4];
    return w;
}

function setDirty(node) {
    node.graph?.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}

function addPanel(node, className, minWidth = 340) {
    const el = document.createElement("div");
    el.className = `star-panel ${className}`;
    const widget = node.addDOMWidget(`star_${className}`, "starPanel", el, {
        hideOnZoom: false,
    });
    widget.serializeValue = () => undefined;
    // Report the real content height to the node layout so nothing
    // spills outside the node frame.
    widget.computeSize = (width) => [width || minWidth, (el.scrollHeight || 60) + 6];
    fitNodeToContent(node, el, minWidth);
    return el;
}

// Grow the node so the whole DOM panel fits inside it. The element is not
// laid out yet when onNodeCreated runs, so measure over a few frames.
function fitNodeToContent(node, el, minWidth = 340) {
    const apply = () => {
        try {
            const neededW = Math.max(minWidth, el.scrollWidth + 30, node.size[0]);
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

// ---------------------------------------------------------------------------
// ⭐ Star Save Image+ - mode segmented control + format chips
// ---------------------------------------------------------------------------

const SAVE_NODE_NAME = "⭐ Star Save Image+";
const LOAD_NODE_NAME = "StarLoadImagePlus";

const SAVE_FORMATS = [
    ["png", "PNG"],
    ["jpg", "JPG"],
    ["webp", "WEBP"],
    ["psd", "PSD"],
];

app.registerExtension({
    name: "StarNodesV2.SaveImage",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== SAVE_NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            const node = this;

            const modeWidget = hideWidget(node, "mode");
            const formatsWidget = hideWidget(node, "formats");

            const panel = addPanel(node, "star-save", 340);
            panel.appendChild(makeHeader("⭐ STAR SAVE +", "multi-format · custom metadata"));

            // --- Mode segmented control -----------------------------------
            const seg = document.createElement("div");
            seg.className = "star-seg";
            const btnSave = document.createElement("button");
            btnSave.className = "star-seg-btn";
            btnSave.innerHTML = "💾 <span>Save</span>";
            const btnPreview = document.createElement("button");
            btnPreview.className = "star-seg-btn";
            btnPreview.innerHTML = "👁 <span>Preview</span>";
            seg.appendChild(btnSave);
            seg.appendChild(btnPreview);
            panel.appendChild(seg);

            // --- Format chips ----------------------------------------------
            const fmtLabel = document.createElement("div");
            fmtLabel.className = "star-row-label";
            fmtLabel.textContent = "Formats";
            panel.appendChild(fmtLabel);

            const chips = document.createElement("div");
            chips.className = "star-chips";
            panel.appendChild(chips);

            const hint = document.createElement("div");
            hint.className = "star-hint";
            hint.textContent = "Preview mode writes a temporary PNG only.";
            panel.appendChild(hint);

            // Status line: shows the exact files that were written.
            const status = document.createElement("div");
            status.className = "star-status";
            panel.appendChild(status);

            const getMode = () => (modeWidget?.value || "save").toLowerCase();
            const getFormats = () =>
                (formatsWidget?.value || "png")
                    .split(",")
                    .map((s) => s.trim().toLowerCase())
                    .filter(Boolean);

            function renderMode() {
                const mode = getMode();
                btnSave.classList.toggle("active", mode !== "preview");
                btnPreview.classList.toggle("active", mode === "preview");
                chips.classList.toggle("disabled", mode === "preview");
                hint.style.display = mode === "preview" ? "block" : "none";
            }

            function renderFormats() {
                const active = getFormats();
                chips.querySelectorAll(".star-chip").forEach((chip) => {
                    chip.classList.toggle("active", active.includes(chip.dataset.fmt));
                });
            }

            btnSave.addEventListener("click", () => {
                if (modeWidget) modeWidget.value = "save";
                renderMode();
                setDirty(node);
            });
            btnPreview.addEventListener("click", () => {
                if (modeWidget) modeWidget.value = "preview";
                renderMode();
                setDirty(node);
            });

            SAVE_FORMATS.forEach(([fmt, label]) => {
                const chip = document.createElement("button");
                chip.className = "star-chip";
                chip.dataset.fmt = fmt;
                chip.textContent = label;
                chip.addEventListener("click", () => {
                    if (getMode() === "preview") return;
                    let active = getFormats();
                    if (active.includes(fmt)) {
                        active = active.filter((f) => f !== fmt);
                        if (active.length === 0) active = ["png"]; // never empty
                    } else {
                        active.push(fmt);
                    }
                    const order = SAVE_FORMATS.map(([f]) => f);
                    active.sort((a, b) => order.indexOf(a) - order.indexOf(b));
                    if (formatsWidget) formatsWidget.value = active.join(",");
                    renderFormats();
                    setDirty(node);
                });
                chips.appendChild(chip);
            });

            renderMode();
            renderFormats();

            const onExecuted = node.onExecuted;
            node.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                const files = message?.star_files?.[0];
                if (Array.isArray(files) && files.length > 0) {
                    if (getMode() === "preview") {
                        status.textContent = "👁 previewed (temp only - no metadata)";
                        status.title = "";
                    } else {
                        const first = files[0];
                        const more = files.length > 1 ? `  (+${files.length - 1} more)` : "";
                        status.textContent = `✅ saved: ${first}${more}`;
                        status.title = files.join("\n");
                    }
                    status.classList.add("active");
                    fitNodeToContent(node, panel, 340);
                }
            };

            const onConfigure = node.onConfigure;
            node.onConfigure = function () {
                onConfigure?.apply(this, arguments);
                migrateOldWorkflow(node, formatsWidget, getFormats);
                renderMode();
                renderFormats();
            };

            return r;
        };
    },
});

// Migrate workflows saved with the original ⭐ Star Save Image+ node:
// its widgets ended with [save_jpg (bool), jpg_quality], which now land on
// [jpg_quality, webp_quality]. Detect the boolean and restore sane values.
function migrateOldWorkflow(node, formatsWidget, getFormats) {
    const jpgQ = getWidget(node, "jpg_quality");
    const webpQ = getWidget(node, "webp_quality");
    if (jpgQ && typeof jpgQ.value === "boolean") {
        const saveJpg = jpgQ.value;
        const oldQuality = webpQ && typeof webpQ.value === "number" ? webpQ.value : 95;
        jpgQ.value = oldQuality;
        if (webpQ) webpQ.value = 90;
        if (saveJpg && formatsWidget) {
            const wanted = new Set(getFormats());
            wanted.add("jpg");
            const order = SAVE_FORMATS.map(([f]) => f);
            formatsWidget.value = order.filter((f) => wanted.has(f)).join(",");
        }
    }
}

// ---------------------------------------------------------------------------
// ⭐ Star Metadata Saver Option - 5 custom fields (slim info panel)
// ---------------------------------------------------------------------------

app.registerExtension({
    name: "StarNodesV2.MetadataSaverOption",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "StarMetadataSaverOption") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            const node = this;

            const panel = addPanel(node, "star-option", 320);
            panel.appendChild(makeHeader("⭐ METADATA OPTIONS", "5 custom fields · embedded on save"));

            const tip = document.createElement("div");
            tip.className = "star-tip";
            tip.textContent =
                "💡 Fill in the key/value widgets below. Every non-empty value is stored as " +
                "StarMetaData 1-5 inside PNG, JPG and WEBP files - read it back with " +
                "⭐ Star Load Image+ and ⭐ Star Image Loader Options. No workflow data is saved.";
            tip.style.fontSize = "10px";
            panel.appendChild(tip);

            return r;
        };
    },
});

// ---------------------------------------------------------------------------
// Clipboard paste helper (shared by loader)
// ---------------------------------------------------------------------------

async function uploadImageFromBlob(node, blob) {
    const formData = new FormData();
    const filename = `pasted_image_${Date.now()}.png`;
    formData.append("image", blob, filename);
    formData.append("overwrite", "true");

    const response = await api.fetchApi("/upload/image", {
        method: "POST",
        body: formData,
    });

    if (response.status === 200) {
        const data = await response.json();
        const imageWidget = getWidget(node, "image");
        if (imageWidget) {
            imageWidget.value = data.name;
            imageWidget.callback?.(data.name);
            setDirty(node);
        }
    } else {
        const errorText = await response.text();
        alert("Failed to upload image: " + errorText);
    }
}

async function pasteFromClipboard(node) {
    try {
        const clipboardItems = await navigator.clipboard.read();
        for (const clipboardItem of clipboardItems) {
            for (const type of clipboardItem.types) {
                if (type.startsWith("image/")) {
                    const blob = await clipboardItem.getType(type);
                    await uploadImageFromBlob(node, blob);
                    return;
                }
            }
        }
        alert("No image found in clipboard");
    } catch (err) {
        console.error("Failed to paste image from clipboard:", err);
        alert("Failed to paste image. Copy an image first and grant clipboard permissions.");
    }
}

// ---------------------------------------------------------------------------
// ⭐ Star Load Image+ - paste button + metadata badge
// ---------------------------------------------------------------------------

app.registerExtension({
    name: "StarNodesV2.LoadImage",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== LOAD_NODE_NAME) return;

        // Context menu entry (like V1).
        const origMenu = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (_, options) {
            origMenu?.apply(this, arguments);
            options.unshift({
                content: "📋 Paste Clipboard Image",
                callback: () => pasteFromClipboard(this),
            });
        };

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            const node = this;

            const panel = addPanel(node, "star-load", 280);

            const bar = document.createElement("div");
            bar.className = "star-loadbar";

            const pasteBtn = document.createElement("button");
            pasteBtn.className = "star-btn";
            pasteBtn.innerHTML = "📋 <span>Paste Image</span>";
            pasteBtn.title = "Paste an image from the clipboard";
            pasteBtn.addEventListener("click", () => pasteFromClipboard(node));
            bar.appendChild(pasteBtn);

            const badge = document.createElement("div");
            badge.className = "star-badge";
            badge.textContent = "📎 no metadata yet";
            bar.appendChild(badge);

            panel.appendChild(bar);

            const tip = document.createElement("div");
            tip.className = "star-tip";
            tip.textContent = "💡 files saved by Star Save live in the output folder - pick entries ending with [output]";
            panel.appendChild(tip);

            const onExecuted = node.onExecuted;
            node.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                const entries = message?.star_metadata?.[0];
                const count = Array.isArray(entries) ? entries.length : 0;
                badge.textContent =
                    count > 0 ? `📎 ${count} metadata ${count === 1 ? "entry" : "entries"}` : "📎 no metadata found";
                badge.classList.toggle("active", count > 0);
            };

            return r;
        };
    },
});

// ---------------------------------------------------------------------------
// ⭐ Star Image Loader Options - scrollable metadata list with copy buttons
// ---------------------------------------------------------------------------

app.registerExtension({
    name: "StarNodesV2.ImageLoaderOptions",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "StarImageLoaderOptions") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            const node = this;

            const panel = addPanel(node, "star-list", 360);
            panel.appendChild(makeHeader("⭐ IMAGE METADATA", "everything found in the file"));

            const list = document.createElement("div");
            list.className = "star-metalist";
            const empty = document.createElement("div");
            empty.className = "star-metalist-empty";
            empty.textContent = "Run the workflow to inspect the metadata…";
            list.appendChild(empty);
            panel.appendChild(list);

            function renderEntries(entries) {
                list.innerHTML = "";
                if (!Array.isArray(entries) || entries.length === 0) {
                    const e = document.createElement("div");
                    e.className = "star-metalist-empty";
                    e.textContent = "No metadata found in this image.";
                    list.appendChild(e);
                    return;
                }
                entries.forEach(([key, value]) => {
                    const row = document.createElement("div");
                    row.className = "star-metarow";

                    const k = document.createElement("div");
                    k.className = "star-metakey";
                    k.textContent = key;
                    k.title = key;
                    row.appendChild(k);

                    const v = document.createElement("div");
                    v.className = "star-metaval";
                    const text = String(value ?? "");
                    v.textContent = text;
                    v.title = text;
                    row.appendChild(v);

                    const copy = document.createElement("button");
                    copy.className = "star-copy";
                    copy.textContent = "⧉";
                    copy.title = "Copy value";
                    copy.addEventListener("click", async () => {
                        try {
                            await navigator.clipboard.writeText(text);
                            copy.textContent = "✓";
                            setTimeout(() => (copy.textContent = "⧉"), 900);
                        } catch (e) {
                            console.warn("copy failed", e);
                        }
                    });
                    row.appendChild(copy);

                    list.appendChild(row);
                });
            }

            const onExecuted = node.onExecuted;
            node.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                renderEntries(message?.star_metadata?.[0]);
                fitNodeToContent(node, panel, 360);
            };

            return r;
        };
    },
});

// ---------------------------------------------------------------------------
// Drag-link suggestions for Star Save / Load Image+
// ---------------------------------------------------------------------------

const SAVE_CLASS_KEY = "⭐ Star Save Image+";
const LOAD_CLASS_KEY = "⭐ Star Load Image+";

function promoteSlotType(slotType, nodeClass, direction) {
    const lg = globalThis.LiteGraph;
    if (!lg) return;

    const mapName = direction === "out"
        ? "slot_types_default_out"
        : "slot_types_default_in";

    if (!lg[mapName]) lg[mapName] = {};
    if (!lg[mapName][slotType]) lg[mapName][slotType] = [];

    const arr = lg[mapName][slotType];

    for (let i = arr.length - 1; i >= 0; i--) {
        const item = arr[i];
        const val = typeof item === "string" ? item : (item?.value || item?.content);
        if (val === nodeClass) {
            arr.splice(i, 1);
        }
    }

    arr.unshift(nodeClass);
}

app.registerExtension({
    name: "StarNodesV2.DragSuggestions",

    setup() {
        setTimeout(() => {
            promoteSlotType("IMAGE", SAVE_CLASS_KEY, "out");
            promoteSlotType("IMAGE", LOAD_CLASS_KEY, "in");
        }, 100);
    },
});