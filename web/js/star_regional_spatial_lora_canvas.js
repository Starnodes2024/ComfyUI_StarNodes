import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// Star Regional Spatial LoRA Canvas
// Interactive modal canvas on the StarRegionalSpatialLoraCanvas node:
// draw bounding boxes, give every box a prompt + optional LoRA + strength,
// and serialize everything into the hidden regions_json widget.

const PALETTE = [
    "#4dd0e1", "#ff8a65", "#aed581", "#ba68c8",
    "#ffd54f", "#7986cb", "#f06292", "#4db6ac",
];

// Mirror of ⭐ Star Advanced Ratio/Latent (star_latent_ratio_megapixel.py) so
// the modal shows exactly the pixel size the backend will produce.
const RATIO_MAP = {
    "1:1": [1, 1], "1:2": [1, 2], "3:4": [3, 4], "2:3": [2, 3],
    "5:7": [5, 7], "9:16": [9, 16], "9:21": [9, 21], "10:16": [10, 16],
    "4:3": [4, 3], "16:10": [16, 10], "3:2": [3, 2], "2:1": [2, 1],
    "7:5": [7, 5], "16:9": [16, 9], "21:9": [21, 9],
};
const BASE_AREA = 1024 * 1024;
const SPECIAL_AREAS = {
    "SD": 512 * 512,
    "SDXL": 1024 * 1024,
    "Qwen Image": 1328 * 1328,
    "WAN HD": 1280 * 720,
    "WAN FullHD": 1920 * 1080,
};

function round16(v) { return Math.max(16, Math.round(v / 16) * 16); }

function computeCanvasSize(node) {
    const widgetVal = (name) => node.widgets.find((w) => w.name === name)?.value;
    const resolution = widgetVal("resolution") || "";
    if (resolution === "custom") {
        return [round16(+widgetVal("custom_width") || 1920), round16(+widgetVal("custom_height") || 1080)];
    }

    let ar = RATIO_MAP[widgetVal("ratio")];
    if (!ar) { // "custom": parse the custom_ratio text widget ("21:9" or "21x9")
        const txt = String(widgetVal("custom_ratio") || "").trim();
        const parts = txt.includes(":") ? txt.split(":") : txt.toLowerCase().split("x");
        const arW = parseInt(parts[0]), arH = parseInt(parts[1]);
        ar = (arW > 0 && arH > 0) ? [arW, arH] : [1, 1];
    }

    const base = resolution.split(" (")[0];
    let area = SPECIAL_AREAS[base];
    if (area === undefined) area = Math.max(0.1, parseFloat(base) || 4) * BASE_AREA;

    const idealW = Math.sqrt(area * (ar[0] / ar[1]));
    return [round16(idealW), round16(idealW * (ar[1] / ar[0]))];
}

function makeEl(tag, parent, style = {}, text = null) {
    const el = document.createElement(tag);
    Object.assign(el.style, style);
    if (text !== null) el.textContent = text;
    if (parent) parent.appendChild(el);
    return el;
}

function openSpatialCanvas(node) {
    const jsonWidget = node.widgets.find((w) => w.name === "regions_json");
    const [initW, initH] = computeCanvasSize(node);

    let boxes = [];
    try {
        boxes = JSON.parse(jsonWidget.value || "[]");
        if (!Array.isArray(boxes)) boxes = [];
    } catch (e) {
        boxes = [];
    }
    boxes = boxes.map((b, i) => ({
        id: b.id ?? i + 1,
        x: +b.x || 0, y: +b.y || 0, w: +b.w || 0.1, h: +b.h || 0.1,
        prompt: b.prompt || "",
        lora_name: b.lora_name || "none",
        lora_strength: b.lora_strength ?? 1.0,
        face_detail: !!b.face_detail,
    }));

    let workW = initW;
    let workH = initH;
    let selected = -1;
    let drag = null; // {mode: 'new'|'move'|'resize', ...}
    let loraList = ["none"];

    // --- overlay + layout -------------------------------------------------
    const overlay = makeEl("div", document.body, {
        position: "fixed", inset: "0", background: "rgba(0,0,0,0.7)",
        zIndex: 10000, display: "flex", alignItems: "center", justifyContent: "center",
    });
    const panel = makeEl("div", overlay, {
        background: "#1e1e1e", borderRadius: "8px", padding: "12px",
        display: "flex", flexDirection: "column", gap: "8px",
        boxShadow: "0 8px 40px rgba(0,0,0,0.8)", maxWidth: "96vw", maxHeight: "94vh",
        fontFamily: "sans-serif", color: "#ddd", fontSize: "13px",
    });

    const header = makeEl("div", panel, { display: "flex", justifyContent: "space-between", alignItems: "center" });
    makeEl("div", header, { fontWeight: "bold", fontSize: "15px" }, "⭐ Star Regional Spatial LoRA Canvas");
    const closeX = makeEl("button", header, { cursor: "pointer" }, "✕");
    closeX.onclick = () => cleanup();

    const toolbar = makeEl("div", panel, { display: "flex", gap: "6px", alignItems: "center", flexWrap: "wrap" });
    makeEl("span", toolbar, { opacity: 0.7 },
        `Canvas: ${workW} x ${workH} px (from the node's ratio / resolution / latent_channels widgets)`);
    const clearBtn = makeEl("button", toolbar, { cursor: "pointer", marginLeft: "auto" }, "Clear all boxes");
    clearBtn.onclick = () => { boxes = []; selected = -1; refreshAll(); };

    const body = makeEl("div", panel, { display: "flex", gap: "10px" });
    const canvas = makeEl("canvas", body, {
        background: "#111", border: "1px solid #444", borderRadius: "4px",
        cursor: "crosshair", display: "block",
    });
    const ctx = canvas.getContext("2d");

    const side = makeEl("div", body, {
        width: "250px", display: "flex", flexDirection: "column", gap: "6px", minWidth: "250px",
    });
    makeEl("div", side, { fontWeight: "bold" }, "Selected box");
    const propsTitle = makeEl("div", side, { fontSize: "12px", opacity: 0.7 }, "(click a box to edit)");
    const promptLabel = makeEl("div", side, {}, "Prompt");
    const promptInput = makeEl("textarea", side, {
        width: "100%", height: "70px", resize: "vertical", boxSizing: "border-box",
        background: "#2a2a2a", color: "#ddd", border: "1px solid #555", borderRadius: "3px",
    });
    promptInput.oninput = () => { if (selected >= 0) boxes[selected].prompt = promptInput.value; };

    makeEl("div", side, {}, "LoRA");
    const loraSearch = makeEl("input", side, {
        width: "100%", boxSizing: "border-box", background: "#2a2a2a", color: "#ddd",
        border: "1px solid #555", borderRadius: "3px", marginBottom: "4px",
    });
    loraSearch.type = "text";
    loraSearch.placeholder = "search Loras...";
    const loraSelect = makeEl("select", side, {
        width: "100%", background: "#2a2a2a", color: "#ddd", border: "1px solid #555", borderRadius: "3px",
        size: 8, // scrollable list, filtered live by the search field
    });

    function populateLoraSelect(filter) {
        const keep = loraSelect.value; // preserve the current selection across refilters
        const q = (filter || "").trim().toLowerCase();
        loraSelect.innerHTML = "";
        for (const name of loraList) {
            if (q && !name.toLowerCase().includes(q)) continue;
            const opt = document.createElement("option");
            opt.value = name;
            opt.textContent = name;
            loraSelect.appendChild(opt);
        }
        if ([...loraSelect.options].some((o) => o.value === keep)) loraSelect.value = keep;
    }
    loraSearch.oninput = () => populateLoraSelect(loraSearch.value);
    loraSelect.onchange = () => { if (selected >= 0) boxes[selected].lora_name = loraSelect.value; };

    const strRow = makeEl("div", side, { display: "flex", alignItems: "center", gap: "6px" });
    makeEl("span", strRow, {}, "LoRA strength");
    const strRange = makeEl("input", strRow, { flex: "1" });
    strRange.type = "range"; strRange.min = 0.0; strRange.max = 1.5; strRange.step = 0.05;
    const strNum = makeEl("span", strRow, { width: "36px", textAlign: "right" }, "1.00");
    strRange.oninput = () => {
        strNum.textContent = parseFloat(strRange.value).toFixed(2);
        if (selected >= 0) boxes[selected].lora_strength = parseFloat(strRange.value);
    };

    const faceRow = makeEl("label", side, { display: "flex", alignItems: "center", gap: "6px", cursor: "pointer", marginTop: "4px" });
    const faceCheck = makeEl("input", faceRow, {});
    faceCheck.type = "checkbox";
    makeEl("span", faceRow, {}, "Face detailer (YOLO)");
    faceCheck.onchange = () => { if (selected >= 0) { boxes[selected].face_detail = faceCheck.checked; draw(); } };

    const delBtn = makeEl("button", side, { cursor: "pointer" }, "Delete box");
    delBtn.onclick = () => {
        if (selected >= 0) { boxes.splice(selected, 1); selected = -1; refreshAll(); }
    };
    makeEl("div", side, { fontSize: "11px", opacity: 0.55, lineHeight: "1.4" },
        "Draw: click + drag on empty space. Move: drag inside a box. Resize: drag the corner dot. Boxes with 'none' LoRA are prompt-only zones. Face detailer needs CLIP + VAE on the sampler.");

    const footer = makeEl("div", panel, { display: "flex", gap: "8px", justifyContent: "flex-end" });
    const cancelBtn = makeEl("button", footer, { cursor: "pointer" }, "Cancel");
    const applyBtn = makeEl("button", footer, {
        cursor: "pointer", background: "#2d6cdf", color: "#fff", border: "none",
        borderRadius: "4px", padding: "5px 14px", fontWeight: "bold",
    }, "Apply");
    cancelBtn.onclick = () => cleanup();
    applyBtn.onclick = () => {
        jsonWidget.value = JSON.stringify(boxes.map((b, i) => ({
            id: i + 1, x: +b.x.toFixed(4), y: +b.y.toFixed(4), w: +b.w.toFixed(4), h: +b.h.toFixed(4),
            prompt: b.prompt, lora_name: b.lora_name, lora_strength: +b.lora_strength,
            face_detail: !!b.face_detail,
        })));
        if (node.graph) node.graph.setDirtyCanvas(true, true);
        cleanup();
    };

    // --- layout / drawing -------------------------------------------------
    function layoutCanvas() {
        const maxW = Math.min(window.innerWidth * 0.62, 900);
        // Cap the height too, otherwise portrait canvases grow taller than the
        // screen (header + toolbar + footer take ~150px of the modal).
        const maxH = Math.max(300, window.innerHeight - 170);
        const scale = Math.min(maxW / workW, maxH / workH);
        const dispW = Math.round(workW * scale);
        const dispH = Math.round(workH * scale);
        canvas.style.width = `${dispW}px`;
        canvas.style.height = `${dispH}px`;
        const dpr = window.devicePixelRatio || 1;
        canvas.width = Math.round(dispW * dpr);
        canvas.height = Math.round(dispH * dpr);
        ctx.setTransform(canvas.width / workW, 0, 0, canvas.height / workH, 0, 0);
        draw();
    }

    function toCanvas(e) {
        const r = canvas.getBoundingClientRect();
        return {
            x: (e.clientX - r.left) * (workW / r.width),
            y: (e.clientY - r.top) * (workH / r.height),
        };
    }

    function draw() {
        ctx.clearRect(0, 0, workW, workH);
        // grid
        ctx.strokeStyle = "rgba(255,255,255,0.08)";
        ctx.lineWidth = workW / 500;
        for (let i = 1; i < 4; i++) {
            ctx.beginPath(); ctx.moveTo((workW / 4) * i, 0); ctx.lineTo((workW / 4) * i, workH); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(0, (workH / 4) * i); ctx.lineTo(workW, (workH / 4) * i); ctx.stroke();
        }
        boxes.forEach((b, i) => {
            const color = PALETTE[i % PALETTE.length];
            const bx = b.x * workW, by = b.y * workH, bw = b.w * workW, bh = b.h * workH;
            ctx.fillStyle = color + "30";
            ctx.fillRect(bx, by, bw, bh);
            ctx.strokeStyle = color;
            ctx.lineWidth = (i === selected ? 2.5 : 1.5) * (workW / 800);
            ctx.strokeRect(bx, by, bw, bh);
            ctx.font = `${Math.max(12, workW / 60)}px sans-serif`;
            ctx.fillStyle = color;
            ctx.fillText(`#${i + 1}${b.lora_name && b.lora_name !== "none" ? " L" : ""}${b.face_detail ? " F" : ""}`, bx + 4, by + Math.max(12, workW / 60) + 2);
            if (i === selected) {
                // resize handle
                const hs = Math.max(6, workW / 120);
                ctx.fillStyle = color;
                ctx.fillRect(bx + bw - hs / 2, by + bh - hs / 2, hs, hs);
            }
        });
    }

    function refreshProps() {
        const has = selected >= 0 && selected < boxes.length;
        promptInput.disabled = !has;
        loraSelect.disabled = !has;
        strRange.disabled = !has;
        faceCheck.disabled = !has;
        delBtn.disabled = !has;
        if (has) {
            const b = boxes[selected];
            propsTitle.textContent = `Box #${selected + 1}`;
            promptInput.value = b.prompt;
            loraSearch.value = "";
            populateLoraSelect("");
            loraSelect.value = b.lora_name;
            strRange.value = b.lora_strength;
            strNum.textContent = parseFloat(b.lora_strength).toFixed(2);
            faceCheck.checked = !!b.face_detail;
        } else {
            propsTitle.textContent = "(click a box to edit)";
            promptInput.value = "";
            faceCheck.checked = false;
        }
    }

    function refreshAll() { draw(); refreshProps(); }

    // --- interaction ------------------------------------------------------
    function hitHandle(pos) {
        if (selected < 0) return false;
        const b = boxes[selected];
        const hs = Math.max(6, workW / 120) * 2; // generous hitbox
        const hx = (b.x + b.w) * workW, hy = (b.y + b.h) * workH;
        return Math.abs(pos.x - hx) <= hs && Math.abs(pos.y - hy) <= hs;
    }

    canvas.onmousedown = (e) => {
        const pos = toCanvas(e);
        if (hitHandle(pos)) {
            drag = { mode: "resize" };
            return;
        }
        const hit = boxes.map((b, i) => ({ b, i })).reverse().find(({ b }) =>
            pos.x >= b.x * workW && pos.x <= (b.x + b.w) * workW &&
            pos.y >= b.y * workH && pos.y <= (b.y + b.h) * workH);
        if (hit) {
            selected = hit.i;
            drag = { mode: "move", dx: pos.x - hit.b.x * workW, dy: pos.y - hit.b.y * workH };
        } else {
            boxes.push({ id: boxes.length + 1, x: pos.x / workW, y: pos.y / workH, w: 0, h: 0, prompt: "", lora_name: "none", lora_strength: 1.0, face_detail: false });
            selected = boxes.length - 1;
            drag = { mode: "new" };
        }
        refreshAll();
    };

    window.onmousemove = (e) => {
        if (!drag) return;
        const pos = toCanvas(e);
        const b = boxes[selected];
        if (!b) return;
        if (drag.mode === "new" || drag.mode === "resize") {
            const x0 = b.x * workW, y0 = b.y * workH;
            const minSize = workW / 50;
            b.w = Math.max(minSize, pos.x - x0) / workW;
            b.h = Math.max(minSize, pos.y - y0) / workH;
            b.w = Math.min(b.w, 1 - b.x);
            b.h = Math.min(b.h, 1 - b.y);
        } else if (drag.mode === "move") {
            b.x = Math.max(0, Math.min(1 - b.w, (pos.x - drag.dx) / workW));
            b.y = Math.max(0, Math.min(1 - b.h, (pos.y - drag.dy) / workH));
        }
        draw();
    };

    window.onmouseup = () => { if (drag) { drag = null; refreshAll(); } };
    canvas.oncontextmenu = (e) => e.preventDefault();

    document.addEventListener("keydown", function esc(e) {
        if (e.key === "Escape") { cleanup(); e.stopPropagation(); }
    }, { once: true });

    function cleanup() {
        window.onmousemove = null;
        window.onmouseup = null;
        overlay.remove();
    }

    // --- init -------------------------------------------------------------
    api.fetchApi("/models/loras").then(async (r) => {
        if (r.ok) loraList = ["none", ...(await r.json())];
    }).catch(() => {}).finally(() => {
        populateLoraSelect(loraSearch.value);
        refreshAll();
    });

    layoutCanvas();
    refreshAll();
}

app.registerExtension({
    name: "Comfy.StarNodes.RegionalSpatialLoraCanvas",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "StarRegionalSpatialLoraCanvas") return;
        const orig = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            orig?.apply(this, arguments);

            const jsonWidget = this.widgets?.find((w) => w.name === "regions_json");
            if (jsonWidget) {
                jsonWidget.hidden = true;
                jsonWidget.type = "hidden";
                jsonWidget.computeSize = () => [0, -4];
            }

            this.addWidget("button", "🎨 Open Spatial Canvas", null, () => openSpatialCanvas(this));
            this.setSize([Math.max(this.size[0], 320), this.size[1]]);
        };
    },
});
