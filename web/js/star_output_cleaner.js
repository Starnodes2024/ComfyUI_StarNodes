import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";

const NODE_NAME = "StarOutputCleaner";

const CSS = `
.star-cleaner { display:flex; flex-direction:column; gap:6px; width:100%; height:100%;
    box-sizing:border-box; padding:6px; background:#1b1b21; border-radius:6px;
    font-family:sans-serif; user-select:none; }
.star-cleaner .sc-row { display:flex; flex:none; align-items:center; gap:6px; flex-wrap:wrap; }
.star-cleaner .sc-btn { background:#2e2e38; color:#ddd; border:1px solid #454550;
    border-radius:4px; padding:4px 10px; font-size:12px; cursor:pointer; }
.star-cleaner .sc-btn:hover { background:#3b3b48; }
.star-cleaner .sc-btn:disabled { opacity:.4; cursor:default; }
.star-cleaner .sc-delete { background:#5c1f1f; border-color:#8a2f2f; color:#ffb3b3; }
.star-cleaner .sc-delete:hover { background:#7a2626; }
.star-cleaner .sc-zip { background:#1f3a5c; border-color:#2f5a8a; color:#b3d4ff; }
.star-cleaner .sc-zip:hover { background:#26496e; }
.star-cleaner .sc-lbl { color:#999; font-size:11px; }
.star-cleaner .sc-count { margin-left:auto; color:#999; font-size:11px; }
.star-cleaner .sc-status { color:#999; font-size:11px; }
.star-cleaner .sc-input, .star-cleaner .sc-select { background:#2e2e38; color:#ddd;
    border:1px solid #454550; border-radius:4px; padding:4px 8px; font-size:12px;
    color-scheme:dark; }
.star-cleaner .sc-num { width:64px; }
.star-cleaner .sc-folderinput { flex:1; min-width:160px; }
.star-cleaner .sc-pagelbl { color:#ccc; font-size:12px; min-width:52px; text-align:center; }
.star-cleaner .sc-grid { flex:1 1 auto; min-height:0; overflow-y:auto; display:grid;
    grid-template-columns:repeat(auto-fill, 200px); justify-content:center;
    gap:8px; align-content:start; padding:2px; }
.star-cleaner .sc-item { position:relative; width:200px; height:200px;
    border:2px solid #333; border-radius:6px; overflow:hidden; cursor:pointer;
    background:#121216; box-sizing:content-box; }
.star-cleaner .sc-item img { width:200px; height:200px; object-fit:contain;
    display:block; pointer-events:none; }
.star-cleaner .sc-item input[type=checkbox] { position:absolute; top:6px; left:6px;
    width:18px; height:18px; margin:0; accent-color:#ffb02e; pointer-events:none; }
.star-cleaner .sc-item .sc-label { position:absolute; left:0; right:0; bottom:0;
    padding:2px 6px; background:rgba(0,0,0,.65); color:#ccc; font-size:10px;
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis; pointer-events:none; }
.star-cleaner .sc-item.sc-selected { border-color:#ffb02e;
    box-shadow:0 0 8px rgba(255,176,46,.55); }
.star-cleaner .sc-empty { grid-column:1/-1; padding:24px; color:#777;
    font-size:12px; text-align:center; }
.sc-modal-overlay { position:fixed; inset:0; background:rgba(0,0,0,.65);
    display:flex; align-items:center; justify-content:center; z-index:10000; }
.sc-modal { background:#23232b; border:1px solid #8a2f2f; border-radius:8px;
    padding:18px 20px; max-width:400px; box-shadow:0 8px 30px rgba(0,0,0,.6); }
.sc-modal-title { color:#ff8a8a; font-size:15px; font-weight:bold; margin-bottom:10px; }
.sc-modal-body { color:#ccc; font-size:12px; line-height:1.6; margin-bottom:16px; }
.sc-modal-body b { color:#ffb3b3; }
.sc-modal-buttons { display:flex; justify-content:flex-end; gap:8px; }
.sc-modal-buttons .sc-btn { font-size:12px; padding:6px 14px; }
.sc-modal-confirm { background:#a32727; border-color:#d33; color:#fff; font-weight:bold; }
.sc-modal-confirm:hover { background:#c22f2f; }
.sc-browser { width:480px; max-width:92vw; border-color:#2f5a8a; }
.sc-browser .sc-modal-title { color:#ffd28a; }
.sc-browser-pathrow { display:flex; align-items:center; gap:8px; margin-bottom:8px; }
.sc-browser-path { color:#999; font-size:11px; word-break:break-all; }
.sc-browser-list { height:260px; overflow-y:auto; background:#1b1b21;
    border:1px solid #3a3a44; border-radius:6px; padding:4px; margin-bottom:12px; }
.sc-browser-item { padding:5px 8px; color:#ccc; font-size:12px; cursor:pointer;
    border-radius:4px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.sc-browser-item:hover { background:#2e2e38; }
.sc-browser-info { color:#777; cursor:default; }
.sc-browser-info:hover { background:transparent; }
.sc-browser-select { background:#1f3a5c; border-color:#2f5a8a; color:#b3d4ff; font-weight:bold; }
.sc-browser-select:hover { background:#26496e; }
`;

function injectCSS() {
    if (document.getElementById("star-cleaner-css")) return;
    const style = document.createElement("style");
    style.id = "star-cleaner-css";
    style.textContent = CSS;
    document.head.appendChild(style);
}

function fmtSize(bytes) {
    if (bytes < 1024) return bytes + " B";
    const units = ["KB", "MB", "GB"];
    let v = bytes / 1024, i = 0;
    while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
    return v.toFixed(1) + " " + units[i];
}

function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
}

/** Modal warning dialog for irreversible deletes. Resolves true/false. */
function confirmPermanentDelete(count, folderLabel) {
    return new Promise((resolve) => {
        const overlay = el("div", "sc-modal-overlay");
        const modal = el("div", "sc-modal");
        const title = el("div", "sc-modal-title", `⚠ Permanently delete ${count} image(s)?`);
        const body = el("div", "sc-modal-body");
        const b1 = el("div", null, `Folder: ${folderLabel}`);
        const b2 = el("div");
        b2.innerHTML = "The selected images will be deleted from disk <b>permanently</b>. " +
            "They will <b>NOT</b> be moved to the recycle bin and <b>cannot be restored</b> afterwards.";
        const buttons = el("div", "sc-modal-buttons");
        const cancelBtn = el("button", "sc-btn", "Cancel");
        const okBtn = el("button", "sc-btn sc-modal-confirm", "Delete permanently");
        buttons.append(cancelBtn, okBtn);
        body.append(b1, b2);
        modal.append(title, body, buttons);
        overlay.appendChild(modal);
        document.body.appendChild(overlay);

        const close = (val) => { overlay.remove(); resolve(val); };
        cancelBtn.addEventListener("click", () => close(false));
        okBtn.addEventListener("click", () => close(true));
        overlay.addEventListener("click", (e) => { if (e.target === overlay) close(false); });
    });
}

/** Server-side folder picker dialog. Calls onSelect(path) on confirm. */
function openFolderBrowser(initialPath, onSelect) {
    const overlay = el("div", "sc-modal-overlay");
    const modal = el("div", "sc-modal sc-browser");
    const title = el("div", "sc-modal-title", "📂 Choose a folder");

    const pathRow = el("div", "sc-browser-pathrow");
    const upBtn = el("button", "sc-btn", "↑ Up");
    const drivesBtn = el("button", "sc-btn", "💻 Drives");
    drivesBtn.style.display = "none"; // shown only on Windows (when drives are reported)
    const pathLbl = el("span", "sc-browser-path");
    pathRow.append(upBtn, drivesBtn, pathLbl);

    const list = el("div", "sc-browser-list");

    const btnRow = el("div", "sc-modal-buttons");
    const cancelBtn = el("button", "sc-btn", "Cancel");
    const selectBtn = el("button", "sc-btn sc-browser-select", "Select this folder");
    btnRow.append(cancelBtn, selectBtn);

    modal.append(title, pathRow, list, btnRow);
    overlay.appendChild(modal);
    document.body.appendChild(overlay);

    let current = "";
    let parent = "";

    const joinPath = (base, name) => {
        if (!base) return name;                       // Windows drive entries are full paths
        const sep = base.includes("\\") ? "\\" : "/";
        return base.endsWith(sep) ? base + name : base + sep + name;
    };
    const isWinDriveRoot = () => /^[A-Za-z]:[\\/]?$/.test(current);

    async function load(p) {
        list.innerHTML = "";
        pathLbl.textContent = p || "…";
        list.appendChild(el("div", "sc-browser-item sc-browser-info", "loading…"));
        try {
            const res = await api.fetchApi(`/star_output_cleaner/browse?path=${encodeURIComponent(p)}`);
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || res.statusText);
            current = data.path || "";
            parent = data.parent || "";
            pathLbl.textContent = current || "(choose a drive)";
            drivesBtn.style.display = (data.drives && data.drives.length) ? "" : "none";
            upBtn.disabled = !parent && !isWinDriveRoot();
            selectBtn.disabled = !current;
            list.innerHTML = "";
            if (!data.dirs?.length) {
                list.appendChild(el("div", "sc-browser-item sc-browser-info", "(no subfolders)"));
            }
            for (const d of data.dirs || []) {
                const item = el("div", "sc-browser-item", "📁 " + d);
                item.title = joinPath(current, d);
                item.addEventListener("click", () => load(joinPath(current, d)));
                list.appendChild(item);
            }
        } catch (e) {
            list.innerHTML = "";
            list.appendChild(el("div", "sc-browser-item sc-browser-info", String(e.message || e)));
        }
    }

    upBtn.addEventListener("click", () => load(isWinDriveRoot() ? "" : parent));
    drivesBtn.addEventListener("click", () => load(""));   // back to the drive list
    cancelBtn.addEventListener("click", () => overlay.remove());
    overlay.addEventListener("click", (e) => { if (e.target === overlay) overlay.remove(); });
    selectBtn.addEventListener("click", () => {
        overlay.remove();
        if (current) onSelect(current);
    });

    load((initialPath || "").trim());
}

function buildUI(node) {
    const state = { images: [], selected: new Set(), loading: false, page: 1, pages: 1, total: 0, perPage: 100 };
    const itemEls = new Map(); // path -> { item, cb }
    let refreshTimer = null;

    const findW = (name) => node.widgets?.find(w => w.name === name);
    const widgetVal = (name) => findW(name)?.value;

    // Permanently hide the canvas widgets that are driven by the node's panel.
    // (They stay serialized in the workflow; hiding them once at creation avoids
    // the layout glitches that show/hide toggling causes on some frontends.)
    function hideWidget(w) {
        if (!w || w._scHidden) return;
        w._scHidden = true;
        w.computeSize = () => [0, -4];
        w.hidden = true;
    }
    ["custom_folder", "amount", "unit", "start_date", "end_date"]
        .forEach((n) => hideWidget(findW(n)));

    // ----- DOM scaffold -----------------------------------------------------
    const container = el("div", "star-cleaner");

    // row 1: toolbar
    const toolbar = el("div", "sc-row");
    const refreshBtn = el("button", "sc-btn", "⟳ Refresh");
    const allBtn = el("button", "sc-btn", "Select page");
    const noneBtn = el("button", "sc-btn", "Select none");
    const countEl = el("span", "sc-count");
    toolbar.append(refreshBtn, allBtn, noneBtn, countEl);

    // row 2: custom-folder path + browse (visible in 'custom folder' source)
    const folderRow = el("div", "sc-row");
    const folderLbl = el("span", "sc-lbl", "folder");
    const folderInput = el("input", "sc-input sc-folderinput");
    folderInput.type = "text";
    folderInput.placeholder = "Folder path, e.g. /home/user/images or D:\\renders";
    folderInput.spellcheck = false;
    const browseBtn = el("button", "sc-btn", "📂 Browse…");
    folderRow.append(folderLbl, folderInput, browseBtn);

    // row 3a: 'last N' controls (amount + unit)
    const lastRow = el("div", "sc-row");
    const lastLbl = el("span", "sc-lbl", "show last");
    const amountInput = el("input", "sc-input sc-num");
    amountInput.type = "number"; amountInput.min = "0"; amountInput.step = "1";
    amountInput.title = "0 = show all images";
    const unitSel = el("select", "sc-select");
    for (const u of ["days", "weeks", "months"]) {
        const o = el("option", null, u); o.value = u; unitSel.appendChild(o);
    }
    lastRow.append(lastLbl, amountInput, unitSel);

    // row 3b: date-range pickers
    const dateRow = el("div", "sc-row");
    const fromLbl = el("span", "sc-lbl", "from");
    const startInput = el("input", "sc-input");
    startInput.type = "date";
    const toLbl = el("span", "sc-lbl", "to");
    const endInput = el("input", "sc-input");
    endInput.type = "date";
    dateRow.append(fromLbl, startInput, toLbl, endInput);

    // row 4: pager
    const pager = el("div", "sc-row");
    const prevBtn = el("button", "sc-btn", "◀ Prev");
    const pageLbl = el("span", "sc-pagelbl", "1 / 1");
    const nextBtn = el("button", "sc-btn", "Next ▶");
    const perLbl = el("span", "sc-lbl", "per page:");
    const perSel = el("select", "sc-select");
    for (const n of [50, 100, 200, 500]) {
        const o = el("option", null, String(n)); o.value = n; perSel.appendChild(o);
    }
    perSel.value = "100";
    pager.append(prevBtn, pageLbl, nextBtn, perLbl, perSel);

    const grid = el("div", "sc-grid");

    // footer
    const footer = el("div", "sc-row");
    const deleteBtn = el("button", "sc-btn sc-delete", "🗑 Delete selected");
    const zipBtn = el("button", "sc-btn sc-zip", "📦 Download ZIP");
    const statusEl = el("span", "sc-status");
    footer.append(deleteBtn, zipBtn, statusEl);

    container.append(toolbar, folderRow, lastRow, dateRow, pager, grid, footer);

    node.addDOMWidget("star_cleaner_ui", "starCleanerUI", container, {
        getValue() { return ""; },
        setValue() { },
        getMinHeight() { return 470; },
        hideOnZoom: false,
    });

    // ----- folder source / params --------------------------------------------
    const isCustomFolder = () => widgetVal("source") === "custom folder";
    const customFolder = () => (widgetVal("custom_folder") || "").trim();
    function folderParams() {
        return isCustomFolder()
            ? `folder=custom&root=${encodeURIComponent(customFolder())}`
            : "folder=output";
    }
    const folderLabel = () => isCustomFolder() ? (customFolder() || "(no folder set)") : "output folder";

    // ----- visibility (DOM rows only — the canvas layout never changes) -------
    function applyVisibility() {
        folderRow.style.display = isCustomFolder() ? "flex" : "none";
        const isRange = widgetVal("mode") === "date range";
        dateRow.style.display = isRange ? "flex" : "none";
        lastRow.style.display = isRange ? "none" : "flex";
    }

    // ----- sync helpers --------------------------------------------------------
    function syncInputsFromWidgets() {
        startInput.value = widgetVal("start_date") || "";
        endInput.value = widgetVal("end_date") || "";
        folderInput.value = widgetVal("custom_folder") || "";
        amountInput.value = widgetVal("amount") ?? 0;
        unitSel.value = widgetVal("unit") || "days";
    }
    function pushToWidget(name, value) {
        const w = findW(name);
        if (w && w.value !== value) { w.value = value; w.callback?.(value); }
        debouncedRefresh();
    }
    startInput.addEventListener("change", () => pushToWidget("start_date", startInput.value));
    endInput.addEventListener("change", () => pushToWidget("end_date", endInput.value));
    folderInput.addEventListener("change", () => pushToWidget("custom_folder", folderInput.value.trim()));
    amountInput.addEventListener("change", () =>
        pushToWidget("amount", Math.max(0, parseInt(amountInput.value, 10) || 0)));
    unitSel.addEventListener("change", () => pushToWidget("unit", unitSel.value));
    browseBtn.addEventListener("click", () => {
        openFolderBrowser(folderInput.value, (chosen) => {
            folderInput.value = chosen;
            pushToWidget("custom_folder", chosen);
        });
    });

    // ----- gallery --------------------------------------------------------------
    function updateCount() {
        countEl.textContent = `${state.total} image(s) · ${state.selected.size} selected`;
        deleteBtn.disabled = state.selected.size === 0;
        zipBtn.disabled = state.selected.size === 0;
    }
    function updatePager() {
        pageLbl.textContent = `${state.page} / ${state.pages}`;
        prevBtn.disabled = state.page <= 1;
        nextBtn.disabled = state.page >= state.pages;
    }

    function toggle(path) {
        if (state.selected.has(path)) state.selected.delete(path);
        else state.selected.add(path);
        const entry = itemEls.get(path);
        if (entry) {
            const on = state.selected.has(path);
            entry.item.classList.toggle("sc-selected", on);
            entry.cb.checked = on;
        }
        updateCount();
    }

    function render() {
        itemEls.clear();
        grid.innerHTML = "";
        updateCount();
        updatePager();
        if (!state.images.length) {
            grid.appendChild(el("div", "sc-empty", state.loading ? "Loading…" : "No images found"));
            return;
        }
        const fp = folderParams();
        const frag = document.createDocumentFragment();
        for (const img of state.images) {
            const item = el("div", "sc-item" + (state.selected.has(img.path) ? " sc-selected" : ""));
            item.title = `${img.path}\n${fmtSize(img.size)} · ${new Date(img.mtime * 1000).toLocaleString()}\n(double-click to open full size)`;

            const im = el("img");
            im.loading = "lazy";
            im.draggable = false;
            im.src = api.apiURL(`/star_output_cleaner/thumbnail?${fp}&path=${encodeURIComponent(img.path)}`);

            const cb = el("input");
            cb.type = "checkbox";
            cb.checked = state.selected.has(img.path);
            cb.tabIndex = -1;

            const label = el("div", "sc-label", img.name);

            item.append(im, cb, label);
            item.addEventListener("click", () => toggle(img.path));
            item.addEventListener("dblclick", (e) => {
                e.stopPropagation();
                // open the ORIGINAL image full size (not the thumbnail)
                const url = api.apiURL(`/star_output_cleaner/image?${fp}&path=${encodeURIComponent(img.path)}`);
                window.open(url, "_blank");
            });

            itemEls.set(img.path, { item, cb });
            frag.appendChild(item);
        }
        grid.appendChild(frag);
    }

    // ----- data --------------------------------------------------------------------
    function buildQuery() {
        const base = folderParams();
        const range = widgetVal("mode") === "date range"
            ? `mode=range&start=${encodeURIComponent(widgetVal("start_date") || "")}` +
              `&end=${encodeURIComponent(widgetVal("end_date") || "")}`
            : `mode=last&amount=${widgetVal("amount") ?? 0}` +
              `&unit=${encodeURIComponent(widgetVal("unit") || "days")}`;
        return `${base}&${range}&page=${state.page}&per_page=${state.perPage}`;
    }

    async function refresh() {
        state.loading = true;
        statusEl.textContent = "loading…";
        render();
        try {
            const res = await api.fetchApi(`/star_output_cleaner/list?${buildQuery()}`);
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || res.statusText);
            state.images = data.images || [];
            state.total = data.total ?? state.images.length;
            state.pages = data.pages ?? 1;
            state.page = data.page ?? 1;      // server clamps out-of-range pages
            // note: selections intentionally persist across pages
            statusEl.textContent = "";
        } catch (e) {
            state.images = [];
            state.total = 0; state.pages = 1; state.page = 1;
            statusEl.textContent = String(e.message || e);
        }
        state.loading = false;
        render();
    }

    /** filter changed -> jump back to page 1, then reload */
    function debouncedRefresh() {
        state.page = 1;
        clearTimeout(refreshTimer);
        refreshTimer = setTimeout(refresh, 400);
    }

    function postBody() {
        return JSON.stringify({
            folder: isCustomFolder() ? "custom" : "output",
            root: isCustomFolder() ? customFolder() : "",
            paths: [...state.selected],
        });
    }

    // ----- button events --------------------------------------------------------------
    refreshBtn.addEventListener("click", refresh);           // keeps the current page
    allBtn.addEventListener("click", () => {                 // select everything on this page
        state.images.forEach(i => state.selected.add(i.path));
        render();
    });
    noneBtn.addEventListener("click", () => {                // clears ALL pages
        state.selected.clear();
        render();
    });
    prevBtn.addEventListener("click", () => { if (state.page > 1) { state.page--; refresh(); } });
    nextBtn.addEventListener("click", () => { if (state.page < state.pages) { state.page++; refresh(); } });
    perSel.addEventListener("change", () => {
        state.perPage = parseInt(perSel.value, 10) || 100;
        state.page = 1;
        refresh();
    });
    deleteBtn.addEventListener("click", async () => {
        const n = state.selected.size;
        if (!n) return;
        if (!(await confirmPermanentDelete(n, folderLabel()))) return;
        statusEl.textContent = "deleting…";
        try {
            const res = await api.fetchApi("/star_output_cleaner/delete", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: postBody(),
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || res.statusText);
            const failed = data.errors?.length ? ` · ${data.errors.length} failed` : "";
            statusEl.textContent = `deleted ${data.deleted?.length ?? 0}${failed}`;
            state.selected.clear();
        } catch (e) {
            statusEl.textContent = "delete failed: " + (e.message || e);
        }
        refresh();
    });
    zipBtn.addEventListener("click", async () => {
        const n = state.selected.size;
        if (!n) return;
        statusEl.textContent = "preparing zip…";
        try {
            const res = await api.fetchApi("/star_output_cleaner/zip", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: postBody(),
            });
            if (!res.ok) {
                let msg = res.statusText;
                try { msg = (await res.json()).error || msg; } catch { }
                throw new Error(msg);
            }
            const blob = await res.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `star_output_cleaner_${new Date().toISOString().slice(0, 10)}.zip`;
            document.body.appendChild(a);
            a.click();
            a.remove();
            setTimeout(() => URL.revokeObjectURL(url), 5000);
            statusEl.textContent = `zipped ${n} image(s)`;
        } catch (e) {
            statusEl.textContent = "zip failed: " + (e.message || e);
        }
    });

    // ----- widget hooks ------------------------------------------------------------------
    ["source", "custom_folder", "mode", "amount", "unit"].forEach((name) => {
        const w = findW(name);
        if (!w) return;
        const orig = w.callback;
        w.callback = function () {
            orig?.apply(this, arguments);
            if (name === "mode" || name === "source") applyVisibility();
            if (name === "custom_folder") folderInput.value = w.value || "";
            debouncedRefresh();
        };
    });

    // reload after a saved workflow is configured (widget values are restored then)
    const onConfigure = node.onConfigure;
    node.onConfigure = function (info) {
        onConfigure?.apply(this, arguments);
        syncInputsFromWidgets();
        applyVisibility();
        debouncedRefresh();
    };

    // ----- init ---------------------------------------------------------------------------
    syncInputsFromWidgets();
    applyVisibility();
    requestAnimationFrame(() => {
        const sz = node.computeSize();
        node.setSize([Math.max(sz[0], 560), Math.max(sz[1], 660)]);
        node.graph?.setDirtyCanvas(true, true);
    });
    refresh();
}

injectCSS();

app.registerExtension({
    name: "Star.OutputCleaner",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            buildUI(this);
            return r;
        };
    },
});
