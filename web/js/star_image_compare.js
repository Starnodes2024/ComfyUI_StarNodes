import { app } from "../../../../scripts/app.js";

function getImageUrl(meta) {
    if (!meta || !meta.filename) return "";
    const subfolder = meta.subfolder ? `&subfolder=${encodeURIComponent(meta.subfolder)}` : "";
    const cacheBust = `&t=${Date.now()}`;
    return `/view?filename=${encodeURIComponent(meta.filename)}&type=${meta.type || "temp"}${subfolder}${cacheBust}`;
}

const PLACEHOLDER_IMAGE = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7";

app.registerExtension({
    name: "StarImageCompare",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "StarImageCompare") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            const posWidget = this.widgets?.find(w => w.name === "compare_position");
            if (posWidget) {
                posWidget.hidden = true;
                posWidget.type = "hidden";
                posWidget.value = posWidget.value ?? 0.5;
            }

            const container = document.createElement("div");
            container.style.position = "relative";
            container.style.width = "100%";
            container.style.height = "100%";
            container.style.backgroundColor = "#1a1a1a";
            container.style.border = "2px solid #444";
            container.style.borderRadius = "4px";
            container.style.overflow = "hidden";
            container.style.userSelect = "none";
            container.style.cursor = "ew-resize";
            container.style.boxSizing = "border-box";
            container.style.margin = "0";
            container.style.padding = "0";

            const img2 = document.createElement("img");
            img2.style.position = "absolute";
            img2.style.top = "0";
            img2.style.left = "0";
            img2.style.width = "100%";
            img2.style.height = "100%";
            img2.style.objectFit = "contain";
            img2.style.userSelect = "none";
            img2.style.pointerEvents = "none";
            img2.draggable = false;
            img2.alt = "Image 2";
            img2.src = PLACEHOLDER_IMAGE;

            const img1 = document.createElement("img");
            img1.style.position = "absolute";
            img1.style.top = "0";
            img1.style.left = "0";
            img1.style.width = "100%";
            img1.style.height = "100%";
            img1.style.objectFit = "contain";
            img1.style.userSelect = "none";
            img1.style.pointerEvents = "none";
            img1.style.clipPath = "inset(0 50% 0 0)";
            img1.draggable = false;
            img1.alt = "Image 1";
            img1.src = PLACEHOLDER_IMAGE;

            const divider = document.createElement("div");
            divider.style.position = "absolute";
            divider.style.top = "0";
            divider.style.width = "4px";
            divider.style.height = "100%";
            divider.style.backgroundColor = "rgba(255, 255, 255, 0.85)";
            divider.style.boxShadow = "0 0 4px rgba(0,0,0,0.5)";
            divider.style.transform = "translateX(-50%)";
            divider.style.cursor = "ew-resize";
            divider.style.pointerEvents = "none";
            divider.style.left = "50%";

            const slider = document.createElement("input");
            slider.type = "range";
            slider.min = "0";
            slider.max = "1";
            slider.step = "0.01";
            slider.value = "0.5";
            slider.title = "Drag to compare images. Right = Image 1, Left = Image 2.";
            slider.style.position = "absolute";
            slider.style.bottom = "10px";
            slider.style.left = "10px";
            slider.style.width = "calc(100% - 20px)";
            slider.style.zIndex = "10";
            slider.style.opacity = "0.8";
            slider.style.cursor = "pointer";

            const label = document.createElement("div");
            label.textContent = "Image 1 | Image 2";
            label.style.position = "absolute";
            label.style.top = "6px";
            label.style.left = "0";
            label.style.width = "100%";
            label.style.textAlign = "center";
            label.style.color = "rgba(255,255,255,0.8)";
            label.style.fontSize = "12px";
            label.style.pointerEvents = "none";
            label.style.userSelect = "none";
            label.style.zIndex = "5";

            container.appendChild(img2);
            container.appendChild(img1);
            container.appendChild(divider);
            container.appendChild(slider);
            container.appendChild(label);

            const placeholder = document.createElement("div");
            placeholder.textContent = "Connect two images and run the workflow.";
            placeholder.style.position = "absolute";
            placeholder.style.top = "50%";
            placeholder.style.left = "50%";
            placeholder.style.transform = "translate(-50%, -50%)";
            placeholder.style.color = "rgba(255,255,255,0.6)";
            placeholder.style.fontSize = "14px";
            placeholder.style.textAlign = "center";
            placeholder.style.pointerEvents = "none";
            placeholder.style.zIndex = "4";
            placeholder.style.display = "block";
            container.appendChild(placeholder);

            const update = (val) => {
                let pos = parseFloat(val);
                if (isNaN(pos)) pos = 0.5;
                pos = Math.max(0, Math.min(1, pos));
                const pct = pos * 100;

                img1.style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
                divider.style.left = `${pct}%`;
                slider.value = pos;

                if (posWidget) {
                    posWidget.value = pos;
                    if (posWidget.callback) {
                        posWidget.callback(pos);
                    }
                    if (this.onPropertyChanged) {
                        this.onPropertyChanged("compare_position", pos);
                    }
                }
            };

            slider.addEventListener("input", (e) => update(e.target.value));
            slider.addEventListener("pointerdown", (e) => e.stopPropagation());

            let dragging = false;
            const getPosFromEvent = (e) => {
                const rect = container.getBoundingClientRect();
                return Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
            };

            container.addEventListener("pointerdown", (e) => {
                dragging = true;
                update(getPosFromEvent(e));
                e.preventDefault();
            });
            container.addEventListener("pointermove", (e) => {
                if (dragging) {
                    update(getPosFromEvent(e));
                }
            });
            container.addEventListener("pointerup", () => {
                dragging = false;
            });
            container.addEventListener("pointerleave", () => {
                dragging = false;
            });

            const widget = this.addDOMWidget("star_image_compare", "viewer", container, {
                serialize: false,
                hideOnZoom: false
            });
            widget.container = container;
            widget.img1 = img1;
            widget.img2 = img2;
            widget.update = update;
            widget.placeholder = placeholder;
            this.compareWidget = widget;

            const FIXED_OFFSET = 200;

            widget.computeSize = (width) => {
                return [Math.max(256, width - 10), 400];
            };

            const updateContainerSize = (width, height) => {
                const h = Math.max(256, height - FIXED_OFFSET);
                const w = Math.max(256, width - 10);
                container.style.width = w + "px";
                container.style.height = h + "px";
            };

            const originalOnResize = this.onResize;
            this.onResize = function(size) {
                if (originalOnResize) originalOnResize.apply(this, arguments);
                updateContainerSize(size[0], size[1]);
            };

            const w = this.size ? this.size[0] : 0;
            const h = this.size ? this.size[1] : 0;
            this.setSize([Math.max(w, 532), Math.max(h, 600)]);
            if (this.size) {
                updateContainerSize(this.size[0], this.size[1]);
            }

            if (posWidget) {
                update(posWidget.value);
            }

            const onConfigured = this.onConfigure;
            this.onConfigure = function(info) {
                const r = onConfigured ? onConfigured.apply(this, arguments) : undefined;
                if (posWidget) {
                    update(posWidget.value);
                }
                return r;
            };

            return result;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(message) {
            if (onExecuted) {
                onExecuted.apply(this, arguments);
            }

            const output = message || {};
            const dataRaw = output.star_image_compare
                || output.output?.star_image_compare
                || output.ui?.star_image_compare
                || [];
            const data = (Array.isArray(dataRaw) && dataRaw.length > 0 ? dataRaw[0] : dataRaw) || {};

            if (this.compareWidget) {
                const w = this.compareWidget;
                if (data.image1 && data.image2) {
                    w.img1.src = getImageUrl(data.image1);
                    w.img2.src = getImageUrl(data.image2);
                    w.update(data.compare_position ?? 0.5);
                    if (w.placeholder) w.placeholder.style.display = "none";
                } else {
                    if (w.placeholder) w.placeholder.style.display = "block";
                }
            }
        };
    }
});
