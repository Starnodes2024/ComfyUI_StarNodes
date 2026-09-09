import { app } from "../../../../scripts/app.js";

function getVideoUrl(meta) {
    if (!meta || !meta.filename) return "";
    const subfolder = meta.subfolder ? `&subfolder=${encodeURIComponent(meta.subfolder)}` : "";
    const cacheBust = `&t=${Date.now()}`;
    return `/view?filename=${encodeURIComponent(meta.filename)}&type=${meta.type || "temp"}${subfolder}${cacheBust}`;
}

app.registerExtension({
    name: "StarVideoCompare",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "StarVideoCompare") return;

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

            // Video 2 (background)
            const vid2 = document.createElement("video");
            vid2.style.position = "absolute";
            vid2.style.top = "0";
            vid2.style.left = "0";
            vid2.style.width = "100%";
            vid2.style.height = "100%";
            vid2.style.objectFit = "contain";
            vid2.style.pointerEvents = "none";
            vid2.muted = true;
            vid2.playsInline = true;
            vid2.preload = "auto";

            // Clip wrapper for the wipe effect
            const clipWrapper = document.createElement("div");
            clipWrapper.style.position = "absolute";
            clipWrapper.style.top = "0";
            clipWrapper.style.left = "0";
            clipWrapper.style.width = "100%";
            clipWrapper.style.height = "100%";
            clipWrapper.style.pointerEvents = "none";
            clipWrapper.style.clipPath = "inset(0 50% 0 0)";

            // Video 1 (foreground, clipped)
            const vid1 = document.createElement("video");
            vid1.style.position = "absolute";
            vid1.style.top = "0";
            vid1.style.left = "0";
            vid1.style.width = "100%";
            vid1.style.height = "100%";
            vid1.style.objectFit = "contain";
            vid1.style.pointerEvents = "none";
            vid1.muted = true;
            vid1.playsInline = true;
            vid1.preload = "auto";

            clipWrapper.appendChild(vid1);

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
            divider.style.zIndex = "2";

            const slider = document.createElement("input");
            slider.type = "range";
            slider.min = "0";
            slider.max = "1";
            slider.step = "0.01";
            slider.value = "0.5";
            slider.title = "Drag: Compare | Scroll: Zoom | Right/Middle/Shift+Click: Pan | DblClick: Reset";
            slider.style.position = "absolute";
            slider.style.bottom = "10px";
            slider.style.left = "10px";
            slider.style.width = "calc(100% - 20px)";
            slider.style.zIndex = "10";
            slider.style.opacity = "0.8";
            slider.style.cursor = "pointer";

            const label = document.createElement("div");
            label.textContent = "Video 1 | Video 2";
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

            const placeholder = document.createElement("div");
            placeholder.textContent = "Connect two videos and run the workflow.";
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

            container.appendChild(vid2);
            container.appendChild(clipWrapper);
            container.appendChild(divider);
            container.appendChild(slider);
            container.appendChild(label);

            // --- Zoom & Pan ---
            let zoom = 1;
            let panX = 0;
            let panY = 0;

            const updateTransform = () => {
                const t = `translate(${panX}px, ${panY}px) scale(${zoom})`;
                vid1.style.transform = t;
                vid2.style.transform = t;
            };

            const update = (val) => {
                let pos = parseFloat(val);
                if (isNaN(pos)) pos = 0.5;
                pos = Math.max(0, Math.min(1, pos));
                const pct = pos * 100;

                clipWrapper.style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
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

            let draggingSlider = false;
            let panning = false;
            let lastPanMouseX = 0;
            let lastPanMouseY = 0;

            const getPosFromEvent = (e) => {
                const rect = container.getBoundingClientRect();
                return Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
            };

            container.addEventListener("wheel", (e) => {
                e.preventDefault();
                e.stopPropagation();
                const zoomDelta = e.deltaY > 0 ? 0.9 : 1.1;
                const newZoom = Math.max(0.1, Math.min(zoom * zoomDelta, 20));
                const rect = container.getBoundingClientRect();
                const mouseX = e.clientX - rect.left;
                const mouseY = e.clientY - rect.top;
                panX = mouseX - (mouseX - panX) * (newZoom / zoom);
                panY = mouseY - (mouseY - panY) * (newZoom / zoom);
                zoom = newZoom;
                updateTransform();
            });

            container.addEventListener("contextmenu", (e) => {
                e.preventDefault();
                e.stopPropagation();
            });

            container.addEventListener("pointerdown", (e) => {
                e.preventDefault();
                e.stopPropagation();
                if (e.button === 1 || e.button === 2 || (e.button === 0 && e.shiftKey)) {
                    panning = true;
                    lastPanMouseX = e.clientX;
                    lastPanMouseY = e.clientY;
                    container.style.cursor = "grabbing";
                } else if (e.button === 0) {
                    draggingSlider = true;
                    update(getPosFromEvent(e));
                    container.style.cursor = "ew-resize";
                }
            });

            container.addEventListener("pointermove", (e) => {
                if (panning) {
                    e.preventDefault();
                    e.stopPropagation();
                    panX += e.clientX - lastPanMouseX;
                    panY += e.clientY - lastPanMouseY;
                    lastPanMouseX = e.clientX;
                    lastPanMouseY = e.clientY;
                    updateTransform();
                } else if (draggingSlider) {
                    e.preventDefault();
                    e.stopPropagation();
                    update(getPosFromEvent(e));
                }
            });

            const stopDrag = () => {
                panning = false;
                draggingSlider = false;
                container.style.cursor = "ew-resize";
            };

            container.addEventListener("pointerup", stopDrag);
            container.addEventListener("pointerleave", stopDrag);

            container.addEventListener("dblclick", (e) => {
                e.preventDefault();
                e.stopPropagation();
                zoom = 1;
                panX = 0;
                panY = 0;
                updateTransform();
            });

            const widget = this.addDOMWidget("star_video_compare", "viewer", container, {
                serialize: false,
                hideOnZoom: false
            });
            widget.container = container;
            widget.vid1 = vid1;
            widget.vid2 = vid2;
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
            const dataRaw = output.star_video_compare
                || output.output?.star_video_compare
                || output.ui?.star_video_compare
                || [];
            const data = (Array.isArray(dataRaw) && dataRaw.length > 0 ? dataRaw[0] : dataRaw) || {};

            if (this.compareWidget) {
                const w = this.compareWidget;
                if (data.video1 && data.video2) {
                    const url1 = getVideoUrl(data.video1);
                    const url2 = getVideoUrl(data.video2);

                    // Sync playback: play both, keep timestamps aligned
                    w.vid2.src = url2;
                    w.vid2.loop = data.loop !== false;
                    w.vid2.playbackRate = 1.0;
                    w.vid2.currentTime = 0;

                    w.vid1.src = url1;
                    w.vid1.loop = data.loop !== false;
                    w.vid1.playbackRate = 1.0;
                    w.vid1.currentTime = 0;

                    // Keep video 1 synced to video 2
                    const syncVideos = () => {
                        const diff = Math.abs(w.vid1.currentTime - w.vid2.currentTime);
                        if (diff > 0.1) {
                            w.vid1.currentTime = w.vid2.currentTime;
                        }
                    };

                    const playBoth = () => {
                        w.vid2.play().catch(() => {});
                        w.vid1.play().catch(() => {});
                    };

                    const pauseBoth = () => {
                        w.vid2.pause();
                        w.vid1.pause();
                    };

                    w.vid2.removeEventListener("timeupdate", syncVideos);
                    w.vid2.addEventListener("timeupdate", syncVideos);
                    w.vid2.onplay = playBoth;
                    w.vid2.onpause = pauseBoth;

                    playBoth();
                    if (w.placeholder) w.placeholder.style.display = "none";
                    w.update(data.compare_position ?? 0.5);
                } else {
                    if (w.placeholder) w.placeholder.style.display = "block";
                }
            }
        };
    }
});
