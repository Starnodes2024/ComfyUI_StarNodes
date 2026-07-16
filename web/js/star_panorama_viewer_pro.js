import { app } from "../../../../scripts/app.js";

const RATIO_MAP = {
    "custom": null,
    "1:1": [1, 1],
    "1:2": [1, 2],
    "3:4": [3, 4],
    "2:3": [2, 3],
    "5:7": [5, 7],
    "9:16": [9, 16],
    "9:21": [9, 21],
    "10:16": [10, 16],
    "4:3": [4, 3],
    "16:10": [16, 10],
    "3:2": [3, 2],
    "2:1": [2, 1],
    "7:5": [7, 5],
    "16:9": [16, 9],
    "21:9": [21, 9],
};

function getWidgetValue(node, name) {
    if (!node.widgets) return null;
    for (const w of node.widgets) {
        if (w.name === name) return w.value;
    }
    return null;
}

function computeExportDims(node) {
    const resolution = getWidgetValue(node, "resolution") || "Full HD (1920x1080)";
    const ratio = getWidgetValue(node, "ratio") || "16:9";
    const customRatio = getWidgetValue(node, "custom_ratio") || "21:9";

    const baseWidth = resolution.includes("Full HD") ? 1920 : 1280;
    let rw = 16, rh = 9;

    if (ratio === "custom") {
        const parts = customRatio.split(/[:x]/);
        if (parts.length === 2) {
            const a = parseInt(parts[0].trim());
            const b = parseInt(parts[1].trim());
            if (a > 0 && b > 0) { rw = a; rh = b; }
        }
    } else if (RATIO_MAP[ratio]) {
        rw = RATIO_MAP[ratio][0];
        rh = RATIO_MAP[ratio][1];
    }

    let width = baseWidth - (baseWidth % 2);
    let height = Math.round(baseWidth * rh / rw);
    height = height - (height % 2);
    return { width, height };
}

function updateGreenFrame(node) {
    if (!node.panoramaWidget) return;
    const widget = node.panoramaWidget;
    const container = widget.container;
    const createFrames = getWidgetValue(node, "create_video_frames");
    const dims = computeExportDims(node);

    if (!widget.greenFrame) {
        const frame = document.createElement("div");
        frame.style.position = "absolute";
        frame.style.border = "3px solid #00ff00";
        frame.style.boxShadow = "0 0 8px rgba(0,255,0,0.6)";
        frame.style.pointerEvents = "none";
        frame.style.zIndex = "5";
        frame.style.left = "50%";
        frame.style.top = "50%";
        frame.style.transform = "translate(-50%, -50%)";
        container.appendChild(frame);
        widget.greenFrame = frame;

        const label = document.createElement("div");
        label.style.position = "absolute";
        label.style.top = "-22px";
        label.style.left = "50%";
        label.style.transform = "translateX(-50%)";
        label.style.background = "rgba(0,180,0,0.85)";
        label.style.color = "#fff";
        label.style.fontSize = "11px";
        label.style.padding = "2px 8px";
        label.style.borderRadius = "3px";
        label.style.whiteSpace = "nowrap";
        label.style.fontWeight = "bold";
        frame.appendChild(label);
        widget.greenFrameLabel = label;
    }

    const containerW = container.clientWidth || 512;
    const containerH = container.clientHeight || 512;
    const aspect = dims.width / dims.height;
    const containerAspect = containerW / containerH;

    let frameW, frameH;
    if (aspect > containerAspect) {
        frameW = containerW - 8;
        frameH = frameW / aspect;
    } else {
        frameH = containerH - 8;
        frameW = frameH * aspect;
    }

    widget.greenFrame.style.width = `${frameW}px`;
    widget.greenFrame.style.height = `${frameH}px`;
    widget.greenFrameLabel.textContent = `${dims.width}x${dims.height}`;
    widget._lastDims = `${dims.width}x${dims.height}`;

    const showFrame = (createFrames === true || createFrames === "true");
    widget.greenFrame.style.display = showFrame ? "block" : "none";
}

app.registerExtension({
    name: "starnodes.panorama_viewer_pro",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "StarPanoramaViewerPro") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                this.serialize_widgets = false;

                const container = document.createElement("div");
                container.style.width = "512px";
                container.style.height = "512px";
                container.style.position = "relative";
                container.style.backgroundColor = "#000";
                container.style.border = "2px solid #444";
                container.style.borderRadius = "4px";
                container.style.overflow = "hidden";

                const canvas = document.createElement("canvas");
                canvas.width = 512;
                canvas.height = 512;
                canvas.style.display = "block";
                canvas.style.width = "100%";
                canvas.style.height = "100%";
                container.appendChild(canvas);

                const loadingText = document.createElement("div");
                loadingText.textContent = "Waiting for panorama...";
                loadingText.style.position = "absolute";
                loadingText.style.top = "50%";
                loadingText.style.left = "50%";
                loadingText.style.transform = "translate(-50%, -50%)";
                loadingText.style.color = "#fff";
                loadingText.style.fontSize = "14px";
                loadingText.style.pointerEvents = "none";
                loadingText.style.zIndex = "3";
                container.appendChild(loadingText);

                const widget = this.addDOMWidget("panorama_viewer_pro", "viewer", container, {
                    serialize: false,
                    hideOnZoom: false
                });

                widget.canvas = canvas;
                widget.container = container;
                widget.loadingText = loadingText;
                widget.viewer = null;
                widget.greenFrame = null;

                this.panoramaWidget = widget;
                const w = this.size ? this.size[0] : 0;
                const h = this.size ? this.size[1] : 0;
                this.setSize([Math.max(w, 532), Math.max(h, 600)]);

                const self = this;
                widget._pollInterval = setInterval(() => {
                    const dims = computeExportDims(self);
                    const createFrames = getWidgetValue(self, "create_video_frames");
                    const showFrame = (createFrames === true || createFrames === "true");
                    const key = `${dims.width}x${dims.height}_${showFrame}`;
                    if (widget._lastDims !== key) {
                        updateGreenFrame(self);
                    }
                }, 500);

                updateGreenFrame(this);

                return result;
            };

            const onExecuted = nodeType.prototype.onExecuted;

            nodeType.prototype.onExecuted = function(message) {
                if (onExecuted) {
                    onExecuted.apply(this, arguments);
                }

                const output = message || {};
                const panos = output.panoramas || output.output?.panoramas || output.ui?.panoramas;

                if (panos && panos.length > 0) {
                    const panoData = panos[0];
                    const subfolder = panoData.subfolder ? `${panoData.subfolder}` : "";
                    const cacheBust = `&t=${Date.now()}`;
                    const imageUrl = `/view?filename=${encodeURIComponent(panoData.filename)}&type=${panoData.type}${subfolder ? "&subfolder=" + encodeURIComponent(subfolder) : ""}${cacheBust}`;
                    const depthUrl = panoData.depth_filename
                        ? `/view?filename=${encodeURIComponent(panoData.depth_filename)}&type=${panoData.type}${subfolder ? "&subfolder=" + encodeURIComponent(subfolder) : ""}${cacheBust}`
                        : null;

                    if (this.panoramaWidget) {
                        initPanoramaViewerPro(this.panoramaWidget, imageUrl, panoData.layout, depthUrl);
                        updateGreenFrame(this);
                    }
                }
            };

            const onRemoved = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function() {
                if (this.panoramaWidget && this.panoramaWidget._pollInterval) {
                    clearInterval(this.panoramaWidget._pollInterval);
                }
                if (onRemoved) {
                    onRemoved.apply(this, arguments);
                }
            };
        }
    }
});

function loadThreeJS(callback) {
    if (window.THREE) {
        callback();
        return;
    }
    const script = document.createElement("script");
    script.src = "/extensions/comfyui_starnodes/js/three.min.js";
    script.onload = callback;
    script.onerror = () => { console.error("Failed to load local Three.js"); };
    document.head.appendChild(script);
}

function makeControlButton(label, title) {
    const btn = document.createElement("button");
    btn.textContent = label;
    btn.title = title;
    btn.style.width = "28px";
    btn.style.height = "28px";
    btn.style.padding = "0";
    btn.style.border = "1px solid #555";
    btn.style.borderRadius = "4px";
    btn.style.background = "#222";
    btn.style.color = "#eee";
    btn.style.fontSize = "14px";
    btn.style.cursor = "pointer";
    btn.style.lineHeight = "1";
    btn.addEventListener("mouseenter", () => { btn.style.background = "#3a3a3a"; });
    btn.addEventListener("mouseleave", () => { btn.style.background = "#222"; });
    return btn;
}

function initPanoramaViewerPro(widget, imageUrl, layout, depthUrl) {
    loadThreeJS(() => {
        const canvas = widget.canvas;
        const container = widget.container;
        const loadingText = widget.loadingText;

        if (widget.viewerState) {
            const old = widget.viewerState;
            cancelAnimationFrame(old.frameId);
            old.abort.abort();
            old.controls.remove();
            widget.viewerState = null;
        }

        if (loadingText) { loadingText.style.display = "none"; }

        const THREE = window.THREE;

        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
        camera.position.set(0, 0, 0.1);

        const renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
        renderer.setSize(512, 512);

        const geometry = new THREE.SphereGeometry(500, 60, 40);
        geometry.scale(-1, 1, 1);

        const textureLoader = new THREE.TextureLoader();
        textureLoader.crossOrigin = "anonymous";

        let material;
        let parallaxEnabled = layout === "SBS" || layout === "Top/Bottom";
        let mouseX = 0, mouseY = 0;
        let targetRotationX = 0, targetRotationY = 0;
        let currentRotationX = 0, currentRotationY = 0;

        const state = {
            frameId: 0,
            abort: new AbortController(),
            controls: null,
            autoRotate: false,
            autoSpeed: 1,
            heldX: 0,
            heldY: 0
        };
        widget.viewerState = state;
        const signal = state.abort.signal;

        textureLoader.load(imageUrl, (texture) => {
            if (layout === "Top/Bottom") {
                texture.repeat.set(1, 0.5);
                texture.offset.set(0, 0.5);
            } else if (layout === "SBS") {
                texture.repeat.set(0.5, 1);
                texture.offset.set(0, 0);
            }

            material = new THREE.MeshBasicMaterial({ map: texture, side: THREE.FrontSide });
            const sphere = new THREE.Mesh(geometry, material);
            scene.add(sphere);

            let isDragging = false;
            let previousMouseX = 0, previousMouseY = 0;

            canvas.addEventListener("mousedown", (e) => {
                isDragging = true;
                previousMouseX = e.clientX;
                previousMouseY = e.clientY;
            }, { signal });

            canvas.addEventListener("mousemove", (e) => {
                const rect = canvas.getBoundingClientRect();
                mouseX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
                mouseY = -((e.clientY - rect.top) / rect.height) * 2 + 1;

                if (isDragging) {
                    const deltaX = e.clientX - previousMouseX;
                    const deltaY = e.clientY - previousMouseY;
                    targetRotationY += deltaX * 0.005;
                    targetRotationX += deltaY * 0.005;
                    targetRotationX = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, targetRotationX));
                    previousMouseX = e.clientX;
                    previousMouseY = e.clientY;
                }

                if (parallaxEnabled && material && material.map) {
                    const parallaxStrength = 0.02;
                    const offsetX = mouseX * parallaxStrength;
                    const offsetY = mouseY * parallaxStrength;
                    if (layout === "Top/Bottom") {
                        const baseOffset = 0.5;
                        const newOffsetY = baseOffset + offsetY * 0.5;
                        material.map.offset.set(offsetX * 0.5, Math.max(0, Math.min(0.5, newOffsetY)));
                    } else if (layout === "SBS") {
                        const newOffsetX = offsetX * 0.5;
                        material.map.offset.set(Math.max(0, Math.min(0.5, newOffsetX)), offsetY * 0.5);
                    }
                }
            }, { signal });

            canvas.addEventListener("mouseup", () => { isDragging = false; }, { signal });
            canvas.addEventListener("mouseleave", () => { isDragging = false; }, { signal });

            function setZoom(delta) {
                camera.fov = Math.max(30, Math.min(120, camera.fov + delta));
                camera.updateProjectionMatrix();
            }

            canvas.addEventListener("wheel", (e) => {
                e.preventDefault();
                setZoom(e.deltaY * 0.05);
            }, { signal });

            const controls = document.createElement("div");
            controls.style.position = "absolute";
            controls.style.bottom = "8px";
            controls.style.left = "50%";
            controls.style.transform = "translateX(-50%)";
            controls.style.display = "flex";
            controls.style.alignItems = "center";
            controls.style.gap = "4px";
            controls.style.padding = "5px 8px";
            controls.style.background = "rgba(0, 0, 0, 0.65)";
            controls.style.borderRadius = "6px";
            controls.style.userSelect = "none";
            controls.style.zIndex = "10";
            controls.addEventListener("mousedown", (e) => e.stopPropagation(), { signal });
            controls.addEventListener("wheel", (e) => e.stopPropagation(), { signal });
            container.appendChild(controls);
            state.controls = controls;

            function addHoldArrow(label, title, dx, dy) {
                const btn = makeControlButton(label, title);
                const press = (e) => { e.preventDefault(); state.heldX = dx; state.heldY = dy; };
                const release = () => { state.heldX = 0; state.heldY = 0; };
                btn.addEventListener("pointerdown", press, { signal });
                btn.addEventListener("pointerup", release, { signal });
                btn.addEventListener("pointerleave", release, { signal });
                controls.appendChild(btn);
            }

            addHoldArrow("\u25C0", "Pan left (hold)", 0, 1);
            addHoldArrow("\u25B2", "Pan up (hold)", 1, 0);
            addHoldArrow("\u25BC", "Pan down (hold)", -1, 0);
            addHoldArrow("\u25B6", "Pan right (hold)", 0, -1);

            const resetBtn = makeControlButton("\u2302", "Reset view");
            resetBtn.addEventListener("click", () => {
                targetRotationX = 0; targetRotationY = 0;
                camera.fov = 75; camera.updateProjectionMatrix();
            }, { signal });
            controls.appendChild(resetBtn);

            const zoomOutBtn = makeControlButton("\u2212", "Zoom out");
            zoomOutBtn.addEventListener("click", () => setZoom(5), { signal });
            controls.appendChild(zoomOutBtn);

            const zoomInBtn = makeControlButton("+", "Zoom in");
            zoomInBtn.addEventListener("click", () => setZoom(-5), { signal });
            controls.appendChild(zoomInBtn);

            const playBtn = makeControlButton("\u25B6\uFE0E", "Start/stop auto-rotation");
            playBtn.style.marginLeft = "6px";
            playBtn.addEventListener("click", () => {
                state.autoRotate = !state.autoRotate;
                playBtn.textContent = state.autoRotate ? "\u23F8\uFE0E" : "\u25B6\uFE0E";
            }, { signal });
            controls.appendChild(playBtn);

            const speedSlider = document.createElement("input");
            speedSlider.type = "range";
            speedSlider.min = "-5"; speedSlider.max = "5"; speedSlider.step = "0.1"; speedSlider.value = "1";
            speedSlider.title = "Auto-rotation speed (negative = reverse)";
            speedSlider.style.width = "70px";
            speedSlider.style.cursor = "pointer";
            speedSlider.addEventListener("input", () => { state.autoSpeed = parseFloat(speedSlider.value); }, { signal });
            controls.appendChild(speedSlider);

            const fsBtn = makeControlButton("\u26F6", "Toggle fullscreen");
            fsBtn.style.marginLeft = "6px";
            fsBtn.addEventListener("click", () => {
                if (document.fullscreenElement === container) {
                    document.exitFullscreen();
                } else {
                    container.requestFullscreen();
                }
            }, { signal });
            controls.appendChild(fsBtn);

            document.addEventListener("fullscreenchange", () => {
                if (document.fullscreenElement === container) {
                    renderer.setSize(container.clientWidth, container.clientHeight);
                    camera.aspect = container.clientWidth / container.clientHeight;
                } else {
                    renderer.setSize(512, 512);
                    camera.aspect = 1;
                }
                camera.updateProjectionMatrix();
            }, { signal });

            function animate() {
                state.frameId = requestAnimationFrame(animate);
                if (state.heldX || state.heldY) {
                    targetRotationX += state.heldX * 0.02;
                    targetRotationY += state.heldY * 0.02;
                    targetRotationX = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, targetRotationX));
                }
                if (state.autoRotate) { targetRotationY += state.autoSpeed * 0.003; }
                currentRotationX += (targetRotationX - currentRotationX) * 0.1;
                currentRotationY += (targetRotationY - currentRotationY) * 0.1;
                camera.rotation.x = currentRotationX;
                camera.rotation.y = currentRotationY;
                renderer.render(scene, camera);
            }

            animate();
        }, undefined, (error) => {
            console.error("Error loading panorama texture:", error);
            if (loadingText) {
                loadingText.textContent = "Error loading panorama";
                loadingText.style.display = "block";
                loadingText.style.color = "#f44";
            }
        });
    });
}
