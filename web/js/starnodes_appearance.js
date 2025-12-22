import { app } from "../../../../scripts/app.js";
import { $el } from "../../../../scripts/ui.js";

// Extension to apply custom colors to all StarNodes
app.registerExtension({
    name: "StarNodes.appearance",
    settings: [
        {
            id: "StarNodes.ApplyToAllNodes",
            name: "Apply StarNodes Style to All Nodes",
            type: "boolean",
            defaultValue: false,
            tooltip: "Apply StarNodes custom colors and appearance to all nodes in ComfyUI (Requires Page Reload)"
        },
        {
            id: "StarNodes.Theme",
            name: "StarNodes Theme",
            type: () => {
                const THEMES = [
                    { id: "default", name: "ComfyUI Default (No Overrides)", bg: null, title: null, frame: null },
                    { id: "starnodes_purple", name: "StarNodes Purple", bg: "#3d124d", title: "#19124d", frame: "#ffffff" },
                    { id: "midnight", name: "Midnight", bg: "#0b1220", title: "#0f2a5f", frame: "#4cc9f0" },
                    { id: "emerald", name: "Emerald", bg: "#0e2a1f", title: "#145a32", frame: "#34d399" },
                    { id: "sunset", name: "Sunset", bg: "#2b0b10", title: "#7c2d12", frame: "#fb923c" },
                    { id: "ocean", name: "Ocean", bg: "#062a3a", title: "#075985", frame: "#22d3ee" },
                    { id: "rose", name: "Rose", bg: "#2a0b1e", title: "#9f1239", frame: "#fb7185" },
                    { id: "lavender", name: "Lavender", bg: "#1c1630", title: "#5b21b6", frame: "#c4b5fd" },
                    { id: "amber", name: "Amber", bg: "#1f1407", title: "#92400e", frame: "#fbbf24" },
                    { id: "forest", name: "Forest", bg: "#0f1f12", title: "#14532d", frame: "#86efac" },
                    { id: "ice", name: "Ice", bg: "#0b1d26", title: "#164e63", frame: "#a5f3fc" },
                    { id: "mono", name: "Monochrome", bg: "#1a1a1a", title: "#333333", frame: "#bdbdbd" },
                    { id: "coffee", name: "Coffee", bg: "#1a0f0a", title: "#7c3f1d", frame: "#e7c6a5" }
                ];

                const getTheme = () => {
                    let v = "starnodes_purple";
                    try { v = app.extensionManager?.setting?.get?.("StarNodes.Theme") ?? v; } catch (_) {}
                    const t = THEMES.find(x => x.id === v) || THEMES[1];
                    return t;
                };

                const applyThemeToExistingNodes = () => {
                    const applyToAll = app.extensionManager?.setting?.get?.("StarNodes.ApplyToAllNodes");
                    const theme = getTheme();
                    const nodes = app.graph?._nodes || [];
                    for (const n of nodes) {
                        const isStar = n?.constructor?.nodeData?.category && String(n.constructor.nodeData.category).startsWith("⭐");
                        if (!applyToAll && !isStar) continue;

                        if (!n.properties) n.properties = {};

                        // If theme is default, clear overrides completely.
                        if (theme.id === "default") {
                            delete n.properties.starnodes_custom_bgcolor;
                            delete n.properties.starnodes_custom_titlebarcolor;
                            delete n.properties.starnodes_frame_color;
                            delete n.properties.starnodes_frame_width;
                            n._starnodes_frame_color = null;
                            n._starnodes_frame_width = 0;
                            n.bgcolor = undefined;
                            n.color = undefined;
                            continue;
                        }

                        // Do not overwrite per-node custom colors if user has set them.
                        if (!n.properties.starnodes_custom_bgcolor) {
                            n.bgcolor = theme.bg;
                        }
                        if (!n.properties.starnodes_custom_titlebarcolor) {
                            n.color = theme.title;
                        }

                        if (!n.properties.starnodes_frame_color) {
                            n._starnodes_frame_color = theme.frame;
                        } else {
                            n._starnodes_frame_color = n.properties.starnodes_frame_color;
                        }

                        const fw = n.properties.starnodes_frame_width;
                        n._starnodes_frame_width = (typeof fw === "number" && isFinite(fw)) ? fw : 1;
                    }
                    if (app.graph) app.graph.change();
                    if (app.canvas) app.canvas.setDirty(true, true);
                };

                let currentTheme = "starnodes_purple";
                try { currentTheme = app.extensionManager?.setting?.get?.("StarNodes.Theme") ?? currentTheme; } catch (_) {}

                const select = $el("select", {
                    style: { minWidth: "220px" },
                    onchange: function() {
                        const val = this.value;
                        app.extensionManager?.setting?.set?.("StarNodes.Theme", val);
                        applyThemeToExistingNodes();
                    }
                });
                for (const t of THEMES) {
                    const opt = document.createElement("option");
                    opt.value = t.id;
                    opt.textContent = t.name;
                    if (String(t.id) === String(currentTheme)) opt.selected = true;
                    select.appendChild(opt);
                }

                const resetBtn = $el("button", {
                    textContent: "Reset All Nodes to ComfyUI Defaults",
                    style: {
                        padding: "6px 10px",
                        backgroundColor: "#d73502",
                        color: "white",
                        border: "none",
                        borderRadius: "4px",
                        cursor: "pointer",
                        fontSize: "12px",
                        fontWeight: "bold",
                        marginLeft: "10px"
                    },
                    onclick: function() {
                        if (!confirm("Reset all node colors/frames to ComfyUI LiteGraph defaults? This clears StarNodes appearance overrides.")) {
                            return;
                        }
                        // Switch to default theme but do NOT toggle ApplyToAllNodes.
                        app.extensionManager?.setting?.set?.("StarNodes.Theme", "default");
                        select.value = "default";

                        const nodes = app.graph?._nodes || [];
                        for (const n of nodes) {
                            if (!n?.properties) n.properties = {};
                            delete n.properties.starnodes_custom_bgcolor;
                            delete n.properties.starnodes_custom_titlebarcolor;
                            delete n.properties.starnodes_frame_color;
                            delete n.properties.starnodes_frame_width;
                            n._starnodes_frame_color = null;
                            n._starnodes_frame_width = 0;
                            n.bgcolor = undefined;
                            n.color = undefined;
                        }
                        if (app.graph) app.graph.change();
                        if (app.canvas) app.canvas.setDirty(true, true);
                    }
                });

                return $el("tr", [
                    $el("td", [ $el("label", { textContent: "Theme" }) ]),
                    $el("td", [ select, resetBtn ])
                ]);
            },
            defaultValue: "starnodes_purple",
            tooltip: "Choose a ready-to-use StarNodes color theme. Border/frame width defaults to 1px for all themes."
        }
    ],
    async setup() {
        // This runs once when the extension is loaded
        console.log("StarNodes appearance extension setup");
    },
    
    async beforeRegisterNodeDef(nodeType, nodeData) {
        // Check if we should apply to all nodes
        const applyToAll = app.extensionManager?.setting?.get?.("StarNodes.ApplyToAllNodes");

        const THEMES = {
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
            coffee: { bg: "#1a0f0a", title: "#7c3f1d", frame: "#e7c6a5" }
        };

        const getThemeState = () => {
            let themeId = "starnodes_purple";
            try { themeId = app.extensionManager?.setting?.get?.("StarNodes.Theme") ?? themeId; } catch (_) {}
            const theme = THEMES[themeId] || THEMES.starnodes_purple;
            return { themeId, theme };
        };

        // Check if this is a StarNode by looking at the category or if global override is enabled
        if (applyToAll || (nodeData.category && nodeData.category.startsWith("⭐"))) {
            if (nodeData.category && nodeData.category.startsWith("⭐")) {
                 console.log(`Found StarNode: ${nodeData.name}, applying custom colors`);
            } else if (applyToAll) {
                 // Less verbose logging for all nodes to avoid console spam
            }
            
            // Define our colors
            const { themeId: initThemeId, theme: initTheme } = getThemeState();
            const backgroundColor = initTheme.bg || "#3d124d";  // Purple background
            const titleColor = initTheme.title || "#19124d";       // Dark blue title

            const applyStarNodeColors = (node) => {
                if (!node) {
                    return;
                }
                if (!node.properties) {
                    node.properties = {};
                }
                const { themeId, theme } = getThemeState();
                const bgDefault = theme.bg || backgroundColor;
                const titleDefault = theme.title || titleColor;
                if (themeId === "default") {
                    node.bgcolor = undefined;
                    node.color = undefined;
                    return;
                }
                const bg = node.properties.starnodes_custom_bgcolor || bgDefault;
                const titlebar = node.properties.starnodes_custom_titlebarcolor || titleDefault;
                node.bgcolor = bg;
                node.color = titlebar;
            };

            const applyStarNodeFrame = (node) => {
                if (!node) {
                    return;
                }
                if (!node.properties) {
                    node.properties = {};
                }
                const { themeId, theme } = getThemeState();
                if (themeId === "default") {
                    node._starnodes_frame_color = null;
                    node._starnodes_frame_width = 0;
                    return;
                }
                const frameColor = node.properties.starnodes_frame_color || theme.frame || null;
                const frameWidth = node.properties.starnodes_frame_width;
                node._starnodes_frame_color = frameColor;
                node._starnodes_frame_width = (typeof frameWidth === "number" && isFinite(frameWidth)) ? frameWidth : 1;
            };

            const openColorPicker = (node) => {
                const input = document.createElement("input");
                input.type = "color";
                const current = (node?.properties?.starnodes_custom_bgcolor || node?.bgcolor || backgroundColor);
                input.value = (typeof current === "string" && /^#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})$/.test(current)) ? current : backgroundColor;
                const canvasEl = app.canvas?.canvas;
                const rect = canvasEl?.getBoundingClientRect?.();
                const ds = app.canvas?.ds;
                const scale = ds?.scale ?? 1;
                const rawOffset = ds?.offset ?? ds?.origin ?? [0, 0];
                const offsetX = (Array.isArray(rawOffset) ? rawOffset[0] : rawOffset?.[0]) ?? rawOffset?.x ?? 0;
                const offsetY = (Array.isArray(rawOffset) ? rawOffset[1] : rawOffset?.[1]) ?? rawOffset?.y ?? 0;
                const nodeX = (node?.pos?.[0] ?? 0) + (node?.size?.[0] ?? 0);
                const nodeY = (node?.pos?.[1] ?? 0);
                const screenX = (rect?.left ?? 0) + (nodeX + offsetX) * scale;
                const screenY = (rect?.top ?? 0) + (nodeY + offsetY) * scale;

                input.style.position = "fixed";
                input.style.left = `${Math.max(0, Math.min(window.innerWidth - 1, Math.round(screenX)))}px`;
                input.style.top = `${Math.max(0, Math.min(window.innerHeight - 1, Math.round(screenY)))}px`;
                input.style.opacity = "0";
                input.style.width = "1px";
                input.style.height = "1px";
                document.body.appendChild(input);

                const cleanup = () => {
                    try {
                        document.body.removeChild(input);
                    } catch (_) {}
                };

                input.addEventListener("input", () => {
                    if (!node.properties) {
                        node.properties = {};
                    }
                    node.properties.starnodes_custom_bgcolor = input.value;
                    applyStarNodeColors(node);
                    if (node.graph) {
                        node.graph.change();
                    }
                    if (app.canvas) {
                        app.canvas.setDirty(true, true);
                    }
                });

                input.addEventListener("change", () => cleanup(), { once: true });
                input.addEventListener("blur", () => cleanup(), { once: true });
                input.click();
            };

            const openTitleBarColorPicker = (node) => {
                const input = document.createElement("input");
                input.type = "color";
                const current = (node?.properties?.starnodes_custom_titlebarcolor || node?.color || titleColor);
                input.value = (typeof current === "string" && /^#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})$/.test(current)) ? current : titleColor;
                const canvasEl = app.canvas?.canvas;
                const rect = canvasEl?.getBoundingClientRect?.();
                const ds = app.canvas?.ds;
                const scale = ds?.scale ?? 1;
                const rawOffset = ds?.offset ?? ds?.origin ?? [0, 0];
                const offsetX = (Array.isArray(rawOffset) ? rawOffset[0] : rawOffset?.[0]) ?? rawOffset?.x ?? 0;
                const offsetY = (Array.isArray(rawOffset) ? rawOffset[1] : rawOffset?.[1]) ?? rawOffset?.y ?? 0;
                const nodeX = (node?.pos?.[0] ?? 0) + (node?.size?.[0] ?? 0);
                const nodeY = (node?.pos?.[1] ?? 0) + 30;
                const screenX = (rect?.left ?? 0) + (nodeX + offsetX) * scale;
                const screenY = (rect?.top ?? 0) + (nodeY + offsetY) * scale;

                input.style.position = "fixed";
                input.style.left = `${Math.max(0, Math.min(window.innerWidth - 1, Math.round(screenX)))}px`;
                input.style.top = `${Math.max(0, Math.min(window.innerHeight - 1, Math.round(screenY)))}px`;
                input.style.opacity = "0";
                input.style.width = "1px";
                input.style.height = "1px";
                document.body.appendChild(input);

                const cleanup = () => {
                    try {
                        document.body.removeChild(input);
                    } catch (_) {}
                };

                input.addEventListener("input", () => {
                    if (!node.properties) {
                        node.properties = {};
                    }
                    node.properties.starnodes_custom_titlebarcolor = input.value;
                    applyStarNodeColors(node);
                    if (node.graph) {
                        node.graph.change();
                    }
                    if (app.canvas) {
                        app.canvas.setDirty(true, true);
                    }
                });

                input.addEventListener("change", () => cleanup(), { once: true });
                input.addEventListener("blur", () => cleanup(), { once: true });
                input.click();
            };

            const openFrameColorPicker = (node) => {
                const input = document.createElement("input");
                input.type = "color";
                const current = (node?.properties?.starnodes_frame_color || "#ffffff");
                input.value = (typeof current === "string" && /^#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})$/.test(current)) ? current : "#ffffff";
                const canvasEl = app.canvas?.canvas;
                const rect = canvasEl?.getBoundingClientRect?.();
                const ds = app.canvas?.ds;
                const scale = ds?.scale ?? 1;
                const rawOffset = ds?.offset ?? ds?.origin ?? [0, 0];
                const offsetX = (Array.isArray(rawOffset) ? rawOffset[0] : rawOffset?.[0]) ?? rawOffset?.x ?? 0;
                const offsetY = (Array.isArray(rawOffset) ? rawOffset[1] : rawOffset?.[1]) ?? rawOffset?.y ?? 0;
                const nodeX = (node?.pos?.[0] ?? 0) + (node?.size?.[0] ?? 0);
                const nodeY = (node?.pos?.[1] ?? 0) + 20;
                const screenX = (rect?.left ?? 0) + (nodeX + offsetX) * scale;
                const screenY = (rect?.top ?? 0) + (nodeY + offsetY) * scale;

                input.style.position = "fixed";
                input.style.left = `${Math.max(0, Math.min(window.innerWidth - 1, Math.round(screenX)))}px`;
                input.style.top = `${Math.max(0, Math.min(window.innerHeight - 1, Math.round(screenY)))}px`;
                input.style.opacity = "0";
                input.style.width = "1px";
                input.style.height = "1px";
                document.body.appendChild(input);

                const cleanup = () => {
                    try {
                        document.body.removeChild(input);
                    } catch (_) {}
                };

                input.addEventListener("input", () => {
                    if (!node.properties) {
                        node.properties = {};
                    }
                    node.properties.starnodes_frame_color = input.value;
                    applyStarNodeFrame(node);
                    if (node.graph) {
                        node.graph.change();
                    }
                    if (app.canvas) {
                        app.canvas.setDirty(true, true);
                    }
                });

                input.addEventListener("change", () => cleanup(), { once: true });
                input.addEventListener("blur", () => cleanup(), { once: true });
                input.click();
            };

            const setFrameWidthPrompt = (node) => {
                const current = node?.properties?.starnodes_frame_width;
                const currentText = (typeof current === "number" && isFinite(current)) ? String(current) : "1";
                const raw = window.prompt("Frame width (pixels)", currentText);
                if (raw === null) {
                    return;
                }
                const v = parseFloat(raw);
                if (!isFinite(v) || v < 0) {
                    return;
                }
                if (!node.properties) {
                    node.properties = {};
                }
                node.properties.starnodes_frame_width = v;
                applyStarNodeFrame(node);
                if (node.graph) {
                    node.graph.change();
                }
                if (app.canvas) {
                    app.canvas.setDirty(true, true);
                }
            };

            const THEME_PRESETS = [
                { id: "default", name: "ComfyUI Default (Clear Overrides)", bg: null, title: null, frame: null },
                { id: "starnodes_purple", name: "StarNodes Purple", bg: "#3d124d", title: "#19124d", frame: "#ffffff" },
                { id: "midnight", name: "Midnight", bg: "#0b1220", title: "#0f2a5f", frame: "#4cc9f0" },
                { id: "emerald", name: "Emerald", bg: "#0e2a1f", title: "#145a32", frame: "#34d399" },
                { id: "sunset", name: "Sunset", bg: "#2b0b10", title: "#7c2d12", frame: "#fb923c" },
                { id: "ocean", name: "Ocean", bg: "#062a3a", title: "#075985", frame: "#22d3ee" },
                { id: "rose", name: "Rose", bg: "#2a0b1e", title: "#9f1239", frame: "#fb7185" },
                { id: "lavender", name: "Lavender", bg: "#1c1630", title: "#5b21b6", frame: "#c4b5fd" },
                { id: "amber", name: "Amber", bg: "#1f1407", title: "#92400e", frame: "#fbbf24" },
                { id: "forest", name: "Forest", bg: "#0f1f12", title: "#14532d", frame: "#86efac" },
                { id: "ice", name: "Ice", bg: "#0b1d26", title: "#164e63", frame: "#a5f3fc" },
                { id: "mono", name: "Monochrome", bg: "#1a1a1a", title: "#333333", frame: "#bdbdbd" },
                { id: "coffee", name: "Coffee", bg: "#1a0f0a", title: "#7c3f1d", frame: "#e7c6a5" }
            ];

            const getSelectedNodes = () => {
                const sel = app.canvas?.selected_nodes;
                if (!sel) {
                    return [];
                }
                if (Array.isArray(sel)) {
                    return sel;
                }
                return Object.values(sel);
            };

            const applyThemePresetToNode = (node, presetId) => {
                const preset = THEME_PRESETS.find(t => t.id === presetId) || THEME_PRESETS[1];
                if (!node) {
                    return;
                }
                if (!node.properties) {
                    node.properties = {};
                }

                if (preset.id === "default") {
                    delete node.properties.starnodes_custom_bgcolor;
                    delete node.properties.starnodes_custom_titlebarcolor;
                    delete node.properties.starnodes_frame_color;
                    delete node.properties.starnodes_frame_width;
                } else {
                    node.properties.starnodes_custom_bgcolor = preset.bg;
                    node.properties.starnodes_custom_titlebarcolor = preset.title;
                    node.properties.starnodes_frame_color = preset.frame;
                    node.properties.starnodes_frame_width = 1;
                }

                applyStarNodeColors(node);
                applyStarNodeFrame(node);
            };

            const applyThemePresetToTarget = (targetNode, presetId) => {
                const selected = getSelectedNodes();
                const multiple = selected.length > 1;
                const targets = multiple ? selected : [targetNode];
                for (const n of targets) {
                    applyThemePresetToNode(n, presetId);
                }
                if (targetNode?.graph) {
                    targetNode.graph.change();
                }
                if (app.canvas) {
                    app.canvas.setDirty(true, true);
                }
            };
            
            // Store the original onNodeCreated function
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            const onConfigure = nodeType.prototype.onConfigure;
            const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
            
            // Override the onNodeCreated function
            nodeType.prototype.onNodeCreated = function() {
                // Call the original onNodeCreated if it exists
                if (onNodeCreated) {
                    onNodeCreated.apply(this, arguments);
                }
                
                applyStarNodeColors(this);
                applyStarNodeFrame(this);
                
                // Store the original drawTitleBar function
                const originalDrawTitleBar = this.drawTitleBar;
                
                // Override the drawTitleBar function to use our custom title color
                this.drawTitleBar = function(ctx, title_height) {
                    // Call the original function first
                    originalDrawTitleBar.call(this, ctx, title_height);
                };
                
                console.log(`Applied custom colors to StarNode: ${this.type}`);
            };

            nodeType.prototype.onConfigure = function() {
                if (onConfigure) {
                    onConfigure.apply(this, arguments);
                }
                applyStarNodeColors(this);
                applyStarNodeFrame(this);
            };

            nodeType.prototype.getExtraMenuOptions = function(_, options) {
                if (getExtraMenuOptions) {
                    getExtraMenuOptions.apply(this, arguments);
                }
                options.push(
                    {
                        content: "⭐ Change Color",
                        callback: () => openColorPicker(this)
                    },
                    {
                        content: "⭐ Title Bar",
                        callback: () => openTitleBarColorPicker(this)
                    },
                    {
                        content: "⭐ Reset Color",
                        callback: () => {
                            if (!this.properties) {
                                this.properties = {};
                            }
                            delete this.properties.starnodes_custom_bgcolor;
                            delete this.properties.starnodes_custom_titlebarcolor;
                            applyStarNodeColors(this);
                            if (this.graph) {
                                this.graph.change();
                            }
                            if (app.canvas) {
                                app.canvas.setDirty(true, true);
                            }
                        }
                    },
                    {
                        content: "⭐ Frame Color",
                        callback: () => openFrameColorPicker(this)
                    },
                    {
                        content: "⭐ Frame Width",
                        callback: () => setFrameWidthPrompt(this)
                    },
                    {
                        content: "⭐ Reset Frame",
                        callback: () => {
                            if (!this.properties) {
                                this.properties = {};
                            }
                            delete this.properties.starnodes_frame_color;
                            delete this.properties.starnodes_frame_width;
                            applyStarNodeFrame(this);
                            if (this.graph) {
                                this.graph.change();
                            }
                            if (app.canvas) {
                                app.canvas.setDirty(true, true);
                            }
                        }
                    },
                    {
                        content: "⭐ Theme Presets",
                        has_submenu: true,
                        submenu: {
                            options: THEME_PRESETS.map((t) => ({
                                content: t.name,
                                callback: () => applyThemePresetToTarget(this, t.id)
                            }))
                        }
                    }
                );
            };

            const onDrawForeground = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function(ctx) {
                if (onDrawForeground) {
                    onDrawForeground.apply(this, arguments);
                }

                const frameColor = this._starnodes_frame_color;
                const frameWidthPx = this._starnodes_frame_width;
                if (!frameColor || !frameWidthPx || frameWidthPx <= 0) {
                    return;
                }

                const ds = app.canvas?.ds;
                const scale = ds?.scale ?? 1;
                const lw = frameWidthPx / (scale || 1);

                ctx.save();
                ctx.strokeStyle = frameColor;
                ctx.lineWidth = lw;
                const inset = lw * 0.5;
                ctx.strokeRect(inset, inset, this.size[0] - lw, this.size[1] - lw);
                ctx.restore();
            };
        }
    }
});
