import { app } from "../../../../scripts/app.js";

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
        }
    ],
    async setup() {
        // This runs once when the extension is loaded
        console.log("StarNodes appearance extension setup");
    },
    
    async beforeRegisterNodeDef(nodeType, nodeData) {
        // Check if we should apply to all nodes
        const applyToAll = app.extensionManager?.setting?.get?.("StarNodes.ApplyToAllNodes");

        // Check if this is a StarNode by looking at the category or if global override is enabled
        if (applyToAll || (nodeData.category && nodeData.category.startsWith("⭐"))) {
            if (nodeData.category && nodeData.category.startsWith("⭐")) {
                 console.log(`Found StarNode: ${nodeData.name}, applying custom colors`);
            } else if (applyToAll) {
                 // Less verbose logging for all nodes to avoid console spam
            }
            
            // Define our colors
            const backgroundColor = "#3d124d";  // Purple background
            const titleColor = "#19124d";       // Dark blue title

            const applyStarNodeColors = (node) => {
                if (!node) {
                    return;
                }
                if (!node.properties) {
                    node.properties = {};
                }
                const bg = node.properties.starnodes_custom_bgcolor || backgroundColor;
                const titlebar = node.properties.starnodes_custom_titlebarcolor || titleColor;
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
                const frameColor = node.properties.starnodes_frame_color || null;
                const frameWidth = node.properties.starnodes_frame_width;
                node._starnodes_frame_color = frameColor;
                node._starnodes_frame_width = (typeof frameWidth === "number" && isFinite(frameWidth)) ? frameWidth : 0;
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
                const currentText = (typeof current === "number" && isFinite(current)) ? String(current) : "2";
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
