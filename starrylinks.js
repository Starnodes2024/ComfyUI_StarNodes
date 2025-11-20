import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import { $el } from "/scripts/ui.js";
// RopePhysics removed in visual-only mode; do not import RopePhysicsManager

// Constants for link render types - match LiteGraph's order
// (Removed unused legacy constants)

// Visual-only mode: rope physics is fully removed

// Helper function to create a decimal number input that preserves decimal points while typing
function createDecimalInput(settingId, label, min, max, decimals, defaultValue, onChangeCallback) {
    return (value) => {
        // Get the current value from settings or use the passed value or default
        let currentValue = value;
        if (currentValue === undefined || currentValue === null || isNaN(currentValue)) {
            // Try to get from settings
            if (app.extensionManager?.setting?.get) {
                currentValue = app.extensionManager.setting.get(settingId);
            }
            // Still invalid? Use default
            if (currentValue === undefined || currentValue === null || isNaN(currentValue)) {
                currentValue = defaultValue;
            }
        }
        
        const inputElement = $el("input", {
            type: "text",
            value: parseFloat(currentValue).toFixed(decimals),
            style: { width: "100px" },
            oninput: function(e) {
                const val = this.value;
                // Allow typing decimal points and negative for some fields
                if (val.match(/^-?\d*\.?\d*$/)) {
                    const numVal = parseFloat(val);
                    if (!isNaN(numVal) && numVal >= min && numVal <= max) {
                        // Update the setting value
                        app.extensionManager.setting.set(settingId, numVal);
                        if (onChangeCallback) {
                            onChangeCallback(numVal);
                        }
                    }
                }
            },
            onblur: function(e) {
                // Format on blur
                const val = parseFloat(this.value);
                if (!isNaN(val) && val >= min && val <= max) {
                    this.value = val.toFixed(decimals);
                } else {
                    // Revert to current setting value
                    const settingValue = app.extensionManager.setting.get(settingId) ?? defaultValue;
                    this.value = parseFloat(settingValue).toFixed(decimals);
                }
            }
        });
        
        return $el("tr", [
            $el("td", [
                $el("label", { textContent: label })
            ]),
            $el("td", [inputElement])
        ]);
    };
}

app.registerExtension({
    name: "StarryLinks",
    
    // Define settings for the extension (visual-only)
    settings: [
        // Visual embellishments
        {
            id: "StarryLinks.PurpleDotsEnabled",
            name: "Purple Dot Chain",
            type: "boolean",
            defaultValue: true,
            tooltip: "Overlay small purple points along the rope"
        },
        {
            id: "StarryLinks.DotStep",
            name: "Dot Step (points)",
            type: "number",
            defaultValue: 2,
            tooltip: "Place a purple dot every N rope points (1-10)",
            attrs: { min: 1, max: 10, step: 1 }
        },
        {
            id: "StarryLinks.DotSize",
            name: "Dot Size",
            type: createDecimalInput("StarryLinks.DotSize", "Dot Size", 0.5, 6.0, 1, 2.5, null),
            defaultValue: 2.5,
            tooltip: "Radius of the purple dots (0.5-6.0)"
        },
        {
            id: "StarryLinks.StarsEnabled",
            name: "Stars",
            type: "boolean",
            defaultValue: true,
            tooltip: "Show a few golden stars along the rope"
        },
        {
            id: "StarryLinks.StarCount",
            name: "Star Count",
            type: "number",
            defaultValue: 3,
            tooltip: "How many stars to place per rope (0-10)",
            attrs: { min: 0, max: 10, step: 1 }
        },
        {
            id: "StarryLinks.StarSize",
            name: "Star Size",
            type: createDecimalInput("StarryLinks.StarSize", "Star Size", 3, 20, 1, 7, null),
            defaultValue: 7,
            tooltip: "Outer radius of the star shape (3-20)"
        },
        {
            id: "StarryLinks.LineWidth",
            name: "Line Width",
            type: createDecimalInput("StarryLinks.LineWidth", "Line Width", 1, 12, 1, 1, null),
            defaultValue: 1,
            tooltip: "Stroke width of link lines (1-12)"
        },
        {
            id: "StarryLinks.ZResetDefaults",
            name: "Reset All Settings",
            type: () => {
                return $el("tr", [
                    $el("td", { colspan: 2, style: { textAlign: "center", paddingTop: "10px" } }, [
                        $el("button", {
                            textContent: "Reset All StarryLinks Settings to Defaults",
                            style: {
                                padding: "8px 16px",
                                backgroundColor: "#d73502",
                                color: "white",
                                border: "none",
                                borderRadius: "4px",
                                cursor: "pointer",
                                fontSize: "14px",
                                fontWeight: "bold"
                            },
                            onmouseover: function() {
                                this.style.backgroundColor = "#b82d02";
                            },
                            onmouseout: function() {
                                this.style.backgroundColor = "#d73502";
                            },
                            onclick: function() {
                                if (confirm("Are you sure you want to reset all StarryLinks settings to their default values?")) {
                                    // Reset all StarryLinks settings
                                    const defaults = {
                                        // Visual-only defaults
                                        'StarryLinks.PurpleDotsEnabled': true,
                                        'StarryLinks.DotStep': 2,
                                        'StarryLinks.DotSize': 2.5,
                                        'StarryLinks.StarsEnabled': true,
                                        'StarryLinks.StarCount': 3,
                                        'StarryLinks.StarSize': 7,
                                                        'StarryLinks.LineWidth': 1
                                    };
                                    
                                    // Apply all defaults
                                    for (const [key, value] of Object.entries(defaults)) {
                                        app.extensionManager.setting.set(key, value);
                                    }
                                    
                                    // No physics updates in visual-only mode
                                    
                                    alert("All StarryLinks settings have been reset to defaults. Please refresh the settings dialog to see the updated values.");
                                }
                            }
                        })
                    ])
                ]);
            },
            defaultValue: null,
            tooltip: "Reset all StarryLinks settings to their default values"
        }
    ],
    
    async init() {
        // Add StarryLink to LiteGraph modes early
        if (window.LiteGraph) {
            if (!LiteGraph.LINK_RENDER_MODES.includes("StarryLink")) {
                LiteGraph.LINK_RENDER_MODES.push("StarryLink");
            }
            // Store the index of our mode
            LiteGraph.STARRYLINK_LINK = LiteGraph.LINK_RENDER_MODES.indexOf("StarryLink");
        }
    },

    async setup() {
        // Wait for LiteGraph to be available
        if (!window.LiteGraph || !window.LGraphCanvas) {
            return;
        }

        // Ensure StarryLink is in the link render modes and index is set
        if (!LiteGraph.LINK_RENDER_MODES.includes("StarryLink")) {
            LiteGraph.LINK_RENDER_MODES.push("StarryLink");
        }
        LiteGraph.STARRYLINK_LINK = LiteGraph.LINK_RENDER_MODES.indexOf("StarryLink");

        // Initialize default values for settings if they haven't been set
        const initializeDefaultValue = (id, defaultValue) => {
            if (app.extensionManager?.setting?.get && app.extensionManager?.setting?.set) {
                const currentValue = app.extensionManager.setting.get(id);
                // If the value is exactly 1 (the ComfyUI default for uninitialized numbers), set our default
                if (currentValue === 1 || currentValue === undefined || currentValue === null) {
                    app.extensionManager.setting.set(id, defaultValue);
                }
            }
        };

        // Set defaults after a short delay to ensure settings are loaded
        setTimeout(() => {
            // Visual-only defaults
            initializeDefaultValue('StarryLinks.PurpleDotsEnabled', true);
            initializeDefaultValue('StarryLinks.DotStep', 2);
            initializeDefaultValue('StarryLinks.DotSize', 2.5);
            initializeDefaultValue('StarryLinks.StarsEnabled', true);
            initializeDefaultValue('StarryLinks.StarCount', 3);
            initializeDefaultValue('StarryLinks.StarSize', 7);
            initializeDefaultValue('StarryLinks.LineColor', '#AAAAAA');
            initializeDefaultValue('StarryLinks.LineWidth', 1);
        }, 100);
        
        // Set up execution event listeners
        if (api) {
            const safeRedraw = () => {
                const c = app.canvas;
                if (!c) return;
                const menuOpen = !!(c.current_menu || c.canvas_menu);
                if (!menuOpen) c.setDirty(false, true);
            };

            api.addEventListener('executing', () => safeRedraw());
            api.addEventListener('progress', () => safeRedraw());
            api.addEventListener('executed', () => safeRedraw());
        }

        // Add StarryLink option to the settings
        const modifyLinkRenderSetting = () => {
            // Check if extensionManager is available (it's the workspace store)
            if (!app.extensionManager || !app.extensionManager.setting || !app.extensionManager.setting.settings) {
                setTimeout(modifyLinkRenderSetting, 100);
                return;
            }

            // Access the settings through extensionManager
            const settings = app.extensionManager.setting.settings;
            const linkRenderSetting = settings['Comfy.LinkRenderMode'];
            
            if (linkRenderSetting && linkRenderSetting.options) {
                // Check if StarryLink is already in the options
                const hasStarryLink = linkRenderSetting.options.some(opt => 
                    (typeof opt === 'object' ? opt.value : opt) === LiteGraph.STARRYLINK_LINK
                );

                if (!hasStarryLink) {
                    // Add StarryLink option
                    linkRenderSetting.options.push({
                        value: LiteGraph.STARRYLINK_LINK,
                        text: "StarryLink"
                    });
                    
                    // Clear the timeout to prevent further retries
                    if (window.starryLinkRetryTimeout) {
                        clearTimeout(window.starryLinkRetryTimeout);
                        window.starryLinkRetryTimeout = null;
                    }
                }
            } else {
                window.starryLinkRetryTimeout = setTimeout(modifyLinkRenderSetting, 100);
            }
        };

        // Start the modification process with a max retry count
        let retryCount = 0;
        const maxRetries = 50; // 5 seconds max
        
        const tryModifyWithRetryLimit = () => {
            if (retryCount++ < maxRetries) {
                modifyLinkRenderSetting();
            }
        };
        
        tryModifyWithRetryLimit();

        // Store original method for fallback
        const originalRenderLink = LGraphCanvas.prototype.renderLink;

        // Helper function to find a bezier curve point - optimized for performance
        function findPointOnCurve(out, start, end, cp1, cp2, t) {
            const t2 = t * t;
            const t3 = t2 * t;
            const mt = 1 - t;
            const mt2 = mt * mt;
            const mt3 = mt2 * mt;
            const mt2t3 = 3 * mt2 * t;
            const mtt23 = 3 * mt * t2;

            out[0] = mt3 * start[0] + mt2t3 * cp1[0] + mtt23 * cp2[0] + t3 * end[0];
            out[1] = mt3 * start[1] + mt2t3 * cp1[1] + mtt23 * cp2[1] + t3 * end[1];
        }

        // Override renderLink for visual-only StarryLink mode
        LGraphCanvas.prototype.renderLink = function(
            ctx,
            a,
            b,
            link,
            skip_border,
            flow,
            color,
            start_dir,
            end_dir,
            num_sublines
        ) {
            // Visual-only mode: draw original ComfyUI line with overlays
            if (this.links_render_mode === LiteGraph.STARRYLINK_LINK) {
                // First, let original renderLink draw the line with proper colors
                const result = originalRenderLink.apply(this, arguments);
                
                // Then draw our custom bezier curve with overlays on top
                try {
                    const dx = b[0] - a[0];
                    const dy = b[1] - a[1];
                    const dist = Math.hypot(dx, dy);
                    const K = Math.min(80, dist * 0.5);
                    const cp1 = [a[0] + (start_dir === LiteGraph.RIGHT ? K : start_dir === LiteGraph.LEFT ? -K : 0), a[1] + (start_dir === LiteGraph.DOWN ? K : start_dir === LiteGraph.UP ? -K : 0)];
                    const cp2 = [b[0] + (end_dir === LiteGraph.RIGHT ? K : end_dir === LiteGraph.LEFT ? -K : 0), b[1] + (end_dir === LiteGraph.DOWN ? K : end_dir === LiteGraph.UP ? -K : 0)];
                    
                    // Use ComfyUI system link colors (no custom override)
                    const lineWidthSetting = parseFloat(app.extensionManager?.setting?.get?.('StarryLinks.LineWidth') || 1);
                    
                    ctx.save();
                    ctx.strokeStyle = color || "#AAAAAA";
                    ctx.lineWidth = Math.max(1, Math.min(12, lineWidthSetting));
                    ctx.beginPath();
                    ctx.moveTo(a[0], a[1]);
                    ctx.bezierCurveTo(cp1[0], cp1[1], cp2[0], cp2[1], b[0], b[1]);
                    ctx.stroke();
                    ctx.restore();
                    
                    // Draw visual overlays on top
                    this._renderStarsOnCurve(ctx, a, b, link, color, start_dir, end_dir);
                } catch (_) {}
                
                return result;
            }

            // Fall back to original rendering for non-StarryLink modes
            return originalRenderLink.apply(this, arguments);
        };
        
        // Helper to render stars/dots along the bezier between a and b using start/dir dirs
        LGraphCanvas.prototype._renderStarsOnCurve = function(ctx, a, b, link, color, start_dir, end_dir) {
            // Derive control points similar to LiteGraph curves
            const dx = b[0] - a[0];
            const dy = b[1] - a[1];
            const dist = Math.hypot(dx, dy);
            const K = Math.min(80, dist * 0.5);

            function cpFrom(p, dir, k) {
                switch (dir) {
                    case LiteGraph.LEFT:  return [p[0] - k, p[1]];
                    case LiteGraph.RIGHT: return [p[0] + k, p[1]];
                    case LiteGraph.UP:    return [p[0], p[1] - k];
                    case LiteGraph.DOWN:  return [p[0], p[1] + k];
                    default: return [p[0] + dx * 0.5, p[1] + dy * 0.5];
                }
            }

            const cp1 = cpFrom(a, start_dir, K);
            const cp2 = cpFrom(b, end_dir, K);

            // Sample the curve
            const samples = Math.max(12, Math.min(80, Math.round(dist / 10)));
            const points = new Array(samples + 1);
            for (let i = 0; i <= samples; i++) {
                const t = i / samples;
                const out = [0,0];
                findPointOnCurve(out, a, b, cp1, cp2, t);
                points[i] = out;
            }

            // Settings (visual-only; static stars)
            const getSetting = (id, def) => app.extensionManager?.setting?.get?.(id) ?? def;
            const dotsEnabled = !!getSetting('StarryLinks.PurpleDotsEnabled', true);
            const dotStep = Math.max(1, parseInt(getSetting('StarryLinks.DotStep', 2) || 2));
            const dotSize = parseFloat(getSetting('StarryLinks.DotSize', 2.5) || 2.5);
            const starsEnabled = !!getSetting('StarryLinks.StarsEnabled', true);
            const starCount = Math.max(0, parseInt(getSetting('StarryLinks.StarCount', 3) || 3));
            const starSize = parseFloat(getSetting('StarryLinks.StarSize', 7) || 7);

            // Draw dots
            if (dotsEnabled && points.length) {
                ctx.save();
                ctx.fillStyle = '#a76cff'; // Always use purple for dots
                for (let i = 0; i < points.length; i += dotStep) {
                    const p = points[i];
                    ctx.beginPath();
                    ctx.arc(p[0], p[1], dotSize, 0, Math.PI * 2);
                    ctx.fill();
                }
                ctx.restore();
            }

            // Helpers for stars
            function drawStar(ctx2, x, y, spikes, outerR, innerR) {
                let rot = Math.PI / 2 * 3;
                let cx = x; let cy = y;
                ctx2.beginPath();
                ctx2.moveTo(cx, cy - outerR);
                for (let i = 0; i < spikes; i++) {
                    cx = x + Math.cos(rot) * outerR; cy = y + Math.sin(rot) * outerR; ctx2.lineTo(cx, cy); rot += Math.PI / 5;
                    cx = x + Math.cos(rot) * innerR; cy = y + Math.sin(rot) * innerR; ctx2.lineTo(cx, cy); rot += Math.PI / 5;
                }
                ctx2.lineTo(x, y - outerR);
                ctx2.closePath();
            }

            function hash32(x) { x |= 0; x = x + 0x7ed55d16 + (x << 12) | 0; x = x ^ 0xc761c23c ^ (x >>> 19);
                x = x + 0x165667b1 + (x << 5) | 0; x = x + 0xd3a2646c ^ (x << 9);
                x = x + 0xfd7046c5 + (x << 3) | 0; x = x ^ 0xb55a4f09 ^ (x >>> 16); return x >>> 0; }

            if (starsEnabled && starCount > 0 && points.length > 2) {
                const baseHue = 48;
                // Use broad-compatible CSS hsl syntax (commas)
                const baseColor = `hsl(${baseHue}, 90%, 55%)`;
                const glowColor = `hsl(${baseHue}, 100%, 70%)`;
                const seed = hash32(((link?.id) ?? 0) ^ (a[0] | 0) ^ (b[1] | 0));

                for (let k = 0; k < starCount; k++) {
                    const idx = Math.floor(((k + 1) / (starCount + 1)) * (points.length - 1));
                    const p = points[idx];
                    if (!p) continue;
                    // Static stars: fixed alpha
                    const alpha = 0.9;

                    ctx.save();
                    ctx.translate(p[0], p[1]);
                    ctx.globalAlpha = alpha;
                    ctx.fillStyle = baseColor;
                    ctx.shadowColor = glowColor;
                    ctx.shadowBlur = 12;
                    ctx.strokeStyle = '#ffd24a';
                    ctx.lineWidth = 1;
                    drawStar(ctx, 0, 0, 5, starSize, starSize * 0.5);
                    ctx.fill();
                    ctx.stroke();
                    ctx.restore();
                }
            }
        };



        // Override getCanvasMenuOptions to add StarryLink option to context menu
        const originalGetCanvasMenuOptions = LGraphCanvas.prototype.getCanvasMenuOptions;
        LGraphCanvas.prototype.getCanvasMenuOptions = function() {
            try {
                const options = originalGetCanvasMenuOptions ? originalGetCanvasMenuOptions.apply(this, arguments) : [];
                
                if (!Array.isArray(options)) {
                    return options;
                }
                
                // Find the Links submenu
                const linksMenu = options.find(opt => opt && opt.content === "Links");
                if (linksMenu && Array.isArray(linksMenu.submenu)) {
                    // Find the render mode submenu
                    const renderModeMenu = linksMenu.submenu.find(opt => opt && opt.content === "Render mode");
                    if (renderModeMenu && Array.isArray(renderModeMenu.submenu)) {
                        // Add StarryLink option if not already present
                        const hasStarryLink = renderModeMenu.submenu.some(opt => opt && opt.content === "StarryLink");
                        if (!hasStarryLink) {
                            renderModeMenu.submenu.push({
                                content: "StarryLink",
                                callback: () => {
                                    if (LiteGraph.STARRYLINK_LINK !== undefined) {
                                        this.links_render_mode = LiteGraph.STARRYLINK_LINK;
                                        this.setDirty(false, true);
                                    }
                                }
                            });
                        }
                    }
                }
                
                return options;
            } catch (error) {
                console.warn("StarryLinks: Error in context menu override", error);
                return originalGetCanvasMenuOptions ? originalGetCanvasMenuOptions.apply(this, arguments) : [];
            }
        };
    }
});
