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

// Helper to create a simple SELECT from an array of {label, value}
function createSimpleSelect(settingId, label, options, defaultValue, onChangeCallback) {
    return () => {
        let currentValue = defaultValue;
        try { currentValue = app.extensionManager?.setting?.get?.(settingId) ?? defaultValue; } catch (_) {}
        const select = $el('select', {
            style: { minWidth: '160px' },
            onchange: function() {
                const val = this.value;
                app.extensionManager?.setting?.set?.(settingId, val);
                if (onChangeCallback) onChangeCallback(val);
            }
        });
        options.forEach(({ label: lab, value }) => {
            const opt = document.createElement('option');
            opt.value = value;
            opt.textContent = lab;
            if (String(value) === String(currentValue)) opt.selected = true;
            select.appendChild(opt);
        });
        return $el('tr', [
            $el('td', [ $el('label', { textContent: label }) ]),
            $el('td', [ select ])
        ]);
    };
}

// Helper to create a color SELECT with preset palette
function createColorSelect(settingId, label, palette, defaultValue, onChangeCallback) {
    return () => {
        let currentValue = defaultValue;
        try { currentValue = app.extensionManager?.setting?.get?.(settingId) ?? defaultValue; } catch (_) {}
        const select = $el('select', {
            style: { minWidth: '160px' },
            onchange: function() {
                const val = this.value;
                app.extensionManager?.setting?.set?.(settingId, val);
                if (onChangeCallback) onChangeCallback(val);
            }
        });
        // Build options
        palette.forEach(({ name, value }) => {
            const opt = document.createElement('option');
            opt.value = value;
            opt.textContent = `${name} (${value})`;
            if (value.toLowerCase() === String(currentValue).toLowerCase()) opt.selected = true;
            select.appendChild(opt);
        });
        return $el('tr', [
            $el('td', [ $el('label', { textContent: label }) ]),
            $el('td', [ select ])
        ]);
    };
}

app.registerExtension({
    name: "StarryLinks",
    
    // Define settings for the extension (visual-only)
    settings: [
        // Visual embellishments
        {
            id: "StarryLinks.DotsEnabled",
            name: "Dots Enabled",
            type: "boolean",
            defaultValue: true,
            tooltip: "Overlay small points along the rope"
        },
        {
            id: "StarryLinks.DotColor",
            name: "Dot Color",
            type: createColorSelect(
                "StarryLinks.DotColor",
                "Dot Color",
                [
                    { name: 'White', value: '#ffffff' },
                    { name: 'Yellow', value: '#ffd24a' },
                    { name: 'Gold', value: '#ffcc00' },
                    { name: 'Orange', value: '#ffa500' },
                    { name: 'Red', value: '#ff4d4d' },
                    { name: 'Pink', value: '#ff77aa' },
                    { name: 'Purple', value: '#a76cff' },
                    { name: 'Blue', value: '#66aaff' },
                    { name: 'Cyan', value: '#33cccc' },
                    { name: 'Teal', value: '#2dd4bf' },
                    { name: 'Green', value: '#66cc66' }
                ],
                "#ffffff",
                null
            ),
            defaultValue: "#ffffff",
            tooltip: "Color of the dots"
        },
        {
            id: "StarryLinks.DotStep",
            name: "Dot Step (points)",
            type: "number",
            defaultValue: 10,
            tooltip: "Place a dot every N rope samples (1-20)",
            attrs: { min: 1, max: 20, step: 1 }
        },
        {
            id: "StarryLinks.DotSize",
            name: "Dot Size",
            type: createDecimalInput("StarryLinks.DotSize", "Dot Size", 0.5, 6.0, 1, 2.5, null),
            defaultValue: 1.0,
            tooltip: "Radius of the dots (0.5-6.0)"
        },
        {
            id: "StarryLinks.DotBlinkEnabled",
            name: "Dots Blinking",
            type: "boolean",
            defaultValue: true,
            tooltip: "Animate a subtle blinking effect on dots"
        },
        {
            id: "StarryLinks.DotBlinkSpeed",
            name: "Dot Blink Speed",
            type: createDecimalInput("StarryLinks.DotBlinkSpeed", "Dot Blink Speed", 0.1, 5.0, 2, 1.0, null),
            defaultValue: 1.0,
            tooltip: "How fast dots blink (0.1-5.0)"
        },
        {
            id: "StarryLinks.DotBlinkStrength",
            name: "Dot Blink Strength",
            type: createDecimalInput("StarryLinks.DotBlinkStrength", "Dot Blink Strength", 0.0, 1.0, 2, 0.6, null),
            defaultValue: 1.0,
            tooltip: "How strongly dots vary brightness (0.0-1.0)"
        },
        {
            id: "StarryLinks.TwinkleEnabled",
            name: "Twinkling Stars",
            type: "boolean",
            defaultValue: true,
            tooltip: "Animate a subtle twinkling effect on stars"
        },
        {
            id: "StarryLinks.TwinkleSpeed",
            name: "Twinkle Speed",
            type: createDecimalInput("StarryLinks.TwinkleSpeed", "Twinkle Speed", 0.1, 5.0, 2, 1.0, null),
            defaultValue: 3.0,
            tooltip: "How fast stars twinkle (0.1-5.0)"
        },
        {
            id: "StarryLinks.TwinkleStrength",
            name: "Twinkle Strength",
            type: createDecimalInput("StarryLinks.TwinkleStrength", "Twinkle Strength", 0.0, 1.0, 2, 0.6, null),
            defaultValue: 1.0,
            tooltip: "How strongly stars vary brightness (0.0-1.0)"
        },
        {
            id: "StarryLinks.StarsEnabled",
            name: "Stars",
            type: "boolean",
            defaultValue: true,
            tooltip: "Show a few golden stars along the rope"
        },
        {
            id: "StarryLinks.StarColor",
            name: "Star Color",
            type: createColorSelect(
                "StarryLinks.StarColor",
                "Star Color",
                [
                    { name: 'White', value: '#ffffff' },
                    { name: 'Yellow', value: '#ffd24a' },
                    { name: 'Gold', value: '#ffcc00' },
                    { name: 'Orange', value: '#ffa500' },
                    { name: 'Red', value: '#ff4d4d' },
                    { name: 'Pink', value: '#ff77aa' },
                    { name: 'Purple', value: '#a76cff' },
                    { name: 'Blue', value: '#66aaff' },
                    { name: 'Cyan', value: '#33cccc' },
                    { name: 'Teal', value: '#2dd4bf' },
                    { name: 'Green', value: '#66cc66' }
                ],
                "#ffd24a",
                null
            ),
            defaultValue: "#ffd24a",
            tooltip: "Fill color of the stars"
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
            defaultValue: 10.0,
            tooltip: "Outer radius of the star shape (3-20)"
        },
        {
            id: "StarryLinks.LineWidth",
            name: "Line Width",
            type: createDecimalInput("StarryLinks.LineWidth", "Line Width", 1, 12, 1, 1, null),
            defaultValue: 1,
            tooltip: "Stroke width of link lines (1-12)"
        },
        // Cute Otters climbing the ropes
        {
            id: "StarryLinks.OttersEnabled",
            name: "Otters",
            type: "boolean",
            defaultValue: false,
            tooltip: "Enable cute little purple otters climbing along the ropes"
        },
        {
            id: "StarryLinks.OtterCount",
            name: "Otter Count",
            type: "number",
            defaultValue: 1,
            tooltip: "How many otters per rope (0-5)",
            attrs: { min: 0, max: 5, step: 1 }
        },
        {
            id: "StarryLinks.OtterSpeed",
            name: "Otter Speed",
            type: createDecimalInput("StarryLinks.OtterSpeed", "Otter Speed", 0.2, 3.0, 2, 1.2, null),
            defaultValue: 1.2,
            tooltip: "Otter climbing speed (in curve-length units per second)"
        },
        {
            id: "StarryLinks.OtterScale",
            name: "Otter Scale",
            type: createDecimalInput("StarryLinks.OtterScale", "Otter Scale", 0.5, 2.0, 2, 1.0, null),
            defaultValue: 1.0,
            tooltip: "Size multiplier for otters"
        },
        {
            id: "StarryLinks.OtterDirection",
            name: "Otter Direction",
            type: createSimpleSelect(
                "StarryLinks.OtterDirection",
                "Otter Direction",
                [
                    { label: 'Up', value: 'up' },
                    { label: 'Down', value: 'down' },
                    { label: 'Both (ping-pong)', value: 'both' }
                ],
                'up',
                null
            ),
            defaultValue: 'up',
            tooltip: "Direction otters climb along the rope"
        },
        {
            id: "StarryLinks.OtterCutenessFX",
            name: "Otter Cuteness FX",
            type: "boolean",
            defaultValue: true,
            tooltip: "Occasional hearts/sparkles and subtle bobbing"
        },
        // PNG sprite otters (optional)
        {
            id: "StarryLinks.OtterSpritesEnabled",
            name: "Otter Sprites (PNG)",
            type: "boolean",
            defaultValue: false,
            tooltip: "Use PNG sprite otters from web/img/otters instead of vector"
        },
        {
            id: "StarryLinks.OtterSpriteFrames",
            name: "Otter Sprite Frames",
            type: "number",
            defaultValue: 9,
            tooltip: "Number of frames (1=otter.png, >1=otter_0..N-1)",
            attrs: { min: 1, max: 24, step: 1 }
        },
        {
            id: "StarryLinks.OtterSpriteFPS",
            name: "Otter Sprite FPS",
            type: "number",
            defaultValue: 6,
            tooltip: "Animation speed in frames per second (0 disables frame animation)",
            attrs: { min: 0, max: 24, step: 1 }
        },
        {
            id: "StarryLinks.OtterSpritePathBase",
            name: "Otter Sprite Path Base",
            type: "string",
            defaultValue: "",
            tooltip: "Optional: full URL base to the folder containing sprite PNGs. Example: /extensions/ComfyUI_StarBetaNodes/web/img/otters"
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
                                        'StarryLinks.DotsEnabled': true,
                                        'StarryLinks.DotColor': '#ffffff',
                                        'StarryLinks.DotStep': 10,
                                        'StarryLinks.DotSize': 1.0,
                                        'StarryLinks.DotBlinkEnabled': true,
                                        'StarryLinks.DotBlinkSpeed': 1.0,
                                        'StarryLinks.DotBlinkStrength': 1.0,
                                        'StarryLinks.TwinkleEnabled': true,
                                        'StarryLinks.TwinkleSpeed': 3.0,
                                        'StarryLinks.TwinkleStrength': 1.0,
                                        'StarryLinks.StarsEnabled': true,
                                        'StarryLinks.StarColor': '#ffd24a',
                                        'StarryLinks.StarCount': 3,
                                        'StarryLinks.StarSize': 10.0,
                                        'StarryLinks.LineWidth': 1.0,
                                        // Otters
                                        'StarryLinks.OttersEnabled': false,
                                        'StarryLinks.OtterCount': 1,
                                        'StarryLinks.OtterSpeed': 1.2,
                                        'StarryLinks.OtterScale': 1.0,
                                        'StarryLinks.OtterDirection': 'up',
                                        'StarryLinks.OtterCutenessFX': true,
                                        // Otter sprites
                                        'StarryLinks.OtterSpritesEnabled': false,
                                        'StarryLinks.OtterSpriteFrames': 9,
                                        'StarryLinks.OtterSpriteFPS': 6,
                                        'StarryLinks.OtterSpritePathBase': ''
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
            try {
                if (app.extensionManager?.setting?.get && app.extensionManager?.setting?.set) {
                    const currentValue = app.extensionManager.setting.get(id);
                    if (typeof currentValue === 'undefined' || currentValue === null) {
                        app.extensionManager.setting.set(id, defaultValue);
                    }
                }
            } catch (_) {}
        };

        // Draw a simple, cute purple otter using vector shapes
        function drawCuteOtter(ctx, p, p2, scale = 1.0, fx = true) {
            const angle = Math.atan2(p2[1] - p[1], p2[0] - p[0]);
            const base = '#a76cff'; // purple
            const dark = '#6a3fb8';
            const eye = '#2b2140';
            const belly = '#d9c6ff';
            const size = 10 * scale; // base size matches star size scale
            const now = (typeof performance !== 'undefined' ? performance.now() : Date.now());
            const bob = fx ? Math.sin(now * 0.005 + (p[0] + p[1]) * 0.01) * 1.5 : 0;

            ctx.save();
            ctx.translate(p[0], p[1] + bob);
            ctx.rotate(angle);
            ctx.scale(1, 1);

            // tiny shadow
            ctx.save();
            ctx.translate(0, size * 0.9);
            ctx.fillStyle = 'rgba(0,0,0,0.12)';
            ctx.beginPath();
            ctx.ellipse(0, 0, size * 0.9, size * 0.35, 0, 0, Math.PI * 2);
            ctx.fill();
            ctx.restore();

            // body (rounded capsule)
            ctx.fillStyle = base;
            ctx.strokeStyle = dark;
            ctx.lineWidth = 1;
            const w = size * 1.1;
            const h = size * 1.8;
            const r = size * 0.55;
            ctx.beginPath();
            ctx.moveTo(-w/2 + r, -h/2);
            ctx.lineTo(w/2 - r, -h/2);
            ctx.quadraticCurveTo(w/2, -h/2, w/2, -h/2 + r);
            ctx.lineTo(w/2, h/2 - r);
            ctx.quadraticCurveTo(w/2, h/2, w/2 - r, h/2);
            ctx.lineTo(-w/2 + r, h/2);
            ctx.quadraticCurveTo(-w/2, h/2, -w/2, h/2 - r);
            ctx.lineTo(-w/2, -h/2 + r);
            ctx.quadraticCurveTo(-w/2, -h/2, -w/2 + r, -h/2);
            ctx.closePath();
            ctx.fill();
            ctx.stroke();

            // belly patch
            ctx.fillStyle = belly;
            ctx.beginPath();
            ctx.ellipse(0, size * 0.1, w * 0.45, h * 0.35, 0, 0, Math.PI * 2);
            ctx.fill();

            // head
            const hr = size * 0.7;
            ctx.fillStyle = base;
            ctx.strokeStyle = dark;
            ctx.beginPath();
            ctx.arc(0, -h * 0.55, hr, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();

            // ears
            ctx.beginPath();
            ctx.arc(-hr * 0.6, -h * 0.55 - hr * 0.4, hr * 0.25, 0, Math.PI * 2);
            ctx.arc(hr * 0.6, -h * 0.55 - hr * 0.4, hr * 0.25, 0, Math.PI * 2);
            ctx.fill();

            // eyes
            ctx.fillStyle = eye;
            const blink = (Math.floor(now / 1200 + (p[0] + p[1]) * 0.001) % 6) === 0; // occasional blink
            if (!blink) {
                ctx.beginPath();
                ctx.arc(-hr * 0.3, -h * 0.55, hr * 0.12, 0, Math.PI * 2);
                ctx.arc(hr * 0.3, -h * 0.55, hr * 0.12, 0, Math.PI * 2);
                ctx.fill();
            } else {
                ctx.lineWidth = 1.2;
                ctx.beginPath();
                ctx.moveTo(-hr * 0.42, -h * 0.55);
                ctx.lineTo(-hr * 0.18, -h * 0.55);
                ctx.moveTo(hr * 0.18, -h * 0.55);
                ctx.lineTo(hr * 0.42, -h * 0.55);
                ctx.stroke();
            }

            // nose/mouth
            ctx.strokeStyle = eye;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(0, -h * 0.48);
            ctx.lineTo(0, -h * 0.44);
            ctx.moveTo(0, -h * 0.44);
            ctx.quadraticCurveTo(-hr * 0.15, -h * 0.40, -hr * 0.25, -h * 0.40);
            ctx.moveTo(0, -h * 0.44);
            ctx.quadraticCurveTo(hr * 0.15, -h * 0.40, hr * 0.25, -h * 0.40);
            ctx.stroke();

            // paws (simple ovals)
            ctx.fillStyle = dark;
            ctx.beginPath();
            ctx.ellipse(-w*0.25, 0, size*0.2, size*0.12, 0.2, 0, Math.PI*2);
            ctx.ellipse(w*0.25, 0, size*0.2, size*0.12, -0.2, 0, Math.PI*2);
            ctx.fill();

            // optional heart sparkles
            if (fx) {
                const heartY = -h * 0.9 + Math.sin(now * 0.006 + (p[0] - p[1]) * 0.01) * 3;
                const heartAlpha = 0.5 + 0.5 * Math.sin(now * 0.006);
                ctx.globalAlpha = heartAlpha;
                ctx.fillStyle = '#ff77aa';
                ctx.beginPath();
                const hs = size * 0.35;
                // small heart shape
                ctx.moveTo(0, heartY);
                ctx.bezierCurveTo(-hs*0.5, heartY - hs*0.8, -hs, heartY - hs*0.1, 0, heartY + hs*0.6);
                ctx.bezierCurveTo(hs, heartY - hs*0.1, hs*0.5, heartY - hs*0.8, 0, heartY);
                ctx.fill();
                ctx.globalAlpha = 1.0;
            }

            ctx.restore();
        }

        // Set defaults after a short delay to ensure settings are loaded
        setTimeout(() => {
            // Visual-only defaults
            initializeDefaultValue('StarryLinks.DotsEnabled', true);
            initializeDefaultValue('StarryLinks.DotColor', '#ffffff');
            initializeDefaultValue('StarryLinks.DotStep', 10);
            initializeDefaultValue('StarryLinks.DotSize', 1.0);
            initializeDefaultValue('StarryLinks.DotBlinkEnabled', true);
            initializeDefaultValue('StarryLinks.DotBlinkSpeed', 1.0);
            initializeDefaultValue('StarryLinks.DotBlinkStrength', 1.0);
            initializeDefaultValue('StarryLinks.TwinkleEnabled', true);
            initializeDefaultValue('StarryLinks.TwinkleSpeed', 3.0);
            initializeDefaultValue('StarryLinks.TwinkleStrength', 1.0);
            initializeDefaultValue('StarryLinks.StarsEnabled', true);
            initializeDefaultValue('StarryLinks.StarColor', '#ffd24a');
            initializeDefaultValue('StarryLinks.StarCount', 3);
            initializeDefaultValue('StarryLinks.StarSize', 10.0);
            initializeDefaultValue('StarryLinks.LineColor', '#AAAAAA');
            initializeDefaultValue('StarryLinks.LineWidth', 1.0);
            // Otters
            initializeDefaultValue('StarryLinks.OttersEnabled', false);
            initializeDefaultValue('StarryLinks.OtterCount', 1);
            initializeDefaultValue('StarryLinks.OtterSpeed', 1.2);
            initializeDefaultValue('StarryLinks.OtterScale', 1.0);
            initializeDefaultValue('StarryLinks.OtterDirection', 'up');
            initializeDefaultValue('StarryLinks.OtterCutenessFX', true);
            // Sprites
            initializeDefaultValue('StarryLinks.OtterSpritesEnabled', false);
            initializeDefaultValue('StarryLinks.OtterSpriteFrames', 9);
            initializeDefaultValue('StarryLinks.OtterSpriteFPS', 6);
            initializeDefaultValue('StarryLinks.OtterSpritePathBase', '');
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
                // Force base draw as SPLINE using original renderer to keep system colors/styles
                const savedMode = this.links_render_mode;
                let splineModeIndex = 0;
                try {
                    const modes = LiteGraph.LINK_RENDER_MODES || [];
                    const found = modes.findIndex(m => typeof m === 'string' && /spline/i.test(m));
                    if (found >= 0) splineModeIndex = found; else if (modes.length > 1) splineModeIndex = 1;
                } catch (_) {}
                this.links_render_mode = splineModeIndex;
                const result = originalRenderLink.apply(this, arguments);
                this.links_render_mode = savedMode;
                try {
                    this._renderStarsOnCurve(ctx, a, b, link, color, start_dir, end_dir);
                } catch (_) {}
                return result;
            }

            // Fall back to original rendering for non-StarryLink modes
            return originalRenderLink.apply(this, arguments);
        };
        
        // Resolve base URL for this extension to load assets (robust to query strings)
        let __starryBaseUrl = null;
        let __starryScriptSrc = null;
        function getStarryBaseUrl() {
            if (__starryBaseUrl) return __starryBaseUrl;
            try {
                const scripts = document.getElementsByTagName('script');
                for (let i = 0; i < scripts.length; i++) {
                    const src = scripts[i].src || '';
                    if (src.includes('/web/js/starrylinks.js')) {
                        __starryScriptSrc = src;
                        // strip trailing /web/js/starrylinks.js with optional query string
                        const m = src.match(/^(.*)\/web\/js\/starrylinks\.js(?:\?.*)?$/);
                        __starryBaseUrl = m ? m[1] : '';
                        return __starryBaseUrl;
                    }
                }
            } catch (_) {}
            // Fallback to relative root
            __starryBaseUrl = '';
            return __starryBaseUrl;
        }

        function buildOtterSpriteCandidates(filename) {
            const candidates = [];
            
            // 1) User-configured base path (highest priority)
            try {
                const userBase = (app.extensionManager?.setting?.get?.('StarryLinks.OtterSpritePathBase') || '').trim();
                if (userBase) {
                    const clean = userBase.replace(/\/$/, '');
                    candidates.push(`${clean}/${filename}`);
                    return candidates; // Return immediately if user path is set
                }
            } catch (_) {}
            
            // 2) Direct relative path from current script location
            try {
                // Get the directory where this script is running
                const scriptUrl = new URL(import.meta.url || document.currentScript?.src || location.href);
                const basePath = scriptUrl.href.replace('/starrylinks.js', '');
                candidates.push(`${basePath}/otters/${filename}`);
            } catch (_) {
                // Fallback: construct from known structure
                candidates.push('./otters/' + filename);
            }
            
            // 3) ComfyUI standard paths
            candidates.push('/extensions/ComfyUI_StarBetaNodes/web/js/otters/' + filename);
            candidates.push('/custom_nodes/ComfyUI_StarBetaNodes/web/js/otters/' + filename);
            
            // Return filtered paths
            return candidates.filter(url => url && url.length > 0);
        }

        function loadImageWithFallback(img, urls, onload, onfail) {
            let idx = 0;
            const tryNext = () => {
                if (idx >= urls.length) { onfail && onfail(); return; }
                const url = urls[idx++];
                img.onload = onload;
                img.onerror = tryNext;
                img.src = url;
            };
            tryNext();
        }

        // Simple sprite cache
        const spriteCache = {
            key: null,
            frames: 0,
            fps: 0,
            images: [],
            ready: false,
            loading: false
        };

        function loadOtterSprites(frames) {
            const key = `frames:${frames}`;
            if (spriteCache.key === key && (spriteCache.ready || spriteCache.loading)) return;
            spriteCache.key = key;
            spriteCache.frames = frames;
            spriteCache.images = [];
            spriteCache.ready = false;
            spriteCache.loading = true;
            let loaded = 0;
            const total = frames;
            
            const onImageLoad = () => {
                loaded++;
                if (loaded === total) {
                    spriteCache.ready = true;
                    spriteCache.loading = false;
                    console.log(`StarryLinks: Loaded ${total} otter sprite frames`);
                }
            };
            
            const onImageError = (url) => {
                console.warn(`StarryLinks: Failed to load sprite: ${url}`);
                spriteCache.loading = false;
            };
            
            if (frames === 1) {
                const img = new Image();
                const urls = buildOtterSpriteCandidates('otter.png');
                console.log(`StarryLinks: Loading single sprite from:`, urls);
                loadImageWithFallback(
                    img,
                    urls,
                    () => { 
                        spriteCache.images[0] = img; 
                        spriteCache.ready = true; 
                        spriteCache.loading = false; 
                        console.log('StarryLinks: Loaded single otter sprite');
                    },
                    () => { 
                        spriteCache.loading = false; 
                        console.warn('StarryLinks: Failed to load single otter sprite');
                    }
                );
            } else {
                for (let i = 0; i < frames; i++) {
                    const img = new Image();
                    const urls = buildOtterSpriteCandidates(`otter_${i}.png`);
                    console.log(`StarryLinks: Loading sprite ${i} from:`, urls);
                    loadImageWithFallback(
                        img,
                        urls,
                        onImageLoad,
                        () => onImageError(urls[0])
                    );
                    spriteCache.images[i] = img;
                }
            }
        }

        // Simple global state for occasional sprite "cuteness" (red tint + heart)
        // Keyed by `${linkId}:${otterIndex}` to persist across frames
        if (!window.__starryOtterLoveStates) window.__starryOtterLoveStates = {};

        // Tiny pixel heart made of rectangles (no external images)
        function drawPixelHeart(ctx, yOffset, pixelScale) {
            const s = pixelScale; // base pixel size
            const x = -2 * s;
            const y = yOffset - 3 * s;
            ctx.fillStyle = '#ff4d6d';
            // 6x6 pixel grid for a small heart
            // Layout (1 = fill):
            // . 1 1 . 1 1 .
            // 1 1 1 1 1 1 1
            // 1 1 1 1 1 1 1
            // . 1 1 1 1 1 .
            // . . 1 1 1 . .
            // . . . 1 . . .
            const dots = [
                [0,1],[1,1],[3,1],[4,1],
                [0,2],[1,2],[2,2],[3,2],[4,2],[5,2],[6,2],
                [0,3],[1,3],[2,3],[3,3],[4,3],[5,3],[6,3],
                [1,4],[2,4],[3,4],[4,4],[5,4],
                [2,5],[3,5],[4,5],
                [3,6]
            ];
            for (const [gx, gy] of dots) {
                ctx.fillRect(x + gx * s, y + gy * s, s, s);
            }
        }

        function drawOtterSprite(ctx, p, p2, scale = 1.0, fx = true, fps = 6, loveProgress = 0) {
            if (!spriteCache.ready || spriteCache.images.length === 0) return false;
            const angle = Math.atan2(p2[1] - p[1], p2[0] - p[0]);
            const now = (typeof performance !== 'undefined' ? performance.now() : Date.now());
            const bob = fx ? Math.sin(now * 0.005 + (p[0] + p[1]) * 0.01) * 1.5 : 0;
            let frameIdx = 0;
            if (fps > 0 && spriteCache.images.length > 1) {
                frameIdx = Math.floor((now / 1000) * fps) % spriteCache.images.length;
            }
            const img = spriteCache.images[frameIdx] || spriteCache.images[0];
            if (!img) return false;

            const size = 10 * scale; // base vector size; map to pixels
            const targetW = 48 * scale;
            const targetH = 38 * scale; // sprite native height is 38px

            ctx.save();
            ctx.translate(p[0], p[1] + bob);
            ctx.rotate(angle);
            // If in love state, tint the sprite pixels in a soft oval region over the head, then draw heart
            if (loveProgress > 0) {
                const ease = loveProgress < 0.5 ? (2 * loveProgress * loveProgress) : (1 - Math.pow(-2 * loveProgress + 2, 2) / 2);
                // Build a tinted sprite on an offscreen canvas (per-pixel tint)
                const tintCanvas = document.createElement('canvas');
                tintCanvas.width = Math.max(2, Math.ceil(targetW));
                tintCanvas.height = Math.max(2, Math.ceil(targetH));
                const tctx = tintCanvas.getContext('2d');
                if (tctx) {
                    // 1) Draw original sprite
                    tctx.drawImage(img, 0, 0, targetW, targetH);
                    // 2) Multiply a soft red elliptical gradient onto the head area
                    const rx = 15 * scale, ry = 9 * scale;
                    const cxLocal = 22 * scale, cyLocal = 15 * scale;
                    tctx.save();
                    tctx.translate(cxLocal, cyLocal);
                    tctx.scale(rx, ry);
                    const grad = tctx.createRadialGradient(0, 0, 0, 0, 0, 1);
                    grad.addColorStop(0, `rgba(255,59,59,${0.8 * ease})`);
                    grad.addColorStop(1, `rgba(255,59,59,0)`);
                    tctx.fillStyle = grad;
                    tctx.globalCompositeOperation = 'multiply';
                    tctx.beginPath();
                    tctx.arc(0, 0, 1, 0, Math.PI * 2);
                    tctx.fill();
                    tctx.restore();
                    // Optional slight additive pass to keep highlights
                    tctx.save();
                    tctx.translate(cxLocal, cyLocal);
                    tctx.scale(rx, ry);
                    const grad2 = tctx.createRadialGradient(0, 0, 0, 0, 0, 1);
                    grad2.addColorStop(0, `rgba(255,80,80,${0.25 * ease})`);
                    grad2.addColorStop(1, `rgba(255,80,80,0)`);
                    tctx.globalCompositeOperation = 'lighter';
                    tctx.fillStyle = grad2;
                    tctx.beginPath();
                    tctx.arc(0, 0, 1, 0, Math.PI * 2);
                    tctx.fill();
                    tctx.restore();
                    // 3) Draw tinted sprite instead of base
                    ctx.drawImage(tintCanvas, -targetW / 2, -targetH / 2);
                } else {
                    // Fallback: draw original
                    ctx.drawImage(img, -targetW / 2, -targetH / 2, targetW, targetH);
                }

                // Heart above head (vector)
                const cx = -targetW / 2 + 22 * scale;
                const floatY = -targetH / 2 - (8 * scale) - (4 * scale * (1 - ease));
                ctx.save();
                ctx.globalAlpha = 0.9 * ease;
                ctx.translate(cx, floatY);
                ctx.fillStyle = '#ff77aa';
                const hs = 6 * Math.max(0.6, scale);
                ctx.beginPath();
                ctx.moveTo(0, 0);
                ctx.bezierCurveTo(-hs, -hs, -2 * hs, -0.2 * hs, 0, 1.2 * hs);
                ctx.bezierCurveTo(2 * hs, -0.2 * hs, hs, -hs, 0, 0);
                ctx.fill();
                ctx.restore();
            } else {
                // Normal draw when no love tint
                ctx.drawImage(img, -targetW / 2, -targetH / 2, targetW, targetH);
            }
            ctx.restore();
            return true;
        }

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

            // Sample the curve with higher resolution and prepare cumulative lengths
            const samples = Math.max(32, Math.min(240, Math.round(dist / 4)));
            const points = new Array(samples + 1);
            const lens = new Array(samples + 1);
            let totalLen = 0;
            let prev = null;
            for (let i = 0; i <= samples; i++) {
                const t = i / samples;
                const out = [0, 0];
                findPointOnCurve(out, a, b, cp1, cp2, t);
                points[i] = out;
                if (i === 0) {
                    lens[i] = 0;
                    prev = out;
                } else {
                    const seg = Math.hypot(out[0] - prev[0], out[1] - prev[1]);
                    totalLen += seg;
                    lens[i] = totalLen;
                    prev = out;
                }
            }

            function pointAtArcLength(s) {
                if (s <= 0) return points[0];
                if (s >= totalLen) return points[points.length - 1];
                // binary search in lens
                let lo = 0, hi = lens.length - 1;
                while (lo < hi) {
                    const mid = (lo + hi) >> 1;
                    if (lens[mid] < s) lo = mid + 1; else hi = mid;
                }
                const i = Math.max(1, lo);
                const l0 = lens[i - 1];
                const l1 = lens[i];
                const p0 = points[i - 1];
                const p1 = points[i];
                const t = (s - l0) / Math.max(1e-6, l1 - l0);
                return [p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t];
            }

            // Settings
            const getSetting = (id, def) => app.extensionManager?.setting?.get?.(id) ?? def;
            // Backward compatibility: migrate old PurpleDotsEnabled to DotsEnabled if present
            let dotsEnabledRaw = app.extensionManager?.setting?.get?.('StarryLinks.DotsEnabled');
            if (typeof dotsEnabledRaw === 'undefined') {
                const old = app.extensionManager?.setting?.get?.('StarryLinks.PurpleDotsEnabled');
                if (typeof old !== 'undefined') dotsEnabledRaw = old;
                else dotsEnabledRaw = true;
            }
            const dotsEnabled = !!dotsEnabledRaw;
            const dotColor = getSetting('StarryLinks.DotColor', '#ffffff') || '#ffffff';
            const dotStep = Math.max(1, parseInt(getSetting('StarryLinks.DotStep', 10) || 10));
            const dotSize = parseFloat(getSetting('StarryLinks.DotSize', 1.0) || 1.0);
            const dotBlinkEnabled = !!getSetting('StarryLinks.DotBlinkEnabled', true);
            const dotBlinkSpeed = parseFloat(getSetting('StarryLinks.DotBlinkSpeed', 1.0) || 1.0);
            const dotBlinkStrength = Math.max(0, Math.min(1, parseFloat(getSetting('StarryLinks.DotBlinkStrength', 1.0) || 1.0)));
            const starsEnabled = !!getSetting('StarryLinks.StarsEnabled', true);
            const starColor = getSetting('StarryLinks.StarColor', '#ffd24a') || '#ffd24a';
            const starCount = Math.max(0, parseInt(getSetting('StarryLinks.StarCount', 3) || 3));
            const starSize = parseFloat(getSetting('StarryLinks.StarSize', 10.0) || 10.0);
            const twinkleEnabled = !!getSetting('StarryLinks.TwinkleEnabled', true);
            const twinkleSpeed = parseFloat(getSetting('StarryLinks.TwinkleSpeed', 3.0) || 3.0);
            const twinkleStrength = Math.max(0, Math.min(1, parseFloat(getSetting('StarryLinks.TwinkleStrength', 1.0) || 1.0)));
            const ottersEnabled = !!getSetting('StarryLinks.OttersEnabled', false);
            const otterCount = Math.max(0, Math.min(5, parseInt(getSetting('StarryLinks.OtterCount', 1) || 1)));
            const otterSpeed = Math.max(0.2, Math.min(3.0, parseFloat(getSetting('StarryLinks.OtterSpeed', 1.2) || 1.2)));
            const otterScale = Math.max(0.5, Math.min(2.0, parseFloat(getSetting('StarryLinks.OtterScale', 1.0) || 1.0)));
            const otterDir = String(getSetting('StarryLinks.OtterDirection', 'up') || 'up');
            const otterFX = !!getSetting('StarryLinks.OtterCutenessFX', true);
            const otterSpritesEnabled = !!getSetting('StarryLinks.OtterSpritesEnabled', false);
            const otterSpriteFrames = Math.max(1, Math.min(24, parseInt(getSetting('StarryLinks.OtterSpriteFrames', 6) || 6)));
            const otterSpriteFPS = Math.max(0, Math.min(24, parseInt(getSetting('StarryLinks.OtterSpriteFPS', 6) || 6)));

            // Draw dots
            if (dotsEnabled && points.length) {
                ctx.save();
                ctx.fillStyle = dotColor;
                const now = (typeof performance !== 'undefined' ? performance.now() : Date.now());
                // Place dots approximately every N samples but using arc-length to keep spacing consistent
                const approxCount = Math.floor(points.length / dotStep);
                const spacing = approxCount > 0 ? (totalLen / approxCount) : totalLen;
                let placed = 0;
                for (let s = 0; s <= totalLen + 1e-3; s += spacing) {
                    const p = pointAtArcLength(s);
                    let alpha = 1.0;
                    if (dotBlinkEnabled) {
                        const phase = ((p[0] * 73856093 ^ p[1] * 19349663) >>> 0) % 1000 / 1000 * Math.PI * 2;
                        const sgn = Math.sin((now / 1000) * dotBlinkSpeed * Math.PI * 2 + phase);
                        const variation = (sgn * 0.5 + 0.5) * dotBlinkStrength;
                        alpha = 0.5 + 0.5 * variation;
                    }
                    ctx.globalAlpha = alpha;
                    ctx.beginPath();
                    ctx.arc(p[0], p[1], dotSize, 0, Math.PI * 2);
                    ctx.fill();
                    placed++;
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
                const baseColor = starColor;
                const glowColor = starColor;
                const seed = hash32(((link?.id) ?? 0) ^ (a[0] | 0) ^ (b[1] | 0));
                const now = (typeof performance !== 'undefined' ? performance.now() : Date.now());

                for (let k = 0; k < starCount; k++) {
                    const s = ((k + 1) / (starCount + 1)) * totalLen;
                    const p = pointAtArcLength(s);
                    if (!p) continue;
                    // Twinkle alpha
                    let alpha = 0.9;
                    if (twinkleEnabled) {
                        // per-star random phase and slight frequency variance from seed
                        const phase = ((hash32(seed + k * 97) % 1000) / 1000) * Math.PI * 2;
                        const freqJitter = 0.75 + ((hash32(seed ^ (k * 131)) % 500) / 500) * 0.5; // 0.75..1.25
                        const s = Math.sin((now / 1000) * twinkleSpeed * freqJitter * Math.PI * 2 + phase);
                        const variation = (s * 0.5 + 0.5) * twinkleStrength; // 0..strength
                        alpha = 0.5 + 0.5 * variation; // 0.5..1 depending on strength
                    }

                    ctx.save();
                    ctx.translate(p[0], p[1]);
                    ctx.globalAlpha = alpha;
                    ctx.fillStyle = baseColor;
                    ctx.shadowColor = glowColor;
                    ctx.shadowBlur = 12;
                    ctx.strokeStyle = baseColor;
                    ctx.lineWidth = 1;
                    drawStar(ctx, 0, 0, 5, starSize, starSize * 0.5);
                    ctx.fill();
                    ctx.stroke();
                    ctx.restore();
                }
            }

            // Otters climbing along the rope
            if (ottersEnabled && otterCount > 0 && points.length > 2 && totalLen > 0) {
                const nowMs = (typeof performance !== 'undefined' ? performance.now() : Date.now());
                const tSec = nowMs / 1000;
                // Interpret otter speed as pixels per second along the curve
                const speedPx = otterSpeed * 50; // tuned multiplier for pleasant default pacing

                // Helper to sample two nearby points for tangent
                const sampleLengthToPoint = (s) => pointAtArcLength(Math.max(0, Math.min(totalLen, s)));

                // Ensure sprites are loading if enabled
                if (otterSpritesEnabled) {
                    loadOtterSprites(otterSpriteFrames);
                }

                for (let i = 0; i < otterCount; i++) {
                    const phase = (i / otterCount) * totalLen;
                    let dirMul = 1;
                    let s;
                    if (otterDir === 'down') {
                        dirMul = -1;
                    } else if (otterDir === 'both') {
                        // ping-pong between 0..totalLen
                        const cycle = (tSec * speedPx + phase * 0.25) / Math.max(1e-6, totalLen);
                        const frac = cycle - Math.floor(cycle);
                        dirMul = (Math.floor(cycle) % 2 === 0) ? 1 : -1;
                        const sRaw = frac * totalLen;
                        s = dirMul > 0 ? sRaw : (totalLen - sRaw);
                    }
                    if (otterDir !== 'both') {
                        const sUnwrapped = (tSec * speedPx * dirMul + phase);
                        s = ((sUnwrapped % totalLen) + totalLen) % totalLen;
                    }
                    const p = sampleLengthToPoint(s);
                    const p2 = sampleLengthToPoint(s + 1);

                    // Cute sprite-only love effect: random, brief, with cooldown
                    let loveProgress = 0;
                    if (otterSpritesEnabled && otterFX) {
                        const key = `${(link?.id) ?? 'L'}:${i}`;
                        let st = window.__starryOtterLoveStates[key];
                        if (!st) {
                            st = { activeUntil: 0, cooldownUntil: 0, startedAt: 0, duration: 900 };
                            window.__starryOtterLoveStates[key] = st;
                        }
                        if (nowMs < st.activeUntil) {
                            const elapsed = nowMs - st.startedAt;
                            loveProgress = Math.max(0, Math.min(1, elapsed / st.duration));
                        } else {
                            // possibly trigger
                            if (nowMs > st.cooldownUntil) {
                                // Low probability per frame
                                if (Math.random() < 0.0025) {
                                    st.startedAt = nowMs;
                                    st.duration = 900 + Math.random() * 400; // 0.9-1.3s
                                    st.activeUntil = st.startedAt + st.duration;
                                    st.cooldownUntil = st.activeUntil + (8000 + Math.random() * 7000); // 8-15s
                                    loveProgress = 0.01;
                                }
                            }
                        }
                    }
                    // Try sprite first if enabled and ready
                    let drawn = false;
                    if (otterSpritesEnabled && spriteCache.ready) {
                        drawn = drawOtterSprite(ctx, p, p2, otterScale, otterFX, otterSpriteFPS, loveProgress);
                    }
                    if (!drawn) {
                        drawCuteOtter(ctx, p, p2, otterScale, otterFX);
                    }
                }
            }
        };

        // Lightweight animation loop to refresh canvas for twinkling
        try {
            if (!window.__starryLinksRAF) {
                const tick = () => {
                    try {
                        const settings = app.extensionManager?.setting;
                        const twinkleOn = settings?.get?.('StarryLinks.TwinkleEnabled');
                        const blinkOn = settings?.get?.('StarryLinks.DotBlinkEnabled');
                        const ottersOn = settings?.get?.('StarryLinks.OttersEnabled');
                        if ((twinkleOn || blinkOn || ottersOn) && app?.canvas) {
                            // Avoid forcing redraw if a menu is open
                            const c = app.canvas;
                            const menuOpen = !!(c.current_menu || c.canvas_menu);
                            if (!menuOpen) c.setDirty(false, true);
                        }
                    } catch (_) {}
                    window.__starryLinksRAF = requestAnimationFrame(tick);
                };
                window.__starryLinksRAF = requestAnimationFrame(tick);
            }
        } catch (_) {}



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
