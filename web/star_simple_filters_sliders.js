// Star Simple Filters - Temperature Slider
// Custom gradient slider for temperature control
import { app } from "../../scripts/app.js";

class TemperatureSlider {
    constructor(node, widgetName, config = {}) {
        this.node = node;
        this.widgetName = widgetName;
        this.config = {
            min: config.min || -100.0,
            max: config.max || 100.0,
            step: config.step || 1.0,
            decimals: config.decimals || 0,
            default: config.default || 0.0
        };

        this.value = config.default;
        this.normalizedPos = this.valueToPosition(this.value);

        this.isDragging = false;
    }

    valueToPosition(value) {
        return (value - this.config.min) / (this.config.max - this.config.min);
    }

    positionToValue(pos) {
        const value = this.config.min + (this.config.max - this.config.min) * pos;
        return Math.round(value / this.config.step) * this.config.step;
    }

    setValue(value) {
        this.value = Math.max(this.config.min, Math.min(this.config.max, value));
        this.normalizedPos = this.valueToPosition(this.value);
    }

    draw(ctx, y, width) {
        const margin = 10;
        const labelWidth = 90;
        const sliderWidth = width - labelWidth - 75;
        const sliderHeight = 10;
        const centerY = y + 15;

        // Draw label "Temperature"
        ctx.fillStyle = "#ffffff";
        ctx.font = "14px sans-serif";
        ctx.textAlign = "left";
        ctx.fillText("Temperature", margin, centerY + 4);

        const sliderStart = margin + labelWidth;

        // Background track
        ctx.fillStyle = "rgba(30, 30, 30, 0.8)";
        ctx.beginPath();
        ctx.roundRect(sliderStart, centerY - sliderHeight/2, sliderWidth, sliderHeight, 5);
        ctx.fill();

        // Blue to Red gradient (full track)
        const gradient = ctx.createLinearGradient(sliderStart, 0, sliderStart + sliderWidth, 0);
        gradient.addColorStop(0, "#0099ff");    // Blue (cold)
        gradient.addColorStop(0.5, "#888888");  // Gray (neutral)
        gradient.addColorStop(1, "#ff3300");    // Red (warm)

        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.roundRect(sliderStart, centerY - sliderHeight/2, sliderWidth, sliderHeight, 5);
        ctx.fill();

        // Cursor/handle - Star shape
        const cursorX = sliderStart + sliderWidth * this.normalizedPos;
        const starSize = 10;
        const spikes = 5;
        const outerRadius = starSize;
        const innerRadius = starSize * 0.4;
        
        ctx.save();
        ctx.translate(cursorX, centerY);
        ctx.rotate(-Math.PI / 2); // Rotate to point upward
        
        // Draw star
        ctx.beginPath();
        for (let i = 0; i < spikes * 2; i++) {
            const radius = i % 2 === 0 ? outerRadius : innerRadius;
            const angle = (Math.PI / spikes) * i;
            const x = Math.cos(angle) * radius;
            const y = Math.sin(angle) * radius;
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.closePath();
        
        // Fill star with gradient based on position
        ctx.fillStyle = "#ffd700"; // Gold color
        ctx.fill();
        
        // Star outline
        ctx.strokeStyle = "#333333";
        ctx.lineWidth = 1.5;
        ctx.stroke();
        
        ctx.restore();

        // Value display
        ctx.fillStyle = "#ffffff";
        ctx.font = "16px monospace";
        ctx.textAlign = "right";
        ctx.fillText(this.value.toFixed(this.config.decimals), width - 6, centerY + 4);
    }

    handleMouseDown(e, localX, localY, width) {
        const margin = 10;
        const labelWidth = 90;
        const sliderStart = margin + labelWidth;
        const sliderWidth = width - labelWidth - 75;

        if (localY >= 5 && localY <= 25) {
            if (localX >= sliderStart && localX <= sliderStart + sliderWidth) {
                this.isDragging = true;
                this.updateFromMouse(localX, width);
                return true;
            }

            // Click on value to enter manually
            if (localX >= width - 65 && localX <= width - 5) {
                return "prompt";
            }
        }
        return false;
    }

    handleMouseMove(e, localX, localY, width) {
        if (this.isDragging) {
            this.updateFromMouse(localX, width);
            return true;
        }
        return false;
    }

    handleMouseUp(e) {
        if (this.isDragging) {
            this.isDragging = false;
            return true;
        }
        return false;
    }

    updateFromMouse(localX, width) {
        const margin = 10;
        const labelWidth = 90;
        const sliderStart = margin + labelWidth;
        const sliderWidth = width - labelWidth - 75;

        let pos = (localX - sliderStart) / sliderWidth;
        pos = Math.max(0, Math.min(1, pos));

        const newValue = this.positionToValue(pos);

        if (newValue !== this.value) {
            this.setValue(newValue);

            const widget = this.node.widgets?.find(w => w.name === this.widgetName);
            if (widget) widget.value = this.value;

            return true;
        }
        return false;
    }
}

app.registerExtension({
    name: "star.simple.filters.temperature.slider",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "StarSimpleFilters") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const result = onNodeCreated?.apply(this, arguments);

            // Only create custom slider for temperature
            const tempWidget = this.widgets?.find(w => w.name === "temperature");
            if (tempWidget) {
                const savedValue = tempWidget.value;
                tempWidget.hidden = true;
                tempWidget.type = "hidden";

                this.temperatureSlider = new TemperatureSlider(this, "temperature", {
                    min: -100,
                    max: 100,
                    step: 1,
                    decimals: 0,
                    default: savedValue
                });

                this.temperatureSlider.setValue(savedValue);
            }

            this.sliderHeight = 40;

            // Adjust node size
            const originalComputeSize = this.computeSize;
            this.computeSize = function(out) {
                const size = originalComputeSize ? originalComputeSize.call(this, out) : [this.size[0], this.size[1]];
                size[1] += this.sliderHeight || 0;
                return size;
            };

            this.size = this.computeSize();

            // Load saved value from workflow
            const originalOnConfigure = this.onConfigure;
            this.onConfigure = function(info) {
                const r = originalOnConfigure?.apply(this, arguments);

                const w = this.widgets?.find(w => w.name === "temperature");
                if (w && this.temperatureSlider) {
                    this.temperatureSlider.setValue(w.value);
                }

                return r;
            };

            return result;
        };

        // Draw temperature slider
        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function(ctx) {
            const result = onDrawForeground?.apply(this, arguments);

            if (this.flags.collapsed) return result;

            if (this.temperatureSlider) {
                const sliderY = this.size[1] - this.sliderHeight + 5;

                ctx.save();
                this.temperatureSlider.draw(ctx, sliderY, this.size[0]);
                ctx.restore();

                const tempWidget = this.widgets?.find(w => w.name === "temperature");
                if (tempWidget && tempWidget.value !== this.temperatureSlider.value) {
                    tempWidget.value = this.temperatureSlider.value;
                    if (this.onPropertyChanged) {
                        this.onPropertyChanged("temperature", this.temperatureSlider.value);
                    }
                }
            }

            return result;
        };

        // Mouse down handler
        const onMouseDown = nodeType.prototype.onMouseDown;
        nodeType.prototype.onMouseDown = function(e, localPos, canvas) {
            if (this.temperatureSlider) {
                const sliderY = this.size[1] - this.sliderHeight + 5;
                const localX = localPos[0];
                const localY = localPos[1] - sliderY;

                const result = this.temperatureSlider.handleMouseDown(e, localX, localY, this.size[0]);

                if (result === "prompt") {
                    canvas.prompt(
                        "Temperature",
                        this.temperatureSlider.value,
                        (v) => {
                            const num = parseFloat(v);
                            if (!isNaN(num)) {
                                this.temperatureSlider.setValue(num);
                                this.setDirtyCanvas(true, true);
                            }
                        },
                        e
                    );
                    return true;
                }

                if (result) {
                    this.setDirtyCanvas(true, true);
                    return true;
                }
            }

            return onMouseDown?.apply(this, arguments);
        };

        // Mouse move handler
        const onMouseMove = nodeType.prototype.onMouseMove;
        nodeType.prototype.onMouseMove = function(e, localPos, canvas) {
            if (this.temperatureSlider && this.temperatureSlider.isDragging) {
                const sliderY = this.size[1] - this.sliderHeight + 5;
                const localX = localPos[0];
                const localY = localPos[1] - sliderY;

                if (this.temperatureSlider.handleMouseMove(e, localX, localY, this.size[0])) {
                    this.setDirtyCanvas(true, true);
                    return true;
                }
            }

            return onMouseMove?.apply(this, arguments);
        };

        // Mouse up handler
        const onMouseUp = nodeType.prototype.onMouseUp;
        nodeType.prototype.onMouseUp = function(e, localPos, canvas) {
            if (this.temperatureSlider) {
                if (this.temperatureSlider.handleMouseUp(e)) {
                    this.setDirtyCanvas(true, true);
                    return true;
                }
            }

            return onMouseUp?.apply(this, arguments);
        };
    }
});
