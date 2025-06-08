import { app } from "../../../../scripts/app.js";

// Extension to apply custom colors to all StarNodes
app.registerExtension({
    name: "StarNodes.appearance",
    async setup() {
        // This runs once when the extension is loaded
        console.log("StarNodes appearance extension setup");
    },
    
    async beforeRegisterNodeDef(nodeType, nodeData) {
        // Check if this is a StarNode by looking at the category
        if (nodeData.category && nodeData.category.startsWith("⭐")) {
            console.log(`Found StarNode: ${nodeData.name}, applying custom colors`);
            
            // Define our colors
            const backgroundColor = "#3d124d";  // Purple background
            const titleColor = "#19124d";       // Dark blue title
            const textColor = "#051b34";        // Dark blue text (was white)
            
            // Store the original onNodeCreated function
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            // Override the onNodeCreated function
            nodeType.prototype.onNodeCreated = function() {
                // Call the original onNodeCreated if it exists
                if (onNodeCreated) {
                    onNodeCreated.apply(this, arguments);
                }
                
                // Apply custom colors
                this.bgcolor = backgroundColor;
                this.color = textColor;
                
                // Store the original drawTitleBar function
                const originalDrawTitleBar = this.drawTitleBar;
                
                // Override the drawTitleBar function to use our custom title color
                this.drawTitleBar = function(ctx, title_height) {
                    // Call the original function first
                    originalDrawTitleBar.call(this, ctx, title_height);
                    
                    // Draw the title text with our custom color
                    if (this.flags.collapsed) {
                        return;
                    }
                    ctx.font = this.title_font || LiteGraph.DEFAULT_TITLE_FONT;
                    const title = this.getTitle();
                    if (title) {
                        ctx.save();
                        ctx.fillStyle = titleColor;
                        ctx.fillText(title, 10, title_height * 0.75);
                        ctx.restore();
                    }
                };
                
                console.log(`Applied custom colors to StarNode: ${this.type}`);
            };
        }
    }
});
