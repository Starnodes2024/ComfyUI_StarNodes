// Copy and paste this entire script into your browser console
// when you have ComfyUI open to immediately change the color of FluxFillSampler nodes

(function() {
    // Function to apply blue color to all existing FluxFillSampler nodes
    function applyBlueToExistingNodes() {
        // Get all nodes in the graph
        const nodes = app.graph._nodes;
        let count = 0;
        
        // Loop through all nodes
        for (let i = 0; i < nodes.length; i++) {
            const node = nodes[i];
            
            // Check if this is a FluxFillSampler node
            if (node.type === "FluxFillSampler") {
                // Apply blue background
                node.bgcolor = "#0066cc";
                node.color = "#ffffff";
                count++;
            }
        }
        
        // Force canvas redraw
        app.canvas.setDirty(true);
        app.canvas.draw(true, true);
        
        return count;
    }
    
    // Function to modify the node constructor to apply blue to new nodes
    function setupNodeColorChange() {
        // Get the node constructor
        const nodeConstructors = LiteGraph.registered_node_types;
        
        if (nodeConstructors["FluxFillSampler"]) {
            // Get the original node constructor
            const originalNodeConstructor = nodeConstructors["FluxFillSampler"];
            
            // Create a wrapper that adds our color
            function modifiedNodeConstructor() {
                // Call the original constructor
                originalNodeConstructor.apply(this, arguments);
                
                // Apply our color
                this.bgcolor = "#0066cc";
                this.color = "#ffffff";
            }
            
            // Copy all properties from the original constructor
            for (const prop in originalNodeConstructor) {
                if (originalNodeConstructor.hasOwnProperty(prop)) {
                    modifiedNodeConstructor[prop] = originalNodeConstructor[prop];
                }
            }
            
            // Replace the constructor
            nodeConstructors["FluxFillSampler"] = modifiedNodeConstructor;
            
            return true;
        }
        
        return false;
    }
    
    // Apply colors to existing nodes
    const existingCount = applyBlueToExistingNodes();
    
    // Set up color change for new nodes
    const setupSuccess = setupNodeColorChange();
    
    console.log(`Applied blue color to ${existingCount} existing FluxFillSampler nodes`);
    console.log(`Setup for new nodes: ${setupSuccess ? "Success" : "Failed"}`);
    
    // Add a message to the UI
    const message = document.createElement("div");
    message.style.position = "fixed";
    message.style.top = "10px";
    message.style.left = "50%";
    message.style.transform = "translateX(-50%)";
    message.style.backgroundColor = "#0066cc";
    message.style.color = "white";
    message.style.padding = "10px";
    message.style.borderRadius = "5px";
    message.style.zIndex = "9999";
    message.style.fontWeight = "bold";
    message.textContent = `Applied blue color to ${existingCount} FluxFillSampler nodes`;
    
    document.body.appendChild(message);
    
    // Remove the message after 3 seconds
    setTimeout(() => {
        document.body.removeChild(message);
    }, 3000);
})();
