import { app } from "../../../../scripts/app.js";

// Extension to handle dynamic inputs for StarNodes
app.registerExtension({
    name: "StarNodes.dynamic",
    async setup() {
        console.log("StarNodes dynamic inputs extension setup");
    },
    
    async beforeRegisterNodeDef(nodeType, nodeData) {
        // List of nodes that should have dynamic inputs
        const dynamicNodes = [
            "StarImageSwitch",      // StarNode.py
            "StarLatentSwitch"      // starlatentinput.py
        ];
        
        // Check if this node should have dynamic inputs
        if (dynamicNodes.includes(nodeData.name)) {
            console.log(`Setting up dynamic inputs for: ${nodeData.name}`);
            
            // Override the getExtraMenuOptions to add a "Add Input" option
            const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
            nodeType.prototype.getExtraMenuOptions = function(_, options) {
                if (getExtraMenuOptions) {
                    getExtraMenuOptions.apply(this, arguments);
                }
                
                if (nodeData.name === "StarImageSwitch") {
                    options.push({
                        content: "Add Image Input",
                        callback: () => {
                            this.addDynamicInput("Image", "IMAGE");
                        }
                    });
                } else if (nodeData.name === "StarLatentSwitch") {
                    options.push({
                        content: "Add Latent Input",
                        callback: () => {
                            this.addDynamicInput("Latent", "LATENT");
                        }
                    });
                }
            };
            
            // Add method to add dynamic input
            nodeType.prototype.addDynamicInput = function(prefix, type) {
                // Find the highest numbered input
                let highestIndex = 0;
                for (const input of this.inputs || []) {
                    if (input.name.startsWith(prefix)) {
                        const match = input.name.match(/(\d+)$/);
                        if (match) {
                            const index = parseInt(match[1]);
                            highestIndex = Math.max(highestIndex, index);
                        }
                    }
                }
                
                // Add a new input with the next number
                const newIndex = highestIndex + 1;
                const newInputName = `${prefix} ${newIndex}`;
                this.addInput(newInputName, type);
                
                // Force a graph change to update the UI
                if (this.graph) {
                    this.graph.change();
                }
            };
            
            // Store the original onConnectionsChange function
            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            
            // Override the onConnectionsChange function
            nodeType.prototype.onConnectionsChange = function(type, index, connected, link_info) {
                // Call the original onConnectionsChange if it exists
                if (onConnectionsChange) {
                    onConnectionsChange.apply(this, arguments);
                }
                
                // Only handle input connections
                if (type !== LiteGraph.INPUT) {
                    return;
                }
                
                // Handle dynamic inputs based on node type
                if (nodeData.name === "StarImageSwitch") {
                    this.handleDynamicInputs("Image", "IMAGE");
                }
                else if (nodeData.name === "StarLatentSwitch") {
                    this.handleDynamicInputs("Latent", "LATENT");
                }
                
                // Force a graph change to update the UI
                if (this.graph) {
                    this.graph.change();
                }
            };
            
            // Add method to handle dynamic inputs for image and latent nodes
            nodeType.prototype.handleDynamicInputs = function(prefix, type) {
                // Find the highest numbered input and check connections
                let highestIndex = 0;
                let allConnected = true;
                let lastConnectedIndex = 0;
                
                // Check all inputs
                for (let i = 0; i < this.inputs.length; i++) {
                    const input = this.inputs[i];
                    if (input.name.startsWith(prefix)) {
                        const match = input.name.match(/(\d+)$/);
                        if (match) {
                            const index = parseInt(match[1]);
                            highestIndex = Math.max(highestIndex, index);
                            
                            // Check if this input is connected
                            if (input.link) {
                                lastConnectedIndex = Math.max(lastConnectedIndex, index);
                            } else {
                                allConnected = false;
                            }
                        }
                    }
                }
                
                // If all inputs are connected, add a new one
                if (allConnected && highestIndex > 0) {
                    const newIndex = highestIndex + 1;
                    const newInputName = `${prefix} ${newIndex}`;
                    this.addInput(newInputName, type);
                }
                
                // Remove any disconnected inputs that are higher than the last connected input
                // except keep one empty input after the last connected one
                for (let i = this.inputs.length - 1; i >= 0; i--) {
                    const input = this.inputs[i];
                    if (input.name.startsWith(prefix)) {
                        const match = input.name.match(/(\d+)$/);
                        if (match) {
                            const index = parseInt(match[1]);
                            if (!input.link && index > lastConnectedIndex + 1) {
                                this.removeInput(i);
                            }
                        }
                    }
                }
            };
            
            // Override the onNodeCreated function to initialize dynamic inputs
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                // Call the original onNodeCreated if it exists
                if (onNodeCreated) {
                    onNodeCreated.apply(this, arguments);
                }
                
                if (nodeData.name === "StarImageSwitch") {
                    // Remove any "Image 1" inputs that might be duplicated
                    const inputsToRemove = [];
                    for (let i = 0; i < this.inputs.length; i++) {
                        if (this.inputs[i].name === "Image 1") {
                            inputsToRemove.push(i);
                        }
                    }
                    
                    // Remove them in reverse order
                    for (let i = inputsToRemove.length - 1; i >= 0; i--) {
                        this.removeInput(inputsToRemove[i]);
                    }
                    
                    // Add the dynamic input
                    this.addInput("Image 1", "IMAGE");
                }
                else if (nodeData.name === "StarLatentSwitch") {
                    // Remove any "Latent 1" inputs that might be duplicated
                    const inputsToRemove = [];
                    for (let i = 0; i < this.inputs.length; i++) {
                        if (this.inputs[i].name === "Latent 1") {
                            inputsToRemove.push(i);
                        }
                    }
                    
                    // Remove them in reverse order
                    for (let i = inputsToRemove.length - 1; i >= 0; i--) {
                        this.removeInput(inputsToRemove[i]);
                    }
                    
                    // Add the dynamic input
                    this.addInput("Latent 1", "LATENT");
                }
            };
        }
    }
});
