import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";

// Store workflow state
const workflowState = {
    stopped: {},  // Track stopped nodes by unique_id
    waiting: {},  // Track nodes waiting for user input
};

app.registerExtension({
    name: "StarNodes.StarStopAndGo",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "StarStopAndGo") {
            
            // Add custom widgets for buttons
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                if (onNodeCreated) {
                    onNodeCreated.apply(this, arguments);
                }
                
                const node = this;
                
                // Add Stop button
                const stopButton = node.addWidget(
                    "button",
                    "🛑 STOP Workflow",
                    null,
                    async () => {
                        const nodeId = String(node.id);
                        
                        // Send decision to backend
                        try {
                            await fetch("/starnodes/stop_and_go/decision", {
                                method: "POST",
                                headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({ node_id: nodeId, decision: "stop" })
                            });
                            console.log(`[StarStopAndGo] Sent STOP decision for node ${nodeId}`);
                        } catch (error) {
                            console.error(`[StarStopAndGo] Error sending STOP decision:`, error);
                        }
                        
                        // Mark workflow as stopped
                        workflowState.stopped[nodeId] = true;
                        workflowState.waiting[nodeId] = false;
                        
                        // Update node appearance
                        node.bgcolor = "#AA3333";
                        node.title = "⭐ Star Stop And Go [STOPPED]";
                        
                        // Interrupt the current execution
                        api.interrupt();
                    },
                    { serialize: false }
                );
                
                // Add Go button
                const goButton = node.addWidget(
                    "button",
                    "✅ GO Continue",
                    null,
                    async () => {
                        const nodeId = String(node.id);
                        
                        // Send decision to backend
                        try {
                            await fetch("/starnodes/stop_and_go/decision", {
                                method: "POST",
                                headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({ node_id: nodeId, decision: "go" })
                            });
                            console.log(`[StarStopAndGo] Sent GO decision for node ${nodeId}`);
                        } catch (error) {
                            console.error(`[StarStopAndGo] Error sending GO decision:`, error);
                        }
                        
                        // Clear stopped state
                        workflowState.stopped[nodeId] = false;
                        workflowState.waiting[nodeId] = false;
                        
                        // Update node appearance
                        node.bgcolor = "#33AA33";
                        node.title = "⭐ Star Stop And Go [GO]";
                        
                        // Reset appearance after a moment
                        setTimeout(() => {
                            node.bgcolor = node.constructor.nodeData?.bgcolor;
                            node.title = "⭐ Star Stop And Go";
                        }, 1500);
                    },
                    { serialize: false }
                );
                
                // Update info based on mode changes
                const updateInfo = () => {
                    const modeWidget = node.widgets?.find(w => w.name === "mode");
                    if (modeWidget) {
                        const mode = modeWidget.value;
                        if (mode === "User Select") {
                            node.title = "⭐ Star Stop And Go [User Select]";
                        } else if (mode === "Pause") {
                            const pauseWidget = node.widgets?.find(w => w.name === "pause_seconds");
                            const seconds = pauseWidget ? pauseWidget.value : 5.0;
                            node.title = `⭐ Star Stop And Go [Pause ${seconds}s]`;
                        } else if (mode === "Bypass") {
                            node.title = "⭐ Star Stop And Go [Bypass]";
                        }
                    }
                };
                
                // Update info when mode changes
                const modeWidget = node.widgets?.find(w => w.name === "mode");
                if (modeWidget) {
                    const originalCallback = modeWidget.callback;
                    modeWidget.callback = function() {
                        if (originalCallback) {
                            originalCallback.apply(this, arguments);
                        }
                        updateInfo();
                    };
                }
                
                // Update info when pause_seconds changes
                const pauseWidget = node.widgets?.find(w => w.name === "pause_seconds");
                if (pauseWidget) {
                    const originalCallback = pauseWidget.callback;
                    pauseWidget.callback = function() {
                        if (originalCallback) {
                            originalCallback.apply(this, arguments);
                        }
                        updateInfo();
                    };
                }
                
                // Initial info update
                updateInfo();
            };
            
            // Handle node removal
            const onRemoved = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function() {
                const nodeId = this.id;
                delete workflowState.stopped[nodeId];
                delete workflowState.waiting[nodeId];
                
                if (onRemoved) {
                    onRemoved.apply(this, arguments);
                }
            };
        }
    },
    
    async setup() {
        // Listen for execution events
        api.addEventListener("execution_start", ({ detail }) => {
            // Clear stopped states when new execution starts
            for (const nodeId in workflowState.stopped) {
                workflowState.stopped[nodeId] = false;
            }
            for (const nodeId in workflowState.waiting) {
                workflowState.waiting[nodeId] = false;
            }
        });
        
        api.addEventListener("execution_error", ({ detail }) => {
            console.log("[StarStopAndGo] Execution error detected");
        });
        
        api.addEventListener("executed", ({ detail }) => {
            // Check if any node is in stop state
            const nodeId = detail.node;
            if (workflowState.stopped[nodeId]) {
                console.log(`[StarStopAndGo] Stopping execution at node ${nodeId}`);
                api.interrupt();
            }
        });
    }
});
