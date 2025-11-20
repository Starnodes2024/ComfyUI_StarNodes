import { app } from "/scripts/app.js";

app.registerExtension({
  name: "Star.ImageEdit.QwenKontext.UI",
  async setup() {
    let schema = null;
    const fetchSchema = async () => {
      try {
        const res = await fetch("/starbetanodes/editprompts", { cache: "no-store" });
        if (!res.ok) return null;
        return await res.json();
      } catch (e) {
        console.warn("Star ImageEdit UI: cannot fetch /starbetanodes/editprompts", e);
        return null;
      }
    };

    const getTaskParams = (model, task) => {
      if (!schema) return [];
      const m = schema?.models?.[model];
      const t = m?.tasks?.[task];
      return Array.isArray(t?.params) ? t.params : [];
    };

    const isEmpty = (v) => v === undefined || v === null || String(v).trim() === "";

    const styleWidget = (widget, required, missing) => {
      const input = widget?.inputEl || widget?.input_element || widget?.element;
      const baseTitle = widget?.tooltip || widget?.name || "";
      
      if (required) {
        widget.disabled = false;
        const hint = missing ? "⚠️ Required: please fill this parameter" : "✅ Required parameter";
        
        if (input) {
          input.style.backgroundColor = missing ? "#fff3cd" : "#e8f5e8";
          input.style.opacity = "1.0";
          input.style.border = missing ? "2px solid #ff9800" : "1px solid #4caf50";
          input.title = baseTitle ? `${baseTitle} — ${hint}` : hint;
          if (missing && input.placeholder !== undefined) {
            input.placeholder = (input.placeholder || "") + " (required)";
          }
        }
        widget.color = missing ? "#ff9800" : "#4caf50";
      } else {
        widget.disabled = true;
        if (input) {
          input.style.backgroundColor = "#f5f5f5";
          input.style.opacity = "0.5";
          input.style.border = "1px solid #ccc";
          input.title = baseTitle ? `${baseTitle} — 🔒 Not needed for this task` : "🔒 Not needed for this task";
          if (input.placeholder !== undefined) {
            input.placeholder = "not needed";
          }
        }
        widget.color = "#9e9e9e";
      }
    };

    const updateNodeUI = (node) => {
      try {
        if (!node || !node.widgets) return;
        
        const modelWidget = node.widgets.find(w => w.name === "model");
        const taskWidget = node.widgets.find(w => w.name === "task");
        
        if (!modelWidget || !taskWidget) return;
        
        const model = modelWidget.value;
        const task = taskWidget.value;
        
        // Debounce rapid changes
        if (node._uiUpdateTimeout) {
          clearTimeout(node._uiUpdateTimeout);
        }
        
        node._uiUpdateTimeout = setTimeout(() => {
          try {
            const requiredList = getTaskParams(model, task);
            const hasValidReqs = Array.isArray(requiredList) && requiredList.length > 0 && !!model && !!task;
            const requiredSet = new Set(requiredList || []);
            
            // Special handling for "Own Prompt (Empty)" and "Image Restore" - disable all fields except keep_clause
            const isOwnPromptEmpty = task === "Own Prompt (Empty)" || task === "Image Restore";
            
            const paramNames = [
              "subject", "color", "background", "object", "location", "style",
              "surface", "text", "lighting", "expression", "clothing_item", "style_or_color"
            ];

            for (const w of node.widgets || []) {
              if (!paramNames.includes(w.name)) continue;
              
              try {
                const req = hasValidReqs && requiredSet.has(w.name);
                const missing = req && isEmpty(w.value);
                
                // Special case: disable all parameter fields for "Own Prompt (Empty)" and "Image Restore"
                if (isOwnPromptEmpty) {
                  w.disabled = true;
                  if (w.inputEl || w.input_element || w.element) {
                    const input = w.inputEl || w.input_element || w.element;
                    if (input && input.style) {
                      input.style.backgroundColor = "#f5f5f5";
                      input.style.opacity = "0.5";
                      input.style.border = "1px solid #ccc";
                      input.title = (w?.tooltip || w?.name || "") + " (disabled for this task)";
                    }
                  }
                  w.color = "#9e9e9e";
                } else {
                  // Normal logic for other tasks
                  w.disabled = !req && hasValidReqs;
                  
                  if (w.inputEl || w.input_element || w.element) {
                    const input = w.inputEl || w.input_element || w.element;
                    if (input && input.style) {
                      if (!hasValidReqs) {
                        // When schema not loaded, show all fields as potentially needed
                        input.style.backgroundColor = "";
                        input.style.opacity = "1.0";
                        input.style.border = "1px solid #ccc";
                        input.title = w?.tooltip || w?.name || "";
                        w.disabled = false; // Enable all when schema not loaded
                      } else if (req) {
                        input.style.backgroundColor = missing ? "#fff3cd" : "#e8f5e8";
                        input.style.opacity = "1.0";
                        input.style.border = missing ? "2px solid #ff9800" : "1px solid #4caf50";
                        input.title = (w?.tooltip || w?.name || "") + (missing ? " (required)" : "");
                      } else {
                        input.style.backgroundColor = "#f5f5f5";
                        input.style.opacity = "0.7";
                        input.style.border = "1px solid #ddd";
                        input.title = (w?.tooltip || w?.name || "") + " (not needed)";
                      }
                    }
                  }
                  
                  w.color = hasValidReqs ? (req ? (missing ? "#ff9800" : "#4caf50") : "#9e9e9e") : undefined;
                }
              } catch (widgetError) {
                // Skip problematic widgets
                console.warn("Error styling widget", w.name, widgetError);
              }
            }
            
            // Add tooltips and advanced field handling here
            for (const aux of ["keep_clause"]) {
              const w = node.widgets?.find(x => x.name === aux);
              if (!w) continue;
              const input = w.inputEl || w.input_element || w.element;
              if (input) {
                input.title = input.title || "Add own or additional Prompt";
                input.style.opacity = input.style.opacity || "0.85";
              }
              // Advanced fields remain enabled
              w.disabled = false;
            }
            
            if (app?.canvas) {
              app.canvas.setDirty(true);
            }
          } catch (updateError) {
            console.warn("Error in delayed update:", updateError);
          }
        }, 50); // Small delay to prevent rapid updates
      } catch (error) {
        console.warn("Error in updateNodeUI:", error);
      }
    };

    const wireNode = (node) => {
      if (node.__star_qwen_ui_wired) return;
      node.__star_qwen_ui_wired = true;
      
      // Ensure widgets exist before trying to use them
      if (!node.widgets) {
        setTimeout(() => wireNode(node), 100);
        return;
      }
      
      // Apply UI immediately with current schema (even if null)
      updateNodeUI(node);
      
      // Then fetch schema if needed and update again
      const ensure = async () => {
        try {
          if (!schema) {
            schema = await fetchSchema();
            // Re-apply UI with fresh schema
            updateNodeUI(node);
          }
        } catch (e) {
          console.warn("Star ImageEdit UI: failed to fetch schema", e);
        }
        
        // Set up widget callbacks
        for (const w of node.widgets || []) {
          const prevCb = w.callback;
          w.callback = async function () {
            try { 
              if (prevCb) await prevCb.apply(this, arguments); 
            } catch (_) {}
            
            // Handle model changes specially
            if (w.name === "model") {
              try {
                const schemaData = await fetchSchema();
                if (schemaData && schemaData.models) {
                  const modelValue = this.value;
                  const tasks = schemaData.models[modelValue]?.tasks || {};
                  const taskWidget = node.widgets.find(w => w.name === "task");
                  
                  if (taskWidget && taskWidget.type === "combo") {
                    const taskOptions = Object.keys(tasks);
                    if (taskOptions.length > 0) {
                      taskWidget.options.values = taskOptions;
                      
                      // Only change task if current one is invalid
                      if (!taskOptions.includes(taskWidget.value)) {
                        const oldValue = taskWidget.value;
                        taskWidget.value = taskOptions[0];
                        
                        // Trigger change if value actually changed
                        if (oldValue !== taskWidget.value && taskWidget.callback) {
                          setTimeout(() => taskWidget.callback(), 10);
                        }
                      }
                    }
                  }
                }
              } catch (e) {
                console.warn("Error updating task options:", e);
              }
            }
            
            updateNodeUI(node);
          };
          // Some widgets use onChange instead of callback
          const prevOnChange = w.onChange;
          w.onChange = function () {
            try { if (prevOnChange) prevOnChange.apply(this, arguments); } catch (_) {}
            updateNodeUI(node);
          };
        }
      };
      ensure();
    };

    // Use the proper ComfyUI node registration system
    app.registerExtension({
      name: "Star.ImageEdit.QwenKontext.UI.Patch",
      async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "StarImageEditQwenKontext") return;
        
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
          const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
          
          // Use multiple attempts to ensure widgets are ready
          const tryWire = (attempt = 0) => {
            if (attempt > 10) return; // Give up after 10 attempts
            
            if (this.widgets && this.widgets.length > 0) {
              wireNode(this);
            } else {
              setTimeout(() => tryWire(attempt + 1), 100);
            }
          };
          
          tryWire();
          return r;
        };
      }
    });

    // Handle existing nodes immediately
    const handleExistingNodes = () => {
      try {
        const graph = app?.canvas?.graph;
        if (!graph || !graph._nodes) return;
        
        for (const node of graph._nodes) {
          if (node.type === "StarImageEditQwenKontext" && !node.__star_qwen_ui_wired) {
            wireNode(node);
          }
        }
      } catch (e) {
        console.warn("Error handling existing nodes:", e);
      }
    };
    
    // Initial check for existing nodes
    handleExistingNodes();
    
    // Periodic check for new nodes
    let foundNew = false;
    const nodeCheckInterval = setInterval(() => {
      try {
        const graph = app?.canvas?.graph;
        if (!graph || !graph._nodes) return;
        
        for (const node of graph._nodes) {
          if (node.type === "StarImageEditQwenKontext" && !node.__star_qwen_ui_wired) {
            wireNode(node);
            foundNew = true;
          }
        }
        
        // Stop polling after 10 seconds if no new nodes found
        if (foundNew) {
          setTimeout(() => clearInterval(nodeCheckInterval), 10000);
        }
      } catch (_) {}
    }, 500);
    
    // Stop polling after 30 seconds
    setTimeout(() => clearInterval(nodeCheckInterval), 30000);
  }
});
