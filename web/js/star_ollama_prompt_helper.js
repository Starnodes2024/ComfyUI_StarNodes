import { app } from "../../../../scripts/app.js";

app.registerExtension({
    name: "starnodes.ollama_prompt_helper",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "StarOllamaPromptHelper") return;

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (onCreated) onCreated.apply(this, arguments);

            const urlWidget = this.widgets.find(w => w.name === "local_address");
            const modelWidget = this.widgets.find(w => w.name === "model");
            const presetWidget = this.widgets.find(w => w.name === "system_prompt_preset");
            const sysPromptWidget = this.widgets.find(w => w.name === "system_prompt");

            if (!urlWidget || !modelWidget) return;

            const refreshBtn = this.addWidget("button", "🔄 Refresh Models");
            refreshBtn.serialize = false;

            const fetchModels = async () => {
                refreshBtn.name = "⏳ Fetching...";
                this.setDirtyCanvas(true);
                try {
                    const resp = await fetch("/starnodes/ollama/models", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ url: urlWidget.value }),
                    });
                    if (!resp.ok) throw new Error("Request failed");
                    const models = await resp.json();
                    if (models.error) throw new Error(models.error);

                    const prev = modelWidget.value;
                    modelWidget.options.values = models;
                    if (models.includes(prev)) {
                        modelWidget.value = prev;
                    } else if (models.length > 0) {
                        modelWidget.value = models[0];
                    }
                } catch (err) {
                    console.error("[StarOllama] model fetch error:", err);
                    if (app.extensionManager?.toast?.add) {
                        app.extensionManager.toast.add({
                            severity: "error",
                            summary: "Ollama connection error",
                            detail: "Make sure Ollama server is running at the specified address.",
                            life: 5000,
                        });
                    }
                }
                refreshBtn.name = "🔄 Refresh Models";
                this.setDirtyCanvas(true);
            };

            refreshBtn.callback = fetchModels;
            urlWidget.callback = fetchModels;

            const updateSysPromptVisibility = () => {
                if (!presetWidget || !sysPromptWidget) return;
                const isCustom = presetWidget.value === "Custom";
                sysPromptWidget.type = isCustom ? "string" : "hidden";
                if (sysPromptWidget.inputEl) {
                    sysPromptWidget.inputEl.style.display = isCustom ? "" : "none";
                }
                const labelEl = sysPromptWidget.labelEl;
                if (labelEl) {
                    labelEl.style.display = isCustom ? "" : "none";
                }
            };

            if (presetWidget) {
                const origCallback = presetWidget.callback;
                presetWidget.callback = function () {
                    if (origCallback) origCallback.apply(this, arguments);
                    updateSysPromptVisibility();
                };
            }

            updateSysPromptVisibility();
            fetchModels();
        };
    },
});
