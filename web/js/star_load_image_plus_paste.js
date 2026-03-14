import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";

app.registerExtension({
    name: "StarLoadImagePlus.PasteClipboard",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "StarLoadImagePlus") return;

        const originalGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        
        nodeType.prototype.getExtraMenuOptions = function(_, options) {
            if (originalGetExtraMenuOptions) {
                originalGetExtraMenuOptions.apply(this, arguments);
            }

            options.unshift({
                content: "📋 Paste Clipboard Image",
                callback: async () => {
                    try {
                        const clipboardItems = await navigator.clipboard.read();
                        
                        for (const clipboardItem of clipboardItems) {
                            for (const type of clipboardItem.types) {
                                if (type.startsWith('image/')) {
                                    const blob = await clipboardItem.getType(type);
                                    await uploadImageFromBlob(this, blob);
                                    return;
                                }
                            }
                        }
                        
                        alert("No image found in clipboard");
                    } catch (err) {
                        console.error("Failed to paste image from clipboard:", err);
                        alert("Failed to paste image. Make sure you have copied an image to clipboard and granted clipboard permissions.");
                    }
                }
            });
        };

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            
            this.addWidget("button", "⭐ 📋 Paste Image", null, () => {
                navigator.clipboard.read().then(async (clipboardItems) => {
                    for (const clipboardItem of clipboardItems) {
                        for (const type of clipboardItem.types) {
                            if (type.startsWith('image/')) {
                                const blob = await clipboardItem.getType(type);
                                await uploadImageFromBlob(this, blob);
                                return;
                            }
                        }
                    }
                    alert("No image found in clipboard");
                }).catch(err => {
                    console.error("Failed to paste image from clipboard:", err);
                    alert("Failed to paste image. Make sure you have copied an image to clipboard and granted clipboard permissions.");
                });
            });
            
            return result;
        };
    }
});

async function uploadImageFromBlob(node, blob) {
    const formData = new FormData();
    const filename = `pasted_image_${Date.now()}.png`;
    formData.append("image", blob, filename);
    formData.append("overwrite", "true");
    
    try {
        const response = await api.fetchApi("/upload/image", {
            method: "POST",
            body: formData,
        });

        if (response.status === 200) {
            const data = await response.json();
            const imageWidget = node.widgets.find(w => w.name === "image");
            
            if (imageWidget) {
                imageWidget.value = data.name;
                
                if (imageWidget.callback) {
                    imageWidget.callback(data.name);
                }
                
                app.graph.setDirtyCanvas(true);
            }
            
            console.log("Image pasted successfully:", data.name);
        } else {
            const errorText = await response.text();
            console.error("Upload failed:", errorText);
            alert("Failed to upload image: " + errorText);
        }
    } catch (error) {
        console.error("Error uploading image:", error);
        alert("Error uploading image: " + error.message);
    }
}
