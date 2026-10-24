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

            this.addWidget("button", "⭐ 📂 Load Workflow", null, async () => {
                const info = getCurrentImageFile(this);
                if (!info) {
                    alert("No image selected.");
                    return;
                }
                try {
                    await loadWorkflowFromImage(info);
                } catch (err) {
                    console.error("Failed to load embedded workflow:", err);
                    alert("Failed to load workflow: " + err.message);
                }
            });

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

function getCurrentImageFile(node) {
    const imageWidget = node.widgets?.find(w => w.name === "image");
    let value = imageWidget?.value;
    if (!value || typeof value !== "string") return null;
    let type = "input";
    for (const annotation of [" [output]", " [input]", " [temp]"]) {
        if (value.endsWith(annotation)) {
            type = annotation.trim().slice(1, -1);
            value = value.slice(0, -annotation.length);
            break;
        }
    }
    value = value.replace(/\\/g, "/");
    const idx = value.lastIndexOf("/");
    const subfolder = idx !== -1 ? value.slice(0, idx) : "";
    const filename = idx !== -1 ? value.slice(idx + 1) : value;
    return { filename, subfolder, type };
}

// Scan PNG chunks for embedded graph data (workflow / prompt / A1111 parameters).
function pngHasWorkflowData(buffer) {
    const view = new DataView(buffer);
    if (view.byteLength < 8 || view.getUint32(0) !== 0x89504e47) return null; // not a PNG
    const decoder = new TextDecoder();
    let offset = 8;
    while (offset + 8 <= view.byteLength) {
        const length = view.getUint32(offset);
        const chunkType = decoder.decode(new Uint8Array(buffer, offset + 4, 4));
        if (chunkType === "tEXt" || chunkType === "iTXt" || chunkType === "zTXt") {
            const head = decoder.decode(new Uint8Array(buffer, offset + 8, Math.min(length, 12)));
            if (head.startsWith("workflow\0") || head.startsWith("prompt\0") || head.startsWith("parameters\0")) return true;
        }
        if (chunkType === "IEND") break;
        offset += 12 + length;
    }
    return false;
}

async function loadWorkflowFromImage(info) {
    const params = new URLSearchParams({ filename: info.filename, subfolder: info.subfolder, type: info.type });
    const response = await api.fetchApi("/view?" + params.toString());
    if (!response.ok) {
        alert("Could not load the image file from the server.");
        return;
    }
    const blob = await response.blob();
    const buffer = await blob.arrayBuffer();
    if (pngHasWorkflowData(buffer) === false) {
        alert("This image has no embedded workflow or prompt.");
        return;
    }
    const ext = info.filename.split(".").pop().toLowerCase();
    const mime = blob.type || { png: "image/png", jpg: "image/jpeg", jpeg: "image/jpeg", webp: "image/webp", bmp: "image/bmp", gif: "image/gif" }[ext] || "image/png";
    const file = new File([buffer], info.filename, { type: mime });
    // app.handleFile is the same handler used when an image is dropped on the canvas.
    await app.handleFile(file);
    app.graph.setDirtyCanvas(true);
}

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
