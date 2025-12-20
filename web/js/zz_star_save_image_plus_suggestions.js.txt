import { app } from "../../../../scripts/app.js";

function isSameSuggestionItem(item, nodeName) {
	if (typeof item === "string") return item === nodeName;
	if (item && typeof item === "object") {
		return item.value === nodeName || item.content === nodeName;
	}
	return false;
}

function promoteSuggestion(type, nodeName) {
	const lg = globalThis.LiteGraph;
	if (!lg) return;

	for (const mapName of ["slot_types_default_out", "slot_types_default_in"]) {
		if (!lg[mapName]) lg[mapName] = {};
		if (!lg[mapName][type]) lg[mapName][type] = [];

		const arr = lg[mapName][type];
		for (let i = arr.length - 1; i >= 0; i--) {
			if (isSameSuggestionItem(arr[i], nodeName)) {
				arr.splice(i, 1);
			}
		}

		arr.unshift(nodeName);
	}
}

app.registerExtension({
	name: "StarNodes.Suggestions.SaveImagePlusTop",
	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData?.name === "StarSaveImagePlus") {
			nodeData.display_name = "⭐ Star Save Image+";
		}
	},
	async setup() {
		setTimeout(() => {
			try {
				const nt = globalThis.LiteGraph?.registered_node_types?.["StarSaveImagePlus"];
				if (nt) {
					nt.title = "⭐ Star Save Image+";
					if (nt.prototype) nt.prototype.title = "⭐ Star Save Image+";
				}
			} catch (e) {
			}
			promoteSuggestion("IMAGE", "⭐ Star Save Image+");
		}, 0);
	},
});
