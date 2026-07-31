import os
import json
import hashlib
import time
import random


class StarPromptPicker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pick_source": (["Pick From File", "Pick From Folder"], {
                    "default": "Pick From File",
                    "tooltip": "Choose where to pick prompts from. File: one prompt per line. Folder: one .txt file per prompt."
                }),
                "file_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to a .txt file where each line is one prompt (empty lines are ignored). Used when Pick From File is selected."
                }),
                "folder_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to a folder containing .txt files. Each file is treated as a single prompt (full file text). Used when Pick From Folder is selected."
                }),
                "mode": (["Random", "One By One"], {
                    "default": "One By One",
                    "tooltip": "Random: pick a random prompt each run. One By One: iterate prompts sequentially and save progress."
                }),
                "start_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1000000000,
                    "tooltip": "Starting index for One By One mode (used when no saved progress exists, or when reset_progress is enabled)."
                }),
            },
            "optional": {
                "reset_progress": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Reset",
                    "label_off": "Keep",
                    "tooltip": "If enabled, resets saved progress and starts at start_index."
                }),
                "include_subfolders": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Only for Pick From Folder: include .txt files in subfolders."
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Optional seed for Random mode. 0 = use a new random seed each run."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "INT")
    RETURN_NAMES = ("prompt", "current_index", "total_prompts", "remaining_prompts")
    FUNCTION = "pick"
    CATEGORY = "⭐StarNodes/Text And Data"

    def _state_file_for_source(self, pick_source, file_path, folder_path):
        if pick_source == "Pick From Folder":
            return os.path.join(folder_path, ".star_prompt_picker_state.json")

        file_dir = os.path.dirname(file_path) if file_path else ""
        file_hash = hashlib.md5((file_path or "").encode("utf-8")).hexdigest()[:12]
        return os.path.join(file_dir or ".", f".star_prompt_picker_state_{file_hash}.json")

    def _load_counter(self, state_file):
        if os.path.exists(state_file):
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                return int(state.get("counter", 0))
            except Exception:
                return 0
        return 0

    def _save_counter(self, state_file, counter):
        try:
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump({"counter": int(counter)}, f)
        except Exception:
            pass

    def _load_prompts_from_file(self, file_path):
        if not file_path:
            raise ValueError("file_path cannot be empty when Pick From File is selected")
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File '{file_path}' cannot be found.")

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.read().splitlines()

        prompts = [line.strip() for line in lines if line.strip()]
        if not prompts:
            raise ValueError(f"No prompts found in '{file_path}'")
        return prompts

    def _load_prompts_from_folder(self, folder_path, include_subfolders=False):
        if not folder_path:
            raise ValueError("folder_path cannot be empty when Pick From Folder is selected")
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Folder '{folder_path}' cannot be found.")

        txt_files = []
        if include_subfolders:
            for root, _, files in os.walk(folder_path):
                for name in files:
                    if name.lower().endswith(".txt"):
                        txt_files.append(os.path.join(root, name))
        else:
            for name in os.listdir(folder_path):
                if name.lower().endswith(".txt"):
                    full = os.path.join(folder_path, name)
                    if os.path.isfile(full):
                        txt_files.append(full)

        txt_files.sort(key=lambda p: os.path.basename(p).lower())

        prompts = []
        for p in txt_files:
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read().strip()
                if content:
                    prompts.append(content)
            except Exception:
                continue

        if not prompts:
            raise ValueError(f"No valid prompt .txt files found in '{folder_path}'")
        return prompts

    def pick(self, pick_source, file_path, folder_path, mode, start_index, reset_progress=False, include_subfolders=False, seed=0):
        if pick_source == "Pick From Folder":
            prompts = self._load_prompts_from_folder(folder_path, include_subfolders=include_subfolders)
        else:
            prompts = self._load_prompts_from_file(file_path)

        total = len(prompts)

        if mode == "Random":
            used_seed = seed
            if used_seed == 0:
                used_seed = random.randint(0, 0xffffffffffffffff)
            rng = random.Random(used_seed)
            idx = rng.randrange(total)
            remaining = max(total - idx - 1, 0)
            return (prompts[idx], idx, total, remaining)

        state_file = self._state_file_for_source(pick_source, file_path, folder_path)

        if reset_progress:
            counter = int(start_index)
        else:
            counter = self._load_counter(state_file)
            if counter == 0 and int(start_index) != 0 and not os.path.exists(state_file):
                counter = int(start_index)

        if counter < 0:
            counter = 0
        if counter >= total:
            counter = 0

        prompt = prompts[counter]

        next_counter = counter + 1
        if next_counter >= total:
            next_counter = 0
        self._save_counter(state_file, next_counter)

        remaining = max(total - counter - 1, 0)
        return (prompt, counter, total, remaining)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time()


NODE_CLASS_MAPPINGS = {
    "StarPromptPicker": StarPromptPicker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarPromptPicker": "⭐ Star Prompt Picker",
}
