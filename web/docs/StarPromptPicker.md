# ⭐ Star Prompt Picker

## Overview

The **Star Prompt Picker** node outputs a single prompt string selected from either:

- A **text file** where **each line is one prompt**
- A **folder** that contains **individual `.txt` files**, where **each file is one prompt**

It supports **Random** picking or **One By One** sequential picking with **saved progress** between workflow runs (similar behavior to **Star Image Loader 1by1**).

## Inputs

All inputs have helpful tooltips when you hover over them in ComfyUI!

### Required

- **pick_source** (Dropdown)
  - **Pick From File**: read prompts from a `.txt` file, one prompt per line
  - **Pick From Folder**: read prompts from `.txt` files inside a folder (one prompt per file)

- **file_path** (STRING)
  - Path to a `.txt` file where **each line** is a prompt
  - Empty lines are ignored
  - Used when `pick_source = Pick From File`

- **folder_path** (STRING)
  - Path to a folder containing `.txt` files
  - Each `.txt` file is read as one prompt (the full file contents)
  - Used when `pick_source = Pick From Folder`

- **mode** (Dropdown)
  - **Random**: picks a random prompt each run
  - **One By One**: iterates prompts sequentially and saves progress between runs

- **start_index** (INT)
  - Starting index for **One By One** mode
  - Used when there is no saved progress yet, or when `reset_progress` is enabled

### Optional

- **reset_progress** (BOOLEAN)
  - When enabled, resets saved progress and starts at `start_index`

- **include_subfolders** (BOOLEAN)
  - Only for `Pick From Folder`
  - When enabled, includes `.txt` files in subfolders

- **seed** (INT)
  - Only used for **Random** mode
  - `0` means a new random seed is used every run
  - Any non-zero value makes the random selection deterministic

## Outputs

- **prompt** (STRING)
  - The selected prompt

- **current_index** (INT)
  - Index of the selected prompt within the resolved prompt list

- **total_prompts** (INT)
  - Total number of prompts found

- **remaining_prompts** (INT)
  - How many prompts remain after the current one (in One By One mode)

## Progress Saving

- In **One By One** mode, the node saves its progress to a JSON state file.

### When using Pick From Folder

The state file is stored inside the folder:

- `.star_prompt_picker_state.json`

### When using Pick From File

The state file is stored next to the file (same directory), using a hashed filename:

- `.star_prompt_picker_state_<hash>.json`

This avoids collisions when you use multiple different prompt files.

## Tips

- If you want to restart from the beginning, enable **reset_progress** for one run.
- If you want deterministic random picking, set a non-zero **seed**.

## Technical Details

- **Category**: ⭐StarNodes/Text And Data
- **Node Name**: ⭐ Star Prompt Picker
- **Class**: `StarPromptPicker`
- **Return Types**: `STRING`, `INT`, `INT`, `INT`
