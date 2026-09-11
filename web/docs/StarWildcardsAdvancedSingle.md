# Star Wildcards Advanced (Single)

## Description
The Star Wildcards Advanced (Single) node is a streamlined version of **Star Wildcards Advanced** for dynamic prompt generation in ComfyUI. It combines a single manual text input with a single wildcard file selector and produces one processed text output. This is useful when you only need one prompt section but still want the full wildcard and randomization features.

## Inputs

### Required
- **seed**: Integer seed value that determines the randomization of wildcard selections
- **prompt**: Text input with support for wildcards and random selections
- **wildcard**: Wildcard file selection for the prompt (None, Random, or specific file)

## Outputs
- **STRING**: A single processed string containing the prompt text with all inline and file-based wildcards applied

## Usage
1. Enter text in the **prompt** field.
2. Optionally select a wildcard file option in **wildcard**:
   - **None**: No external wildcard file content will be added.
   - **Random**: A random wildcard file will be selected from the wildcards directory.
   - **Specific file**: Choose a specific wildcard file from the dropdown.
3. Set a **seed** value to control the randomization.
4. Connect the output to any node that accepts text input, such as a conditioning node.

## Features

### Text Processing
- **Inline Wildcards**: Use `__wildcard_name__` syntax within the **prompt** text.
- **Random Selection**: Use `{option1|option2|option3}` syntax for random selection.
- **Folder Support**: Use `folder\__wildcard_name__` to access wildcards in subfolders.
- **Nested Wildcards**: Wildcards can contain other wildcards for complex combinations.

### Wildcard File Integration
- **Direct File Selection**: Choose a specific wildcard file from the **wildcard** dropdown.
- **Random File Selection**: Automatically select a random wildcard file with the **Random** option.
- **Combined Processing**: The final output is your **prompt** (including any inline wildcards) plus the optional content from the selected wildcard file.

## Wildcard File Structure
- Wildcard files should be placed in the `ComfyUI/wildcards/` directory.
- Each wildcard file should be named with the wildcard name and have a `.txt` extension.
- Each line in the wildcard file represents a possible replacement option.
- For folder organization, create subfolders within the wildcards directory.

## Technical Details
- The node automatically scans the wildcards directory to populate the **wildcard** dropdown.
- Inline syntax uses the same processing as **Star Wildcards Advanced** and **Star Seven Wildcards`**.
- The `seed` is used for both inline random options and wildcard file selection to ensure reproducible results.
- If a selected wildcard file does not exist or is empty, the wildcard name is used as fallback text.

## Examples

### Basic Usage with File Selection
```text
prompt: "a photo of a"
wildcard: "animals" (selects from animals.txt)
```

### Combining Manual Text with Wildcards
```text
prompt: "a {realistic|stylized} __animal__ with __color__ fur"
wildcard: "None"
```

### Using Random Wildcard Selection
```text
prompt: "a portrait of a"
wildcard: "Random" (selects a random wildcard file)
```

## Notes
- This node is functionally equivalent to using a single row of **Star Wildcards Advanced**.
- Ideal when you only need one prompt section and want to keep your graph compact.
- Wildcard files are shared across all ComfyUI nodes that support wildcards.
