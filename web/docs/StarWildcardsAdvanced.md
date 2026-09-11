# Star Wildcards Advanced

## Description
The Star Wildcards Advanced node is an enhanced text processing tool for dynamic prompt generation in ComfyUI. It combines manual text input with automated wildcard file selection, offering greater flexibility than the basic wildcards node. This node processes up to seven separate text inputs, each with its own dedicated wildcard file selector, and combines them into a single output string. It's particularly useful for creating complex, varied prompts with a mix of manual text and randomized content.

## Inputs

### Required
- **seed**: Integer seed value that determines the randomization of wildcard selections
- **prompt_1**: First text input with support for wildcards and random selections
- **wildcard_1**: Wildcard file selection for the first prompt (None, Random, or specific file)
- **prompt_2**: Second text input with support for wildcards and random selections
- **wildcard_2**: Wildcard file selection for the second prompt
- **prompt_3**: Third text input with support for wildcards and random selections
- **wildcard_3**: Wildcard file selection for the third prompt
- **prompt_4**: Fourth text input with support for wildcards and random selections
- **wildcard_4**: Wildcard file selection for the fourth prompt
- **prompt_5**: Fifth text input with support for wildcards and random selections
- **wildcard_5**: Wildcard file selection for the fifth prompt
- **prompt_6**: Sixth text input with support for wildcards and random selections
- **wildcard_6**: Wildcard file selection for the sixth prompt
- **prompt_7**: Seventh text input with support for wildcards and random selections
- **wildcard_7**: Wildcard file selection for the seventh prompt

## Outputs
- **STRING**: A single combined string containing all processed prompts with wildcards replaced

## Usage
1. Enter text in any or all of the seven prompt inputs
2. For each prompt, select a wildcard file option:
   - **None**: No additional wildcard content will be added
   - **Random**: A random wildcard file will be selected
   - **Specific file**: Choose a specific wildcard file from the dropdown
3. Set a seed value to control the randomization
4. Connect the output to any node that accepts text input, such as a conditioning node

## Features

### Text Processing
- **Inline Wildcards**: Use `__wildcard_name__` syntax within any prompt text
- **Random Selection**: Use `{option1|option2|option3}` syntax for random selection
- **Folder Support**: Use `folder\__wildcard_name__` to access wildcards in subfolders
- **Nested Wildcards**: Wildcards can contain other wildcards for complex combinations

### Wildcard File Integration
- **Direct File Selection**: Choose specific wildcard files from dropdown menus
- **Random File Selection**: Automatically select random wildcard files
- **Combined Processing**: Each prompt section combines manual text with wildcard file content
- **Empty Section Handling**: Empty prompt sections are excluded from the final output

## Wildcard File Structure
- Wildcard files should be placed in the `ComfyUI/wildcards/` directory
- Each wildcard file should be named with the wildcard name and have a `.txt` extension
- Each line in the wildcard file represents a possible replacement option
- For folder organization, create subfolders within the wildcards directory

## Technical Details
- The node automatically scans the wildcards directory to populate the dropdown menus
- Each prompt and wildcard combination uses a different seed offset to ensure variety
- If a selected wildcard file doesn't exist, the wildcard name is used as fallback text
- Empty prompt sections are filtered out when combining the final text

## Examples

### Basic Usage with File Selection
```
prompt_1: "a photo of a"
wildcard_1: "animals" (selects from animals.txt)
prompt_2: "with"
wildcard_2: "colors" (selects from colors.txt)
```

### Combining Manual Text with Wildcards
```
prompt_1: "a {realistic|stylized} __animal__ with __color__ fur"
wildcard_1: "None"
prompt_2: ""
wildcard_2: "environments" (adds a random environment from environments.txt)
```

### Using Random Wildcard Selection
```
prompt_1: "a portrait of a"
wildcard_1: "Random" (selects a random wildcard file)
```

## Notes
- For best results, use a different seed for each generation to get varied outputs
- The node is particularly useful for creating complex prompts with consistent structure but varied content
- Wildcard files are shared across all ComfyUI nodes that support wildcards
- The dropdown menus only show wildcard files in the root wildcards directory, but subfolder wildcards can be accessed using the folder\__wildcard__ syntax in prompt text
