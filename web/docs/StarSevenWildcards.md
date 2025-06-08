# Star Seven Wildcards

## Description
The Star Seven Wildcards node is a powerful text processing tool that enables dynamic prompt generation by incorporating wildcards and random selections. It processes up to seven separate text inputs, each with its own wildcard processing, and combines them into a single output string. This node is particularly useful for creating varied prompts across multiple generations or for organizing complex prompts into logical sections.

## Inputs

### Required
- **seed**: Integer seed value that determines the randomization of wildcard selections
- **prompt_1**: First text input with support for wildcards and random selections
- **prompt_2**: Second text input with support for wildcards and random selections
- **prompt_3**: Third text input with support for wildcards and random selections
- **prompt_4**: Fourth text input with support for wildcards and random selections
- **prompt_5**: Fifth text input with support for wildcards and random selections
- **prompt_6**: Sixth text input with support for wildcards and random selections
- **prompt_7**: Seventh text input with support for wildcards and random selections

## Outputs
- **STRING**: A single combined string containing all processed prompts with wildcards replaced

## Usage
1. Enter text in any or all of the seven prompt inputs
2. Include wildcards and random selections in your text using the supported syntax
3. Set a seed value to control the randomization
4. Connect the output to any node that accepts text input, such as a conditioning node

## Features

### Wildcard Syntax
- **Basic Wildcards**: Use `__wildcard_name__` to randomly select a line from a corresponding wildcard file
- **Folder Support**: Use `folder\__wildcard_name__` to access wildcards in subfolders
- **Random Selection**: Use `{option1|option2|option3}` to randomly select one option from a list
- **Nested Wildcards**: Wildcards can contain other wildcards for complex combinations
- **Seed Offsetting**: Each prompt uses a different seed offset to ensure variety across the seven inputs

## Wildcard File Structure
- Wildcard files should be placed in the `ComfyUI/wildcards/` directory
- Each wildcard file should be named with the wildcard name and have a `.txt` extension
- Each line in the wildcard file represents a possible replacement option
- For folder organization, create subfolders within the wildcards directory

## Technical Details
- The node processes each prompt with a different seed offset to ensure variety
- Wildcards are processed recursively up to a maximum depth of 10 to prevent infinite loops
- If a wildcard file doesn't exist, the wildcard name itself is used as the replacement
- Empty wildcard files will result in the wildcard name being used as the replacement

## Examples

### Basic Usage
```
prompt_1: "a photo of a __animal__"
prompt_2: "with __color__ fur"
prompt_3: "in a __environment__"
```

### Using Random Selection
```
prompt_1: "a {realistic|stylized|detailed} rendering"
prompt_2: "of a {small|large|medium-sized} __animal__"
```

### Using Folder Organization
```
prompt_1: "a photo of a animals\__mammal__"
prompt_2: "with colors\__warm_colors__ fur"
```

## Notes
- While the node is named "Star Seven Wildcards" in the UI, the internal class name is "StarFiveWildcards" (a legacy from when it supported five inputs)
- For best results, use a different seed for each generation to get varied outputs
- The node automatically handles nested wildcards, allowing for complex, hierarchical prompt structures
- Each prompt section is processed independently before being combined, allowing for modular prompt design
