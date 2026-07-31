# Star Easy-Text-Storage

## Description
This node provides a simple and persistent text storage system for ComfyUI workflows. It allows you to save, load, replace, and remove text snippets that persist between ComfyUI sessions. This is particularly useful for storing frequently used prompts, settings, or any text content that you want to reuse across different workflows.

## Inputs

### Required
- **mode**: Operation mode (options: "Save Text", "Load Text", "Remove Text", "Replace Text")
- **Save-Name**: Name to identify the text entry in storage

### Optional
- **Text-Selector**: Dropdown to select from previously saved text entries
- **text_content**: The text content to save or replace (multiline text input)

## Outputs
- **STRING**: The loaded text content or status message based on the selected operation mode

## Usage

### Save Text Mode
1. Set mode to "Save Text"
2. Enter a name in the "Save-Name" field
3. Enter the text content you want to save in the "text_content" field
4. Execute the node to save the text

### Load Text Mode
1. Set mode to "Load Text"
2. Select the text entry to load from the "Text-Selector" dropdown
3. Execute the node to load and output the saved text

### Remove Text Mode
1. Set mode to "Remove Text"
2. Select the text entry to remove from the "Text-Selector" dropdown
3. Execute the node to delete the text from storage

### Replace Text Mode
1. Set mode to "Replace Text"
2. Select the text entry to replace from the "Text-Selector" dropdown
3. Enter the new text content in the "text_content" field
4. Execute the node to update the stored text

## Features
- **Persistent Storage**: Saved texts remain available between ComfyUI sessions
- **Automatic Refresh**: The node automatically updates its dropdown list when texts are added or removed
- **Status Messages**: Provides helpful feedback about operations
- **Duplicate Prevention**: Automatically adds numbering to prevent overwriting existing entries with the same name
- **Cross-Session Compatibility**: Texts saved in one session are available in future sessions

## Storage Location
The node stores text data in a JSON file (`startext.json`) located in the main ComfyUI directory. This ensures that your saved texts persist even when updating the node or ComfyUI itself.

## Notes
- For large text blocks like complex prompts, the multiline text input makes editing convenient
- The node is particularly useful for storing template prompts that you frequently modify
- You can use this node in combination with other text processing nodes to create reusable text components
- The storage system is designed to be simple and reliable, focusing on basic text storage operations
