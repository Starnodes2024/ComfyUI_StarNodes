# Star Face Loader

## Description
The Star Face Loader is a specialized image loading node designed for face images and reference photos. It provides an organized system for managing face images with dedicated folders, automatic organization, and seamless handling of both uploaded and pasted images. This node is particularly useful for workflows involving face swapping, reference-based generation, or any task requiring consistent access to face images.

## Inputs

### Required
- **image**: Selection of available face images from the input directory or faces subdirectory
- **upload_to_face_folder**: Whether to automatically copy uploaded images to the faces folder

## Outputs
- **image**: The loaded image as a tensor
- **mask**: An empty mask matching the image dimensions (for compatibility with mask-based workflows)

## Usage
1. Select an image from the dropdown list, or use the upload button to add a new image
2. Choose whether to save uploaded images to the faces folder
3. The node will load the image and provide it as output along with a blank mask
4. Images can be used directly in workflows requiring face references

## Features

### Dedicated Face Image Management
- Automatically creates and manages a dedicated "faces" folder in the ComfyUI input directory
- Organizes face images separately from other input images
- Maintains original images in the input directory for compatibility

### Support for Multiple Image Sources
- Loads images from the faces directory (prioritized)
- Falls back to the main input directory if no faces directory exists
- Handles uploaded images from the user's computer
- Processes pasted images directly from the clipboard

### Automatic Image Organization
- Saves uploaded images to the input directory
- Optionally copies uploaded images to the faces folder
- Saves pasted images with timestamped filenames to prevent conflicts
- Creates a separate "pasted" directory for clipboard images

### Preview Integration
- Automatically generates image previews in the UI
- Shows the loaded image directly in the workflow

## Technical Details
- Images are converted to RGB format to ensure compatibility
- Loaded images are normalized to the 0-1 range as floating-point tensors
- The node creates an empty mask tensor matching the image dimensions
- File operations include proper error handling and path validation
- Timestamped filenames prevent conflicts between similarly named files

## Notes
- The "faces" folder is created automatically if it doesn't exist
- Images in the faces folder will appear at the top of the selection list
- When "upload_to_face_folder" is enabled, uploaded images are copied to both the input directory and faces folder
- Pasted images are always saved with timestamped filenames to prevent conflicts
- This node is designed to work seamlessly with face-related nodes like face swapping
