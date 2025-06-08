# Starnodes Aspect Ratio

## Description
This node helps generate image dimensions based on standard aspect ratios and desired megapixel counts. It can either use a predefined aspect ratio or automatically detect the closest matching aspect ratio from an input image.

## Inputs
- **aspect_ratio**: Select from predefined aspect ratios (e.g., "16:9", "4:3", "1:1", etc.)
- **megapixel**: Target image size in megapixels (options: 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
- **use_nearest_from_image**: When enabled, the node will analyze the input image and select the closest matching aspect ratio
- **image** (optional): Input image for aspect ratio detection when `use_nearest_from_image` is enabled

## Outputs
- **width**: The calculated width in pixels
- **height**: The calculated height in pixels
- **Resolution**: A formatted string showing the resolution (e.g., "512 x 768")

## Usage
1. Select a desired aspect ratio from the dropdown menu
2. Choose the target megapixel size for your output image
3. Optionally, connect an image and enable `use_nearest_from_image` to automatically match the closest standard aspect ratio
4. Use the output width and height values to configure image generation nodes

This node is particularly useful for:
- Creating images with standard aspect ratios (16:9, 4:3, 1:1, etc.)
- Scaling images to specific megapixel counts while maintaining aspect ratio
- Automatically detecting the aspect ratio of an input image and finding the closest standard ratio
- Generating appropriate dimensions for different output requirements (social media, print, etc.)

The node calculates dimensions that maintain the selected aspect ratio while targeting the specified megapixel count, ensuring optimal image sizes for various use cases.
