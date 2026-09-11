# Star Palette Extractor

## Description
This node extracts the dominant color palette from an input image using K-means clustering. It analyzes the image to identify the most prominent colors and provides detailed color information in various formats (Hex, RGB, CMYK). The node also generates a visual palette image showing each extracted color with its corresponding values.

## Inputs
- **image**: The input image to extract colors from
- **num_colors**: Number of colors to extract from the image (default: 8, range: 2-32)
- **tile_size**: Size of each color tile in the output palette image (default: 128, range: 32-512)
- **color_format**: Format for color information output:
  - `All`: Show all formats (Hex, RGB, CMYK)
  - `Hex`: Show only hexadecimal color codes
  - `RGB`: Show only RGB values
  - `CMYK`: Show only CMYK percentages

## Outputs
- **palette**: A text string containing all extracted colors in the selected format
- **palette_image**: A visual representation of the color palette with color tiles and information
- **Colors**: Same as the palette output (for compatibility)
- **color1** through **color10**: Individual color values for the first 10 extracted colors

## Usage
1. Connect an image source to the input
2. Adjust the number of colors to extract based on your needs
3. Set the tile size for the visual palette preview
4. Select your preferred color format
5. Use the extracted palette information in your workflow or export it for design purposes

This node is particularly useful for:
- Extracting color schemes from reference images
- Creating consistent color palettes for design projects
- Analyzing the color composition of generated images
- Exporting color information in various formats for different applications
