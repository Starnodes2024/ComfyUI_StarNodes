# Star Watermark

## Description
A comprehensive watermarking node that allows you to add text or image watermarks to your generated images. This node supports both single images and batches, with extensive customization options including positioning, opacity, rotation, blend modes, and special text effects.

## Inputs

### Required
- **images**: The input image(s) to watermark
- **mode**: Watermark type (`Text` or `Image`)
- **opacity**: Watermark opacity (0.0-1.0, default: 0.5)
- **position**: Watermark position (`center`, `top-left`, `top-right`, `bottom-left`, `bottom-right`, `top-center`, `bottom-center`, `left-center`, `right-center`)
- **x_offset**: Horizontal offset from position in pixels (default: 0)
- **y_offset**: Vertical offset from position in pixels (default: 0)
- **rotation**: Watermark rotation in degrees (default: 0)
- **blend_mode**: How the watermark blends with the image (`normal`, `multiply`, `overlay`)
- **output_mask**: Whether to output a mask of the watermark (default: false)
- **adaptive_color**: Automatically adjust text color for visibility (default: false)
- **randomize_per_batch**: Randomize watermark position for each image in batch (default: false)

### Text Mode Options
- **text**: Text content for the watermark
- **font**: Font to use (system fonts are detected automatically)
- **font_size**: Size of the text (default: 36)
- **font_color**: Text color (default: white)
- **font_bold**: Use bold text (default: false)
- **font_italic**: Use italic text (default: false)
- **font_underline**: Use underlined text (default: false)
- **effect_outline**: Add outline to text (default: false)
- **effect_shadow**: Add drop shadow to text (default: false)
- **effect_glow**: Add glow effect to text (default: false)
- **insert_datetime**: Add current date/time to text (default: false)

### Image Mode Options
- **watermark_image**: Image to use as watermark
- **watermark_mask**: Optional mask for the watermark image
- **invert_mask**: Invert the watermark mask (default: false)
- **image_scale**: Scale factor for the watermark image (default: 100%)
- **repeat_pattern**: Repeat the watermark in a pattern (default: false)

## Outputs
- **images**: The watermarked image(s)
- **mask**: Watermark mask (if output_mask is enabled)

## Usage
1. Connect your source image(s) to the input
2. Choose between text or image watermark mode
3. Configure watermark appearance using the available options
4. Adjust position, opacity, and other settings as needed
5. The node will output the watermarked image(s)

### Text Watermark Example
For adding a copyright notice to your images:
- Set mode to `Text`
- Enter your copyright text (e.g., "© 2023 Your Name")
- Position at `bottom-right` with small offsets
- Use semi-transparent opacity (0.3-0.5)
- Enable `adaptive_color` for better visibility

### Image Watermark Example
For adding a logo to your images:
- Set mode to `Image`
- Connect your logo image to `watermark_image`
- Adjust `image_scale` to size appropriately
- Position at desired location
- Set blend mode to `normal` or `multiply` depending on your logo

## Notes
- The node automatically detects available system fonts
- For transparent watermarks, use an image with alpha channel
- When using `adaptive_color`, the text color will adjust based on the background for better visibility
- The `randomize_per_batch` option is useful for creating unique watermarks across a batch of images
