# Star Image Compare

## Description

**Star Image Compare** is an interactive before/after image comparison node. Connect two IMAGE inputs and use a draggable wipe slider to compare them directly inside the node.

## Inputs

### Optional
- **image_1**: First image to compare (IMAGE type). Shown when the slider is moved to the **right**.
- **image_2**: Second image to compare (IMAGE type). Shown when the slider is moved to the **left**.
- **compare_position**: Initial slider position from `0.0` (left / Image 2) to `1.0` (right / Image 1). The value is saved in the workflow and updated when you drag the slider.
- **caption_image1**: Caption text for Image 1, drawn onto the **concat_image** output.
- **caption_image2**: Caption text for Image 2, drawn onto the **concat_image** output.
- **split_line_color**: Color of the 3px divider line in the **half_half_image** output — `red` (default), `black`, `white`, `grey`, `blue` or `green`.
- **concat_layout**: Layout of the **concat_image** output — `auto` (default), `left/right` or `top/bottom`.

Both images must be connected for the comparison to appear. If they are missing, the node shows a placeholder message.

## Outputs

- **concat_image**: A single image containing both inputs concatenated. Captions are shown under each image with a black background and white text. The caption bar height and font size scale dynamically with the image size (8% of image height, font proportional to bar height).
  - `auto`: side by side when both images are **portrait**, stacked vertically otherwise.
  - `left/right`: always side by side, matching the height of the smaller image.
  - `top/bottom`: always stacked vertically, matching the width of the smaller image.
  - If one image is larger, it is resized down to match the smaller dimension.
- **half_half_image**: A single image the same size as **image_1**, split vertically down the middle — the left half shows Image 1 and the right half shows the right half of Image 2 (resized to match Image 1). A 3px divider line (color set by **split_line_color**) marks the split. `caption_image1` is drawn at the bottom-left and `caption_image2` at the bottom-right, each with a semi-transparent dark background.

## How to Use

1. Connect an image to the **image_1** input.
2. Connect an image to the **image_2** input.
3. Execute the workflow.
4. Drag the vertical divider or the slider at the bottom of the node:
   - **Far left**: shows only Image 2.
   - **Far right**: shows only Image 1.
   - **Middle**: splits the view between both images.

## Notes

- The first image in each batch is used for the comparison.
- The DOM widget displays a label "Image 1 | Image 2" above the comparison view, indicating which image is on which side.
- If the images have different aspect ratios, they are fit inside the viewer using `object-fit: contain`.
- The `concat_image` output is always available, even if one or both images are missing (a black placeholder is returned in that case).
