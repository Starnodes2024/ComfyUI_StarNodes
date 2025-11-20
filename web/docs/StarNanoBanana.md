# Star Nano Banana (Gemini Image Generation)

A ComfyUI custom node that integrates Google's Gemini 2.5 Flash model for image generation and editing.

## Features

- **Image Generation**: Create images from text prompts using Gemini's native image generation capabilities
- **Image Editing**: Edit existing images with text instructions
- **Model Selection**: Choose between different Gemini models
- **Megapixel Options**: Select from 1 to 15 megapixel sizes
- **Aspect Ratios**: Choose from predefined ratios (1:1, 16:9, etc.)
- **Multiple Input Images**: Optionally provide up to 5 input images for editing
- **API Key Management**: Flexible API key storage options (local or external)

## Setup

1. **Install Dependencies**:
   ```bash
   pip install google-generativeai
   ```

2. **API Key Configuration**:
   - Choose one of the following methods to set your Google Gemini API key:
   
   **Method 1: Direct key in googleapi.ini (recommended for local use)**
   - Edit the `googleapi.ini` file in the custom nodes directory
   - Uncomment and set your key:
   ```ini
   [API_KEY]
   key = your_actual_api_key_here
   ```
   
   **Method 2: Point to an external file (recommended for shared setups)**
   - Edit the `googleapi.ini` file in the custom nodes directory
   - Ensure it points to your external file:
   ```ini
   [API_PATH]
   path = D:\AI\ComfyUINEW\googleapi.ini
   ```
   - Create the external file with:
   ```ini
   [API_KEY]
   key = your_actual_api_key_here
   ```

3. **Get API Key**:
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Add the key using one of the methods above

## Inputs

### Required
- **prompt**: Text description of the image to generate or edit (supports multiline)
- **model**: Gemini model to use
  - `gemini-2.5-flash-image-preview` (recommended)
- **ratio**: Aspect ratio of the output image
  - `1:1` (square)
  - `16:9` (widescreen)
  - `9:16` (portrait)
  - `4:3` (standard)
  - `3:4` (portrait standard)
- **megapixels**: Output image size in megapixels
  - `1 MP (≈1024x1024)` to `15 MP (≈3920x3920)`

### Optional
- **prompt_template**: Choose from ready-made prompt templates or use your own prompt
  - `Use Own Prompt` (default) - Uses the text from the prompt input field
  - `Style Transfer` - Transform to impressionist painting style
  - `Color Enhancement` - Enhance colors and saturation
  - `Background Change` - Replace background with studio setup
  - `Object Removal` - Remove distracting background objects
  - `Lighting Adjustment` - Improve lighting and illumination
  - `Composition Edit` - Improve framing and visual balance
  - `Filter Application` - Apply professional photography filters
  - `Resolution Enhancement` - Improve quality and sharpness
  - `Face Enhancement` - Professional portrait retouching
  - `Texture Change` - Change surface material properties
  - `Mood Change` - Change atmosphere and ambiance
  - `Time of Day` - Change to golden hour lighting
  - `Weather Change` - Add dramatic weather effects
  - `Season Change` - Transform seasonal appearance
  - `Age Progression` - Make subject appear younger
  - `Clothing Change` - Change outfit and attire
  - `Hair Style` - Change hairstyle and texture
  - `Makeup Change` - Apply cosmetic enhancements
  - `Pose Change` - Adjust body posture
  - `Expression Change` - Change facial expression
  - `Multiple Images` - Combine input images seamlessly
  - `Collage Creation` - Create artistic collage layout
  - `Before After` - Create transformation comparison
  - `Product Enhancement` - Enhance product presentation
  - `Logo Addition` - Add branding elements
  - `Text Overlay` - Add text and typography
  - `Border Addition` - Add decorative borders
  - `Frame Addition` - Add picture frame effects
  - `Art Style` - Convert to digital art style
- **image1** to **image5**: Input images for editing (up to 5 images)

## Outputs

- **image**: Generated or edited image
- **prompt**: The final prompt text that was sent to the Gemini API (useful for debugging or reuse)

## Usage Examples

### Using Ready-Made Prompt Templates
1. Connect the node to your workflow
2. Select a template from the `prompt_template` dropdown (e.g., "Landscape: Serene mountain lake...")
3. The template will automatically populate the prompt sent to Gemini
4. Optionally connect the `prompt` output to see the final prompt text

### Custom Prompt with Templates
1. Set `prompt_template` to "Use Own Prompt"
2. Enter your custom prompt in the `prompt` input field
3. The node will use your custom prompt as entered

### Image Editing
1. Connect an input image to `image1`
2. Set a prompt describing the desired edit: "Add a sunset sky in the background"
3. Select model, ratio, and megapixels
4. Execute the node

### Multi-Image Editing
1. Connect multiple input images to `image1`, `image2`, etc.
2. Set a prompt that references the images: "Create a collage combining these images"
3. Execute the node

## Notes

- When providing input images, the node switches to editing mode
- Without input images, it generates new images from scratch
- All generated images include Gemini's SynthID watermark
- The node automatically resizes output to match the selected megapixels and ratio
- Supports up to 15 MP for high-resolution outputs

## Troubleshooting

- **"API key not found"**: Check your `googleapi.ini` configuration and ensure the API key is set correctly
- **"google-generativeai package not installed"**: Install with `pip install google-generativeai`
- **"Gemini API error"**: Verify API key validity, quota limits, and internet connection
- **No image in response**: Try a different prompt, or check if the model refused the content
- **Invalid ratio/megapixels**: Ensure the selected options match the expected format

## Technical Details

- Built using Google's official `google-generativeai` SDK
- Uses Gemini 2.5 Flash with native image generation
- Supports multimodal input (text + images)
- Automatic tensor ↔ PIL image conversion
- Error handling for API failures and missing dependencies
- Flexible API key management with local or external storage options
