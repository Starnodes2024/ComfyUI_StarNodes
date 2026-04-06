# StarNodes v1.9.9 Release Notes

## New Node: ⭐ Star Flux2 Conditioner

A powerful conditioning node for Flux2 models that supports text encoding and multiple reference images.

### Features
- Text prompt encoding with CLIP
- Support for up to 5 reference images
- **Join References** option (default: true) - combines images 2-5 into a 2x2 grid
- Automatic image resizing to 1 megapixel
- VAE encoding of reference images
- Reference latents injection for Flux2 edit models

### Join References Mode
When enabled, images 2-5 are combined into a single 2x2 grid:
- Each cell is 1024x1024 pixels
- Images are scaled to fit while preserving aspect ratio
- White padding centers images in their cells
- Empty cells filled with white
- Final 2048x2048 grid resized to 1MP before encoding

### Files Included
- `star_flux2_conditioner.py` - Main node implementation
- `__init__.py` - Updated with node registration
- `README.md` - Updated with new node documentation
- `pyproject.toml` - Version updated to 1.9.9
- `web/docs/StarFlux2Conditioner.md` - Detailed node documentation

### Usage
1. Connect CLIP and VAE models
2. Enter your text prompt
3. Optionally connect 1-5 reference images
4. Toggle "Join References" based on your needs:
   - **True**: Combines images 2-5 into a grid (easier for models to process)
   - **False**: Processes each image separately
5. Use POS output for positive conditioning
6. Use NEG output for negative conditioning

### Technical Details
- Category: ⭐StarNodes/Conditioning
- Outputs: POS (positive conditioning), NEG (negative conditioning)
- Reference latents injected using `reference_latents` key
- Compatible with Flux2, SDXL, and other reference-aware models

## Bug Fixes
- Fixed tensor dimension mismatch errors in grid creation
- Proper batch dimension handling
- Consistent tensor sizes for model compatibility

## Version
- Version: 1.9.9
- All version files updated
