import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageOps, ImageChops, ImageFilter
import os
import datetime

# Common Windows fonts to include
COMMON_FONTS = [
    "arial.ttf", "arialbd.ttf", "ariali.ttf", "ariblk.ttf",
    "calibri.ttf", "calibrib.ttf", "calibrii.ttf", "calibriz.ttf",
    "cambria.ttc", "cambriab.ttf", "cambriai.ttf", "cambriaz.ttf",
    "comic.ttf", "comicbd.ttf", "consola.ttf", "consolab.ttf",
    "consolai.ttf", "consolaz.ttf", "cour.ttf", "courbd.ttf",
    "couri.ttf", "courbi.ttf", "georgia.ttf", "georgiab.ttf",
    "georgiai.ttf", "georgiaz.ttf", "impact.ttf", "segoeui.ttf",
    "segoeuib.ttf", "segoeuii.ttf", "segoeuiz.ttf", "times.ttf",
    "timesbd.ttf", "timesi.ttf", "timesbi.ttf", "trebuc.ttf",
    "trebucbd.ttf", "trebucit.ttf", "trebucbi.ttf", "verdana.ttf",
    "verdanab.ttf", "verdanai.ttf", "verdanaz.ttf"
]

class StarWatermark:
    """
    ComfyUI Node: StarWatermark
    Adds a text or image watermark to single or batch images with advanced customization.
    """
    def __init__(self):
        # Preset font directory (can be expanded)
        self.font_dir = os.path.join(os.path.dirname(__file__), "fonts")
        if not os.path.exists(self.font_dir):
            os.makedirs(self.font_dir)
    
    @classmethod
    def get_system_fonts(cls):
        """Get a list of available system fonts"""
        # Start with arial as default option
        fonts = ["arial.ttf"]
        
        try:
            # Try to find the Windows fonts directory
            if os.name == 'nt':  # Windows
                # Standard Windows font directory paths to check
                possible_font_dirs = [
                    os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), "Fonts"),
                    os.path.join(os.environ.get('SYSTEMROOT', 'C:\\Windows'), "Fonts"),
                    "C:\\Windows\\Fonts"
                ]
                
                # Find the first valid font directory
                fonts_dir = None
                for dir_path in possible_font_dirs:
                    if os.path.exists(dir_path) and os.path.isdir(dir_path):
                        fonts_dir = dir_path
                        break
                
                if fonts_dir:
                    # Check which common fonts actually exist
                    for font_name in COMMON_FONTS:
                        font_path = os.path.join(fonts_dir, font_name)
                        if os.path.exists(font_path):
                            fonts.append(font_name)
                    
                    # Try to find additional fonts using glob (if available)
                    try:
                        import glob
                        font_extensions = ['*.ttf', '*.ttc', '*.otf']
                        for ext in font_extensions:
                            found_files = glob.glob(os.path.join(fonts_dir, ext))
                            for font_file in found_files:
                                font_name = os.path.basename(font_file)
                                if font_name not in fonts:  # Avoid duplicates
                                    fonts.append(font_name)
                    except ImportError:
                        # If glob is not available, we still have the common fonts
                        pass
                else:
                    # If we couldn't find the fonts directory, just use the common fonts list
                    fonts.extend(COMMON_FONTS)
            else:  # Linux/Mac - simplified approach
                # Add common fonts as fallback
                fonts.extend([
                    "Arial.ttf", "Arial Bold.ttf", "Arial Italic.ttf",
                    "Times New Roman.ttf", "Times New Roman Bold.ttf",
                    "Verdana.ttf", "Verdana Bold.ttf", "Courier.ttf"
                ])
        except Exception as e:
            print(f"Error enumerating system fonts: {e}")
            # If all else fails, add the common fonts anyway
            fonts.extend(COMMON_FONTS)
        
        # Remove duplicates and sort
        fonts = sorted(list(set(fonts)))
        return fonts

    @classmethod
    def INPUT_TYPES(cls):
        # Get system fonts for the dropdown using the class method
        system_fonts = cls.get_system_fonts()
        
        return {
            "required": {
                "images": ("IMAGE",),
                "text": ("STRING", {"default": "Sample Watermark"}),  # Single-line, compact, moved above mode
                "mode": (["text", "image"], {"default": "text"}),
                "image_scale": ("FLOAT", {"default": 100.0, "min": 1.0, "max": 500.0, "step": 1.0}),  # New: scale watermark image %
                "repeat_pattern": ("BOOLEAN", {"default": False}),  # New option for repeating/tile watermark
                "opacity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "position": (["center", "top", "bottom", "left", "right", "top_left", "top_right", "bottom_left", "bottom_right", "custom"], {"default": "bottom_right"}),
                "x_offset": ("INT", {"default": 10, "min": -10000, "max": 10000}),
                "y_offset": ("INT", {"default": 10, "min": -10000, "max": 10000}),
                "rotation": ("INT", {"default": 0, "min": -180, "max": 180}),
                "blend_mode": (["normal", "multiply", "overlay"], {"default": "normal"}),
                "output_mask": ("BOOLEAN", {"default": False}),
                "adaptive_color": ("BOOLEAN", {"default": False}),
                "randomize_per_batch": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "watermark_image": ("IMAGE",),
                "watermark_mask": ("MASK",),
                #"text": ("STRING", {"default": "Sample Watermark", "multiline": True}),
                "font": (cls.get_system_fonts(), {"default": "arial.ttf"}),
                "font_size": ("INT", {"default": 32, "min": 1, "max": 1000}),
                "font_color": ("STRING", {"default": "#FFFFFF"}),
                "font_bold": ("BOOLEAN", {"default": False}),
                "font_italic": ("BOOLEAN", {"default": False}),
                "font_underline": ("BOOLEAN", {"default": False}),
                "effect_outline": ("BOOLEAN", {"default": False}),
                "effect_shadow": ("BOOLEAN", {"default": False}),
                "effect_glow": ("BOOLEAN", {"default": False}),
                "insert_datetime": ("BOOLEAN", {"default": False}),
                "invert_mask": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("watermarked", "mask")
    FUNCTION = "apply_watermark"
    CATEGORY = "⭐StarNodes/Image And Latent"

    def apply_watermark(self, images, mode, opacity, position, x_offset, y_offset, rotation, blend_mode, output_mask, adaptive_color, randomize_per_batch, repeat_pattern=False, watermark_image=None, watermark_mask=None, text=None, font=None, font_size=None, font_color=None, font_bold=None, font_italic=None, font_underline=None, effect_outline=None, effect_shadow=None, effect_glow=None, insert_datetime=None, invert_mask=None, image_scale=100.0):
        import torch
        
        # Set defaults for optional parameters
        if text is None:
            text = "Sample Watermark"
        if font_size is None:
            font_size = 32
        if font_color is None:
            font_color = "#FFFFFF"
        
        # Convert input to batch if it's a single image
        if hasattr(images, 'ndim') and images.ndim == 3:
            images = images[None, ...]
        
        # Convert tensor to numpy if needed
        if 'torch' in str(type(images)):
            images = images.cpu().numpy()
        
        batch = []
        masks = []
        
        for idx, img_np in enumerate(images):
            # Convert tensor to numpy if needed
            if 'torch' in str(type(img_np)):
                img_np = img_np.cpu().numpy()
            
            # Convert numpy array to PIL Image
            img = Image.fromarray(np.clip(img_np * 255, 0, 255).astype(np.uint8))
            
            # Create watermark based on mode
            if mode == "text" and text:
                wm_img, wm_mask = self._create_text_watermark(
                    size=img.size, text=text, font=font, font_size=font_size, font_color=font_color, font_bold=font_bold, font_italic=font_italic, font_underline=font_underline, effect_outline=effect_outline, effect_shadow=effect_shadow, effect_glow=effect_glow, insert_datetime=insert_datetime, adaptive_color=adaptive_color
                )
            elif mode == "image" and watermark_image is not None:
                wm_img, wm_mask = self._prepare_image_watermark(
                    watermark_image=watermark_image, watermark_mask=watermark_mask, target_size=img.size, invert_mask=invert_mask
                )
                # Scale watermark image and mask if needed
                if image_scale != 100.0 and wm_img is not None:
                    scale_factor = image_scale / 100.0
                    new_size = (max(1, int(wm_img.width * scale_factor)), max(1, int(wm_img.height * scale_factor)))
                    wm_img = wm_img.resize(new_size, Image.LANCZOS)
                    if wm_mask is not None:
                        wm_mask = wm_mask.resize(new_size, Image.LANCZOS)
            else:
                # Default to a simple text watermark if no valid mode or missing image
                wm_img, wm_mask = self._create_text_watermark(
                    img.size, "Watermark", None, 32, "#FFFFFF", False, False, False, False, False, False, False, False
                )
            
            # Ensure tuple is always returned
            if wm_img is None or wm_mask is None:
                wm_img = None if wm_img is None else wm_img
                wm_mask = None if wm_mask is None else wm_mask
            
            # Handle repeat_pattern tiling
            if repeat_pattern:
                if wm_img is not None:
                    tiled_img = Image.new("RGBA", img.size, (0, 0, 0, 0))
                    tiled_mask = Image.new("L", img.size, 0) if wm_mask is not None else None
                    w, h = wm_img.size
                    for x in range(0, img.size[0], w):
                        for y in range(0, img.size[1], h):
                            # Determine the region to paste/crop
                            paste_width = min(w, img.size[0] - x)
                            paste_height = min(h, img.size[1] - y)
                            if paste_width < w or paste_height < h:
                                # Crop watermark to fit remaining space at edge
                                cropped_wm = wm_img.crop((0, 0, paste_width, paste_height))
                                tiled_img.alpha_composite(cropped_wm, (x, y))
                                if wm_mask is not None:
                                    cropped_mask = wm_mask.crop((0, 0, paste_width, paste_height))
                                    tiled_mask.paste(cropped_mask, (x, y))
                            else:
                                tiled_img.alpha_composite(wm_img, (x, y))
                                if wm_mask is not None:
                                    tiled_mask.paste(wm_mask, (x, y))
                    wm_img = tiled_img
                    wm_mask = tiled_mask if wm_mask is not None else None
            
            # Apply watermark to image
            # --- Apply watermark to image ---
            if wm_img is not None:
                out_img, out_mask = self._apply_watermark_to_image(
                    img, wm_img, wm_mask, opacity, position, x_offset, y_offset, 
                    rotation, blend_mode, output_mask, randomize_per_batch
                )
            else:
                # If watermark creation failed, return original image and blank mask if needed
                out_img = img
                out_mask = Image.new("L", img.size, 0) if output_mask else None
            
            # Convert back to numpy arrays and normalize to 0-1 range
            batch.append(np.asarray(out_img).astype(np.float32) / 255.0)
            if out_mask is not None:
                masks.append(np.asarray(out_mask).astype(np.float32) / 255.0)
            elif output_mask:
                # Create empty mask if output_mask is True but no mask was generated
                empty_mask = np.zeros((*img_np.shape[:2], 1), dtype=np.float32)
                masks.append(empty_mask)
        
        # Stack images and masks into batches
        batch = np.stack(batch, axis=0)
        
        # Handle masks - create empty batch if no masks were generated
        if masks:
            masks = np.stack(masks, axis=0)
        else:
            if output_mask:
                # Create empty mask batch with same dimensions as image batch
                masks = np.zeros((*batch.shape[:3], 1), dtype=np.float32)
            else:
                # If output_mask is False, create a dummy mask
                masks = np.zeros_like(batch)
        
        # Convert to torch tensors for ComfyUI compatibility
        batch = torch.from_numpy(batch)
        masks = torch.from_numpy(masks)
        
        return (batch, masks)

    def _create_text_watermark(self, size, text, font, font_size, font_color, font_bold, font_italic, font_underline, effect_outline, effect_shadow, effect_glow, insert_datetime, adaptive_color):
        # Compose text with date/time if needed
        if insert_datetime:
            text = text + " " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Ensure we have text to render
        if not text or text.strip() == "":
            text = "Watermark"
        
        # Ensure we have a font size
        if font_size is None or font_size < 8:
            font_size = 32
        
        # Font loading
        try:
            # Define possible font directories for Windows
            possible_font_dirs = []
            if os.name == 'nt':  # Windows
                possible_font_dirs = [
                    os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), "Fonts"),
                    os.path.join(os.environ.get('SYSTEMROOT', 'C:\\Windows'), "Fonts"),
                    "C:\\Windows\\Fonts"
                ]
            
            # Try to find the selected font
            font_path = None
            font_obj = None
            
            # First check if it's in our custom font directory
            custom_font_path = os.path.join(self.font_dir, font)
            if os.path.exists(custom_font_path):
                try:
                    font_obj = ImageFont.truetype(custom_font_path, font_size)
                    font_path = custom_font_path
                except Exception as e:
                    print(f"Error loading custom font {font}: {e}")
            
            # Then check system font directories if custom font failed
            if font_obj is None and os.name == 'nt':
                for font_dir in possible_font_dirs:
                    if not os.path.exists(font_dir):
                        continue
                        
                    system_font_path = os.path.join(font_dir, font)
                    if os.path.exists(system_font_path):
                        try:
                            font_obj = ImageFont.truetype(system_font_path, font_size)
                            font_path = system_font_path
                            break
                        except Exception as e:
                            print(f"Error loading system font {font}: {e}")
            
            # If selected font couldn't be loaded, try Arial as fallback
            if font_obj is None:
                # Try Arial first (most common)
                if os.name == 'nt':
                    for font_dir in possible_font_dirs:
                        if not os.path.exists(font_dir):
                            continue
                            
                        arial_path = os.path.join(font_dir, "arial.ttf")
                        if os.path.exists(arial_path):
                            try:
                                font_obj = ImageFont.truetype(arial_path, font_size)
                                font_path = arial_path
                                print(f"Using Arial as fallback font")
                                break
                            except Exception:
                                pass
                
                # If Arial failed too, try other common fonts
                if font_obj is None:
                    fallback_fonts = ["calibri.ttf", "segoeui.ttf", "verdana.ttf", "tahoma.ttf"]
                    
                    for fb_font in fallback_fonts:
                        if font_obj is not None:
                            break
                            
                        for font_dir in possible_font_dirs:
                            if not os.path.exists(font_dir):
                                continue
                                
                            try_path = os.path.join(font_dir, fb_font)
                            if os.path.exists(try_path):
                                try:
                                    font_obj = ImageFont.truetype(try_path, font_size)
                                    font_path = try_path
                                    print(f"Using {fb_font} as fallback font")
                                    break
                                except Exception:
                                    pass
            
            # If all else fails, use default font
            if font_obj is None:
                print("Using default font as no system fonts were found")
                font_obj = ImageFont.load_default()
            # Log font loading result
            if font_path:
                print(f"Loaded font from {font_path}")
        except Exception as e:
            print(f"Font loading error: {e}")
            font_obj = ImageFont.load_default()
        
        # Create a temporary image to measure text dimensions
        # This approach allows us to handle text styling more accurately
        temp_img = Image.new('RGBA', (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)
        
        # Prepare styled text
        # Since PIL doesn't directly support bold/italic/underline through the font object,
        # we'll simulate these effects when drawing the text
        
        # Calculate text size to create appropriately sized image
        # Using getbbox() for modern PIL versions or getsize() for older versions
        try:
            bbox = font_obj.getbbox(text)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            try:
                text_width, text_height = font_obj.getsize(text)
            except Exception as e:
                print(f"Text size calculation error: {e}")
                # Fallback to a reasonable size if all else fails
                text_width, text_height = len(text) * font_size // 2, font_size * 2
        
        # Adjust dimensions for bold/italic effects
        if font_bold:
            text_width += int(font_size * 0.1)  # Add extra width for bold text
        if font_italic:
            text_width += int(font_size * 0.1)  # Add extra width for italic slant
        
        # Add padding and account for effects
        padding = max(font_size // 2, 10)
        if effect_outline or effect_shadow or effect_glow:
            padding += font_size // 4
        
        # Create watermark image with transparent background
        # Make sure the watermark is not larger than the target image
        target_width, target_height = size
        wm_width = min(text_width + padding * 2, target_width)
        wm_height = min(text_height + padding * 2, target_height)
        
        wm_img = Image.new("RGBA", (wm_width, wm_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(wm_img)
        
        # Adaptive color: sample background to determine contrast
        if adaptive_color:
            # For now, use a simple approach - could be improved to sample actual image
            bg_lum = 255  # Assume white for now
            font_color = "#111111" if bg_lum > 128 else "#FFFFFF"
        
        # Convert font_color from string to tuple if needed
        if isinstance(font_color, str):
            if font_color.startswith('#'):
                # Convert hex color to RGB
                font_color = font_color.lstrip('#')
                if len(font_color) == 3:
                    font_color = ''.join([c*2 for c in font_color])
                try:
                    r, g, b = tuple(int(font_color[i:i+2], 16) for i in (0, 2, 4))
                    font_color = (r, g, b, 255)  # Add alpha channel
                except Exception as e:
                    print(f"Color parsing error: {e}")
                    font_color = (255, 255, 255, 255)  # Default to white
            else:
                # Handle non-hex color strings
                font_color = (255, 255, 255, 255)  # Default to white
        
        # Position text in the center of the watermark image
        x, y = padding, padding
        
        # Apply effects
        if effect_shadow:
            shadow_color = (0, 0, 0, 255)  # Black with full opacity
            draw.text((x+2, y+2), text, font=font_obj, fill=shadow_color)
        
        if effect_outline:
            # Use contrasting color for outline
            if isinstance(font_color, tuple) and len(font_color) >= 3:
                # Invert the RGB components for contrast
                r, g, b = font_color[:3]
                outline_color = (255-r, 255-g, 255-b, 255)
            else:
                outline_color = (0, 0, 0, 255)  # Default to black
            
            # Draw outline by offsetting text in 8 directions
            for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                draw.text((x+dx, y+dy), text, font=font_obj, fill=outline_color)
        
        # Implement bold effect by drawing the text multiple times with slight offsets
        if font_bold:
            for offset in range(1, 2):
                draw.text((x+offset, y), text, font=font_obj, fill=font_color)
                draw.text((x, y+offset), text, font=font_obj, fill=font_color)
        
        # Draw main text
        draw.text((x, y), text, font=font_obj, fill=font_color)
        
        # Implement italic effect using PIL's transform method for proper slant
        if font_italic:
            # Create a new image with the text
            text_img = wm_img.copy()
            
            # Calculate shear matrix for italic effect
            # This creates a proper slant transformation for the text
            shear_factor = 0.3  # Adjust this value to control the slant amount
            width, height = text_img.size
            
            # Apply shear transformation (slant)
            # This matrix applies a horizontal shear which creates the italic effect
            m = (1, shear_factor, 0, 0, 1, 0)
            
            # We need a slightly larger image to accommodate the sheared text
            new_width = width + int(height * shear_factor)
            italicized = Image.new("RGBA", (new_width, height), (0, 0, 0, 0))
            
            # Apply the transformation and paste it centered in the new image
            transformed = text_img.transform(
                (width, height),
                Image.AFFINE,
                m,
                Image.BICUBIC
            )
            
            # Paste the transformed text into the new image
            italicized.paste(transformed, (0, 0))
            
            # Replace the original image with the italicized version
            wm_img = italicized
        
        # Implement underline effect
        if font_underline:
            underline_y = y + text_height - int(font_size * 0.1)  # Position slightly below text
            underline_thickness = max(1, int(font_size * 0.05))  # Scale thickness with font size
            
            # Draw the underline
            for i in range(underline_thickness):
                draw.line([(x, underline_y + i), (x + text_width, underline_y + i)], fill=font_color, width=1)
        
        # Apply glow effect if requested
        if effect_glow:
            try:
                # Create a blurred version of the text for the glow effect
                glow_img = wm_img.copy()
                glow_radius = max(1, font_size//10)  # Ensure minimum radius of 1
                glow_img = glow_img.filter(ImageFilter.GaussianBlur(radius=glow_radius))
                
                # Enhance the glow brightness
                glow_img = ImageEnhance.Brightness(glow_img).enhance(1.5)
                
                # Create a new image and composite the glow under the original text
                final_img = Image.new("RGBA", wm_img.size, (0, 0, 0, 0))
                final_img = Image.alpha_composite(final_img, glow_img)
                final_img = Image.alpha_composite(final_img, wm_img)
                wm_img = final_img
            except Exception as e:
                print(f"Glow effect error: {e}")
                # Continue without glow if there's an error
        
        # Create mask from alpha channel
        mask = wm_img.split()[-1]
        
        return wm_img, mask

    def _prepare_image_watermark(self, watermark_image, watermark_mask, target_size, invert_mask=False):
        # Handle the watermark image
        if watermark_image is None:
            return None, None
        
        # Convert tensor to numpy if needed
        wmi = watermark_image
        if 'torch' in str(type(wmi)):
            wmi = wmi.cpu().numpy()
        
        # Handle batch dimension if present
        if wmi.ndim == 4 and wmi.shape[0] == 1:  # If it's a batch of 1
            wmi = wmi[0]  # Take the first image from the batch
        
        # Ensure correct channel order (HWC for PIL)
        if wmi.shape[0] == 3 or wmi.shape[0] == 4:  # If in CHW format
            wmi = wmi.transpose(1, 2, 0)  # Convert to HWC
        
        # Convert to PIL Image
        wm_img = Image.fromarray(np.clip(wmi * 255, 0, 255).astype(np.uint8))
        wm_mask = None
        
        # Process mask if provided
        if watermark_mask is not None:
            print(f"Processing watermark mask with shape: {watermark_mask.shape if hasattr(watermark_mask, 'shape') else 'unknown'}")
            # Convert tensor to numpy if needed
            wmm = watermark_mask
            if 'torch' in str(type(wmm)):
                wmm = wmm.cpu().numpy()
            
            # Handle batch dimension if present
            if wmm.ndim == 4 and wmm.shape[0] == 1:  # If it's a batch of 1
                wmm = wmm[0]  # Take the first mask from the batch
            
            # Handle different mask formats
            # ComfyUI masks can be in various formats, we need to normalize them
            if wmm.ndim == 3:
                print(f"Mask has 3 dimensions: {wmm.shape}")
                # Check if it's a 3D tensor with shape (H, W, 1) or (1, H, W) or (C, H, W)
                if wmm.shape[0] == 1:  # (1, H, W) format
                    wmm = wmm[0]  # Convert to (H, W)
                    print(f"Converted (1, H, W) to (H, W): {wmm.shape}")
                elif wmm.shape[2] == 1:  # (H, W, 1) format
                    wmm = wmm[:, :, 0]  # Convert to (H, W)
                    print(f"Converted (H, W, 1) to (H, W): {wmm.shape}")
                elif wmm.shape[0] == 3 or wmm.shape[0] == 4:  # (C, H, W) format
                    # Take first channel and transpose
                    wmm = wmm[0]  # Convert to (H, W)
                    print(f"Converted (C, H, W) to (H, W): {wmm.shape}")
                else:
                    # Try to reshape based on target size
                    try:
                        h, w = target_size
                        if wmm.size == h * w:
                            wmm = wmm.reshape(h, w)
                            print(f"Reshaped to target size: {wmm.shape}")
                    except Exception as e:
                        print(f"Error reshaping mask: {e}")
            
            # Ensure we have a 2D array for the mask
            if wmm.ndim != 2:
                print(f"Mask still not 2D after processing: {wmm.shape}")
                # If we still don't have a 2D array, try to reshape or flatten
                try:
                    # Try to get a sensible shape based on the target size
                    h, w = target_size
                    if wmm.size == h * w:  # If total elements match target size
                        wmm = wmm.reshape(h, w)
                        print(f"Reshaped to match target size: {wmm.shape}")
                    else:
                        # Just use the first channel or create a blank mask
                        print(f"Warning: Mask shape {wmm.shape} not compatible with target size {target_size}. Creating blank mask.")
                        wmm = np.ones(target_size, dtype=np.float32)
                except Exception as e:
                    print(f"Error reshaping mask: {e}. Creating blank mask.")
                    wmm = np.ones(target_size, dtype=np.float32)
            
            # Convert to PIL Image
            try:
                # Ensure mask is properly normalized
                if wmm.max() > 1.0 + 1e-5:  # Allow for small floating point errors
                    wmm = wmm / 255.0
                
                wm_mask = Image.fromarray(np.clip(wmm * 255, 0, 255).astype(np.uint8))
                print(f"Created mask image with mode: {wm_mask.mode} and size: {wm_mask.size}")
                
                # If mask is RGB, convert to grayscale
                if wm_mask.mode != 'L':
                    wm_mask = wm_mask.convert('L')
                    print(f"Converted mask to grayscale mode: {wm_mask.mode}")
                
                # Resize mask to match watermark image if needed
                if wm_mask.size != wm_img.size:
                    print(f"Resizing mask from {wm_mask.size} to match watermark size {wm_img.size}")
                    wm_mask = wm_mask.resize(wm_img.size, Image.LANCZOS)
            except Exception as e:
                print(f"Error converting mask to PIL Image: {e}. Creating blank mask.")
                wm_mask = Image.new('L', wm_img.size, 255)  # White mask (fully opaque)

            # Invert the mask if requested
            if invert_mask and wm_mask is not None:
                from PIL import ImageOps
                wm_mask = ImageOps.invert(wm_mask)
            # Apply the mask as alpha channel to the watermark image
            if wm_img.mode != 'RGBA':
                wm_img = wm_img.convert('RGBA')
            wm_img.putalpha(wm_mask)
        else:
            # If no mask is provided, use the alpha channel of the watermark image if it has one
            if wm_img.mode == 'RGBA':
                wm_mask = wm_img.split()[-1]
                print(f"Using alpha channel from watermark as mask")
            else:
                # If no alpha channel, create a solid mask
                wm_mask = Image.new('L', wm_img.size, 255)
                print(f"Created default white mask of size {wm_mask.size}")
        # Always return a tuple
        return wm_img, wm_mask

    # Apply the watermark to an image
    def _apply_watermark_to_image(self, img, wm_img, wm_mask, opacity, position, x_offset, y_offset, rotation, blend_mode, output_mask, randomize_per_batch):
        import random
        if wm_img is None:
            return img, None
        
        # Ensure the watermark is in RGBA mode for proper blending
        if wm_img.mode != "RGBA":
            wm_img = wm_img.convert("RGBA")
        
        # Apply rotation if needed
        if rotation != 0:
            try:
                wm_img = wm_img.rotate(rotation, expand=True, resample=Image.BICUBIC)
                if wm_mask is not None:
                    wm_mask = wm_mask.rotate(rotation, expand=True, resample=Image.BICUBIC)
            except Exception as e:
                print(f"Rotation error: {e}")
        
        # Apply opacity adjustment
        if opacity < 1.0:
            try:
                # Create a copy to avoid modifying the original
                wm_img = wm_img.copy()
                # Get alpha channel
                r, g, b, alpha = wm_img.split()
                # Adjust alpha channel brightness to control opacity
                alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
                # Recombine channels
                wm_img = Image.merge("RGBA", (r, g, b, alpha))
            except Exception as e:
                print(f"Opacity adjustment error: {e}")
        
        # Get dimensions
        W, H = img.size
        w, h = wm_img.size
        
        # Handle randomization if enabled
        if randomize_per_batch:
            # Random position within image bounds
            max_x = max(0, W - w)
            max_y = max(0, H - h)
            x_offset = random.randint(0, max_x) if max_x > 0 else 0
            y_offset = random.randint(0, max_y) if max_y > 0 else 0
    def _apply_watermark_to_image(self, img, wm_img, wm_mask, opacity, position, x_offset, y_offset, rotation, blend_mode, output_mask, randomize_per_batch):
        import random
        if wm_img is None:
            return img, None
        
        # Ensure the watermark is in RGBA mode for proper blending
        if wm_img.mode != "RGBA":
            wm_img = wm_img.convert("RGBA")
        
        # Apply rotation if needed
        if rotation != 0:
            try:
                wm_img = wm_img.rotate(rotation, expand=True, resample=Image.BICUBIC)
                if wm_mask is not None:
                    wm_mask = wm_mask.rotate(rotation, expand=True, resample=Image.BICUBIC)
            except Exception as e:
                print(f"Rotation error: {e}")
        
        # Apply opacity adjustment
        if opacity < 1.0:
            try:
                # Create a copy to avoid modifying the original
                wm_img = wm_img.copy()
                # Get alpha channel
                r, g, b, alpha = wm_img.split()
                # Adjust alpha channel brightness to control opacity
                alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
                # Recombine channels
                wm_img = Image.merge("RGBA", (r, g, b, alpha))
            except Exception as e:
                print(f"Opacity adjustment error: {e}")
        
        # Get dimensions
        W, H = img.size
        w, h = wm_img.size
        
        # Handle randomization if enabled
        if randomize_per_batch:
            # Random position within image bounds
            max_x = max(0, W - w)
            max_y = max(0, H - h)
            x_offset = random.randint(0, max_x) if max_x > 0 else 0
            y_offset = random.randint(0, max_y) if max_y > 0 else 0
            
            # Random opacity variation
            if opacity < 1.0:
                random_opacity = random.uniform(max(0.1, opacity-0.2), min(1.0, opacity+0.2))
                try:
                    r, g, b, alpha = wm_img.split()
                    alpha = ImageEnhance.Brightness(alpha).enhance(random_opacity / opacity)  # Adjust relative to current opacity
                    wm_img = Image.merge("RGBA", (r, g, b, alpha))
                except Exception as e:
                    print(f"Random opacity adjustment error: {e}")
        
        # Calculate position based on selected placement option
        try:
            if position == "top_left":
                pos = (x_offset, y_offset)
            elif position == "top_right":
                pos = (max(0, W-w-x_offset), y_offset)
            elif position == "bottom_left":
                pos = (x_offset, max(0, H-h-y_offset))
            elif position == "bottom_right":
                pos = (max(0, W-w-x_offset), max(0, H-h-y_offset))
            elif position == "center":
                pos = (max(0, (W-w)//2 + x_offset), max(0, (H-h)//2 + y_offset))
            elif position == "top":
                pos = (max(0, (W-w)//2 + x_offset), y_offset)
            elif position == "bottom":
                pos = (max(0, (W-w)//2 + x_offset), max(0, H-h-y_offset))
            elif position == "left":
                pos = (x_offset, max(0, (H-h)//2 + y_offset))
            elif position == "right":
                pos = (max(0, W-w-x_offset), max(0, (H-h)//2 + y_offset))
            elif position == "custom":
                # Ensure position is within image bounds
                pos = (min(max(0, x_offset), W-w if w < W else 0), 
                       min(max(0, y_offset), H-h if h < H else 0))
            else:
                # Default to bottom right if position is not recognized
                print(f"Unrecognized position: {position}, defaulting to bottom_right")
                pos = (max(0, W-w-x_offset), max(0, H-h-y_offset))
        except Exception as e:
            print(f"Position calculation error: {e}")
            # Default to top-left if there's an error
            pos = (0, 0)
        
        # Handle tiling separately
        if position == "tile":
            try:
                # Convert image to RGBA for compositing
                base = img.convert("RGBA")
                
                # Tile the watermark across the entire image
                for x in range(0, W, w):
                    for y in range(0, H, h):
                        # Make sure we don't go out of bounds
                        if x < W and y < H:
                            # Calculate the portion of the watermark that fits
                            paste_width = min(w, W - x)
                            paste_height = min(h, H - y)
                            
                            if paste_width > 0 and paste_height > 0:
                                # Crop watermark if needed to fit at the edges
                                if paste_width < w or paste_height < h:
                                    cropped_wm = wm_img.crop((0, 0, paste_width, paste_height))
                                    base.alpha_composite(cropped_wm, dest=(x, y))
                                else:
                                    base.alpha_composite(wm_img, dest=(x, y))
                
                # Create mask if requested
                mask_out = None
                if output_mask:
                    mask_out = Image.new("L", img.size, 0)
                    alpha = wm_img.split()[-1]
                    
                    for x in range(0, W, w):
                        for y in range(0, H, h):
                            # Make sure we don't go out of bounds
                            if x < W and y < H:
                                paste_width = min(w, W - x)
                                paste_height = min(h, H - y)
                                
                                if paste_width > 0 and paste_height > 0:
                                    if paste_width < w or paste_height < h:
                                        cropped_alpha = alpha.crop((0, 0, paste_width, paste_height))
                                        mask_out.paste(cropped_alpha, (x, y))
                                    else:
                                        mask_out.paste(alpha, (x, y))
                
                return base.convert("RGB"), mask_out
            
            except Exception as e:
                print(f"Tiling error: {e}")
                # Fall back to non-tiled if tiling fails
                position = "center"
                pos = ((W-w)//2, (H-h)//2)
        
        # Apply the watermark using the appropriate blend mode
        try:
            # Convert base image to RGBA for compositing
            base = img.convert("RGBA")
            
            # Ensure position is valid (watermark at least partially within image)
            x, y = pos
            if x >= W or y >= H or x + w <= 0 or y + h <= 0:
                # Watermark would be completely outside image, use center position instead
                x = max(0, (W - w) // 2)
                y = max(0, (H - h) // 2)
                pos = (x, y)
            
            # Apply blend mode
            if blend_mode == "normal" or blend_mode is None:
                # Simple alpha compositing
                base.alpha_composite(wm_img, dest=pos)
            else:
                try:
                    # Calculate the region where the watermark will be placed
                    crop_x = max(0, x)
                    crop_y = max(0, y)
                    crop_right = min(W, x + w)
                    crop_bottom = min(H, y + h)
                    
                    if crop_right > crop_x and crop_bottom > crop_y:
                        # Get the region of the base image where the watermark will be placed
                        region = base.crop((crop_x, crop_y, crop_right, crop_bottom))
                        
                        # Calculate the corresponding region of the watermark
                        wm_x = max(0, -x)
                        wm_y = max(0, -y)
                        wm_region = wm_img.crop((wm_x, wm_y, wm_x + (crop_right - crop_x), wm_y + (crop_bottom - crop_y)))
                        
                        # Apply the selected blend mode
                        if blend_mode == "multiply":
                            blended = ImageChops.multiply(region, wm_region)
                        elif blend_mode == "overlay":
                            blended = ImageChops.overlay(region, wm_region)
                        else:  # Fallback to normal
                            blended = wm_region
                        
                        # Paste the blended region back into the base image
                        base.paste(blended, (crop_x, crop_y), wm_region)
                except Exception as e:
                    print(f"Blend mode error: {e}")
                    # Fallback to normal blend mode
                    base.alpha_composite(wm_img, dest=pos)
            
            # Create mask if requested
            mask_out = None
            if output_mask:
                try:
                    mask_out = Image.new("L", img.size, 0)
                    alpha = wm_img.split()[-1]
                    
                    # Calculate paste coordinates that are within bounds
                    paste_x = max(0, x)
                    paste_y = max(0, y)
                    
                    # Calculate the region of the alpha channel to use
                    alpha_x = max(0, -x)
                    alpha_y = max(0, -y)
                    alpha_width = min(w - alpha_x, W - paste_x)
                    alpha_height = min(h - alpha_y, H - paste_y)
                    
                    if alpha_width > 0 and alpha_height > 0:
                        alpha_region = alpha.crop((alpha_x, alpha_y, alpha_x + alpha_width, alpha_y + alpha_height))
                        mask_out.paste(alpha_region, (paste_x, paste_y))
                except Exception as e:
                    print(f"Mask creation error: {e}")
            
            return base.convert("RGB"), mask_out
        
        except Exception as e:
            print(f"Watermark application error: {e}")
            # Return original image if watermark application fails
            return img, None


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_CLASS_MAPPINGS["StarWatermark"] = StarWatermark
NODE_DISPLAY_NAME_MAPPINGS["StarWatermark"] = "⭐ Star Watermark"
