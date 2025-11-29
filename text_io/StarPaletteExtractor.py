import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Dict
import webcolors
import colorsys

class StarPaletteExtractor:
    """
    Extracts the dominant color palette from an image and creates a preview image with square tiles for each color.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "num_colors": ("INT", {"default": 8, "min": 2, "max": 32}),
                "tile_size": ("INT", {"default": 128, "min": 32, "max": 512}),
                "color_format": (["All", "Hex", "RGB", "CMYK"], {"default": "All"}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE", "STRING") + tuple(["STRING"]*10)
    RETURN_NAMES = ("palette", "palette_image", "Colors") + tuple([f"color{i+1}" for i in range(10)])
    FUNCTION = "extract_palette"
    CATEGORY = "⭐StarNodes/Color"

    @staticmethod
    def extract_palette(image, num_colors: int = 8, tile_size: int = 64, color_format: str = "All"):
        # Handle input: torch.Tensor or np.ndarray
        try:
            import torch
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
        except ImportError:
            pass
        # Ensure image is np.ndarray and scale if needed
        arr_img = image[0]
        if arr_img.dtype != np.uint8:
            arr_img = (arr_img * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(arr_img)
        # Resize for speed
        small_img = pil_img.resize((128, 128), resample=Image.LANCZOS)
        arr = np.array(small_img).reshape(-1, 3)
        # KMeans clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_colors, n_init=5, random_state=42)
        labels = kmeans.fit_predict(arr)
        palette = kmeans.cluster_centers_.astype(np.uint8)
        hex_colors = ['#%02x%02x%02x' % tuple(c) for c in palette]
        
        # Color info extraction functions (removed color names for better accuracy)
        
        def rgb_to_hex(rgb):
            return '#{:02x}{:02x}{:02x}'.format(*rgb)
        
        def rgb_to_rgb_str(rgb):
            return f'RGB({rgb[0]}, {rgb[1]}, {rgb[2]})'
        
        def rgb_to_cmyk(rgb):
            r, g, b = [x/255.0 for x in rgb]
            k = 1 - max(r, g, b)
            if k == 1:
                return (0, 0, 0, 100)  # Black
            c = (1 - r - k) / (1 - k) * 100
            m = (1 - g - k) / (1 - k) * 100
            y = (1 - b - k) / (1 - k) * 100
            k = k * 100
            return (round(c), round(m), round(y), round(k))
        
        def rgb_to_cmyk_str(rgb):
            c, m, y, k = rgb_to_cmyk(rgb)
            return f'CMYK({c}%, {m}%, {y}%, {k}%)'  
        
        def get_all_color_info(rgb):
            hex_val = rgb_to_hex(rgb)
            rgb_str = rgb_to_rgb_str(rgb)
            cmyk_str = rgb_to_cmyk_str(rgb)
            return {
                "hex": hex_val,
                "rgb": rgb_str,
                "cmyk": cmyk_str
            }
        
        # Build palette string with all color information
        palette_lines = ["Dominant Colors of the Image:"]
        color_details = []
        
        for idx, rgb in enumerate(palette):
            rgb_tuple = tuple(rgb.tolist())
            color_info = get_all_color_info(rgb_tuple)
            color_details.append(color_info)
            
            # Format line based on selected color format
            if color_format == "Hex":
                line = f"Color {idx+1}: {color_info['hex']}"
            elif color_format == "RGB":
                line = f"Color {idx+1}: {color_info['rgb']}"
            elif color_format == "CMYK":
                line = f"Color {idx+1}: {color_info['cmyk']}"
            else:  # "All"
                line = f"Color {idx+1}: {color_info['hex']} | {color_info['rgb']} | {color_info['cmyk']}"
            
            palette_lines.append(line)
        
        palette_str = "\n".join(palette_lines)
        
        # Create palette image as a more landscape-oriented grid
        import math
        
        # Calculate grid dimensions - try to make it more horizontal
        aspect_ratio = 1.5  # Target wider than tall
        total_area = num_colors
        grid_cols = max(1, math.ceil(math.sqrt(total_area * aspect_ratio)))
        grid_rows = math.ceil(num_colors / grid_cols)
        
        # Adjusted caption height for 3 lines of text
        caption_height = int(tile_size * 0.5)  # Shorter caption height
        
        # Calculate tile dimensions - make tiles landscape oriented
        tile_width = tile_size
        tile_height = int(tile_size * 0.75)  # Make tiles wider than tall
        
        img_width = grid_cols * tile_width
        img_height = grid_rows * (tile_height + caption_height)
        palette_img = Image.new("RGBA", (img_width, img_height), (255, 255, 255, 255))
        draw = ImageDraw.Draw(palette_img)
        
        # Setup smaller fonts to fit text better
        try:
            # Adjust font sizes for landscape layout
            font = ImageFont.truetype("arial.ttf", max(8, tile_width // 10))
            small_font = ImageFont.truetype("arial.ttf", max(7, tile_width // 12))
        except Exception:
            font = ImageFont.load_default()
            small_font = font
            
        for idx, color in enumerate(palette):
            row = idx // grid_cols
            col = idx % grid_cols
            x0 = col * tile_width
            y0 = row * (tile_height + caption_height)
            
            # Draw color rectangle (landscape oriented)
            draw.rectangle([x0, y0, x0 + tile_width, y0 + tile_height], fill=tuple(color.tolist()) + (255,))
            
            # Draw white rectangle for caption background
            draw.rectangle([x0, y0 + tile_height, x0 + tile_width, y0 + tile_height + caption_height], 
                          fill=(255, 255, 255, 255))
            
            # Get color information
            rgb_tuple = tuple(color.tolist())
            color_info = color_details[idx]
            
            # Draw color information with shorter format and no redundancy
            # Note: In the image display, we strip the RGB/CMYK prefixes but keep the values
            rgb_display = color_info['rgb'].replace('RGB', '').strip()
            cmyk_display = color_info['cmyk'].replace('CMYK', '').strip()
            
            text_items = [
                ("Hex: " + color_info['hex'], 0),
                ("RGB: " + rgb_display, 1),
                ("CMYK: " + cmyk_display, 2)
            ]
            
            for text_item, line_num in text_items:
                try:
                    bbox = draw.textbbox((0, 0), text_item, font=small_font)
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                except AttributeError:
                    # Fallback for older PIL versions
                    try:
                        w, h = small_font.getsize(text_item)
                    except:
                        w, h = (tile_size // 2, tile_size // 12)
                        
                line_height = max(h + 1, caption_height // 3)  # Distribute space evenly
                text_x = x0 + 2  # Smaller left padding
                text_y = y0 + tile_height + (line_num * line_height)
                
                # Ensure text fits within the tile width
                if w > tile_width - 4:  # If text is too wide, try even smaller font
                    try:
                        tiny_font = ImageFont.truetype("arial.ttf", max(6, tile_width // 16))
                        draw.text((text_x, text_y), text_item, fill=(0, 0, 0), font=tiny_font)
                    except:
                        draw.text((text_x, text_y), text_item, fill=(0, 0, 0), font=small_font)
                else:
                    draw.text((text_x, text_y), text_item, fill=(0, 0, 0), font=small_font)

        # Output as torch tensor (B, H, W, C), float32, 0-1
        try:
            import torch
            palette_img_tensor = torch.from_numpy(np.array(palette_img)).float() / 255.0
            palette_img_tensor = palette_img_tensor.unsqueeze(0)  # add batch dim
        except ImportError:
            # fallback: still return numpy, but warn
            palette_img_tensor = np.array(palette_img)[None, ...] / 255.0
            
        # Prepare color outputs based on selected format (up to 10)
        color_outputs = []
        for i in range(10):
            if i < len(palette):
                rgb_tuple = tuple(palette[i].tolist())
                color_info = color_details[i]
                
                if color_format == "Hex":
                    output = color_info['hex']
                elif color_format == "RGB":
                    output = color_info['rgb']
                elif color_format == "CMYK":
                    output = color_info['cmyk']
                else:  # "All"
                    output = f"{color_info['hex']} | {color_info['rgb']} | {color_info['cmyk']}"
                    
                color_outputs.append(output)
            else:
                color_outputs.append("")
                
        return (palette_str, palette_img_tensor, palette_str, *color_outputs)

NODE_CLASS_MAPPINGS = {
    "StarPaletteExtractor": StarPaletteExtractor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarPaletteExtractor": "⭐ Star Palette Extractor"
}