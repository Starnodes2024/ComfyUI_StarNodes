import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math
import json
import os

class StarGridComposer:
    """
    Composes multiple images into a grid layout with automatic sizing.
    - Supports batch input or individual image inputs
    - Automatically adjusts grid cell sizes to maintain aspect ratio
    - Dynamic input handling via JavaScript
    """
    CATEGORY = '⭐StarNodes/Image And Latent'
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "compose_grid"
    OUTPUT_NODE = False
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get available system fonts
        font_list = ["arial", "times", "courier", "calibri", "verdana", "tahoma"]
        try:
            # Try to get actual system fonts if possible
            import matplotlib.font_manager as fm
            fonts = fm.findSystemFonts()
            system_fonts = [os.path.splitext(os.path.basename(f))[0].lower() for f in fonts]
            # Remove duplicates and sort
            system_fonts = sorted(list(set(system_fonts)))
            if system_fonts:
                font_list = system_fonts
        except:
            pass
            
        # Use a more compact input structure
        return {
            "required": {
                # Grid settings
                "max_width": ("INT", {"default": 2048, "min": 64, "max": 8192, "step": 8}),
                "max_height": ("INT", {"default": 2048, "min": 64, "max": 8192, "step": 8}),
                "cols": ("INT", {"default": 2, "min": 1, "max": 4, "step": 1}),
                "rows": ("INT", {"default": 2, "min": 1, "max": 4, "step": 1}),
                
                # Image inputs - connect to batcher nodes
                "Grid Image Batch": ("IMAGE", {"forceInput": True}),  # Single image input that can accept batches
                
                # Caption settings
                "add_caption": ("BOOLEAN", {"default": False}),
                "caption_font_size": ("INT", {"default": 80, "min": 8, "max": 500, "step": 1}),
                "main_caption_font_size": ("INT", {"default": 120, "min": 12, "max": 500, "step": 1}),
                
                # Caption style options
                "caption_bg_color": (["black", "white", "gray", "red", "green", "blue", "transparent"], {"default": "black"}),
                "caption_text_color": (["white", "black", "gray", "red", "green", "blue", "yellow"], {"default": "white"}),
                "font_family": (font_list, {"default": "arial"}),
                "font_bold": ("BOOLEAN", {"default": False}),
                "main_caption_bg_color": (["black", "white", "gray", "red", "green", "blue", "transparent"], {"default": "black"}),
                "main_caption_text_color": (["white", "black", "gray", "red", "green", "blue", "yellow"], {"default": "white"}),
            },
            "optional": {
                "input_caption": ("STRING", {"multiline": True, "default": ""}),  # Optional main caption
                "Grid Captions Batch": ("STRING", {"forceInput": True}),  # Optional caption input for all images
            }
        }
        
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Force reprocess when any input changes
        return float("nan")
        
    @classmethod
    def VALIDATE_INPUTS(s, **kwargs):
        return True
        
    # Removed RETURN_TYPES and FUNCTION methods as they're now class variables


    def compose_grid(self, max_width, max_height, cols, rows, add_caption=False, 
                 caption_font_size=14, main_caption_font_size=32, caption_bg_color="black", 
                 caption_text_color="white", font_family="arial", font_bold=False,
                 main_caption_bg_color="black", main_caption_text_color="white", 
                 input_caption="", **kwargs):
        # Get the image batch and captions batch from kwargs
        images = kwargs.get("Grid Image Batch")
        captions = kwargs.get("Grid Captions Batch", "")
        from PIL import ImageFont
        
        # Convert color names to RGB tuples
        color_map = {
            "black": (0, 0, 0),
            "white": (255, 255, 255),
            "gray": (128, 128, 128),
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
            "transparent": None
        }
        caption_bg_color_rgb = color_map.get(caption_bg_color)
        caption_font_color = color_map.get(caption_text_color)
        main_caption_bg_color_rgb = color_map.get(main_caption_bg_color)
        main_caption_font_color = color_map.get(main_caption_text_color)
        
        # Load font
        try:
            import os
            from matplotlib import font_manager
            font_path = None
            for f in font_manager.findSystemFonts():
                if os.path.splitext(os.path.basename(f))[0].lower() == font_family.lower():
                    font_path = f
                    break
            
            if font_path:
                if font_bold:
                    # Try to find bold version
                    font_dir = os.path.dirname(font_path)
                    font_name = os.path.splitext(os.path.basename(font_path))[0]
                    bold_suffixes = ["bold", "bd", "b"]
                    
                    for suffix in bold_suffixes:
                        bold_path = os.path.join(font_dir, f"{font_name}{suffix}.ttf")
                        if os.path.exists(bold_path):
                            font_path = bold_path
                            break
                
                font = ImageFont.truetype(font_path, caption_font_size)
                main_font = ImageFont.truetype(font_path, main_caption_font_size)
            else:
                font = ImageFont.truetype(font_family, caption_font_size)
                main_font = ImageFont.truetype(font_family, main_caption_font_size)
        except:
            # Fallback to default font
            font = ImageFont.load_default()
            main_font = ImageFont.load_default()
        
        # Calculate caption height
        caption_margin = 1
        caption_height = font.getbbox("Ag")[3] + caption_margin * 2
        main_caption_margin = 0
        
        # Prepare images and captions
        total_slots = cols * rows
        image_list = []
        caption_list = []
        
        # Process the input images
        if images is not None:
            # Check if it's a batch of images
            if len(images.shape) == 4:  # [B, H, W, C] format
                batch_size = images.shape[0]
                for i in range(min(batch_size, total_slots)):
                    image_list.append(images[i:i+1])
            else:  # Single image
                image_list.append(images)
        
        # Fill remaining slots with None
        while len(image_list) < total_slots:
            image_list.append(None)
        
        # Process captions
        if isinstance(captions, str):
            # Split by newlines or commas
            if '\n' in captions:
                caption_parts = captions.split('\n')
            else:
                caption_parts = captions.split(',')
            
            # Clean up and add captions
            for i, part in enumerate(caption_parts):
                if i < total_slots:
                    caption_list.append(part.strip())
        
        # Fill remaining captions with empty strings
        while len(caption_list) < total_slots:
            caption_list.append("")
        
        # Limit to total_slots
        image_list = image_list[:total_slots]
        caption_list = caption_list[:total_slots]
        
        # Reserve space for captions under images
        cell_caption_height = caption_height if any(bool(c) for c in caption_list) else 0
        # If main caption is to be added, reserve space at bottom
        add_main_caption = add_caption and input_caption.strip() != ""
        main_caption_height = main_font.getbbox("Ag")[3] + main_caption_margin * 2 if add_main_caption else 0
        
        # Calculate cell dimensions
        cell_width = max_width // cols
        cell_height = (max_height - main_caption_height) // rows - cell_caption_height
        # Create output image
        out_height = rows * (cell_height + cell_caption_height) + main_caption_height
        result = Image.new('RGB', (cols * cell_width, out_height), (0, 0, 0))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(result)
        # Place each image in the grid
        for i in range(total_slots):
            img_tensor = image_list[i]
            if img_tensor is None:
                continue
            # Convert tensor to PIL Image
            img = img_tensor.cpu().numpy()
            img = (img * 255).astype(np.uint8)
            if len(img.shape) == 4:
                img = img[0]  # Remove batch dimension if present
            if img.shape[0] == 1:
                img = img.transpose(1, 2, 0)  # CHW to HWC for grayscale
            elif img.shape[0] == 3:
                img = img.transpose(1, 2, 0)  # CHW to HWC for RGB
            img_pil = Image.fromarray(img)
            # Calculate position in grid
            row = i // cols
            col = i % cols
            # Calculate position and size maintaining aspect ratio
            img_ratio = img_pil.width / img_pil.height
            cell_ratio = cell_width / cell_height
            if img_ratio > cell_ratio:
                new_width = cell_width
                new_height = int(cell_width / img_ratio)
            else:
                new_height = cell_height
                new_width = int(cell_height * img_ratio)
            img_pil = img_pil.resize((new_width, new_height), Image.LANCZOS)
            x = col * cell_width + (cell_width - new_width) // 2
            y = row * (cell_height + cell_caption_height)
            result.paste(img_pil, (x, y))
            # Draw caption if present
            caption = caption_list[i]
            if caption:
                cx = col * cell_width
                cy = y + new_height
                # Draw caption background rectangle (if not transparent)
                if caption_bg_color_rgb is not None:
                    draw.rectangle([cx, cy, cx + cell_width, cy + cell_caption_height], fill=caption_bg_color_rgb)
                # Draw text centered
                text_w = font.getlength(caption)
                text_x = cx + (cell_width - text_w) // 2
                text_y = cy + caption_margin
                draw.text((text_x, text_y), caption, font=font, fill=caption_font_color)
        # Draw main caption if needed
        if add_main_caption:
            cy = out_height - main_caption_height
            # Draw main caption background (if not transparent)
            if main_caption_bg_color_rgb is not None:
                draw.rectangle([0, cy, cols * cell_width, out_height], fill=main_caption_bg_color_rgb)
            text = input_caption.strip()
            text_w = main_font.getlength(text)
            text_x = (cols * cell_width - text_w) // 2
            text_y = cy + main_caption_margin
            draw.text((text_x, text_y), text, font=main_font, fill=main_caption_font_color)
        result_np = np.array(result).astype(np.float32) / 255.0
        result_tensor = torch.from_numpy(result_np).unsqueeze(0)
        return (result_tensor,)


    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Force reprocess when any input changes
        return float("nan")

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "StarGridComposer": StarGridComposer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarGridComposer": "⭐ Star Grid Composer"
}
