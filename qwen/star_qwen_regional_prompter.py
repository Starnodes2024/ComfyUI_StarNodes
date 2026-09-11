"""
Star Qwen Regional Prompter Node
Simplified regional prompting for Qwen2-VL + Qwen-Image workflows.
Automatically divides the image into 4 quadrants for easy regional control.
"""


class StarQwenRegionalPrompter:
    """
    Simplified regional prompting node for Qwen2-VL CLIP + Qwen-Image.
    
    Automatically divides the image into 4 equal quadrants:
    - Upper Left, Upper Right, Lower Left, Lower Right
    
    Uses combined mode (spatial descriptions + grounding tokens) for best results.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", ),
                "background_prompt": ("STRING", {
                    "multiline": True, 
                    "dynamicPrompts": True,
                    "default": "A beautiful scene"
                }),
                "image_width": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 8,
                    "tooltip": "Width of the target image"
                }),
                "image_height": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 8,
                    "tooltip": "Height of the target image"
                }),
            },
            "optional": {
                "region_upper_left": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": True,
                    "default": "",
                    "tooltip": "Prompt for upper left quadrant (leave empty to skip)"
                }),
                "region_upper_right": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": True,
                    "default": "",
                    "tooltip": "Prompt for upper right quadrant (leave empty to skip)"
                }),
                "region_lower_left": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": True,
                    "default": "",
                    "tooltip": "Prompt for lower left quadrant (leave empty to skip)"
                }),
                "region_lower_right": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": True,
                    "default": "",
                    "tooltip": "Prompt for lower right quadrant (leave empty to skip)"
                }),
            },
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode_regional"
    CATEGORY = "⭐StarNodes/Conditioning"
    
    def normalize_coordinate(self, coord, max_val):
        """
        Normalize coordinate to [0, 1000) range as per Qwen2-VL specification.
        
        Args:
            coord: Original coordinate value
            max_val: Maximum value (image width or height)
            
        Returns:
            Normalized coordinate in [0, 1000) range
        """
        if max_val <= 0:
            return 0
        normalized = int((coord / max_val) * 1000)
        # Clamp to [0, 999]
        return max(0, min(999, normalized))
    
    def format_bbox(self, x1, y1, x2, y2, img_width, img_height):
        """
        Format bounding box coordinates in Qwen2-VL format.
        
        Args:
            x1, y1: Top-left corner coordinates
            x2, y2: Bottom-right corner coordinates
            img_width, img_height: Image dimensions for normalization
            
        Returns:
            Formatted bounding box string
        """
        # Normalize coordinates to [0, 1000) range
        norm_x1 = self.normalize_coordinate(x1, img_width)
        norm_y1 = self.normalize_coordinate(y1, img_height)
        norm_x2 = self.normalize_coordinate(x2, img_width)
        norm_y2 = self.normalize_coordinate(y2, img_height)
        
        # Format: <|box_start|>(x1,y1),(x2,y2)<|box_end|>
        return f"<|box_start|>({norm_x1},{norm_y1}),({norm_x2},{norm_y2})<|box_end|>"
    
    def encode_regional(self, clip, background_prompt, image_width, image_height,
                       region_upper_left="", region_upper_right="", 
                       region_lower_left="", region_lower_right=""):
        """
        Encode regional prompts using fixed quadrants with combined mode.
        
        The image is automatically divided into 4 equal quadrants:
        - Upper Left: (0, 0) to (width/2, height/2)
        - Upper Right: (width/2, 0) to (width, height/2)
        - Lower Left: (0, height/2) to (width/2, height)
        - Lower Right: (width/2, height/2) to (width, height)
        
        Uses combined mode: spatial descriptions + grounding tokens for best results.
        """
        
        # Start with background prompt
        full_prompt = background_prompt.strip()
        
        # Calculate quadrant dimensions
        half_width = image_width // 2
        half_height = image_height // 2
        
        # Define quadrants: (name, prompt, x, y, width, height, spatial_desc)
        quadrants = [
            ("Upper Left", region_upper_left, 0, 0, half_width, half_height, "in the upper left area"),
            ("Upper Right", region_upper_right, half_width, 0, half_width, half_height, "in the upper right area"),
            ("Lower Left", region_lower_left, 0, half_height, half_width, half_height, "in the lower left area"),
            ("Lower Right", region_lower_right, half_width, half_height, half_width, half_height, "in the lower right area"),
        ]
        
        # Collect regions with prompts
        regions = []
        for name, prompt, x, y, width, height, spatial_desc in quadrants:
            if prompt and prompt.strip():
                prompt_text = prompt.strip()
                
                # Calculate bottom-right corner
                x2 = x + width
                y2 = y + height
                
                # Format bounding box with grounding tokens
                bbox = self.format_bbox(x, y, x2, y2, image_width, image_height)
                
                # Combined mode: spatial description + grounding tokens
                regional_text = f"{prompt_text} {spatial_desc} <|object_ref_start|>{prompt_text}<|object_ref_end|> {bbox}"
                regions.append(regional_text)
        
        # Combine background and regional prompts
        if regions:
            if full_prompt:
                full_prompt += ", "
            full_prompt += ", ".join(regions)
        
        # Encode the combined prompt using CLIP
        tokens = clip.tokenize(full_prompt)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        
        return (conditioning,)


NODE_CLASS_MAPPINGS = {
    "StarQwenRegionalPrompter": StarQwenRegionalPrompter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarQwenRegionalPrompter": "⭐ Star Qwen Regional Prompter",
}
