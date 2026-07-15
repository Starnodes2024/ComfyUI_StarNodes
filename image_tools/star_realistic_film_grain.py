import torch
import torch.nn.functional as F
import math

class StarRealisticFilmGrain:
    """
    ComfyUI Custom Node: Star Realistic Film Grain
    Generates highly realistic, analog film grain based on real film profiles,
    luminance masking (grain mostly in midtones), and organic variance (clumping).
    """
    
    FILM_PROFILES = {
        # Format: "Name": (is_color, base_size, organic_variance, default_strength)
        "Custom": (False, 1.0, 0.5, 0.5),
        "Kodak Tri-X 400": (False, 1.2, 0.65, 0.6),
        "Ilford HP5 Plus 400": (False, 1.0, 0.5, 0.5),
        "Ilford Delta 3200": (False, 2.6, 0.85, 0.75),   # Very coarse, high-speed grain
        "Kodak T-Max 100": (False, 0.4, 0.25, 0.25),     # Very fine, clean grain
        "Kodak Portra 400": (True, 0.8, 0.45, 0.4),      # Popular, soft color grain
        "Fujifilm Superia 400": (True, 1.4, 0.6, 0.55),  # Prominent, structured color grain
        "Cinestill 800T": (True, 1.6, 0.7, 0.65)         # Motion picture grain with cool/shadow characteristics
    }

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "demo_mode": ("BOOLEAN", {"default": False}),
                "film_profile": (list(s.FILM_PROFILES.keys()),),
                "blend_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.02}),
                "auto_scale_grain": ("BOOLEAN", {"default": True}),
                "manual_grain_size": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 8.0, "step": 0.1}),
                "organic_variance": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "chroma_grain": ("BOOLEAN", {"default": False}), 
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_film_grain"
    CATEGORY = "⭐StarNodes/Image & Latent Selection"

    def _apply_grain_to_image(self, image, current_size, current_variance, use_chroma, blend_strength, auto_scale_grain):
        """Core grain application logic extracted for reuse."""
        orig_device = image.device
        orig_dtype = image.dtype
        
        b, h, w, c = image.shape
        img_tensor = image.permute(0, 3, 1, 2)

        if auto_scale_grain:
            scale_factor = max(h, w) / 1024.0
            actual_size = max(0.2, current_size * scale_factor)
        else:
            actual_size = max(0.2, current_size)

        noise_h = max(4, int(h / actual_size))
        noise_w = max(4, int(w / actual_size))

        noise_channels = 3 if use_chroma else 1
        noise = torch.randn((b, noise_channels, noise_h, noise_w), device=orig_device, dtype=orig_dtype)
        noise = F.interpolate(noise, size=(h, w), mode='bilinear', align_corners=False)
        
        if not use_chroma:
            noise = noise.repeat(1, 3, 1, 1)

        if current_variance > 0.0:
            cloud_h = max(2, int(noise_h / 8))
            cloud_w = max(2, int(noise_w / 8))
            clouds = torch.randn((b, 1, cloud_h, cloud_w), device=orig_device, dtype=orig_dtype)
            clouds = F.interpolate(clouds, size=(h, w), mode='bilinear', align_corners=False)
            clouds = torch.sigmoid(clouds) * 2.0 - 1.0
            noise = noise * (1.0 + current_variance * clouds)

        noise = torch.clamp(noise * 0.15, -0.5, 0.5)

        luminance = (img_tensor[:, 0:1, :, :] * 0.2126 + 
                     img_tensor[:, 1:2, :, :] * 0.7152 + 
                     img_tensor[:, 2:3, :, :] * 0.0722)
        
        grain_mask = 4.0 * luminance * (1.0 - luminance)
        grain_mask = torch.clamp(grain_mask, 0.0, 1.0)
        effective_noise = noise * grain_mask * blend_strength

        noise_shifted = effective_noise + 0.5
        output = (1.0 - 2.0 * noise_shifted) * (img_tensor ** 2) + 2.0 * noise_shifted * img_tensor
        output = torch.clamp(output, 0.0, 1.0)
        output = output.permute(0, 2, 3, 1)
        
        return output

    def _draw_text_on_image(self, image, text):
        """Draw text label on a black bar at the bottom of the image."""
        device = image.device
        dtype = image.dtype
        h, w, c = image.shape
        
        # Scale bar and font based on image size
        bar_height = max(30, int(h * 0.06))
        
        result = image.clone()
        result[-bar_height:, :, :] = 0.0
        
        # Scale character width based on image width
        char_width = max(8, int(w * 0.016))
        text_width = len(text) * char_width
        x_offset = max(2, (w - text_width) // 2)
        y_offset = h - int(bar_height * 0.6)
        
        simple_font = {
            'A': [[0,1,1,0],[1,0,0,1],[1,1,1,1],[1,0,0,1],[1,0,0,1]],
            'B': [[1,1,1,0],[1,0,0,1],[1,1,1,0],[1,0,0,1],[1,1,1,0]],
            'C': [[0,1,1,1],[1,0,0,0],[1,0,0,0],[1,0,0,0],[0,1,1,1]],
            'D': [[1,1,1,0],[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,1,1,0]],
            'E': [[1,1,1,1],[1,0,0,0],[1,1,1,0],[1,0,0,0],[1,1,1,1]],
            'F': [[1,1,1,1],[1,0,0,0],[1,1,1,0],[1,0,0,0],[1,0,0,0]],
            'G': [[0,1,1,1],[1,0,0,0],[1,0,1,1],[1,0,0,1],[0,1,1,1]],
            'H': [[1,0,0,1],[1,0,0,1],[1,1,1,1],[1,0,0,1],[1,0,0,1]],
            'I': [[1,1,1],[0,1,0],[0,1,0],[0,1,0],[1,1,1]],
            'K': [[1,0,0,1],[1,0,1,0],[1,1,0,0],[1,0,1,0],[1,0,0,1]],
            'L': [[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,1,1,1]],
            'M': [[1,0,0,0,1],[1,1,0,1,1],[1,0,1,0,1],[1,0,0,0,1],[1,0,0,0,1]],
            'N': [[1,0,0,1],[1,1,0,1],[1,0,1,1],[1,0,0,1],[1,0,0,1]],
            'O': [[0,1,1,0],[1,0,0,1],[1,0,0,1],[1,0,0,1],[0,1,1,0]],
            'P': [[1,1,1,0],[1,0,0,1],[1,1,1,0],[1,0,0,0],[1,0,0,0]],
            'R': [[1,1,1,0],[1,0,0,1],[1,1,1,0],[1,0,1,0],[1,0,0,1]],
            'S': [[0,1,1,1],[1,0,0,0],[0,1,1,0],[0,0,0,1],[1,1,1,0]],
            'T': [[1,1,1,1,1],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0]],
            'U': [[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,0,1],[0,1,1,0]],
            'V': [[1,0,0,0,1],[1,0,0,0,1],[0,1,0,1,0],[0,1,0,1,0],[0,0,1,0,0]],
            'X': [[1,0,0,1],[0,1,1,0],[0,1,1,0],[0,1,1,0],[1,0,0,1]],
            'Y': [[1,0,0,0,1],[0,1,0,1,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0]],
            '-': [[0,0,0,0],[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]],
            ' ': [[0,0],[0,0],[0,0],[0,0],[0,0]],
            '0': [[0,1,1,0],[1,0,0,1],[1,0,0,1],[1,0,0,1],[0,1,1,0]],
            '1': [[0,1,0],[1,1,0],[0,1,0],[0,1,0],[1,1,1]],
            '2': [[1,1,1,0],[0,0,0,1],[0,1,1,0],[1,0,0,0],[1,1,1,1]],
            '3': [[1,1,1,0],[0,0,0,1],[0,1,1,0],[0,0,0,1],[1,1,1,0]],
            '4': [[1,0,0,1],[1,0,0,1],[1,1,1,1],[0,0,0,1],[0,0,0,1]],
            '5': [[1,1,1,1],[1,0,0,0],[1,1,1,0],[0,0,0,1],[1,1,1,0]],
            '6': [[0,1,1,0],[1,0,0,0],[1,1,1,0],[1,0,0,1],[0,1,1,0]],
            '7': [[1,1,1,1],[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]],
            '8': [[0,1,1,0],[1,0,0,1],[0,1,1,0],[1,0,0,1],[0,1,1,0]],
            '9': [[0,1,1,0],[1,0,0,1],[0,1,1,1],[0,0,0,1],[0,1,1,0]],
        }
        
        # Scale font based on image size (2x for 500px images)
        font_scale = max(1, int(w / 250))
        
        x_pos = x_offset
        for char in text.upper():
            if char in simple_font:
                pattern = simple_font[char]
                for row_idx, row in enumerate(pattern):
                    for col_idx, pixel in enumerate(row):
                        if pixel == 1:
                            # Draw scaled pixels
                            for sy in range(font_scale):
                                for sx in range(font_scale):
                                    py = y_offset + (row_idx * font_scale) + sy
                                    px = x_pos + (col_idx * font_scale) + sx
                                    if 0 <= py < h and 0 <= px < w:
                                        result[py, px, :] = 1.0
                x_pos += (len(pattern[0]) * font_scale) + font_scale
            else:
                x_pos += 3 * font_scale
        
        return result

    def apply_film_grain(self, image, demo_mode, film_profile, blend_strength, auto_scale_grain, manual_grain_size, organic_variance, chroma_grain):
        if demo_mode:
            return self._create_demo_grid(image, blend_strength, auto_scale_grain)
        
        # Normal mode - single profile application
        is_color, p_size, p_variance, p_strength = self.FILM_PROFILES[film_profile]
        
        if film_profile != "Custom":
            current_size = p_size
            current_variance = p_variance
            use_chroma = is_color
        else:
            current_size = manual_grain_size
            current_variance = organic_variance
            use_chroma = chroma_grain

        output = self._apply_grain_to_image(image, current_size, current_variance, use_chroma, blend_strength, auto_scale_grain)
        return (output,)

    def _create_demo_grid(self, image, blend_strength, auto_scale_grain):
        """Create a grid showing all film profiles."""
        device = image.device
        dtype = image.dtype
        
        # Take only first image from batch
        if image.shape[0] > 1:
            image = image[0:1]
        
        # Get all profiles except "Custom"
        profiles = [name for name in self.FILM_PROFILES.keys() if name != "Custom"]
        num_profiles = len(profiles)
        
        # Calculate grid dimensions
        grid_cols = math.ceil(math.sqrt(num_profiles))
        grid_rows = math.ceil(num_profiles / grid_cols)
        
        # Resize input image to 500x500 for better visibility
        tile_size = 500
        b, h, w, c = image.shape
        img_resized = image.permute(0, 3, 1, 2)
        img_resized = F.interpolate(img_resized, size=(tile_size, tile_size), mode='bilinear', align_corners=False)
        img_resized = img_resized.permute(0, 2, 3, 1)
        
        # Boost blend strength for demo mode to make grain more visible
        # Use at least 0.8 or the user's value if higher
        demo_blend_strength = max(0.8, blend_strength)
        
        # Create tiles for each profile
        tiles = []
        for profile_name in profiles:
            is_color, p_size, p_variance, p_strength = self.FILM_PROFILES[profile_name]
            
            # Apply grain to the resized image with boosted strength
            grain_img = self._apply_grain_to_image(
                img_resized, 
                p_size, 
                p_variance, 
                is_color, 
                demo_blend_strength, 
                auto_scale_grain
            )
            
            # Add text label
            labeled_img = self._draw_text_on_image(grain_img[0], profile_name)
            tiles.append(labeled_img)
        
        # Pad with black tiles if needed
        total_tiles_needed = grid_rows * grid_cols
        while len(tiles) < total_tiles_needed:
            black_tile = torch.zeros((tile_size, tile_size, c), device=device, dtype=dtype)
            tiles.append(black_tile)
        
        # Assemble grid
        rows = []
        for row_idx in range(grid_rows):
            row_tiles = []
            for col_idx in range(grid_cols):
                tile_idx = row_idx * grid_cols + col_idx
                row_tiles.append(tiles[tile_idx])
            row = torch.cat(row_tiles, dim=1)  # Concatenate horizontally
            rows.append(row)
        
        grid = torch.cat(rows, dim=0)  # Concatenate vertically
        grid = grid.unsqueeze(0)  # Add batch dimension
        
        return (grid,)


NODE_CLASS_MAPPINGS = {
    "StarRealisticFilmGrain": StarRealisticFilmGrain,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarRealisticFilmGrain": "⭐ Star Realistic Film Grain",
}
