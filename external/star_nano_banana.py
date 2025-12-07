import os
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import configparser
import re
import math
import time
from typing import Optional, Tuple

try:
    import google.generativeai as genai
except ImportError:
    genai = None

class StarNanoBanana:
    def __init__(self):
        self.api_key = self._load_api_key()
        self.model = None
        if self.api_key and genai:
            genai.configure(api_key=self.api_key)
        
    @staticmethod
    def _load_api_key() -> Optional[str]:
        """Load API key using a pointer ini or a fixed fallback path.

        Resolution order:
        1) Root-level ini as pointer: comfyui_starnodes/googleapi.ini with section [API_PATH] and key 'path' -> points to external ini containing [API_KEY] key.
        2) Root-level ini direct key: comfyui_starnodes/googleapi.ini with section [API_KEY] key='...'.
        3) Environment variable: GOOGLE_API_KEY
        """
        # Get the root directory of comfyui_starnodes (two levels up from external/)
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        node_ini = os.path.join(root_dir, "googleapi.ini")

        def _read_key_from_ini(ini_path: str) -> Optional[str]:
            cfg = configparser.ConfigParser()
            try:
                cfg.read(ini_path)
                return cfg.get("API_KEY", "key", fallback=None)
            except Exception:
                return None

        # 1) Pointer ini in root directory
        if os.path.exists(node_ini):
            cfg = configparser.ConfigParser()
            try:
                cfg.read(node_ini)
                # Pointer mode
                if cfg.has_section("API_PATH"):
                    pointer_path = cfg.get("API_PATH", "path", fallback=None)
                    if pointer_path:
                        pointer_path = os.path.expandvars(pointer_path)
                        if os.path.exists(pointer_path):
                            key_val = _read_key_from_ini(pointer_path)
                            if key_val:
                                return key_val
                # Direct key mode (backward compatible)
                key_val = cfg.get("API_KEY", "key", fallback=None)
                if key_val:
                    return key_val
            except Exception:
                pass

        # 3) Environment variable fallback
        env_key = os.environ.get("GOOGLE_API_KEY")
        if env_key:
            return env_key

        return None

    @classmethod
    def INPUT_TYPES(cls):
        models = [
            "gemini-2.5-flash-image-preview",  "gemini-3-pro-image-preview",# Official Name for Nano Banana
        ]

        ratios = [
            "1:1",
            "2:1",
            "16:9",
            "9:16",
            "4:3",
            "3:4",
        ]

        megapixels = [
            "1 MP (≈ 1024x1024)",
            "2 MP (≈ 1448x1448)",
            "3 MP (≈ 1774x1774)",
            "4 MP (≈ 2048x2048)",
            "5 MP (≈ 2289x2289)",
            "6 MP (≈ 2508x2508)",
            "7 MP (≈ 2715x2715)",
            "8 MP (≈ 2867x2867)",
            "9 MP (≈ 3072x3072)",
            "10 MP (≈ 3184x3184)",
            "11 MP (≈ 3337x3337)",
            "12 MP (≈ 3488x3488)",
            "13 MP (≈ 3634x3634)",
            "14 MP (≈ 3778x3778)",
            "15 MP (≈ 3920x3920)",
        ]

        prompt_templates = [
            "Use Own Prompt",
            "Style Transfer: Transform the image to impressionist painting style with visible brush strokes and vibrant colors",
            "Color Enhancement: Enhance colors, increase saturation, improve contrast, make image more vibrant and appealing",
            "Background Change: Replace the background with a studio setup, clean white backdrop, professional lighting",
            "Object Removal: Remove the distracting objects in background, keep only the main subject, clean composition",
            "Lighting Adjustment: Brighten the image, improve lighting, add dramatic shadows, enhance overall illumination",
            "Composition Edit: Improve image composition, better framing, apply rule of thirds, enhance visual balance",
            "Filter Application: Apply professional photography filter, enhance mood, adjust tones, cinematic look",
            "Resolution Enhancement: Improve image quality, sharpen details, reduce noise, enhance clarity and sharpness",
            "Face Enhancement: Enhance facial features, improve skin texture, brighten eyes, professional portrait retouching",
            "Texture Change: Change surface textures, make materials look more realistic, enhance material properties",
            "Mood Change: Change image atmosphere to warm and inviting, enhance emotional impact, improve ambiance",
            "Time of Day: Change time to golden hour lighting, warm sunset colors, long shadows, magical atmosphere",
            "Weather Change: Add dramatic stormy weather, rain effects, dark clouds, moody atmospheric lighting",
            "Season Change: Transform to autumn season, add fall colors, warm tones, seasonal foliage changes",
            "Age Progression: Make subject appear 10 years younger, smooth skin, enhance youthful features",
            "Clothing Change: Change outfit to formal business attire, professional suit, elegant appearance",
            "Hair Style: Change hairstyle to modern trendy cut, enhance hair texture, professional styling",
            "Makeup Change: Apply natural makeup look, enhance features, professional cosmetic enhancement",
            "Pose Change: Adjust body posture to more confident stance, improve posture, dynamic composition",
            "Expression Change: Change facial expression to happy and smiling, warm friendly appearance",
            "Multiple Images: Combine all input images into single cohesive scene, blend seamlessly together",
            "Collage Creation: Create artistic collage layout, arrange images creatively, unified design theme",
            "Before After: Create side-by-side comparison showing transformation, highlight changes clearly",
            "Product Enhancement: Enhance product appearance, improve lighting, make more appealing, professional presentation",
            "Logo Addition: Add company logo watermark, subtle placement, maintain image quality",
            "Text Overlay: Add inspirational quote overlay, elegant typography, complementary design",
            "Border Addition: Add decorative artistic border, enhance presentation, professional framing",
            "Frame Addition: Add vintage picture frame effect, classic styling, elegant presentation",
            "Art Style: Convert to digital art style, vibrant colors, artistic interpretation, creative enhancement"
        ]

        return {
            "required": {
                "prompt": ("STRING", {"default": "A beautiful fantasy landscape, cinematic lighting", "multiline": True}),
                "model": (models, {"default": "gemini-3-pro-image-preview"}),
                "ratio": (ratios, {"default": "16:9"}),
                "megapixels": (megapixels, {"default": "4 MP (≈ 2048x2048)"}),
            },
            "optional": {
                "prompt_template": (prompt_templates, {"default": "Use Own Prompt"}),
                "image1": ("IMAGE", {}),
                "image2": ("IMAGE", {}),
                "image3": ("IMAGE", {}),
                "image4": ("IMAGE", {}),
                "image5": ("IMAGE", {}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always rerun this node, even if inputs haven't changed
        # Use a changing value so ComfyUI does not cache the output
        return time.time()

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "prompt")
    FUNCTION = "generate_image"
    CATEGORY = "⭐StarNodes/Image Generation"

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert ComfyUI tensor to PIL Image"""
        if tensor.ndim == 4:
            tensor = tensor.squeeze(0)
        image_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(image_np)

    def _pil_to_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        """Convert PIL Image to ComfyUI tensor"""
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        image_np = np.array(pil_image).astype(np.float32) / 255.0
        return torch.from_numpy(image_np).unsqueeze(0)

    def _create_error_image(self, text: str = "PROMPT DECLINED", detail: str = None) -> torch.Tensor:
        """Create a 1024x1024 black image with bold red text and optional white detail text below"""
        from PIL import ImageDraw, ImageFont
        
        # Create black image
        img = Image.new('RGB', (1024, 1024), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Try to load fonts
        main_font = None
        detail_font = None
        try:
            # Try to find a system font
            import matplotlib.font_manager as fm
            font_path = None
            for f in fm.findSystemFonts():
                font_name = os.path.basename(f).lower()
                # Look for bold fonts
                if 'bold' in font_name and ('arial' in font_name or 'verdana' in font_name):
                    font_path = f
                    break
            
            # If no bold font found, try regular arial or verdana
            if not font_path:
                for f in fm.findSystemFonts():
                    font_name = os.path.basename(f).lower()
                    if 'arial' in font_name or 'verdana' in font_name:
                        font_path = f
                        break
            
            if font_path:
                main_font = ImageFont.truetype(font_path, 80)
                detail_font = ImageFont.truetype(font_path, 24)
        except Exception:
            pass
        
        # Fallback to default font if no system font found
        if main_font is None:
            try:
                main_font = ImageFont.load_default()
                detail_font = ImageFont.load_default()
            except Exception:
                main_font = None
                detail_font = None
        
        # Draw main text in center
        if main_font:
            # Get text bounding box for centering
            bbox = draw.textbbox((0, 0), text, font=main_font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Calculate center position (shift up if we have detail text)
            y_offset = -40 if detail else 0
            x = (1024 - text_width) // 2
            y = (1024 - text_height) // 2 + y_offset
            
            # Draw main text in red
            draw.text((x, y), text, fill=(255, 0, 0), font=main_font)
            
            # Draw detail text below in white if provided
            if detail and detail_font:
                # Word wrap the detail text to fit within image width
                max_width = 900  # Leave some margin
                words = detail.split()
                lines = []
                current_line = []
                
                for word in words:
                    test_line = ' '.join(current_line + [word])
                    test_bbox = draw.textbbox((0, 0), test_line, font=detail_font)
                    test_width = test_bbox[2] - test_bbox[0]
                    
                    if test_width <= max_width:
                        current_line.append(word)
                    else:
                        if current_line:
                            lines.append(' '.join(current_line))
                            current_line = [word]
                        else:
                            lines.append(word)
                
                if current_line:
                    lines.append(' '.join(current_line))
                
                # Draw each line
                detail_y = y + text_height + 30
                for line in lines:
                    detail_bbox = draw.textbbox((0, 0), line, font=detail_font)
                    detail_width = detail_bbox[2] - detail_bbox[0]
                    detail_x = (1024 - detail_width) // 2
                    draw.text((detail_x, detail_y), line, fill=(255, 255, 255), font=detail_font)
                    detail_y += detail_bbox[3] - detail_bbox[1] + 5
        else:
            # Fallback: draw simple text without font
            draw.text((400, 500), text, fill=(255, 0, 0))
            if detail:
                draw.text((300, 550), detail[:50], fill=(255, 255, 255))  # Truncate if too long
        
        return self._pil_to_tensor(img)

    def generate_image(
        self,
        prompt: str,
        model: str,
        ratio: str,
        megapixels: str,
        prompt_template: str = "Use Own Prompt",
        image1: Optional[torch.Tensor] = None,
        image2: Optional[torch.Tensor] = None,
        image3: Optional[torch.Tensor] = None,
        image4: Optional[torch.Tensor] = None,
        image5: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, str]:
        # Determine final prompt first for error returns
        if prompt_template == "Use Own Prompt":
            final_prompt = prompt
        else:
            # Extract the actual prompt from the template (remove the category prefix)
            if ":" in prompt_template:
                final_prompt = prompt_template.split(":", 1)[1].strip()
            else:
                final_prompt = prompt_template
        
        if not self.api_key:
            error_detail = "Please add your Google Gemini API key to googleapi.ini"
            print(f"[StarNanoBanana] {error_detail}")
            return (self._create_error_image("API KEY MISSING", error_detail), final_prompt)
        if not genai:
            error_detail = "google-generativeai package not installed. Install with: pip install google-generativeai"
            print(f"[StarNanoBanana] {error_detail}")
            return (self._create_error_image("PACKAGE MISSING", error_detail), final_prompt)

        try:
            self.model = genai.GenerativeModel(model)
        except Exception as e:
            error_detail = f"Failed to initialize model '{model}': {str(e)}"
            print(f"[StarNanoBanana] {error_detail}")
            return (self._create_error_image("MODEL INIT FAILED", error_detail), final_prompt)

        contents = []
        input_images = [img for img in [image1, image2, image3, image4, image5] if img is not None]
        for img_tensor in input_images:
            contents.append(self._tensor_to_pil(img_tensor))

        # Add an explicit instruction for image generation to the prompt
        full_prompt = f"Generate an image based on the following prompt: {final_prompt}"
        contents.append(full_prompt)

        try:
            response = self.model.generate_content(contents)
            
            generated_image_data = None
            for part in response.parts:
                if part.inline_data:
                    generated_image_data = part.inline_data
                    break
            
            if generated_image_data:
                img_bytes = generated_image_data.data
                generated_pil = Image.open(BytesIO(img_bytes))

                # Compute target width/height from ratio and megapixel selection
                # 1 MP is defined as 1024x1024 area. Area = MP * 1024 * 1024
                ratio_match = re.match(r"^(\d+):(\d+)$", ratio.strip())
                if not ratio_match:
                    raise ValueError(f"Invalid ratio format: {ratio}. Expected like '1:1', '16:9'.")
                rw, rh = map(int, ratio_match.groups())

                mp_match = re.search(r"(\d+)\s*MP", megapixels, flags=re.IGNORECASE)
                if not mp_match:
                    raise ValueError(f"Invalid megapixels selection: {megapixels}. Expected like '1 MP (...)'.")
                mp_val = int(mp_match.group(1))

                base_area = 1024 * 1024
                target_area = mp_val * base_area

                aspect = rw / rh
                # Solve width*height = target_area and width/height = aspect
                width_f = (target_area * aspect) ** 0.5
                height_f = width_f / aspect

                def _round_to_multiple(x: float, m: int = 64) -> int:
                    return max(m, int(round(x / m) * m))

                width = _round_to_multiple(width_f, 64)
                height = _round_to_multiple(height_f, 64)

                # Resize the generated image to target dimensions
                if generated_pil.size != (width, height):
                    generated_pil = generated_pil.resize((width, height), Image.Resampling.LANCZOS)

                result_tensor = self._pil_to_tensor(generated_pil)
                return (result_tensor, final_prompt)
            else:
                error_message = "PROMPT DECLINED"
                error_detail = None
                
                # Check if response has candidates and handle finish reasons
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    finish_reason = getattr(candidate, 'finish_reason', None)
                    
                    if finish_reason == 8:  # SAFETY
                        error_detail = "The model refused the prompt due to safety policies."
                        print(f"[StarNanoBanana] Safety filter triggered: {error_detail}")
                    elif finish_reason == 3:  # MAX_TOKENS
                        error_detail = "The response was truncated due to length limits."
                        print(f"[StarNanoBanana] {error_detail}")
                    elif finish_reason == 4:  # RECITATION
                        error_detail = "The model detected repetitive content."
                        print(f"[StarNanoBanana] {error_detail}")
                    elif finish_reason == 5:  # LANGUAGE
                        error_detail = "The model detected inappropriate language."
                        print(f"[StarNanoBanana] Language filter triggered: {error_detail}")
                    else:
                        error_detail = f"No image generated. Finish reason: {finish_reason}."
                        print(f"[StarNanoBanana] {error_detail}")
                    
                    # Try to get text response if available and append to detail
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts') and candidate.content.parts:
                        try:
                            text_parts = [part.text for part in candidate.content.parts if hasattr(part, 'text')]
                            if text_parts:
                                api_response = ' '.join(text_parts)
                                print(f"[StarNanoBanana] API Text Response: {api_response}")
                                if error_detail:
                                    error_detail += f" API response: {api_response}"
                                else:
                                    error_detail = f"API response: {api_response}"
                        except Exception:
                            pass  # Ignore if we can't get text
                else:
                    error_detail = "No image was generated by the API. The model may have refused the prompt."
                    print(f"[StarNanoBanana] {error_detail}")
                
                # Return error image for any case where no image was generated
                error_tensor = self._create_error_image(error_message, error_detail)
                return (error_tensor, final_prompt)

        except Exception as e:
            # Log the error details to console
            error_detail = str(e)
            print(f"[StarNanoBanana] Error caught: {error_detail}")
            
            # Return error image for ALL errors to prevent workflow interruption
            error_tensor = self._create_error_image("PROMPT DECLINED", error_detail)
            return (error_tensor, final_prompt)


NODE_CLASS_MAPPINGS = {
    "StarNanoBanana": StarNanoBanana,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarNanoBanana": "⭐ Star Nano Banana (Gemini Image Gen)",
}