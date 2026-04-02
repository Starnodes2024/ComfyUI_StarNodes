import json
from typing import Dict, Any, Tuple


class StarQwenRebalancePrompter:
    """
    Creates structured prompts in the Qwen-Rebalance format with detailed scene composition.
    
    Outputs a JSON-formatted prompt with subject, foreground, midground, background,
    composition, visual_guidance, color_tone, lighting_mood, and caption fields.
    """

    COMPOSITION_PRESETS = [
        "rule of thirds, off-center subject, vertical orientation, depth layering",
        "centered composition, symmetrical balance, horizontal orientation",
        "golden ratio, diagonal leading lines, dynamic perspective",
        "rule of thirds, horizontal orientation, balanced depth",
        "centered subject, radial composition, circular flow",
        "off-center, asymmetric balance, negative space emphasis",
        "Custom (use text field below)",
    ]

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "subject": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "SUBJECT e.g., woman with short dark hair, backless white dress, barefoot, holding straw hat"
                    }
                ),
                "foreground": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "FOREGROUND e.g., wet rocks, shallow water, stone shoreline"
                    }
                ),
                "midground": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "MIDGROUND e.g., woman standing on rocky shore, calm water surface, distant pier posts"
                    }
                ),
                "background": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "BACKGROUND e.g., coastal town with red-tiled roofs, church bell tower, hazy mountains"
                    }
                ),
                "composition_preset": (
                    cls.COMPOSITION_PRESETS,
                    {"default": cls.COMPOSITION_PRESETS[0]}
                ),
                "custom_composition": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "CUSTOM COMPOSITION - Used only when 'Custom' preset is selected"
                    }
                ),
                "color_tone": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "COLOR TONE e.g., soft pastel tones, muted earthy palette, isolated warm orange accents"
                    }
                ),
                "lighting_mood": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "LIGHTING MOOD e.g., soft natural lighting, diffused daylight, serene atmosphere"
                    }
                ),
                "auto_generate_visual_guidance": (
                    "BOOLEAN",
                    {"default": True}
                ),
                "custom_visual_guidance": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "VISUAL GUIDANCE - Used only when auto-generate is disabled"
                    }
                ),
                "auto_generate_caption": (
                    "BOOLEAN",
                    {"default": True}
                ),
                "custom_caption": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "CAPTION - Used only when auto-generate is disabled"
                    }
                ),
                "output_format": (
                    ["JSON", "Plain Text"],
                    {"default": "JSON"}
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "build_prompt"
    CATEGORY = "⭐StarNodes/Prompts"

    def _generate_visual_guidance(
        self,
        subject: str,
        foreground: str,
        composition: str
    ) -> str:
        """Auto-generate visual guidance based on composition and subject."""
        guidance_parts = []
        
        # Extract key elements from foreground for leading lines
        fg_elements = foreground.split(",")[0].strip() if foreground else "elements"
        guidance_parts.append(f"leading lines from {fg_elements} to subject")
        
        # Add lighting contrast hints
        if "light" in subject.lower() or "white" in subject.lower() or "dress" in subject.lower():
            guidance_parts.append("light contrast on subject")
        
        # Add directional cues from composition
        if "off-center" in composition.lower():
            guidance_parts.append("asymmetric visual weight")
        if "rule of thirds" in composition.lower():
            guidance_parts.append("intersection point focus")
        if "depth" in composition.lower():
            guidance_parts.append("layered depth perception")
        
        # Add subject positioning
        if "looking" in subject.lower() or "gaze" in subject.lower():
            guidance_parts.append("directional gaze emphasis")
        
        return ", ".join(guidance_parts)

    def _generate_caption(
        self,
        subject: str,
        foreground: str,
        midground: str,
        background: str,
        lighting_mood: str
    ) -> str:
        """Auto-generate a natural language caption from all elements."""
        # Extract main subject descriptor
        subject_main = subject.split(",")[0].strip() if subject else "subject"
        
        # Build caption
        caption_parts = []
        caption_parts.append(f"A {subject_main}")
        
        if midground:
            mid_action = midground.split(",")[0].strip()
            caption_parts.append(mid_action)
        
        if foreground:
            caption_parts.append(f"on {foreground.split(',')[0].strip()}")
        
        if background:
            bg_main = background.split(",")[0].strip()
            caption_parts.append(f"with {bg_main} in the background")
        
        if lighting_mood:
            mood_main = lighting_mood.split(",")[0].strip()
            caption_parts.append(f"under {mood_main}")
        
        return " ".join(caption_parts) + "."

    def build_prompt(
        self,
        subject: str,
        foreground: str,
        midground: str,
        background: str,
        composition_preset: str,
        custom_composition: str,
        color_tone: str,
        lighting_mood: str,
        auto_generate_visual_guidance: bool,
        custom_visual_guidance: str,
        auto_generate_caption: bool,
        custom_caption: str,
        output_format: str
    ) -> Tuple[str]:
        """Build the structured prompt."""
        
        # Resolve composition
        if composition_preset == "Custom (use text field below)":
            composition = custom_composition.strip() or "balanced composition"
        else:
            composition = composition_preset
        
        # Resolve visual guidance
        if auto_generate_visual_guidance:
            visual_guidance = self._generate_visual_guidance(subject, foreground, composition)
        else:
            visual_guidance = custom_visual_guidance.strip() or "visual flow from foreground to background"
        
        # Resolve caption
        if auto_generate_caption:
            caption = self._generate_caption(subject, foreground, midground, background, lighting_mood)
        else:
            caption = custom_caption.strip() or "A scene composition."
        
        # Build the prompt structure
        prompt_dict = {
            "subject": subject.strip(),
            "foreground": foreground.strip(),
            "midground": midground.strip(),
            "background": background.strip(),
            "composition": composition,
            "visual_guidance": visual_guidance,
            "color_tone": color_tone.strip(),
            "lighting_mood": lighting_mood.strip(),
            "caption": caption,
        }
        
        # Format output
        if output_format == "JSON":
            output = json.dumps(prompt_dict, indent=2, ensure_ascii=False)
        else:
            # Plain text format
            lines = []
            for key, value in prompt_dict.items():
                lines.append(f"{key}: {value}")
            output = "\n".join(lines)
        
        return (output,)


NODE_CLASS_MAPPINGS = {
    "StarQwenRebalancePrompter": StarQwenRebalancePrompter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarQwenRebalancePrompter": "⭐ Star Qwen-Rebalance-Prompter",
}
