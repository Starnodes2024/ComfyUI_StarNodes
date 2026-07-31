class StarSplitSamplerInfo:
    """
    Splits the SAMPLER_INFO output from StarSampler into individual components.
    Each component can be accessed separately for further processing or display.
    """
    BGCOLOR = "#3d124d"
    COLOR = "#19124d"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "split_info": ("SAMPLER_INFO", {"tooltip": "Connect from StarSampler split_info output"}),
            }
        }

    RETURN_TYPES = ("FLOAT", "STRING", "STRING", "STRING", "STRING", "STRING", "FLOAT", "INT", "FLOAT")
    RETURN_NAMES = ("processing_time", "model", "text_encoder", "vae_model", "sampler", "scheduler", "denoise", "steps", "cfg")
    FUNCTION = "split"
    CATEGORY = "⭐StarNodes/Sampler"

    def split(self, split_info):
        """Split the info dict into individual outputs."""
        processing_time = float(split_info.get("processing_time", 0.0))
        model = str(split_info.get("model", "Unknown Model"))
        text_encoder = str(split_info.get("text_encoder", "Embedded in model"))
        vae_model = str(split_info.get("vae_model", "Unknown VAE"))
        sampler = str(split_info.get("sampler", "Unknown"))
        scheduler = str(split_info.get("scheduler", "Unknown"))
        denoise = float(split_info.get("denoise", 1.0))
        steps = int(split_info.get("steps", 20))
        cfg = float(split_info.get("cfg", 7.0))
        
        return (processing_time, model, text_encoder, vae_model, sampler, scheduler, denoise, steps, cfg)


NODE_CLASS_MAPPINGS = {
    "StarSplitSamplerInfo": StarSplitSamplerInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarSplitSamplerInfo": "⭐ Star Split Sampler Info",
}
