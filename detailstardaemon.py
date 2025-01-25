# Detail Deamon adapted by https://github.com/muerrilla/sd-webui-detail-daemon
# Detail Deamon adapted by https://github.com/Jonseed/ComfyUI-Detail-Daemon
class DetailStarDaemon:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "detail_amount": ("FLOAT", {"default": 0.1, "min": -5.0, "max": 5.0, "step": 0.01}),
                "detail_start": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "detail_end": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "detail_bias": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "detail_exponent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("DETAIL_SCHEDULE",)
    RETURN_NAMES = ("detail_schedule",)
    FUNCTION = "create_schedule"
    CATEGORY = "⭐StarNodes"

    def create_schedule(self, detail_amount, detail_start, detail_end, detail_bias, detail_exponent):
        return ({
            "detail_amount": detail_amount,
            "detail_start": detail_start,
            "detail_end": detail_end,
            "detail_bias": detail_bias,
            "detail_exponent": detail_exponent
        },)

NODE_CLASS_MAPPINGS = {
    "DetailStarDaemon": DetailStarDaemon
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DetailStarDaemon": "⭐ Detail Star Daemon"
}
