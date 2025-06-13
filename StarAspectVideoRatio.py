class Starnodes_Aspect_Video_Ratio:
    """
    Node to select a video aspect ratio and calculate the height from a given width.
    Outputs width and height as int and string, and size as WxH string.
    """
    CATEGORY = "⭐StarNodes/Helpers And Tools"
    RETURN_TYPES = ("INT", "STRING", "INT", "STRING", "STRING")
    RETURN_NAMES = ("width", "width_str", "height", "height_str", "size")
    FUNCTION = "calculate"
    OUTPUT_NODE = False
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ratio": ( [
                    "Free Ratio",
                    "1:1 [square]",
                    "8:5 [landscape]",
                    "4:3 [landscape]",
                    "3:2 [landscape]",
                    "7:5 [landscape]",
                    "16:9 [landscape]",
                    "21:9 [landscape]",
                    "19:9 [landscape]",
                    "3:4 [portrait]",
                    "2:3 [portrait]",
                    "5:7 [portrait]",
                    "9:16 [portrait]",
                    "9:21 [portrait]",
                    "5:8 [portrait]",
                    "9:19 [portrait]",
                ], ),
                "width": ("INT", {"default": 750, "min": 1, "max": 8192}),
                "height_free_only": ("INT", {"default": 750, "min": 1, "max": 8192}),
            }
        }

    RATIO_MAP = {
        "1:1 [square]": (1, 1),
        "8:5 [landscape]": (8, 5),
        "4:3 [landscape]": (4, 3),
        "3:2 [landscape]": (3, 2),
        "7:5 [landscape]": (7, 5),
        "16:9 [landscape]": (16, 9),
        "21:9 [landscape]": (21, 9),
        "19:9 [landscape]": (19, 9),
        "3:4 [portrait]": (3, 4),
        "2:3 [portrait]": (2, 3),
        "5:7 [portrait]": (5, 7),
        "9:16 [portrait]": (9, 16),
        "9:21 [portrait]": (9, 21),
        "5:8 [portrait]": (5, 8),
        "9:19 [portrait]": (9, 19),
    }

    def calculate(self, ratio, width, height_free_only):
        if ratio == "Free Ratio":
            height = height_free_only
        else:
            num, den = self.RATIO_MAP.get(ratio, (1, 1))
            height = int(round(width * den / num))
        width_str = str(width)
        height_str = str(height)
        size = f"{width}x{height}"
        return (width, width_str, height, height_str, size)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return True

# For dynamic import
NODE_CLASS_MAPPINGS = {"Starnodes_Aspect_Video_Ratio": Starnodes_Aspect_Video_Ratio}
NODE_DISPLAY_NAME_MAPPINGS = {"Starnodes_Aspect_Video_Ratio": "⭐ Starnodes Aspect Video Ratio"}
