class StarMultiInputsToOne:
    BGCOLOR = "#3d124d"
    COLOR = "#19124d"
    CATEGORY = '⭐StarNodes/Utilities'
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output",)
    FUNCTION = "process_inputs"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "input_1": ("*",),
                "input_2": ("*",),
                "input_3": ("*",),
                "input_4": ("*",),
                "input_5": ("*",),
            }
        }

    def process_inputs(self, **kwargs):
        for i in range(1, 6):
            input_val = kwargs.get(f"input_{i}")
            if input_val is not None:
                return (input_val,)
        
        return (None,)

NODE_CLASS_MAPPINGS = {
    "StarMultiInputsToOne": StarMultiInputsToOne
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarMultiInputsToOne": "⭐ Star Multi Inputs To One"
}
