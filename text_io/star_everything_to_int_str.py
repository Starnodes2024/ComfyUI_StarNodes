import math


class AlwaysEqualProxy(str):
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False


class StarEverythingToIntStr:
    BGCOLOR = "#3d124d"
    COLOR = "#19124d"
    CATEGORY = "⭐StarNodes/Helpers And Tools"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (AlwaysEqualProxy("*"), {"tooltip": "Connect any data type. Outputs a safe INT and STRING conversion."}),
            }
        }

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("int", "str")
    FUNCTION = "convert"

    def _to_int(self, v):
        if v is None:
            return 0

        if isinstance(v, bool):
            return int(v)

        if isinstance(v, int):
            return v

        if isinstance(v, float):
            if math.isfinite(v):
                return int(v)
            return 0

        if isinstance(v, (tuple, list)):
            if len(v) == 1:
                return self._to_int(v[0])
            return 0

        try:
            import torch

            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    return self._to_int(v.item())
                return 0
        except Exception:
            pass

        if isinstance(v, str):
            s = v.strip()
            if s == "":
                return 0
            try:
                return int(s)
            except Exception:
                try:
                    f = float(s)
                    if math.isfinite(f):
                        return int(f)
                except Exception:
                    return 0
            return 0

        try:
            return int(v)
        except Exception:
            return 0

    def convert(self, value):
        int_value = self._to_int(value)
        try:
            str_value = str(value)
        except Exception:
            str_value = ""
        return (int_value, str_value)


NODE_CLASS_MAPPINGS = {
    "StarEverythingToIntStr": StarEverythingToIntStr,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarEverythingToIntStr": "⭐ Star Everything to INT/STR",
}
