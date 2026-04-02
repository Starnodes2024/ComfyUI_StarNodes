import math
import torch

class StarQwenWanRatio:
    @classmethod
    def INPUT_TYPES(cls):
        # Helper to compute best-fit dims for a target total pixels with a given aspect ratio
        def round_to_multiple(x, m=16):
            return int(max(m, round(x / m) * m))

        def best_fit(total_pixels: int, ar_w: int, ar_h: int):
            ideal_w = math.sqrt(total_pixels * (ar_w / ar_h))
            ideal_h = ideal_w * (ar_h / ar_w)
            w = round_to_multiple(ideal_w, 16)
            h = round_to_multiple(ideal_h, 16)

            def score(ww, hh):
                prod = ww * hh
                prod_diff = abs(prod - total_pixels)
                ar_err = abs((ww / hh) - (ar_w / ar_h))
                return prod_diff + ar_err * 1000

            best_w, best_h = w, h
            best_s = score(w, h)

            # local search around the snapped solution
            candidates = []
            for dw in range(-128, 129, 16):
                ww = max(16, w + dw)
                h_ratio = round_to_multiple(ww * ar_h / ar_w, 16)
                candidates.append((ww, h_ratio))
            for dh in range(-128, 129, 16):
                hh = max(16, h + dh)
                w_ratio = round_to_multiple(hh * ar_w / ar_h, 16)
                candidates.append((w_ratio, hh))

            common = {
                (16, 9): [(1280, 720), (1920, 1080)],
                (9, 16): [(720, 1280), (1080, 1920)],
            }
            if (ar_w, ar_h) in common:
                for ww, hh in common[(ar_w, ar_h)]:
                    candidates.append((ww, hh))

            for ww, hh in candidates:
                s = score(ww, hh)
                if s < best_s:
                    best_s, best_w, best_h = s, ww, hh

            return best_w, best_h

        # Qwen fixed mapping (keep as-is), includes label with dimensions
        qwen_map = {
            "1:1 (1328x1328)": (1328, 1328),
            "16:9 (1664x928)": (1664, 928),
            "9:16 (928x1664)": (928, 1664),
            "4:3 (1472x1104)": (1472, 1104),
            "3:4 (1104x1472)": (1104, 1472),
            "3:2 (1584x1056)": (1584, 1056),
            "2:3 (1056x1584)": (1056, 1584),
            "5:7 (1120x1568)": (1120, 1568),
            "7:5 (1568x1120)": (1568, 1120),
            "Free Ratio (custom)": None,
        }

        # WAN HD (1280x720 total pixels) and WAN Full HD (1920x1080) mappings with labels including computed dims
        wan_ratios = [(1, 1), (16, 9), (9, 16), (4, 3), (3, 4), (3, 2), (2, 3), (5, 7), (7, 5)]

        def build_wan_map(total_pixels: int):
            mapping = {}
            for ar_w, ar_h in wan_ratios:
                w, h = best_fit(total_pixels, ar_w, ar_h)
                label = f"{ar_w}:{ar_h} ({w}x{h})"
                mapping[label] = (w, h)
            mapping["Free Ratio (custom)"] = None
            return mapping

        wan_hd_map = build_wan_map(921_600)
        wan_fhd_map = build_wan_map(2_073_600)

        return {
            "required": {
                "model": (["Qwen", "Wan HD", "Wan Full HD"], {"default": "Qwen"}),
                "qwen_ratio": (list(qwen_map.keys()), {"default": "1:1 (1328x1328)"}),
                "wan_hd_ratio": (list(wan_hd_map.keys()), {"default": next(iter(wan_hd_map.keys()))}),
                "wan_fhd_ratio": (list(wan_fhd_map.keys()), {"default": next(iter(wan_fhd_map.keys()))}),
                "use_nearest_image_ratio": ("BOOLEAN", {"default": False}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
                "custom_width": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 16}),
                "custom_height": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 16}),
            },
            "optional": {
                "image": ("IMAGE", {}),
            },
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("latent", "width", "height")
    FUNCTION = "create"
    CATEGORY = "⭐StarNodes/Image And Latent"

    @staticmethod
    def _round_to_multiple(x, m=16):
        return int(max(m, round(x / m) * m))

    @staticmethod
    def _best_fit_dims_for_total_pixels(total_pixels: int, ar_w: int, ar_h: int):
        # Start from the ideal real-valued solution, then snap to multiples of 16 and refine
        ideal_w = math.sqrt(total_pixels * (ar_w / ar_h))
        ideal_h = ideal_w * (ar_h / ar_w)
        w = StarQwenWanRatio._round_to_multiple(ideal_w, 16)
        h = StarQwenWanRatio._round_to_multiple(ideal_h, 16)

        def score(ww, hh):
            # score by absolute pixel difference with a small penalty on aspect error
            prod = ww * hh
            prod_diff = abs(prod - total_pixels)
            ar_err = abs((ww / hh) - (ar_w / ar_h))
            return prod_diff + ar_err * 1000  # weight aspect a bit

        best_w, best_h = w, h
        best_s = score(w, h)

        # Local search around the snapped solution
        candidates = []
        for dw in range(-128, 129, 16):
            ww = max(16, w + dw)
            # adjust h based on target ratio and total pixels nearby
            # try two strategies: keep ratio based h, and product-based h
            h_ratio = StarQwenWanRatio._round_to_multiple(ww * ar_h / ar_w, 16)
            for hh in (h_ratio,):
                candidates.append((ww, hh))
        for dh in range(-128, 129, 16):
            hh = max(16, h + dh)
            w_ratio = StarQwenWanRatio._round_to_multiple(hh * ar_w / ar_h, 16)
            for ww in (w_ratio,):
                candidates.append((ww, hh))

        # Add exact known targets for common 16:9/9:16 with HD/FHD
        common = {
            (16, 9): [(1280, 720), (1920, 1080)],
            (9, 16): [(720, 1280), (1080, 1920)],
        }
        if (ar_w, ar_h) in common:
            for ww, hh in common[(ar_w, ar_h)]:
                candidates.append((ww, hh))

        for ww, hh in candidates:
            s = score(ww, hh)
            if s < best_s:
                best_s, best_w, best_h = s, ww, hh

        return best_w, best_h

    def create(self, model: str, qwen_ratio: str, wan_hd_ratio: str, wan_fhd_ratio: str, use_nearest_image_ratio=False, batch_size=1, custom_width=1024, custom_height=1024, image=None):
        # Decide which selector to use based on model
        # Helper to build WAN maps here too (so we can search nearest by AR)
        def build_wan_map_local(total_pixels: int):
            wan_ratios = [(1, 1), (16, 9), (9, 16), (4, 3), (3, 4), (3, 2), (2, 3), (5, 7), (7, 5)]
            mapping = {}
            for ar_w, ar_h in wan_ratios:
                w, h = self._best_fit_dims_for_total_pixels(total_pixels, ar_w, ar_h)
                label = f"{ar_w}:{ar_h} ({w}x{h})"
                mapping[label] = (w, h)
            mapping["Free Ratio (custom)"] = None
            return mapping

        # Optionally override selection based on nearest image ratio
        width = height = None
        if use_nearest_image_ratio and image is not None:
            try:
                # image is [B,H,W,C] or [H,W,C]
                if image.ndim == 3:
                    H, W = int(image.shape[0]), int(image.shape[1])
                else:
                    H, W = int(image.shape[1]), int(image.shape[2])
                if H > 0 and W > 0:
                    target_ar = W / H
                    if model == "Qwen":
                        qwen_map = {
                            "1:1 (1328x1328)": (1328, 1328),
                            "16:9 (1664x928)": (1664, 928),
                            "9:16 (928x1664)": (928, 1664),
                            "4:3 (1472x1104)": (1472, 1104),
                            "3:4 (1104x1472)": (1104, 1472),
                            "3:2 (1584x1056)": (1584, 1056),
                            "2:3 (1056x1584)": (1056, 1584),
                            "5:7 (1120x1568)": (1120, 1568),
                            "7:5 (1568x1120)": (1568, 1120),
                        }
                        candidates = list(qwen_map.values())
                    else:
                        # WAN HD or FHD
                        if model == "Wan HD":
                            m = build_wan_map_local(921_600)
                        else:
                            m = build_wan_map_local(2_073_600)
                        candidates = [v for v in m.values() if v is not None]
                    # choose closest by aspect ratio
                    best = None
                    best_err = 1e9
                    for (ww, hh) in candidates:
                        ar = ww / hh
                        err = abs(ar - target_ar)
                        if err < best_err:
                            best_err, best = err, (ww, hh)
                    if best is not None:
                        width, height = best
            except Exception:
                # fallback to regular path if anything goes wrong
                width = height = None

        if width is None or height is None:
            if model == "Qwen":
                chosen = qwen_ratio
                # local Qwen mapping
                qwen_map = {
                    "1:1 (1328x1328)": (1328, 1328),
                    "16:9 (1664x928)": (1664, 928),
                    "9:16 (928x1664)": (928, 1664),
                    "4:3 (1472x1104)": (1472, 1104),
                    "3:4 (1104x1472)": (1104, 1472),
                    "3:2 (1584x1056)": (1584, 1056),
                    "2:3 (1056x1584)": (1056, 1584),
                    "5:7 (1120x1568)": (1120, 1568),
                    "7:5 (1568x1120)": (1568, 1120),
                }
                if chosen.startswith("Free"):
                    width, height = custom_width, custom_height
                else:
                    width, height = qwen_map[chosen]
            else:
                # WAN modes: parse the (WxH) from the label
                chosen = wan_hd_ratio if model == "Wan HD" else wan_fhd_ratio
                if chosen.startswith("Free"):
                    width, height = custom_width, custom_height
                else:
                    # label format: "A:B (WxH)"
                    try:
                        dims = chosen.split("(")[-1].split(")")[0]
                        w_str, h_str = dims.split("x")
                        width, height = int(w_str), int(h_str)
                    except Exception:
                        # Fallback: if parsing fails, default to HD/FHD 16:9
                        if model == "Wan HD":
                            width, height = 1280, 720
                        else:
                            width, height = 1920, 1080

        # Enforce divisibility by 16
        width = width - (width % 16)
        height = height - (height % 16)

        # Ensure divisibility by 8 for latent grid
        width_latent = width - (width % 8)
        height_latent = height - (height % 8)

        latent = torch.zeros([batch_size, 4, height_latent // 8, width_latent // 8])
        return ({"samples": latent}, width, height)


NODE_CLASS_MAPPINGS = {
    "StarQwenWanRatio": StarQwenWanRatio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarQwenWanRatio": "⭐ Star Qwen / WAN Ratio",
}
