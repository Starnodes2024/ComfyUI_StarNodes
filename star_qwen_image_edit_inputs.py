import math
import torch
import torch.nn.functional as F


class StarQwenImageEditInputs:
    @classmethod
    def INPUT_TYPES(cls):
        # Qwen fixed mapping
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
        # Prepend special option
        qwen_options = ["Use Best Ratio from Image 1"] + list(qwen_map.keys())

        return {
            "required": {
                "image1": ("IMAGE", {}),
                "qwen_resolution": (qwen_options, {"default": "Use Best Ratio from Image 1"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
                "custom_width": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 16}),
                "custom_height": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 16}),
            },
            "optional": {
                "image2": ("IMAGE", {}),
                "image3": ("IMAGE", {}),
                "image4": ("IMAGE", {}),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT", "INT", "INT")
    RETURN_NAMES = ("stitched", "latent", "width", "height")
    FUNCTION = "process"
    CATEGORY = "⭐StarNodes/Image And Latent"

    # ------------- helpers -------------
    @staticmethod
    def _to_nchw(img):
        # img: [H,W,C] or [1,H,W,C] float
        if img.ndim == 4:
            img1 = img[0]
        else:
            img1 = img
        # [H,W,C] -> [1,C,H,W]
        return img1.permute(2, 0, 1).unsqueeze(0)

    @staticmethod
    def _to_image(nchw):
        # [1,C,H,W] -> [1,H,W,C]
        return nchw.squeeze(0).permute(1, 2, 0).unsqueeze(0)

    @staticmethod
    def _resize_keep_aspect_max_side(img, max_side, only_if_larger=True):
        # img: [1,H,W,C] or [H,W,C]
        if img.ndim == 3:
            img = img.unsqueeze(0)
        H, W = int(img.shape[1]), int(img.shape[2])
        if max_side is None or (only_if_larger and max(H, W) <= max_side):
            return img
        scale = max_side / max(H, W)
        new_h = max(1, int(round(H * scale)))
        new_w = max(1, int(round(W * scale)))
        return StarQwenImageEditInputs._resize_exact(img, new_h, new_w)

    @staticmethod
    def _resize_to_height(img, target_h):
        if img.ndim == 3:
            img = img.unsqueeze(0)
        H, W = int(img.shape[1]), int(img.shape[2])
        if H == target_h:
            return img
        scale = target_h / H
        new_w = max(1, int(round(W * scale)))
        return StarQwenImageEditInputs._resize_exact(img, target_h, new_w)

    @staticmethod
    def _resize_exact(img, target_h, target_w):
        # img: [1,H,W,C]
        nchw = StarQwenImageEditInputs._to_nchw(img)
        out = F.interpolate(nchw, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return StarQwenImageEditInputs._to_image(out)

    @staticmethod
    def _letterbox_fit(img, target_h, target_w):
        # preserve aspect, fit inside target, center with white padding
        if img.ndim == 3:
            img = img.unsqueeze(0)
        H, W = int(img.shape[1]), int(img.shape[2])
        if H == 0 or W == 0:
            canvas = torch.ones((1, target_h, target_w, img.shape[3]), dtype=img.dtype, device=img.device)
            return canvas
        scale = min(target_w / W, target_h / H)
        new_w = max(1, int(round(W * scale)))
        new_h = max(1, int(round(H * scale)))
        resized = StarQwenImageEditInputs._resize_exact(img, new_h, new_w)
        canvas = torch.ones((1, target_h, target_w, img.shape[3]), dtype=img.dtype, device=img.device)
        top = (target_h - new_h) // 2
        left = (target_w - new_w) // 2
        canvas[:, top:top + new_h, left:left + new_w, :] = resized
        return canvas

    @staticmethod
    def _concat_h(img_left, img_right):
        # both [1,H,W,C], same H
        return torch.cat([img_left, img_right], dim=2)

    @staticmethod
    def _choose_qwen_dims_from_image(qwen_map_vals, image1):
        # image1: [1,H,W,C] or [H,W,C]
        if image1 is None:
            return None
        if image1.ndim == 4:
            H, W = int(image1.shape[1]), int(image1.shape[2])
        else:
            H, W = int(image1.shape[0]), int(image1.shape[1])
        if H <= 0 or W <= 0:
            return None
        target_ar = W / H
        best = None
        best_err = 1e9
        for (ww, hh) in qwen_map_vals:
            ar = ww / hh
            err = abs(ar - target_ar)
            if err < best_err:
                best_err, best = err, (ww, hh)
        return best

    # ------------- main -------------
    def process(self, image1, qwen_resolution,
                batch_size=1, custom_width=1024, custom_height=1024,
                image2=None, image3=None, image4=None):
        # Collect images (take first frame if batched)
        imgs = []
        for img in [image1, image2, image3, image4]:
            if img is None:
                continue
            if img.ndim == 4:
                imgs.append(img[0:1])
            else:
                imgs.append(img.unsqueeze(0))
        n = len(imgs)

        # Stitching rules
        stitched = None
        if n <= 1:
            # Only image1 rules
            if n == 1:
                # if bigger than 1328 on one side, scale down longest side to 1328; else keep
                stitched = self._resize_keep_aspect_max_side(imgs[0], 1328, only_if_larger=True)
            else:
                # Edge case: no image provided, create white 1328x1328
                stitched = torch.ones((1, 1328, 1328, 3), dtype=image1.dtype if image1 is not None else torch.float32)
        elif n == 2:
            # 1x2 grid -> make same height then concat, then ensure max height 1328 (downscale only)
            h1 = int(imgs[0].shape[1])
            h2 = int(imgs[1].shape[1])
            target_h = max(h1, h2)
            a = self._resize_to_height(imgs[0], target_h)
            b = self._resize_to_height(imgs[1], target_h)
            stitched = self._concat_h(a, b)
            # Downscale only by height if taller than 1328
            cur_h = int(stitched.shape[1])
            if cur_h > 1328:
                stitched = self._resize_to_height(stitched, 1328)
        elif n in (3, 4):
            # 2x2 grid, final 1328x1328. Fit each into 664x664 with letterbox.
            cell_h = cell_w = 1328 // 2  # 664
            cells = []
            for i in range(4):
                if i < n:
                    cells.append(self._letterbox_fit(imgs[i], cell_h, cell_w))
                else:
                    # blank
                    dt = imgs[0].dtype if n > 0 else torch.float32
                    device = imgs[0].device if n > 0 else None
                    cells.append(torch.ones((1, cell_h, cell_w, 3), dtype=dt, device=device))
            top = self._concat_h(cells[0], cells[1])
            bottom = self._concat_h(cells[2], cells[3])
            stitched = torch.cat([top, bottom], dim=1)  # vertical concat
            # Ensure exact 1328x1328 (already is by construction)
        else:
            # Shouldn't happen because max 4 inputs
            stitched = imgs[0]

        # Ensure dtype/device consistency
        if stitched.dtype != torch.float32:
            stitched = stitched.to(torch.float32)

        # Latent selection (Qwen-only list)
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

        if qwen_resolution == "Use Best Ratio from Image 1":
            chosen = self._choose_qwen_dims_from_image(list(qwen_map.values()), image1)
            if chosen is None:
                width, height = 1328, 1328
            else:
                width, height = chosen
        elif qwen_resolution.startswith("Free"):
            width, height = custom_width, custom_height
        else:
            width, height = qwen_map[qwen_resolution]

        # Enforce divisibility by 16/8
        width = width - (width % 16)
        height = height - (height % 16)
        width_latent = width - (width % 8)
        height_latent = height - (height % 8)

        latent = torch.zeros([batch_size, 4, height_latent // 8, width_latent // 8])

        return (stitched, {"samples": latent}, width, height)


NODE_CLASS_MAPPINGS = {
    "StarQwenImageEditInputs": StarQwenImageEditInputs,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarQwenImageEditInputs": "⭐ Star Qwen Image Edit Inputs",
}
