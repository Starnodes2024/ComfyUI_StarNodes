import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from torch import Tensor
from torch.nn import functional as F
from torchvision.transforms import ToTensor, ToPILImage


# Helper functions for AdaIN and wavelet-based color transfer

def calc_mean_std(feat: Tensor, eps: float = 1e-5):
    size = feat.size()
    assert len(size) == 4, "The input feature should be 4D tensor."
    b, c = size[:2]
    feat_var = feat.view(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(b, c, 1, 1)
    feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat: Tensor, style_feat: Tensor):
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def wavelet_blur(image: Tensor, radius: int):
    kernel_vals = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
    kernel = kernel[None, None]
    kernel = kernel.repeat(3, 1, 1, 1)
    image = F.pad(image, (radius, radius, radius, radius), mode="replicate")
    output = F.conv2d(image, kernel, groups=3, dilation=radius)
    return output


def wavelet_decomposition(image: Tensor, levels: int = 5):
    high_freq = torch.zeros_like(image)
    for i in range(levels):
        radius = 2 ** i
        low_freq = wavelet_blur(image, radius)
        high_freq += image - low_freq
        image = low_freq
    return high_freq, low_freq


def wavelet_reconstruction(content_feat: Tensor, style_feat: Tensor):
    content_high_freq, content_low_freq = wavelet_decomposition(content_feat)
    del content_low_freq
    style_high_freq, style_low_freq = wavelet_decomposition(style_feat)
    del style_high_freq
    return content_high_freq + style_low_freq


def adain_color_fix(target: Image.Image, source: Image.Image) -> Image.Image:
    to_tensor = ToTensor()
    target_tensor = to_tensor(target).unsqueeze(0)
    source_tensor = to_tensor(source).unsqueeze(0)
    result_tensor = adaptive_instance_normalization(target_tensor, source_tensor)
    to_image = ToPILImage()
    return to_image(result_tensor.squeeze(0).clamp_(0.0, 1.0))


def wavelet_color_fix(target: Image.Image, source: Image.Image) -> Image.Image:
    source = source.resize(target.size, resample=Image.Resampling.LANCZOS)
    to_tensor = ToTensor()
    target_tensor = to_tensor(target).unsqueeze(0)
    source_tensor = to_tensor(source).unsqueeze(0)
    result_tensor = wavelet_reconstruction(target_tensor, source_tensor)
    to_image = ToPILImage()
    return to_image(result_tensor.squeeze(0).clamp_(0.0, 1.0))


class StarSimpleFilters:
    BGCOLOR = "#3d124d"  # Background color
    COLOR = "#19124d"  # Title color
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The input image to apply filters to."}),
                "sharpen": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1, "display": "slider", "tooltip": "Increase image sharpness. 0 is no effect."}),
                "blur": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1, "display": "slider", "tooltip": "Apply Gaussian Blur. 0 is no effect."}),
                "saturation": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0, "display": "slider", "tooltip": "Adjust color intensity. 0: Neutral, -100: B&W, 100: Double saturation."}),
                "contrast": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0, "display": "slider", "tooltip": "Adjust contrast. 0 is neutral."}),
                "brightness": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0, "display": "slider", "tooltip": "Adjust brightness. 0 is neutral."}),
                "temperature": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0, "tooltip": "Adjust color temperature. -100: Cooler (Blue), 100: Warmer (Red)."}),
                "filter_strength": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 100.0, "step": 1.0, "display": "slider", "tooltip": "Global opacity of the effect. 100 is full effect, 0 is original image."}),
                "color_match_method": (["None", "wavelet", "adain", "mkl", "hm", "reinhard", "mvgd", "hm-mvgd-hm", "hm-mkl-hm"], {"default": "None", "tooltip": "Method for color matching."}),
            },
            "optional": {
                "color_match_image": ("IMAGE", {"tooltip": "Reference image for color matching."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_filters"
    CATEGORY = "⭐StarNodes/Image And Latent"

    def apply_filters(self, image, sharpen, blur, saturation, contrast, brightness, temperature, filter_strength, color_match_method, color_match_image=None):
        result_images = []

        # Only import ColorMatcher if needed
        if color_match_method not in ["None", "wavelet", "adain"]:
            try:
                from color_matcher import ColorMatcher
            except ImportError:
                raise Exception("Can't import color-matcher, did you install requirements.txt? Manual install: pip install color-matcher")

        for i in range(image.shape[0]):
            img_tensor = image[i]
            pil_img = None  # Will be set by color match or conversion

            # 0. Color Match
            if color_match_method != "None" and color_match_image is not None:
                try:
                    ref_idx = i % color_match_image.shape[0]
                    ref_tensor = color_match_image[ref_idx]

                    if color_match_method in ["wavelet", "adain"]:
                        img_pil_cm = Image.fromarray((img_tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8))
                        ref_pil_cm = Image.fromarray((ref_tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8))

                        if color_match_method == "wavelet":
                            pil_img = wavelet_color_fix(img_pil_cm, ref_pil_cm)
                        else:
                            pil_img = adain_color_fix(img_pil_cm, ref_pil_cm)
                    else:
                        cm = ColorMatcher()
                        target_np = img_tensor.cpu().numpy()
                        ref_np = ref_tensor.cpu().numpy()
                        matched_np = cm.transfer(src=target_np, ref=ref_np, method=color_match_method)
                        matched_np = np.clip(matched_np, 0.0, 1.0)
                        # Convert to PIL for further processing
                        pil_img = Image.fromarray((matched_np * 255.0).clip(0, 255).astype(np.uint8))
                except Exception as e:
                    print(f"Color Match error on image {i}: {e}")

            # Convert tensor [H, W, C] to PIL if not already done by color match
            if pil_img is None:
                img_np = (img_tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np)

            processed_img = pil_img

            # 1. Blur
            if blur > 0:
                processed_img = processed_img.filter(ImageFilter.GaussianBlur(radius=blur))

            # 2. Sharpen
            if sharpen > 0:
                factor = 1.0 + sharpen
                enhancer = ImageEnhance.Sharpness(processed_img)
                processed_img = enhancer.enhance(factor)

            # 3. Saturation
            if saturation != 0:
                factor = 1.0 + (saturation / 100.0)
                factor = max(0.0, factor)
                enhancer = ImageEnhance.Color(processed_img)
                processed_img = enhancer.enhance(factor)

            # 4. Contrast
            if contrast != 0:
                factor = 1.0 + (contrast / 100.0)
                factor = max(0.0, factor)
                enhancer = ImageEnhance.Contrast(processed_img)
                processed_img = enhancer.enhance(factor)

            # 5. Brightness
            if brightness != 0:
                factor = 1.0 + (brightness / 100.0)
                factor = max(0.0, factor)
                enhancer = ImageEnhance.Brightness(processed_img)
                processed_img = enhancer.enhance(factor)

            # 6. Temperature
            if temperature != 0:
                processed_img = processed_img.convert("RGB")
                r, g, b = processed_img.split()

                shift = (temperature / 100.0) * 0.5
                r_factor = 1.0 + shift
                b_factor = 1.0 - shift

                r = r.point(lambda x: int(min(255, max(0, x * r_factor))))
                b = b.point(lambda x: int(min(255, max(0, x * b_factor))))

                processed_img = Image.merge("RGB", (r, g, b))

            # Filter Strength Blend
            if filter_strength < 100.0:
                blend_factor = max(0.0, min(1.0, filter_strength / 100.0))

                true_original_tensor = image[i]
                true_original_np = (true_original_tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                true_original_pil = Image.fromarray(true_original_np)

                processed_img = Image.blend(true_original_pil, processed_img, blend_factor)

            res_np = np.array(processed_img).astype(np.float32) / 255.0
            res_tensor = torch.from_numpy(res_np).unsqueeze(0)
            result_images.append(res_tensor)

        if len(result_images) > 0:
            final_tensor = torch.cat(result_images, dim=0)
        else:
            final_tensor = image

        return (final_tensor,)

NODE_CLASS_MAPPINGS = {
    "StarSimpleFilters": StarSimpleFilters
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarSimpleFilters": "⭐ Star Simple Filters"
}
