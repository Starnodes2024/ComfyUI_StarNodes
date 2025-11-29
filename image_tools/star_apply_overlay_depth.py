import torch
import torch.nn.functional as F
import time
import os
import numpy as np
from PIL import Image
import folder_paths


class StarApplyOverlayDepth:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE", {}),
                "filtered_image": ("IMAGE", {}),
                "strength": (
                    "INT",
                    {"default": 100, "min": 0, "max": 100, "step": 1},
                ),
            },
            "optional": {
                # Provide either a depth/greyscale image OR a mask. One must be connected.
                "depth_image": ("IMAGE", {}),
                "mask": ("MASK", {}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "show_preview": ("BOOLEAN", {"default": True}),
                # Pixel-based blur radius for the mask (in pixels)
                "blur_mask_px": (
                    "INT",
                    {"default": 0, "min": 0, "max": 256, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"
    CATEGORY = "⭐StarNodes/Image And Latent"

    def _ensure_batched(self, t):
        if t is None:
            return None
        if t.ndim == 3:
            # H W C or H W (mask)
            return t.unsqueeze(0)
        return t

    def _to_mask_from_image(self, img):
        # img: [B, H, W, C] in 0..1
        if img.shape[-1] == 1:
            m = img[..., 0]
        else:
            # luminance conversion
            r = img[..., 0]
            g = img[..., 1]
            b = img[..., 2]
            m = 0.299 * r + 0.587 * g + 0.114 * b
        return m.clamp(0.0, 1.0)  # [B, H, W]

    def _resize_to(self, tensor, target_h, target_w, is_mask=False):
        # tensor can be [B,H,W] or [B,H,W,C]
        if tensor.ndim == 4:  # image [B,H,W,C]
            x = tensor.permute(0, 3, 1, 2)  # B C H W
            mode = "bilinear"
            align_corners = False
            x = F.interpolate(x, size=(target_h, target_w), mode=mode, align_corners=align_corners)
            return x.permute(0, 2, 3, 1)
        elif tensor.ndim == 3:  # mask [B,H,W]
            x = tensor.unsqueeze(1)  # B 1 H W
            mode = "nearest" if is_mask else "bilinear"
            if mode == "bilinear":
                x = F.interpolate(x, size=(target_h, target_w), mode=mode, align_corners=False)
            else:
                x = F.interpolate(x, size=(target_h, target_w), mode=mode)
            return x.squeeze(1)
        else:
            return tensor

    def _match_batch(self, tensors):
        # Broadcast batch if one has batch=1
        batches = [t.shape[0] for t in tensors if t is not None]
        if not batches:
            return tensors
        max_b = max(batches)
        new_tensors = []
        for t in tensors:
            if t is None:
                new_tensors.append(None)
                continue
            b = t.shape[0]
            if b == max_b:
                new_tensors.append(t)
            elif b == 1:
                # repeat to match
                reps = [max_b] + [1] * (t.ndim - 1)
                new_tensors.append(t.repeat(*reps))
            else:
                # if mismatched and not 1, truncate to min
                min_b = min(batches)
                new_tensors.append(t[:min_b])
        return new_tensors

    def _gaussian_kernel_1d(self, sigma, device, dtype):
        # Create 1D Gaussian kernel with size covering +/- 3 sigma
        radius = int(torch.ceil(torch.tensor(3.0 * sigma)).item())
        size = 2 * radius + 1
        x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        kernel = torch.exp(-(x ** 2) / (2 * sigma * sigma))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, -1), radius

    def _gaussian_blur(self, mask_2d, sigma):
        # mask_2d: [B,H,W]
        if sigma <= 1e-6:
            return mask_2d
        B, H, W = mask_2d.shape
        device = mask_2d.device
        dtype = mask_2d.dtype
        kx, rx = self._gaussian_kernel_1d(sigma, device, dtype)
        ky, ry = kx, rx
        x = mask_2d.unsqueeze(1)  # B 1 H W
        # horizontal
        x = F.pad(x, (rx, rx, 0, 0), mode="reflect")
        x = F.conv2d(x, kx.view(1, 1, 1, -1))
        # vertical
        x = F.pad(x, (0, 0, ry, ry), mode="reflect")
        x = F.conv2d(x, ky.view(1, 1, -1, 1))
        return x.squeeze(1)

    def apply(self, source_image, filtered_image, strength=100, depth_image=None, mask=None, invert_mask=False, show_preview=True, blur_mask_px=0):
        # Ensure batched tensors
        source_image = self._ensure_batched(source_image)
        filtered_image = self._ensure_batched(filtered_image)
        depth_image = self._ensure_batched(depth_image) if depth_image is not None else None
        mask = self._ensure_batched(mask) if mask is not None else None

        if depth_image is None and mask is None:
            raise ValueError("Please connect either a depth/greyscale image or a mask.")

        # Determine target size from source
        B, H, W, C = source_image.shape

        # Resize inputs to match source
        filtered_image = self._resize_to(filtered_image, H, W)
        if mask is not None:
            mask = self._resize_to(mask, H, W, is_mask=True)
        if depth_image is not None:
            depth_image = self._resize_to(depth_image, H, W)

        # Make batches compatible
        source_image, filtered_image, mask, depth_image = self._match_batch(
            [source_image, filtered_image, mask, depth_image]
        )

        # Build mask from depth or provided mask
        if mask is not None:
            # mask expected in [B,H,W] in 0..1, but could be [B,H,W,1]
            if mask.ndim == 4 and mask.shape[-1] == 1:
                mask_2d = mask[..., 0]
            elif mask.ndim == 3:
                mask_2d = mask
            else:
                # If it's an image masquerading as mask
                mask_2d = self._to_mask_from_image(mask)
        else:
            # Use depth/greyscale image to build mask
            mask_2d = self._to_mask_from_image(depth_image)

        mask_2d = mask_2d.clamp(0.0, 1.0)
        if invert_mask:
            mask_2d = 1.0 - mask_2d

        # Optional Gaussian blur on mask using pixel radius
        blur_mask_px = int(blur_mask_px) if blur_mask_px is not None else 0
        if blur_mask_px > 0:
            # approximate relation: radius ~= 3*sigma  => sigma ~= radius/3
            sigma = max(float(blur_mask_px), 1.0) / 3.0
            mask_2d = self._gaussian_blur(mask_2d, sigma)
            mask_2d = mask_2d.clamp(0.0, 1.0)

        # Strength scaling
        alpha = (strength / 100.0) * mask_2d  # [B,H,W]
        alpha = alpha.unsqueeze(-1)  # [B,H,W,1]

        # Blend
        out = source_image * (1.0 - alpha) + filtered_image * alpha
        out = out.clamp(0.0, 1.0)
        # Ensure proper dtype/contiguous memory for ComfyUI
        if not torch.is_floating_point(out):
            out = out.float()
        out = out.contiguous()
        # Save a UI preview image of the mask (first batch element) like StarIconExporter
        if bool(show_preview):
            try:
                # Prepare preview canvas size
                preview_s = 256
                m = mask_2d[0]  # [H,W]
                # to numpy uint8 grayscale
                m_cpu = m.detach().clamp(0.0, 1.0).mul(255).to(torch.uint8).cpu().numpy()
                pil_gray = Image.fromarray(m_cpu, mode="L")
                # fit into square preview while keeping aspect
                Hm, Wm = pil_gray.height, pil_gray.width
                scale = min(preview_s / max(1, Wm), preview_s / max(1, Hm))
                new_w = max(1, int(round(Wm * scale)))
                new_h = max(1, int(round(Hm * scale)))
                pil_resized = pil_gray.resize((new_w, new_h), resample=Image.LANCZOS)
                canvas = Image.new("RGBA", (preview_s, preview_s), (0, 0, 0, 0))
                ox = (preview_s - new_w) // 2
                oy = (preview_s - new_h) // 2
                # visualize grayscale as RGB
                pil_rgb = Image.merge("RGB", (pil_resized, pil_resized, pil_resized))
                canvas.paste(pil_rgb.convert("RGBA"), (ox, oy))

                # Ensure output subfolder exists
                out_root = folder_paths.get_output_directory()
                # mirror StarIconExporter style: use a named subfolder under outputs
                subfolder = os.path.join("StarOverlay", "MaskPreviews")
                out_dir = os.path.join(out_root, subfolder)
                os.makedirs(out_dir, exist_ok=True)
                # timestamped filename to avoid collisions
                ts = time.strftime("%Y%m%d_%H%M%S")
                filename = f"overlay_mask_{ts}.png"
                save_path = os.path.join(out_dir, filename)
                canvas.save(save_path, format="PNG")

                ui = {
                    "images": [
                        {
                            "filename": filename,
                            "subfolder": subfolder,
                            "type": "output",
                        }
                    ]
                }
                return {"ui": ui, "result": (out,)}
            except Exception:
                # If preview save fails, fall back to returning only the image
                pass

        return (out,)


NODE_CLASS_MAPPINGS = {
    "StarApplyOverlayDepth": StarApplyOverlayDepth,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarApplyOverlayDepth": "⭐ Star Apply Overlay (Depth)",
}
