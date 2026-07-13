import torch
import torch.nn.functional as F

import comfy.sd
import comfy.utils
import folder_paths
import nodes


# Inlined from comfy.ldm.seedvr.color_fix and comfy_extras.nodes_seedvr to avoid version dependencies
D65_WHITE_X = 0.95047
D65_WHITE_Z = 1.08883
CIELAB_DELTA = 6.0 / 29.0
CIELAB_KAPPA = (29.0 / 3.0) ** 3


def div_pad(image, factor):
    height_factor, width_factor = factor
    height, width = image.shape[-2:]
    pad_height = (height_factor - (height % height_factor)) % height_factor
    pad_width = (width_factor - (width % width_factor)) % width_factor
    if pad_height == 0 and pad_width == 0:
        return image
    padding = (0, pad_width, 0, pad_height)
    return torch.nn.functional.pad(image, padding, mode='constant', value=0.0)


def cut_videos(videos):
    t = videos.size(1)
    if t < 1:
        raise ValueError("cut_videos expected at least one frame.")
    if t == 1:
        return videos
    if t <= 4:
        padding = videos[:, -1:].repeat(1, 4 - t + 1, 1, 1, 1)
        return torch.cat([videos, padding], dim=1)
    if (t - 1) % 4 == 0:
        return videos
    padding = videos[:, -1:].repeat(1, 4 - ((t - 1) % 4), 1, 1, 1)
    videos = torch.cat([videos, padding], dim=1)
    if (videos.size(1) - 1) % 4 != 0:
        raise ValueError(f"cut_videos failed to pad video length to 4n+1; got {videos.size(1)} frames.")
    return videos


def wavelet_blur(image, radius):
    kernel_size = 2 * radius + 1
    sigma = radius / 3.0
    x = torch.arange(-radius, radius + 1, dtype=image.dtype, device=image.device)
    gauss = torch.exp(-0.5 * (x / sigma) ** 2)
    gauss = gauss / gauss.sum()
    
    # Horizontal blur: reshape to (B*H, C, W), apply 1D conv, reshape back
    B, C, H, W = image.shape
    kernel_h = gauss.view(1, 1, -1).expand(C, 1, -1)
    img_reshaped = image.permute(0, 2, 1, 3).reshape(B * H, C, W)
    blurred_h = F.conv1d(img_reshaped, kernel_h, padding=radius, groups=C)
    blurred_h = blurred_h.reshape(B, H, C, W).permute(0, 2, 1, 3)
    
    # Vertical blur: reshape to (B*W, C, H), apply 1D conv, reshape back
    img_reshaped = blurred_h.permute(0, 3, 1, 2).reshape(B * W, C, H)
    blurred_v = F.conv1d(img_reshaped, kernel_h, padding=radius, groups=C)
    blurred = blurred_v.reshape(B, W, C, H).permute(0, 2, 3, 1)
    
    return blurred


def wavelet_decomposition(image, levels=5):
    high_freq = torch.zeros_like(image)
    for i in range(levels):
        radius = 2 ** i
        low_freq = wavelet_blur(image, radius)
        high_freq.add_(image).sub_(low_freq)
        image = low_freq
    return high_freq, low_freq


def wavelet_reconstruction(content_feat, style_feat):
    if content_feat.shape != style_feat.shape:
        if len(content_feat.shape) >= 3:
            style_feat = F.interpolate(style_feat, size=content_feat.shape[-2:], mode='bilinear', align_corners=False)
    content_high_freq, content_low_freq = wavelet_decomposition(content_feat)
    del content_low_freq
    style_high_freq, style_low_freq = wavelet_decomposition(style_feat)
    del style_high_freq
    if content_high_freq.shape != style_low_freq.shape:
        style_low_freq = F.interpolate(style_low_freq, size=content_high_freq.shape[-2:], mode='bilinear', align_corners=False)
    content_high_freq.add_(style_low_freq)
    return content_high_freq.clamp_(-1.0, 1.0)


def _histogram_matching_channel(source, reference):
    original_shape = source.shape
    source_flat = source.flatten()
    reference_flat = reference.flatten()
    source_sorted, source_indices = torch.sort(source_flat)
    reference_sorted, _ = torch.sort(reference_flat)
    del reference_flat
    n_source = len(source_sorted)
    n_reference = len(reference_sorted)
    if n_source == n_reference:
        matched_sorted = reference_sorted
    else:
        source_quantiles = torch.linspace(0, 1, n_source, device=source.device)
        ref_indices = (source_quantiles * (n_reference - 1)).long()
        ref_indices.clamp_(0, n_reference - 1)
        matched_sorted = reference_sorted[ref_indices]
        del source_quantiles, ref_indices, reference_sorted
    del source_sorted, source_flat
    inverse_indices = torch.argsort(source_indices)
    del source_indices
    matched_flat = matched_sorted[inverse_indices]
    del matched_sorted, inverse_indices
    return matched_flat.reshape(original_shape)


def _lab_to_rgb_batch(lab, matrix_inv, epsilon, kappa):
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    fy = (L + 16.0) / 116.0
    fx = a.div(500.0).add_(fy)
    fz = fy - b / 200.0
    del L, a, b
    x = torch.where(fx > epsilon, torch.pow(fx, 3.0), fx.mul(116.0).sub_(16.0).div_(kappa))
    y = torch.where(fy > epsilon, torch.pow(fy, 3.0), fy.mul(116.0).sub_(16.0).div_(kappa))
    z = torch.where(fz > epsilon, torch.pow(fz, 3.0), fz.mul(116.0).sub_(16.0).div_(kappa))
    del fx, fy, fz
    x.mul_(D65_WHITE_X)
    z.mul_(D65_WHITE_Z)
    xyz = torch.stack([x, y, z], dim=1)
    del x, y, z
    B, _, H, W = xyz.shape
    xyz_flat = xyz.permute(0, 2, 3, 1).reshape(-1, 3)
    del xyz
    xyz_flat = xyz_flat.to(dtype=matrix_inv.dtype)
    rgb_linear_flat = torch.matmul(xyz_flat, matrix_inv.T)
    del xyz_flat
    rgb_linear = rgb_linear_flat.reshape(B, H, W, 3).permute(0, 3, 1, 2)
    del rgb_linear_flat
    mask = rgb_linear > 0.0031308
    rgb = torch.where(mask, torch.pow(torch.clamp(rgb_linear, min=0.0), 1.0 / 2.4).mul_(1.055).sub_(0.055), rgb_linear * 12.92)
    del mask, rgb_linear
    return torch.clamp(rgb, 0.0, 1.0)


def _rgb_to_lab_batch(rgb, matrix, epsilon, kappa):
    mask = rgb > 0.04045
    rgb_linear = torch.where(mask, torch.pow((rgb + 0.055) / 1.055, 2.4), rgb / 12.92)
    del mask
    B, _, H, W = rgb_linear.shape
    rgb_flat = rgb_linear.permute(0, 2, 3, 1).reshape(-1, 3)
    del rgb_linear
    rgb_flat = rgb_flat.to(dtype=matrix.dtype)
    xyz_flat = torch.matmul(rgb_flat, matrix.T)
    del rgb_flat
    xyz = xyz_flat.reshape(B, H, W, 3).permute(0, 3, 1, 2)
    del xyz_flat
    xyz[:, 0].div_(D65_WHITE_X)
    xyz[:, 2].div_(D65_WHITE_Z)
    epsilon_cubed = epsilon ** 3
    mask = xyz > epsilon_cubed
    f_xyz = torch.where(mask, torch.pow(xyz, 1.0 / 3.0), xyz.mul(kappa).add_(16.0).div_(116.0))
    del xyz, mask
    L = f_xyz[:, 1].mul(116.0).sub_(16.0)
    a = (f_xyz[:, 0] - f_xyz[:, 1]).mul_(500.0)
    b = (f_xyz[:, 1] - f_xyz[:, 2]).mul_(200.0)
    del f_xyz
    return torch.stack([L, a, b], dim=1)


def lab_color_transfer(content_feat, style_feat, luminance_weight=0.8):
    content_feat = wavelet_reconstruction(content_feat, style_feat)
    if content_feat.shape != style_feat.shape:
        style_feat = F.interpolate(style_feat, size=content_feat.shape[-2:], mode='bilinear', align_corners=False)
    device = content_feat.device
    original_dtype = content_feat.dtype
    content_feat = content_feat.float()
    style_feat = style_feat.float()
    rgb_to_xyz_matrix = torch.tensor([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]], dtype=torch.float32, device=device)
    xyz_to_rgb_matrix = torch.tensor([[3.2404542, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560], [0.0556434, -0.2040259, 1.0572252]], dtype=torch.float32, device=device)
    epsilon = CIELAB_DELTA
    kappa = CIELAB_KAPPA
    content_feat.add_(1.0).mul_(0.5).clamp_(0.0, 1.0)
    style_feat.add_(1.0).mul_(0.5).clamp_(0.0, 1.0)
    content_lab = _rgb_to_lab_batch(content_feat, rgb_to_xyz_matrix, epsilon, kappa)
    del content_feat
    style_lab = _rgb_to_lab_batch(style_feat, rgb_to_xyz_matrix, epsilon, kappa)
    del style_feat, rgb_to_xyz_matrix
    matched_a = _histogram_matching_channel(content_lab[:, 1], style_lab[:, 1])
    matched_b = _histogram_matching_channel(content_lab[:, 2], style_lab[:, 2])
    if luminance_weight < 1.0:
        matched_L = _histogram_matching_channel(content_lab[:, 0], style_lab[:, 0])
        result_L = content_lab[:, 0].mul(luminance_weight).add_(matched_L.mul(1.0 - luminance_weight))
        del matched_L
    else:
        result_L = content_lab[:, 0]
    del content_lab, style_lab
    result_lab = torch.stack([result_L, matched_a, matched_b], dim=1)
    del result_L, matched_a, matched_b
    result_rgb = _lab_to_rgb_batch(result_lab, xyz_to_rgb_matrix, epsilon, kappa)
    del result_lab, xyz_to_rgb_matrix
    result = result_rgb.mul_(2.0).sub_(1.0)
    del result_rgb
    result = result.to(original_dtype)
    return result


class StarTiledSeedVRUpscaler:
    def __init__(self):
        self._model_name = None
        self._model = None
        self._vae_name = None
        self._vae = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "The SeedVR2 diffusion model."}),
                "vae_name": (folder_paths.get_filename_list("vae"), {"tooltip": "The SeedVR2 (EMA) VAE."}),
                "scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.25, "tooltip": "Upscale factor for the output image."}),
                "rows": ("INT", {"default": 3, "min": 1, "max": 16, "tooltip": "Number of tile rows. More rows = smaller tiles = less VRAM."}),
                "cols": ("INT", {"default": 3, "min": 1, "max": 16, "tooltip": "Number of tile columns. More columns = smaller tiles = less VRAM."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "⭐StarNodes/Image And Latent"
    DESCRIPTION = "Upscales an image with SeedVR2 by processing it in overlapping tiles to keep VRAM usage low. Tiles are enhanced one by one and blended back together seamlessly."

    TILE_OVERLAP = 0.1
    VAE_TILE = 1024
    VAE_TILE_OVERLAP = 128
    SEED = 0

    def _load_model(self, model_name):
        if self._model is None or self._model_name != model_name:
            self._model = None
            model_path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
            self._model = comfy.sd.load_diffusion_model(model_path)
            self._model_name = model_name
        return self._model

    def _load_vae(self, vae_name):
        if self._vae is None or self._vae_name != vae_name:
            self._vae = None
            vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
            vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))
            vae.throw_exception_if_invalid()
            self._vae = vae
            self._vae_name = vae_name
        return self._vae

    @staticmethod
    def _lanczos(image, width, height):
        samples = image.movedim(-1, 1)
        samples = comfy.utils.common_upscale(samples, width, height, "lanczos", "disabled")
        return samples.movedim(1, -1)

    def _tile_coords(self, height, width, rows, cols):
        tile_h = height // rows
        tile_w = width // cols
        overlap_y = 0 if rows == 1 else min(tile_h // 2, int(tile_h * self.TILE_OVERLAP))
        overlap_x = 0 if cols == 1 else min(tile_w // 2, int(tile_w * self.TILE_OVERLAP))
        coords = []
        for i in range(rows):
            for j in range(cols):
                y1 = i * tile_h
                x1 = j * tile_w
                if i > 0:
                    y1 -= overlap_y
                if j > 0:
                    x1 -= overlap_x
                y2 = height if i == rows - 1 else y1 + tile_h + overlap_y
                x2 = width if j == cols - 1 else x1 + tile_w + overlap_x
                coords.append((y1, x1, y2, x2, i, j))
        return coords, overlap_x, overlap_y

    @staticmethod
    def _merge_tiles(tiles, coords, height, width, overlap_x, overlap_y):
        first = tiles[0]
        out = torch.zeros((1, height, width, first.shape[-1]), dtype=first.dtype, device=first.device)
        for tile, (y1, x1, y2, x2, i, j) in zip(tiles, coords):
            mask = torch.ones((1, y2 - y1, x2 - x1, 1), dtype=tile.dtype, device=tile.device)
            if i > 0 and overlap_y > 0:
                ramp = torch.linspace(0.0, 1.0, overlap_y, dtype=tile.dtype, device=tile.device)
                mask[:, :overlap_y, :, :] *= ramp.view(1, -1, 1, 1)
            if j > 0 and overlap_x > 0:
                ramp = torch.linspace(0.0, 1.0, overlap_x, dtype=tile.dtype, device=tile.device)
                mask[:, :, :overlap_x, :] *= ramp.view(1, 1, -1, 1)
            out[:, y1:y2, x1:x2, :] = tile * mask + out[:, y1:y2, x1:x2, :] * (1.0 - mask)
        return out

    @staticmethod
    def _conditioning(model, latent):
        diffusion_model = model.model.diffusion_model
        if not hasattr(diffusion_model, "positive_conditioning"):
            raise RuntimeError("Star Tiled SeedVR Upscaler: the selected diffusion model is not a SeedVR2 model.")
        cond = latent.movedim(1, -1).contiguous()
        mask = cond.new_ones(cond.shape[:-1] + (1,))
        condition = torch.cat((cond, mask), dim=-1).movedim(-1, 1)
        positive = [[diffusion_model.positive_conditioning.unsqueeze(0), {"condition": condition}]]
        negative = [[diffusion_model.negative_conditioning.unsqueeze(0), {"condition": condition}]]
        return positive, negative

    def _decode_tiled(self, vae, samples):
        temporal_size = 64
        temporal_overlap = 8
        temporal_compression = vae.temporal_compression_decode()
        if temporal_compression is not None:
            temporal_size = max(2, temporal_size // temporal_compression)
            temporal_overlap = max(1, min(temporal_size // 2, temporal_overlap // temporal_compression))
        else:
            temporal_size = None
            temporal_overlap = None
        compression = vae.spacial_compression_decode()
        images = vae.decode_tiled(samples, tile_x=self.VAE_TILE // compression, tile_y=self.VAE_TILE // compression,
                                  overlap=self.VAE_TILE_OVERLAP // compression, tile_t=temporal_size, overlap_t=temporal_overlap)
        if images.ndim == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return images

    def _process_tile(self, tile, model, vae):
        tile_h, tile_w = tile.shape[1], tile.shape[2]

        # SeedVR2 preprocess: pad to 16px multiples and 4n+1 frames as (B, T, H, W, C)
        images = tile.unsqueeze(0).permute(0, 1, 4, 2, 3)
        b, t, c, h, w = images.shape
        images = images.reshape(b * t, c, h, w).clamp(0.0, 1.0)
        images = div_pad(images, (16, 16))
        images = images.reshape(b, t, c, images.shape[-2], images.shape[-1])
        images = cut_videos(images)
        images = images.permute(0, 1, 3, 4, 2).contiguous()

        latent = vae.encode_tiled(images, tile_x=self.VAE_TILE, tile_y=self.VAE_TILE,
                                  overlap=self.VAE_TILE_OVERLAP, tile_t=64, overlap_t=8)
        positive, negative = self._conditioning(model, latent)
        sampled = nodes.common_ksampler(model, self.SEED, 1, 1.0, "euler", "simple",
                                        positive, negative, {"samples": latent}, denoise=1.0)[0]
        decoded = self._decode_tiled(vae, sampled["samples"])

        decoded = decoded[:1, :tile_h, :tile_w, :3]
        if decoded.shape[1] != tile_h or decoded.shape[2] != tile_w:
            decoded = self._lanczos(decoded, tile_w, tile_h)

        reference = tile.to(device=decoded.device, dtype=decoded.dtype)
        fixed = lab_color_transfer(decoded.movedim(-1, 1) * 2.0 - 1.0, reference.movedim(-1, 1) * 2.0 - 1.0)
        return fixed.movedim(1, -1).add(1.0).div(2.0).clamp(0.0, 1.0)

    def upscale(self, image, model_name, vae_name, scale, rows, cols):
        model = self._load_model(model_name)
        vae = self._load_vae(vae_name)

        pbar = comfy.utils.ProgressBar(image.shape[0] * rows * cols)
        results = []
        for b in range(image.shape[0]):
            img = image[b:b + 1, :, :, :3]
            target_h = max(16, round(img.shape[1] * scale))
            target_w = max(16, round(img.shape[2] * scale))
            upscaled = self._lanczos(img, target_w, target_h)

            coords, overlap_x, overlap_y = self._tile_coords(target_h, target_w, rows, cols)
            tiles = []
            for (y1, x1, y2, x2, i, j) in coords:
                tiles.append(self._process_tile(upscaled[:, y1:y2, x1:x2, :], model, vae))
                pbar.update(1)
            results.append(self._merge_tiles(tiles, coords, target_h, target_w, overlap_x, overlap_y))

        return (torch.cat(results, dim=0),)


NODE_CLASS_MAPPINGS = {
    "StarTiledSeedVRUpscaler": StarTiledSeedVRUpscaler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarTiledSeedVRUpscaler": "⭐ Star Tiled SeedVR Upscaler",
}
