import torch

import comfy.sd
import comfy.utils
import folder_paths
import nodes
from comfy.ldm.seedvr.color_fix import lab_color_transfer
from comfy_extras.nodes_seedvr import cut_videos, div_pad


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
