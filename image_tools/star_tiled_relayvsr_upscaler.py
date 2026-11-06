"""Star Tiled RelayVSR Upscaler - image upscaling with the RelayVSR diffusion
model, processed in overlapping tiles to keep VRAM usage low.

Self-contained: the conditioning, one-step sampling and VSR decoding logic is
inlined (adapted from the core RelayVSRConditioning / VSRDecode nodes) instead
of loading the native nodes, so the node keeps working even after core changes.
"""

import logging
import math

import torch

import comfy.model_management
import comfy.sample
import comfy.sd
import comfy.utils
import folder_paths

from ..misc.star_progress import make_event_cb, ProgressReporter

# RelayVSR constants, inlined from comfy.ldm.wan.model_vsr
RELAY_KEYFRAME_INTERVAL = 15


def relay_keyframes(frames):
    return list(range(0, frames - 1, RELAY_KEYFRAME_INTERVAL)) + [frames - 1]


def relay_latent_size(height, width, latent_stride):
    # whole 2x2 patches of the model's pixel stride
    stride = latent_stride * 2
    return math.ceil(height / stride) * 2, math.ceil(width / stride) * 2


class StarTiledRelayVSRUpscaler:
    def __init__(self):
        self._model_name = None
        self._model = None
        self._vae_name = None
        self._vae = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The image to upscale. It goes in at its native size - the RelayVSR checkpoint decides the upscale factor (e.g. x4)."}),
                "model_name": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "The RelayVSR diffusion model (e.g. relayvsr_x4_int8_convrot.safetensors)."}),
                "vae_name": (folder_paths.get_filename_list("vae"), {"tooltip": "The RelayVSR Flash VAE (e.g. relayvsr_flash_vae_x4_fp32.safetensors)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": "fixed", "tooltip": "Random seed for the one-step sampling pass."}),
                "tiled": ("BOOLEAN", {"default": True, "label_on": "tiled", "label_off": "whole image", "tooltip": "On: split the image into overlapping tiles (low VRAM, for large images). Off: process the whole image in a single pass - faster and seam-free, but needs a lot of VRAM on large images. Ignores max_tile_size and tile_overlap."}),
                "max_tile_size": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 64, "tooltip": "Maximum tile width/height in pixels of the input image. The image is split into as many tiles as needed to stay under this size - smaller tiles = less VRAM, larger tiles = fewer seams."}),
                "tile_overlap": ("FLOAT", {"default": 0.25, "min": 0.05, "max": 0.5, "step": 0.05, "tooltip": "Overlap ratio between tiles. Higher values reduce seam artifacts but increase VRAM/time."}),
                "detail": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider", "widget": "slider", "tooltip": "Scales the high-frequency residual the model adds on top of the plain bicubic upscale. 0.0 = pure bicubic upscale, 1.0 = full model output."}),
                "batch_size": ("INT", {"default": 0, "min": 0, "max": 10000, "tooltip": "Frames processed together per tile. 0 = auto: the whole input runs as ONE video pass - the fastest setting for videos (the model only samples every 15th frame anyway). 1 = each image separately (use for unrelated images). >1 = fixed frames per pass (lower VRAM than auto on very long videos)."}),
            },
            "optional": {
                "model_override": ("MODEL", {"tooltip": "Optional model override. When connected, this model will be used instead of the one selected in model_name dropdown. Useful for patched models or custom loaders."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "⭐StarNodes/Image And Latent"
    DESCRIPTION = "Upscales an image with RelayVSR by processing it in overlapping tiles to keep VRAM usage low. Tiles are enhanced one by one with the one-step RelayVSR pass and blended back together seamlessly."

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
            # the RelayVSR Flash decoder reads its upscale_factor from the
            # safetensors metadata, so load and pass it like the core loader
            sd, metadata = comfy.utils.load_torch_file(vae_path, return_metadata=True)
            vae = comfy.sd.VAE(sd=sd, metadata=metadata)
            vae.throw_exception_if_invalid()
            self._vae = vae
            self._vae_name = vae_name
        return self._vae

    @staticmethod
    def _lanczos(image, width, height):
        samples = image.movedim(-1, 1)
        samples = comfy.utils.common_upscale(samples, width, height, "lanczos", "disabled")
        return samples.movedim(1, -1)

    def _tile_coords(self, height, width, max_tile_size, tile_overlap):
        rows = max(1, math.ceil(height / max_tile_size))
        cols = max(1, math.ceil(width / max_tile_size))
        tile_h = math.ceil(height / rows)
        tile_w = math.ceil(width / cols)
        overlap_y = 0 if rows == 1 else min(tile_h // 2, int(tile_h * tile_overlap))
        overlap_x = 0 if cols == 1 else min(tile_w // 2, int(tile_w * tile_overlap))
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
    def _merge_tiles(tiles, coords, height, width, overlap_x, overlap_y, factor):
        first = tiles[0]
        out = torch.zeros((first.shape[0], height, width, first.shape[-1]), dtype=first.dtype, device=first.device)
        for tile, (y1, x1, y2, x2, i, j) in zip(tiles, coords):
            oy1, ox1 = round(y1 * factor), round(x1 * factor)
            ooy, oox = round(overlap_y * factor), round(overlap_x * factor)
            th, tw = tile.shape[1], tile.shape[2]
            mask = torch.ones((1, th, tw, 1), dtype=tile.dtype, device=tile.device)
            if i > 0 and ooy > 0:
                ramp = torch.linspace(0.0, 1.0, ooy, dtype=tile.dtype, device=tile.device)
                mask[:, :ooy, :, :] *= ramp.view(1, -1, 1, 1)
            if j > 0 and oox > 0:
                ramp = torch.linspace(0.0, 1.0, oox, dtype=tile.dtype, device=tile.device)
                mask[:, :, :oox, :] *= ramp.view(1, 1, -1, 1)
            out[:, oy1:oy1 + th, ox1:ox1 + tw, :] = tile * mask + out[:, oy1:oy1 + th, ox1:ox1 + tw, :] * (1.0 - mask)
        return out

    def _conditioning(self, model, tile):
        """RelayVSR conditioning + empty latent for one tile, inlined from the
        core RelayVSRConditioning node. The tile stays at its native size and
        the conditioning holds the fixed prompt - the checkpoint decides the
        upscale factor."""
        diffusion_model = model.model.diffusion_model
        if not hasattr(diffusion_model, "positive_conditioning") or not hasattr(diffusion_model, "latent_stride"):
            raise RuntimeError("Star Tiled RelayVSR Upscaler: the selected diffusion model is not a RelayVSR model.")
        keyframes = relay_keyframes(tile.shape[0])
        positive = [[diffusion_model.positive_conditioning.unsqueeze(0).clone(), {"lq_images": tile[keyframes]}]]
        latent_size = relay_latent_size(tile.shape[1], tile.shape[2], diffusion_model.latent_stride)
        latent = torch.zeros([1, model.model.latent_format.latent_channels, len(keyframes), *latent_size],
                             device=comfy.model_management.intermediate_device())
        return positive, {"samples": latent}

    def _sample(self, model, positive, latent, seed):
        """Single-step RelayVSR pass (1 step, cfg 1.0, euler / simple), inlined
        instead of the core KSampler node. The same conditioning goes to cond
        and uncond - cfg 1.0 drops the uncond batch."""
        latent_image = latent["samples"]
        latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)
        noise = comfy.sample.prepare_noise(latent_image, seed, latent.get("batch_index"))
        samples = comfy.sample.sample(model, noise, 1, 1.0, "euler", "simple", positive, positive, latent_image,
                                      denoise=1.0, force_full_denoise=True, noise_mask=latent.get("noise_mask"),
                                      disable_pbar=True, seed=seed)
        return {"samples": samples}

    def _vsr_decode(self, vae, samples, tile, detail):
        """Inlined VSR Decode: the Flash decoder reads the low-quality frames
        and blends its detail into the bicubic upscale by the detail factor."""
        decoded = vae.decode(samples["samples"], vae_options={"lq_images": tile, "detail": detail})
        return decoded.reshape(-1, *decoded.shape[-3:]).detach()

    def _process_tile(self, tile, model, vae, seed, detail):
        with torch.no_grad():
            positive, latent = self._conditioning(model, tile)
            sampled = self._sample(model, positive, latent, seed)
            return self._vsr_decode(vae, sampled, tile, detail)

    def upscale(self, image, model_name, vae_name, seed, tiled, max_tile_size, tile_overlap, detail, batch_size, unique_id=None, model_override=None):
        import time
        start_time = time.time()
        model = model_override if model_override is not None else self._load_model(model_name)
        vae = self._load_vae(vae_name)

        in_h, in_w = image.shape[1], image.shape[2]
        if tiled:
            coords, overlap_x, overlap_y = self._tile_coords(in_h, in_w, max_tile_size, tile_overlap)
            logging.info(f"Star Tiled RelayVSR: {in_w}x{in_h} input split into {len(coords)} tiles "
                         f"(max tile {max_tile_size}px, overlap {tile_overlap:.2f})")
        else:
            coords, overlap_x, overlap_y = [(0, 0, in_h, in_w, 0, 0)], 0, 0
            logging.info(f"Star Tiled RelayVSR: {in_w}x{in_h} input in a single pass (tiled off)")

        batch_size = int(batch_size)
        if batch_size <= 0:
            # auto: all frames in one video pass - the fastest setting, since
            # the model only samples every 15th frame anyway
            batch_size = image.shape[0]
        chunks = [image[i:i + batch_size] for i in range(0, image.shape[0], batch_size)]
        if len(chunks) > 1 or chunks[0].shape[0] > 1:
            logging.info(f"Star Tiled RelayVSR: {image.shape[0]} frame(s) in {len(chunks)} batch(es) "
                         f"of {batch_size} (video mode: every {RELAY_KEYFRAME_INTERVAL}th frame is a keyframe)")

        total_tiles = len(chunks) * len(coords)
        event_cb = make_event_cb(unique_id)
        reporter = ProgressReporter(total_tiles, label="upscaling", event_cb=event_cb)
        pbar = comfy.utils.ProgressBar(total_tiles)
        results = []
        factor = None
        for chunk in chunks:
            img = chunk[:, :, :, :3]
            in_h, in_w = img.shape[1], img.shape[2]
            tiles = []
            for (y1, x1, y2, x2, i, j) in coords:
                decoded = self._process_tile(img[:, y1:y2, x1:x2, :], model, vae, seed, detail).clamp(0.0, 1.0)
                if factor is None:
                    # upscale factor comes from the checkpoint, measure it once
                    factor = decoded.shape[1] / max(1, y2 - y1)
                # the relay latent rounds up to whole model patches, so resize
                # each tile onto the factor lattice for a seamless merge
                exp_h = max(1, round((y2 - y1) * factor))
                exp_w = max(1, round((x2 - x1) * factor))
                if decoded.shape[1] != exp_h or decoded.shape[2] != exp_w:
                    decoded = self._lanczos(decoded, exp_w, exp_h)
                tiles.append(decoded)
                pbar.update(1)
                reporter.report(fraction=0.0, sub=f"tile {len(tiles)}/{len(coords)}")
                reporter.finish_unit()
            out_h = max(1, round(in_h * factor))
            out_w = max(1, round(in_w * factor))
            results.append(self._merge_tiles(tiles, coords, out_h, out_w, overlap_x, overlap_y, factor))

        reporter.finish_all(time.time() - start_time)
        return (torch.cat(results, dim=0),)


NODE_CLASS_MAPPINGS = {
    "StarTiledRelayVSRUpscaler": StarTiledRelayVSRUpscaler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarTiledRelayVSRUpscaler": "⭐ Star Tiled RelayVSR Upscaler",
}
