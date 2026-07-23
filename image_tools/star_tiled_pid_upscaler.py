"""
⭐ Star Tiled PiD Upscaler

Tiled image upscaler for NVIDIA PiD / PixelDiT models
(Comfy-Org/PixelDiT checkpoints such as pid_qwenimage_1024_to_4096_4step_bf16).

PiD is a latent-conditioned *pixel-space* diffusion upscaler:
the low-res image is VAE-encoded and injected into the diffusion model
("lq_latent"), which then re-renders the image directly in pixel space at
the requested output size in ~4 distilled steps (no VAE decode needed —
a virtual "pixel_space" VAE is used).

This node replicates the reference ComfyUI workflow
(UNETLoader + CLIPLoader(pixeldit) + VAEEncode + PiDConditioning +
ManualSigmas + LCM SamplerCustom + pixel_space VAEDecode + ColorTransfer)
but processes the image in overlapping tiles so VRAM usage stays low and
much bigger upscales are possible.  Tiles are blended back together with
feathered masks and each tile is color-matched to the source image
(wavelet + LAB histogram transfer), which replaces both the PiD color
bias patch and the final ColorTransfer node of the manual workflow.

Requires a recent ComfyUI with native PixelDiT/PiD support
(PiDConditioning core node / comfy.ldm.pixeldit).
"""

import re

import torch
import torch.nn.functional as F

import comfy.latent_formats
import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.sd
import comfy.utils
import folder_paths
import node_helpers

from ..star_progress import make_event_cb, ProgressReporter


# ---------------------------------------------------------------------------
# Color fix — inlined from star_tiled_seedvr_upscaler (originally from
# comfy.ldm.seedvr.color_fix) so this file is self-contained.
# ---------------------------------------------------------------------------
D65_WHITE_X = 0.95047
D65_WHITE_Z = 1.08883
CIELAB_DELTA = 6.0 / 29.0
CIELAB_KAPPA = (29.0 / 3.0) ** 3


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


# ---------------------------------------------------------------------------
# PiD color bias correction (Flux2 backbone) — ported from KJNodes
# "PiD Color Bias Correction" (kijai/ComfyUI-KJNodes, MIT license).
# Only applied for Flux2 PiD models (128-channel latents); the per-tile LAB
# color transfer above already handles drift for all other backbones.
# ---------------------------------------------------------------------------
PID_BIAS_COEF_FLUX2 = torch.tensor([
    [-0.130306, +0.127184, +0.014058],  # R_mean
    [-0.053279, -0.408929, +0.004243],  # G_mean
    [-0.009386, +0.109546, -0.134091],  # B_mean
    [-0.033373, -0.011615, -0.026129],  # R_std
    [+0.180052, +0.062021, +0.071317],  # G_std
    [-0.067958, -0.058595, -0.098645],  # B_std
    [-0.248116, -0.240633, -0.105600],  # R_mean*G_mean
    [+0.304035, +0.322566, +0.093224],  # R_mean*B_mean
    [-0.157648, -0.227127, -0.112368],  # G_mean*B_mean
    [-0.062814, +0.030765, +0.062735],  # intercept
], dtype=torch.float32)


def apply_pid_bias_patch(model, strength):
    """Clone *model* and subtract the predicted per-channel bias from the
    x0 prediction at the first sampling step (Flux2 PiD only)."""
    coef_cpu = PID_BIAS_COEF_FLUX2  # (10, 3)

    def pid_bias_post_cfg(args):
        denoised = args["denoised"]
        try:
            sigmas = args["model_options"]["transformer_options"]["sample_sigmas"]
            sigma = args.get("sigma", args.get("timestep"))
            if sigma is None or not torch.isclose(sigma.max(), sigmas[0]).item():
                return denoised
        except (KeyError, AttributeError):
            sigma = args.get("sigma")
            if sigma is None or sigma.max().item() < 0.95:
                return denoised

        coef = coef_cpu.to(denoised.device, dtype=denoised.dtype)
        rgb_m = denoised.mean(dim=(0, 2, 3))
        rgb_s = denoised.std(dim=(0, 2, 3))
        one = torch.tensor(1.0, device=denoised.device, dtype=denoised.dtype)
        feats = torch.stack([
            rgb_m[0], rgb_m[1], rgb_m[2],
            rgb_s[0], rgb_s[1], rgb_s[2],
            rgb_m[0] * rgb_m[1], rgb_m[0] * rgb_m[2], rgb_m[1] * rgb_m[2],
            one,
        ])
        bias = feats @ coef  # (3,)
        return denoised - strength * bias.view(1, 3, 1, 1)

    m = model.clone()
    m.set_model_sampler_post_cfg_function(pid_bias_post_cfg)
    return m


class StarTiledPiDUpscaler:
    def __init__(self):
        self._model_name = None
        self._model = None
        self._clip_name = None
        self._clip = None
        self._vae_name = None
        self._vae = None
        self._pixel_vae = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "The PiD / PixelDiT diffusion model, e.g. pid_qwenimage_1024_to_4096_4step_bf16.safetensors."}),
                "clip_name": (folder_paths.get_filename_list("text_encoders"), {"tooltip": "Gemma 2 2B text encoder for PixelDiT (gemma_2_2b_it_elm). PiD uses an empty prompt, but the model still needs the text encoder loaded."}),
                "vae_name": (folder_paths.get_filename_list("vae"), {"tooltip": "VAE matching the PiD backbone, used to encode the input image (e.g. qwen_image_vae for qwenimage, ae.safetensors for flux)."}),
                "latent_format": (["qwenimage", "flux", "sd3", "sdxl"], {"default": "qwenimage", "tooltip": "Latent format of the PiD backbone. Flux1 (16-ch) and Flux2 (128-ch) are auto-detected under 'flux'."}),
                "scale": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 8.0, "step": 0.25, "tooltip": "Upscale factor for the output image. PiD models are trained for 4x."}),
                "rows": ("INT", {"default": 2, "min": 1, "max": 16, "tooltip": "Number of tile rows. More rows = smaller tiles = less VRAM."}),
                "cols": ("INT", {"default": 2, "min": 1, "max": 16, "tooltip": "Number of tile columns. More columns = smaller tiles = less VRAM."}),
            },
            # NOTE: degrade_sigma / sigmas / color_bias_fix are intentionally
            # NOT in INPUT_TYPES. ComfyUI only passes "hidden" inputs for its
            # reserved keys (UNIQUE_ID, PROMPT, ...) — custom hidden names are
            # never sent, which would crash with missing arguments. Instead
            # they are keyword defaults on upscale() below: always available,
            # never shown as widgets.
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "⭐StarNodes/Image And Latent"
    DESCRIPTION = "Upscales an image with a PiD / PixelDiT model by processing it in overlapping tiles to keep VRAM usage low. Each tile is re-rendered in pixel space (4 LCM steps) and blended back seamlessly, with per-tile color matching."

    TILE_OVERLAP = 0.1
    SEED = 3

    # ─── model / clip / vae loading (cached) ──────────────────────────────
    def _load_model(self, model_name):
        if self._model is None or self._model_name != model_name:
            self._model = None
            model_path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
            model = comfy.sd.load_diffusion_model(model_path)
            diffusion_model = getattr(model.model, "diffusion_model", None)
            if diffusion_model is None or not hasattr(diffusion_model, "lq_proj"):
                raise RuntimeError("Star Tiled PiD Upscaler: the selected diffusion model is not a PiD / PixelDiT model.")
            self._model = model
            self._model_name = model_name
        return self._model

    def _load_clip(self, clip_name):
        if self._clip is None or self._clip_name != clip_name:
            self._clip = None
            clip_type = getattr(comfy.sd.CLIPType, "PIXELDIT", None)
            if clip_type is None:
                raise RuntimeError("Star Tiled PiD Upscaler: this ComfyUI version has no PixelDiT support — please update ComfyUI.")
            clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
            self._clip = comfy.sd.load_clip(ckpt_paths=[clip_path],
                                            embedding_directory=folder_paths.get_folder_paths("embeddings"),
                                            clip_type=clip_type)
            self._clip_name = clip_name
        return self._clip

    def _load_vae(self, vae_name):
        if self._vae is None or self._vae_name != vae_name:
            self._vae = None
            vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
            vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))
            vae.throw_exception_if_invalid()
            self._vae = vae
            self._vae_name = vae_name
        return self._vae

    def _load_pixel_vae(self):
        # Virtual "pixel_space" VAE (same as VAELoader -> pixel_space):
        # decode is identity + remap from [-1, 1] to [0, 1].
        if self._pixel_vae is None:
            self._pixel_vae = comfy.sd.VAE(sd={"pixel_space_vae": torch.tensor(1.0)})
        return self._pixel_vae

    # ─── small helpers ─────────────────────────────────────────────────────
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
    def _latent_format_cls(latent_format, samples):
        if latent_format == "flux":
            return comfy.latent_formats.Flux2 if samples.shape[1] == 128 else comfy.latent_formats.Flux
        if latent_format == "sd3":
            return comfy.latent_formats.SD3
        if latent_format == "sdxl":
            return comfy.latent_formats.SDXL
        if latent_format == "qwenimage":
            return comfy.latent_formats.Wan21
        raise ValueError(f"Unknown latent_format: {latent_format}")

    # ─── per-tile PiD pass ─────────────────────────────────────────────────
    def _process_tile(self, tile, out_h, out_w, factor, model, vae, pixel_vae, empty_cond,
                      latent_format, degrade_sigma_t, sampler, sigmas_t):
        tile_h, tile_w = tile.shape[1], tile.shape[2]
        # PiD always runs at its native factor of the *input tile* — never at
        # the final scale — so sampling VRAM is independent of `scale`.
        native_h = max(16, tile_h * factor)
        native_w = max(16, tile_w * factor)

        # 1. Encode the low-res tile with the backbone VAE (like VAEEncode).
        samples = vae.encode(tile[:, :, :, :3])

        # 2. PiDConditioning: attach processed lq_latent + degrade_sigma.
        fmt_cls = self._latent_format_cls(latent_format, samples)
        lq_latent = fmt_cls().process_in(samples)
        if lq_latent.ndim == 5:
            lq_latent = lq_latent[:, :, 0]
        positive = node_helpers.conditioning_set_values(
            empty_cond, {"lq_latent": lq_latent, "degrade_sigma": degrade_sigma_t})
        del samples, lq_latent

        # 3. Empty pixel-space latent at the native (factor x) tile size
        #    (like EmptyChromaRadianceLatentImage: PiD works in pixel space).
        latent_image = torch.zeros((1, 3, native_h, native_w),
                                   device=comfy.model_management.intermediate_device())

        # 4. Sample 4 distilled steps with the LCM sampler + manual sigmas
        #    (like KSamplerSelect + ManualSigmas + SamplerCustom, cfg=1).
        noise = comfy.sample.prepare_noise(latent_image, self.SEED)
        disable_pbar = not getattr(comfy.utils, "PROGRESS_BAR_ENABLED", True)
        sampled = comfy.sample.sample_custom(model, noise, 1.0, sampler, sigmas_t,
                                             positive, empty_cond, latent_image,
                                             disable_pbar=disable_pbar, seed=self.SEED)
        del noise, latent_image, positive

        # 5. Pixel-space "decode": identity + remap [-1, 1] -> [0, 1].
        decoded = pixel_vae.decode(sampled)
        del sampled

        # 6. Bring the native (2x/4x) tile to its final slot size on the
        #    output canvas. This is where `scale` is applied: scale > factor
        #    upscales the tile further, scale < factor supersamples it down.
        decoded = decoded[:1, :, :, :3]
        if decoded.shape[1] > native_h or decoded.shape[2] > native_w:
            decoded = decoded[:, :native_h, :native_w, :]
        if decoded.shape[1] != out_h or decoded.shape[2] != out_w:
            decoded = self._lanczos(decoded, out_w, out_h)

        # 7. Color match the tile to the source (replaces the ColorTransfer
        #    node of the manual workflow and keeps the original colors).
        reference = self._lanczos(tile.to(device=decoded.device, dtype=decoded.dtype), out_w, out_h)
        fixed = lab_color_transfer(decoded.movedim(-1, 1) * 2.0 - 1.0,
                                   reference.movedim(-1, 1) * 2.0 - 1.0)
        return fixed.movedim(1, -1).add(1.0).div(2.0).clamp(0.0, 1.0)

    # ─── main entry ────────────────────────────────────────────────────────
    def upscale(self, image, model_name, clip_name, vae_name, latent_format, scale,
                rows, cols, unique_id=None,
                # Hidden advanced settings — used internally, not shown as widgets.
                degrade_sigma=0.1, sigmas="0.999, 0.866, 0.634, 0.342, 0", color_bias_fix=1.0):
        import time
        start_time = time.time()
        model = self._load_model(model_name)
        clip = self._load_clip(clip_name)
        vae = self._load_vae(vae_name)
        pixel_vae = self._load_pixel_vae()
        factor = 4  # Fixed to 4x native upscale

        # PiD uses an empty prompt; the Gemma text encoder still provides the
        # required context embeddings. Same conditioning serves as negative
        # (cfg is 1.0, so the negative branch is skipped anyway).
        tokens = clip.tokenize("")
        empty_cond = clip.encode_from_tokens_scheduled(tokens)

        sigma_values = [float(x) for x in re.findall(r"[-+]?(?:\d*\.*\d+)", sigmas)]
        if len(sigma_values) < 2:
            raise ValueError("Star Tiled PiD Upscaler: sigmas must contain at least two values, e.g. \"0.999, 0.866, 0.634, 0.342, 0\".")
        sigmas_t = torch.FloatTensor(sigma_values)
        sampler = comfy.samplers.sampler_object("lcm")
        degrade_sigma_t = torch.tensor([float(degrade_sigma)], dtype=torch.float32)

        # Optional Flux2 color bias correction (auto-detected: flux format +
        # 128-channel VAE latents). Other backbones rely on LAB color transfer.
        is_flux2 = latent_format == "flux" and getattr(vae, "latent_channels", 0) == 128
        if color_bias_fix != 0.0 and is_flux2:
            model = apply_pid_bias_patch(model, color_bias_fix)

        total_tiles = image.shape[0] * rows * cols
        event_cb = make_event_cb(unique_id)
        reporter = ProgressReporter(total_tiles, label="upscaling", event_cb=event_cb)
        results = []
        for b in range(image.shape[0]):
            img = image[b:b + 1, :, :, :3]
            src_h, src_w = img.shape[1], img.shape[2]
            target_h = max(16, round(src_h * scale))
            target_w = max(16, round(src_w * scale))

            # Tiles are laid out on the *target* canvas; the matching region
            # of the original (low-res) image conditions each tile.
            coords, overlap_x, overlap_y = self._tile_coords(target_h, target_w, rows, cols)
            tiles = []
            for (y1, x1, y2, x2, i, j) in coords:
                iy1 = max(0, min(int(round(y1 / scale)), src_h - 8))
                iy2 = max(iy1 + 8, min(int(round(y2 / scale)), src_h))
                ix1 = max(0, min(int(round(x1 / scale)), src_w - 8))
                ix2 = max(ix1 + 8, min(int(round(x2 / scale)), src_w))
                tile = img[:, iy1:iy2, ix1:ix2, :]
                tiles.append(self._process_tile(tile, y2 - y1, x2 - x1, factor, model, vae,
                                                pixel_vae, empty_cond, latent_format,
                                                degrade_sigma_t, sampler, sigmas_t))
                # finish_unit() pushes the DOM event: each done tile adds 100/total_tiles %
                reporter.finish_unit(sub=f"tile {len(tiles)}/{len(coords)}")
            results.append(self._merge_tiles(tiles, coords, target_h, target_w, overlap_x, overlap_y))

        reporter.finish_all(time.time() - start_time)
        return (torch.cat(results, dim=0),)


NODE_CLASS_MAPPINGS = {
    "StarTiledPiDUpscaler": StarTiledPiDUpscaler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarTiledPiDUpscaler": "⭐ Star Tiled PiD Upscaler",
}
