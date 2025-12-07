"""Star SD Upscale Refiner with optional simple image upscaling."""

import math
import inspect
from copy import deepcopy

import torch
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

import comfy.model_patcher as comfy_model_patcher
import comfy.samplers as comfy_samplers
import comfy.model_management as comfy_model_management
from comfy.controlnet import ControlNet, T2IAdapter
from comfy.utils import common_upscale
from comfy.model_management import processing_interrupted, loaded_models, load_models_gpu
import folder_paths
from nodes import (
    CheckpointLoaderSimple,
    LoraLoader,
    CLIPTextEncode,
    ControlNetApplyAdvanced,
    ControlNetLoader,
)
from comfy_extras.nodes_upscale_model import UpscaleModelLoader, ImageUpscaleWithModel

try:
    from comfy_extras.nodes_custom_sampler import KSamplerSelect, SamplerCustom
    from comfy_extras.nodes_align_your_steps import AlignYourStepsScheduler
    from nodes import VAEDecodeTiled
except ImportError:  # Fallback so ComfyUI can still import the extension
    KSamplerSelect = None
    SamplerCustom = None
    AlignYourStepsScheduler = None
    VAEDecodeTiled = None


def _fourier_filter(x, threshold, scale):
    x_freq = torch.fft.fftn(x.float(), dim=(-2, -1))
    x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))

    b, c, h, w = x_freq.shape
    mask = torch.ones((b, c, h, w), device=x.device)

    crow, ccol = h // 2, w // 2
    mask[..., crow - threshold : crow + threshold, ccol - threshold : ccol + threshold] = scale
    x_freq = x_freq * mask

    x_freq = torch.fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = torch.fft.ifftn(x_freq, dim=(-2, -1)).real

    return x_filtered.to(x.dtype)


def _apply_freeu_v2(model, b1, b2, s1, s2):
    # Some models (e.g. non-UNet / flow models) may not expose model_channels
    # in the standard SD unet_config. In those cases we simply skip FreeU.
    try:
        model_config = getattr(model.model, "model_config", None)
        if model_config is None:
            return model

        unet_config = getattr(model_config, "unet_config", None)
        if unet_config is None:
            # model_config might be a dict-like
            unet_config = getattr(model_config, "get", lambda *_args, **_kwargs: None)("unet_config")
        if not unet_config or "model_channels" not in unet_config:
            return model

        model_channels = unet_config["model_channels"]
    except Exception:
        return model

    scale_dict = {model_channels * 4: (b1, s1), model_channels * 2: (b2, s2)}
    on_cpu_devices = {}

    def output_block_patch(h, hsp, transformer_options):
        scale = scale_dict.get(int(h.shape[1]), None)
        if scale is not None:
            hidden_mean = h.mean(1).unsqueeze(1)
            b = hidden_mean.shape[0]
            hidden_max, _ = torch.max(hidden_mean.view(b, -1), dim=-1, keepdim=True)
            hidden_min, _ = torch.min(hidden_mean.view(b, -1), dim=-1, keepdim=True)
            hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)

            h[:, : h.shape[1] // 2] = h[:, : h.shape[1] // 2] * ((scale[0] - 1) * hidden_mean + 1)

            if hsp.device not in on_cpu_devices:
                try:
                    hsp = _fourier_filter(hsp, threshold=1, scale=scale[1])
                except Exception:
                    on_cpu_devices[hsp.device] = True
                    hsp = _fourier_filter(hsp.cpu(), threshold=1, scale=scale[1]).to(hsp.device)
            else:
                hsp = _fourier_filter(hsp.cpu(), threshold=1, scale=scale[1]).to(hsp.device)

        return h, hsp

    m = model.clone()
    m.set_model_output_block_patch(output_block_patch)
    return m


def _apply_pag(model, scale):
    unet_block = "middle"
    unet_block_id = 0
    m = model.clone()

    def perturbed_attention(q, k, v, extra_options, mask=None):
        return v

    def post_cfg_function(args):
        model_inner = args["model"]
        cond_pred = args["cond_denoised"]
        cond = args["cond"]
        cfg_result = args["denoised"]
        sigma = args["sigma"]
        model_options = args["model_options"].copy()
        x = args["input"]

        if scale == 0:
            return cfg_result

        model_options = comfy_model_patcher.set_model_options_patch_replace(
            model_options,
            perturbed_attention,
            "attn1",
            unet_block,
            unet_block_id,
        )
        (pag,) = comfy_samplers.calc_cond_batch(model_inner, [cond], x, sigma, model_options)

        return cfg_result + (cond_pred - pag) * scale

    m.set_model_sampler_post_cfg_function(post_cfg_function)

    return m


def _make_sampler_pre_cfg_function(
    minimum_sigma_to_disable_uncond=0,
    maximum_sigma_to_enable_uncond=1000000,
    disabled_cond_start=10000,
    disabled_cond_end=10000,
):
    def sampler_pre_cfg_function(sigma, uncond, cond, cond_scale, **kwargs):
        if sigma[0] < minimum_sigma_to_disable_uncond or sigma[0] > maximum_sigma_to_enable_uncond:
            uncond = None
        if sigma[0] <= disabled_cond_start and sigma[0] > disabled_cond_end:
            cond = None
        return uncond, cond, cond_scale

    return sampler_pre_cfg_function


_original_sampling_function = None


def _sampling_function_patched(model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None, **kwargs):
    cond_copy = cond
    uncond_copy = uncond

    for fn in model_options.get("sampler_patch_model_pre_cfg_function", []):
        args = {"model": model, "sigma": timestep, "model_options": model_options}
        model, model_options = fn(args)

    if "sampler_pre_cfg_function" in model_options:
        uncond, cond, cond_scale = model_options["sampler_pre_cfg_function"](
            sigma=timestep,
            uncond=uncond,
            cond=cond,
            cond_scale=cond_scale,
        )

    if math.isclose(cond_scale, 1.0) and model_options.get("disable_cfg1_optimization", False) is False:
        uncond_ = None
    else:
        uncond_ = uncond

    conds = [cond, uncond_]

    out = comfy_samplers.calc_cond_batch(model, conds, x, timestep, model_options)
    cond_pred = out[0]
    uncond_pred = out[1]

    if "sampler_cfg_function" in model_options:
        args = {
            "cond": x - cond_pred,
            "uncond": x - uncond_pred,
            "cond_scale": cond_scale,
            "timestep": timestep,
            "input": x,
            "sigma": timestep,
            "cond_denoised": cond_pred,
            "uncond_denoised": uncond_pred,
            "model": model,
            "model_options": model_options,
            "cond_pos": cond_copy,
            "cond_neg": uncond_copy,
        }
        cfg_result = x - model_options["sampler_cfg_function"](args)
    else:
        cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale

    for fn in model_options.get("sampler_post_cfg_function", []):
        args = {
            "denoised": cfg_result,
            "cond": cond_copy,
            "uncond": uncond_copy,
            "model": model,
            "uncond_denoised": uncond_pred,
            "cond_denoised": cond_pred,
            "sigma": timestep,
            "model_options": model_options,
            "input": x,
        }
        cfg_result = fn(args)

    return cfg_result


def _monkey_patching_comfy_sampling_function():
    global _original_sampling_function

    if _original_sampling_function is None:
        _original_sampling_function = comfy_samplers.sampling_function
    if hasattr(comfy_samplers.sampling_function, "_automatic_cfg_decorated"):
        return
    comfy_samplers.sampling_function = _sampling_function_patched
    comfy_samplers.sampling_function._automatic_cfg_decorated = True


def _get_denoised_ranges(latent, measure="hard", top_k=0.25):
    chans = []
    for x in range(len(latent)):
        max_values = torch.topk(
            latent[x] - latent[x].mean() if measure == "range" else latent[x],
            k=int(len(latent[x]) * top_k),
            largest=True,
        ).values
        min_values = torch.topk(
            latent[x] - latent[x].mean() if measure == "range" else latent[x],
            k=int(len(latent[x]) * top_k),
            largest=False,
        ).values
        max_val = torch.mean(max_values).item()
        min_val = (
            abs(torch.mean(min_values).item())
            if measure == "soft"
            else torch.mean(torch.abs(min_values)).item()
        )
        denoised_range = (max_val + min_val) / 2
        chans.append(denoised_range**2 if measure == "hard_squared" else denoised_range)
    return chans


def _get_sigmin_sigmax(model):
    model_sampling = model.model.model_sampling
    sigmin = model_sampling.sigma(model_sampling.timestep(model_sampling.sigma_min))
    sigmax = model_sampling.sigma(model_sampling.timestep(model_sampling.sigma_max))
    return sigmin, sigmax


def _check_skip(sigma, high_sigma_threshold, low_sigma_threshold):
    return sigma > high_sigma_threshold or sigma < low_sigma_threshold


def _apply_auto_cfg(model):
    m = model.clone()

    model_options_copy = deepcopy(m.model_options)
    _monkey_patching_comfy_sampling_function()

    automatic_cfg = "hard"
    skip_uncond = True
    fake_uncond_start = False
    uncond_sigma_start = 1000.0
    uncond_sigma_end = 1.0
    auto_cfg_topk = 0.25
    auto_cfg_ref = 8.0

    lerp_uncond = False
    lerp_uncond_strength = 1.0
    lerp_uncond_sigma_start = 1000.0
    lerp_uncond_sigma_end = 1.0

    cond_exp = False
    uncond_exp = False
    fake_uncond_exp = False
    fake_uncond_multiplier = 1.0
    fake_uncond_sigma_start = 1000.0
    fake_uncond_sigma_end = 1.0

    attention_modifiers_positive = []
    attention_modifiers_negative = []
    attention_modifiers_fake_negative = []
    eval_string_cond = ""
    eval_string_uncond = ""
    eval_string_fake = ""

    last_cfg_ht_one = {"value": 1.0}
    previous_cond_pred = {"value": None}
    previous_sigma = {"value": None}

    sigmin, sigmax = _get_sigmin_sigmax(m)

    if skip_uncond:
        m.model_options["sampler_pre_cfg_function"] = _make_sampler_pre_cfg_function(
            uncond_sigma_end,
            uncond_sigma_start,
            100000.0,
            100000.0,
        )

    def automatic_cfg_function(args):
        cond_scale = args["cond_scale"]
        input_x = args["input"]
        cond_pred = args["cond_denoised"]
        uncond_pred = args["uncond_denoised"]
        sigma = args["sigma"][0]
        model_options = args["model_options"]

        if previous_cond_pred["value"] is None:
            previous_cond_pred["value"] = cond_pred.clone().detach().to(device=cond_pred.device)
        if previous_sigma["value"] is None:
            previous_sigma["value"] = sigma.item()

        reference_cfg = auto_cfg_ref if auto_cfg_ref > 0 else cond_scale

        def fake_uncond_step():
            return (
                fake_uncond_start
                and skip_uncond
                and (sigma > uncond_sigma_start or sigma < uncond_sigma_end)
                and sigma <= fake_uncond_sigma_start
                and sigma >= fake_uncond_sigma_end
            )

        if fake_uncond_step():
            uncond_pred = cond_pred.clone().detach().to(device=cond_pred.device) * fake_uncond_multiplier

        if cond_exp and sigma <= lerp_uncond_sigma_start and sigma >= lerp_uncond_sigma_end:
            cond_pred = cond_pred
        if uncond_exp and sigma <= lerp_uncond_sigma_start and sigma >= lerp_uncond_sigma_end and not fake_uncond_step():
            uncond_pred = uncond_pred
        if fake_uncond_step() and fake_uncond_exp:
            uncond_pred = uncond_pred

        previous_cond_pred["value"] = cond_pred.clone().detach().to(device=cond_pred.device)

        if sigma >= sigmax or cond_scale > 1:
            last_cfg_ht_one["value"] = cond_scale
        target_intensity = last_cfg_ht_one["value"] / 10.0

        if ((
            _check_skip(sigma, uncond_sigma_start, uncond_sigma_end)
            and skip_uncond
        ) and not fake_uncond_step()) or cond_scale == 1:
            return input_x - cond_pred

        if (
            lerp_uncond
            and not _check_skip(sigma, lerp_uncond_sigma_start, lerp_uncond_sigma_end)
            and lerp_uncond_strength != 1
        ):
            uncond_pred_norm = uncond_pred.norm()
            uncond_pred = torch.lerp(cond_pred, uncond_pred, lerp_uncond_strength)
            uncond_pred = uncond_pred * uncond_pred_norm / uncond_pred.norm()

        cond = input_x - cond_pred
        uncond = input_x - uncond_pred

        if automatic_cfg == "None":
            return uncond + cond_scale * (cond - uncond)

        denoised_tmp = input_x - (uncond + reference_cfg * (cond - uncond))

        for b in range(len(denoised_tmp)):
            denoised_ranges = _get_denoised_ranges(denoised_tmp[b], automatic_cfg, auto_cfg_topk)
            for c in range(len(denoised_tmp[b])):
                fixeds_scale = reference_cfg * target_intensity / denoised_ranges[c]
                denoised_tmp[b][c] = uncond[b][c] + fixeds_scale * (cond[b][c] - uncond[b][c])

        return denoised_tmp

    m.model_options["sampler_cfg_function"] = automatic_cfg_function

    return m


def _ceildiv(big, small):
    return -(big // -small)


class _BBox:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.box = [x, y, x + w, y + h]
        self.slicer = (slice(None), slice(None), slice(y, y + h), slice(x, x + w))

    def __getitem__(self, idx):
        return self.box[idx]


def _repeat_to_batch_size(tensor, batch_size, dim=0):
    if dim == 0 and tensor.shape[dim] == 1:
        return tensor.expand([batch_size] + [-1] * (len(tensor.shape) - 1))
    if tensor.shape[dim] > batch_size:
        return tensor.narrow(dim, 0, batch_size)
    if tensor.shape[dim] < batch_size:
        reps = dim * [1] + [_ceildiv(batch_size, tensor.shape[dim])] + [1] * (len(tensor.shape) - 1 - dim)
        return tensor.repeat(reps).narrow(dim, 0, batch_size)
    return tensor


def _split_bboxes(w, h, tile_w, tile_h, overlap=16, init_weight=1.0):
    cols = _ceildiv((w - overlap), (tile_w - overlap))
    rows = _ceildiv((h - overlap), (tile_h - overlap))
    dx = (w - tile_w) / (cols - 1) if cols > 1 else 0
    dy = (h - tile_h) / (rows - 1) if rows > 1 else 0

    bbox_list = []
    weight = torch.zeros((1, 1, h, w), device=comfy_model_management.get_torch_device(), dtype=torch.float32)
    for row in range(rows):
        y = min(int(row * dy), h - tile_h)
        for col in range(cols):
            x = min(int(col * dx), w - tile_w)
            bbox = _BBox(x, y, tile_w, tile_h)
            bbox_list.append(bbox)
            weight[bbox.slicer] += init_weight

    return bbox_list, weight


class _AbstractDiffusion:
    def __init__(self):
        self.method = self.__class__.__name__
        self.pbar = None

        self.w = 0
        self.h = 0
        self.tile_width = None
        self.tile_height = None
        self.tile_overlap = None
        self.tile_batch_size = None

        self.x_buffer = None
        self._weights = None
        self._init_grid_bbox = None
        self._init_done = None

        self.step_count = 0
        self.inner_loop_count = 0
        self.kdiff_step = -1

        self.enable_grid_bbox = False
        self.tile_w = None
        self.tile_h = None
        self.tile_bs = None
        self.num_tiles = None
        self.num_batches = None
        self.batched_bboxes = []

        self.enable_custom_bbox = False
        self.custom_bboxes = []

        self.enable_controlnet = False
        self.control_tensor_batch_dict = {}
        self.control_tensor_batch = [[]]
        self.control_params = {}
        self.control_tensor_cpu = False
        self.control_tensor_custom = []

        self.draw_background = True
        self.weights = None
        self.imagescale = None
        self.uniform_distribution = None
        self.sigmas = None

    def reset_buffer(self, x_in):
        if self.x_buffer is None or self.x_buffer.shape != x_in.shape:
            self.x_buffer = torch.zeros_like(x_in, device=x_in.device, dtype=x_in.dtype)
        else:
            self.x_buffer.zero_()

    def init_grid_bbox(self, tile_w, tile_h, overlap, tile_bs):
        self.weights = torch.zeros((1, 1, self.h, self.w), device=comfy_model_management.get_torch_device(), dtype=torch.float32)
        self.enable_grid_bbox = True

        self.tile_w = min(tile_w, self.w)
        self.tile_h = min(tile_h, self.h)
        overlap = max(0, min(overlap, min(tile_w, tile_h) - 4))
        bboxes, weights = _split_bboxes(self.w, self.h, self.tile_w, self.tile_h, overlap, self.get_tile_weights())
        self.weights += weights
        self.num_tiles = len(bboxes)
        self.num_batches = _ceildiv(self.num_tiles, tile_bs)
        self.tile_bs = _ceildiv(len(bboxes), self.num_batches)
        self.batched_bboxes = [bboxes[i * self.tile_bs : (i + 1) * self.tile_bs] for i in range(self.num_batches)]

    def get_grid_bbox(self, tile_w, tile_h, overlap, tile_bs, w, h, device, get_tile_weights=lambda: 1.0):
        weights = torch.zeros((1, 1, h, w), device=device, dtype=torch.float32)

        tile_w = min(tile_w, w)
        tile_h = min(tile_h, h)
        overlap = max(0, min(overlap, min(tile_w, tile_h) - 4))
        bboxes, weights_ = _split_bboxes(w, h, tile_w, tile_h, overlap, get_tile_weights())
        weights += weights_
        num_tiles = len(bboxes)
        num_batches = _ceildiv(num_tiles, tile_bs)
        tile_bs = _ceildiv(len(bboxes), num_batches)
        batched_bboxes = [bboxes[i * tile_bs : (i + 1) * tile_bs] for i in range(num_batches)]
        return batched_bboxes

    def get_tile_weights(self):
        return 1.0

    def init_done(self):
        self.total_bboxes = 0
        if self.enable_grid_bbox:
            self.total_bboxes += self.num_batches
        if self.enable_custom_bbox:
            self.total_bboxes += len(self.custom_bboxes)
        assert self.total_bboxes > 0

    def prepare_controlnet_tensors(self, refresh=False, tensor=None):
        if not refresh:
            if self.control_tensor_batch is not None or self.control_params is not None:
                return
        tensors = [tensor]
        self.org_control_tensor_batch = tensors
        self.control_tensor_batch = []
        for i in range(len(tensors)):
            control_tile_list = []
            control_tensor = tensors[i]
            for bboxes in self.batched_bboxes:
                single_batch_tensors = []
                for bbox in bboxes:
                    if len(control_tensor.shape) == 3:
                        control_tensor = control_tensor.unsqueeze(0)
                    control_tile = control_tensor[:, :, bbox[1] * self.compression : bbox[3] * self.compression, bbox[0] * self.compression : bbox[2] * self.compression]
                    single_batch_tensors.append(control_tile)
                control_tile = torch.cat(single_batch_tensors, dim=0)
                if self.control_tensor_cpu:
                    control_tile = control_tile.cpu()
                control_tile_list.append(control_tile)
            self.control_tensor_batch.append(control_tile_list)

            if len(self.custom_bboxes) > 0:
                custom_control_tile_list = []
                for bbox in self.custom_bboxes:
                    if len(control_tensor.shape) == 3:
                        control_tensor = control_tensor.unsqueeze(0)
                    control_tile = control_tensor[:, :, bbox[1] * self.compression : bbox[3] * self.compression, bbox[0] * self.compression : bbox[2] * self.compression]
                    if self.control_tensor_cpu:
                        control_tile = control_tile.cpu()
                    custom_control_tile_list.append(control_tile)
                self.control_tensor_custom.append(custom_control_tile_list)

    def switch_controlnet_tensors(self, batch_id, x_batch_size, tile_batch_size, is_denoise=False):
        if self.control_tensor_batch is None:
            return
        for param_id in range(len(self.control_tensor_batch)):
            control_tile = self.control_tensor_batch[param_id][batch_id]
            if x_batch_size > 1:
                all_control_tile = []
                for i in range(tile_batch_size):
                    this_control_tile = [control_tile[i].unsqueeze(0)] * x_batch_size
                    all_control_tile.append(torch.cat(this_control_tile, dim=0))
                control_tile = torch.cat(all_control_tile, dim=0)
                self.control_tensor_batch[param_id][batch_id] = control_tile

    def process_controlnet(self, x_noisy, c_in, cond_or_uncond, bboxes, batch_size, batch_id, shifts=None, shift_condition=None):
        control = c_in.get("control", None)
        if control is None:
            return
        param_id = -1
        tuple_key = tuple(cond_or_uncond) + tuple(x_noisy.shape)
        while control is not None:
            param_id += 1

            if tuple_key not in self.control_params:
                self.control_params[tuple_key] = [[None]]

            while len(self.control_params[tuple_key]) <= param_id:
                self.control_params[tuple_key].append([None])

            while len(self.control_params[tuple_key][param_id]) <= batch_id:
                self.control_params[tuple_key][param_id].append(None)

            if self.refresh or control.cond_hint is None or not isinstance(self.control_params[tuple_key][param_id][batch_id], torch.Tensor):
                if control.cond_hint is not None:
                    del control.cond_hint
                control.cond_hint = None
                compression_ratio = control.compression_ratio
                if getattr(control, "vae", None) is not None:
                    compression_ratio *= control.vae.downscale_ratio
                else:
                    if getattr(control, "latent_format", None) is not None:
                        raise ValueError("This Controlnet needs a VAE but none was provided, please use a ControlNetApply node with a VAE input and connect it.")
                PH, PW = self.h * compression_ratio, self.w * compression_ratio

                device = getattr(control, "device", x_noisy.device)
                dtype = getattr(control, "manual_cast_dtype", None)
                if dtype is None:
                    dtype = getattr(getattr(control, "control_model", None), "dtype", None)
                if dtype is None:
                    dtype = x_noisy.dtype

                if isinstance(control, T2IAdapter):
                    width, height = control.scale_image_to(PW, PH)
                    cns = common_upscale(control.cond_hint_original, width, height, control.upscale_algorithm, "center").float().to(device=device)
                    if control.channels_in == 1 and control.cond_hint.shape[1] > 1:
                        cns = torch.mean(control.cond_hint, 1, keepdim=True)
                elif control.__class__.__name__ == "ControlLLLiteAdvanced":
                    if getattr(control, "sub_idxs", None) is not None and control.cond_hint_original.shape[0] >= control.full_latent_length:
                        cns = common_upscale(control.cond_hint_original[control.sub_idxs], PW, PH, control.upscale_algorithm, "center").to(dtype=dtype, device=device)
                    else:
                        cns = common_upscale(control.cond_hint_original, PW, PH, control.upscale_algorithm, "center").to(dtype=dtype, device=device)
                else:
                    cns = common_upscale(control.cond_hint_original, PW, PH, control.upscale_algorithm, "center").to(dtype=dtype, device=device)
                    cns = control.preprocess_image(cns)
                    if getattr(control, "vae", None) is not None:
                        loaded_models_ = loaded_models(only_currently_used=True)
                        cns = control.vae.encode(cns.movedim(1, -1))
                        load_models_gpu(loaded_models_)
                    if getattr(control, "latent_format", None) is not None:
                        cns = control.latent_format.process_in(cns)
                    if len(getattr(control, "extra_concat_orig", ())) > 0:
                        to_concat = []
                        for c in control.extra_concat_orig:
                            c = c.to(device=device)
                            c = common_upscale(c, cns.shape[3], cns.shape[2], control.upscale_algorithm, "center")
                            to_concat.append(_repeat_to_batch_size(c, cns.shape[0]))
                        cns = torch.cat([cns] + to_concat, dim=1)

                    cns = cns.to(device=device, dtype=dtype)

                cf = control.compression_ratio
                if cns.shape[0] != batch_size:
                    cns = _repeat_to_batch_size(cns, batch_size)
                if shifts is not None:
                    control.cns = cns
                    sh_h, sh_w = shifts
                    sh_h *= cf
                    sh_w *= cf
                    if (sh_h, sh_w) != (0, 0):
                        if sh_h == 0 or sh_w == 0:
                            cns = control.cns.roll(shifts=(sh_h, sh_w), dims=(-2, -1))
                        else:
                            if shift_condition:
                                cns = control.cns.roll(shifts=sh_h, dims=-2)
                            else:
                                cns = control.cns.roll(shifts=sh_w, dims=-1)
                cns_slices = [cns[:, :, bbox[1] * cf : bbox[3] * cf, bbox[0] * cf : bbox[2] * cf] for bbox in bboxes]
                control.cond_hint = torch.cat(cns_slices, dim=0).to(device=cns.device)
                del cns_slices
                del cns
                self.control_params[tuple_key][param_id][batch_id] = control.cond_hint
            else:
                if hasattr(control, "cns") and shifts is not None:
                    cf = control.compression_ratio
                    cns = control.cns
                    sh_h, sh_w = shifts
                    sh_h *= cf
                    sh_w *= cf
                    if (sh_h, sh_w) != (0, 0):
                        if sh_h == 0 or sh_w == 0:
                            cns = control.cns.roll(shifts=(sh_h, sh_w), dims=(-2, -1))
                        else:
                            if shift_condition:
                                cns = control.cns.roll(shifts=sh_h, dims=-2)
                            else:
                                cns = control.cns.roll(shifts=sh_w, dims=-1)
                    cns_slices = [cns[:, :, bbox[1] * cf : bbox[3] * cf, bbox[0] * cf : bbox[2] * cf] for bbox in bboxes]
                    control.cond_hint = torch.cat(cns_slices, dim=0).to(device=cns.device)
                    del cns_slices
                    del cns
                else:
                    control.cond_hint = self.control_params[tuple_key][param_id][batch_id]
            control = control.previous_controlnet


class _MultiDiffusion(_AbstractDiffusion):
    @torch.inference_mode()
    def __call__(self, model_function, args):
        x_in = args["input"]
        t_in = args["timestep"]
        c_in = args["c"]
        cond_or_uncond = args["cond_or_uncond"]

        N, C, H, W = x_in.shape

        self.refresh = False
        if self.weights is None or self.h != H or self.w != W:
            self.h, self.w = H, W
            self.refresh = True
            self.init_grid_bbox(self.tile_width, self.tile_height, self.tile_overlap, self.tile_batch_size)
            self.init_done()
        self.h, self.w = H, W
        self.reset_buffer(x_in)

        if self.draw_background:
            for batch_id, bboxes in enumerate(self.batched_bboxes):
                if processing_interrupted():
                    return x_in

                x_tile = torch.cat([x_in[bbox.slicer] for bbox in bboxes], dim=0)
                t_tile = _repeat_to_batch_size(t_in, x_tile.shape[0])
                c_tile = {}
                for k, v in c_in.items():
                    if isinstance(v, torch.Tensor):
                        if len(v.shape) == len(x_tile.shape):
                            bboxes_ = bboxes
                            if v.shape[-2:] != x_in.shape[-2:]:
                                cf = x_in.shape[-1] * self.compression // v.shape[-1]
                                bboxes_ = self.get_grid_bbox(
                                    self.width // cf,
                                    self.height // cf,
                                    self.overlap // cf,
                                    self.tile_batch_size,
                                    v.shape[-1],
                                    v.shape[-2],
                                    x_in.device,
                                    self.get_tile_weights,
                                )
                            v = torch.cat([v[bbox_.slicer] for bbox_ in bboxes_[batch_id]])
                        if v.shape[0] != x_tile.shape[0]:
                            v = _repeat_to_batch_size(v, x_tile.shape[0])
                    c_tile[k] = v

                if "control" in c_in:
                    control_obj = c_in["control"]
                    if control_obj is not None:
                        self.process_controlnet(x_tile, c_in, cond_or_uncond, bboxes, N, batch_id)
                        c_tile["control"] = control_obj.get_control_orig(
                            x_tile,
                            t_tile,
                            c_tile,
                            len(cond_or_uncond),
                            c_in["transformer_options"],
                        )

                x_tile_out = model_function(x_tile, t_tile, **c_tile)

                for i, bbox in enumerate(bboxes):
                    self.x_buffer[bbox.slicer] += x_tile_out[i * N : (i + 1) * N, :, :, :]
                del x_tile_out, x_tile, t_tile, c_tile

        x_out = torch.where(self.weights > 1, self.x_buffer / self.weights, self.x_buffer)

        return x_out


def _apply_tiled_diffusion(model, tile_width=1024, tile_height=1024, tile_overlap=128, tile_batch_size=4):
    impl = _MultiDiffusion()
    compression = 4 if "CASCADE" in str(model.model.model_type) else 8
    impl.tile_width = tile_width // compression
    impl.tile_height = tile_height // compression
    impl.tile_overlap = tile_overlap // compression
    impl.tile_batch_size = tile_batch_size

    impl.compression = compression
    impl.width = tile_width
    impl.height = tile_height
    impl.overlap = tile_overlap

    model = model.clone()
    model.set_model_unet_function_wrapper(impl)
    model.model_options["tiled_diffusion"] = True
    return model


class StarSDUpscaleRefiner:
    """Refinement helper node that mirrors the refine01.json workflow.

    Responsibilities (current):
    - Load a checkpoint (MODEL, CLIP, VAE).
    - Apply up to three optional LoRAs in sequence.
    - Encode positive and negative prompts.
    - Output MODEL, VAE, positive CONDITIONING, negative CONDITIONING.
    """

    @classmethod
    def INPUT_TYPES(cls):
        checkpoints = folder_paths.get_filename_list("checkpoints")
        loras = ["None"] + folder_paths.get_filename_list("loras")
        available_upscalers = folder_paths.get_filename_list("upscale_models")
        controlnets = ["None"] + folder_paths.get_filename_list("controlnet")

        # Default checkpoint from workflow if available
        default_checkpoint = "SD1.5\\juggernaut_reborn.safetensors"
        if default_checkpoint not in checkpoints and checkpoints:
            default_checkpoint = checkpoints[0]

        return {
            "required": {
                # Checkpoint
                "checkpoint_name": (checkpoints, {
                    "default": default_checkpoint,
                    "tooltip": "SD1.5 checkpoint model to use",
                }),

                # LoRAs (optional, can be set to None)
                "lora_1_name": (loras, {
                    "default": "None",
                    "tooltip": "First LoRA (applied first). Set to None to disable.",
                }),
                "lora_1_strength": ("FLOAT", {
                    "default": 0.25,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "Strength for first LoRA (model & CLIP)",
                }),

                "lora_2_name": (loras, {
                    "default": "None",
                    "tooltip": "Second LoRA (applied after LoRA 1). Set to None to disable.",
                }),
                "lora_2_strength": ("FLOAT", {
                    "default": 0.1,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "Strength for second LoRA (model & CLIP)",
                }),

                "lora_3_name": (loras, {
                    "default": "None",
                    "tooltip": "Third LoRA (applied last). Set to None to disable.",
                }),
                "lora_3_strength": ("FLOAT", {
                    "default": 0.1,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "Strength for third LoRA (model & CLIP)",
                }),

                # Prompts
                "positive_prompt": ("STRING", {
                    "default": "masterpiece, best quality, highres",
                    "multiline": True,
                    "tooltip": "Positive prompt",
                }),
                "negative_prompt": ("STRING", {
                    "default": "(worst quality, low quality, normal quality:1.5)",
                    "multiline": True,
                    "tooltip": "Negative prompt",
                }),
                "seed": ("INT", {
                    "default": 5398475983,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                }),
                "refine_steps": ("INT", {
                    "default": 25,
                    "min": 1,
                    "max": 500,
                    "tooltip": "Refine Steps",
                }),
                "refine_denoise": ("FLOAT", {
                    "default": 0.30,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Refine Denoise",
                }),
            },
            "optional": {
                # Optional simple image upscale path
                "IMAGE": ("IMAGE",),
                "controlnet_name": (controlnets, {"default": "control_v11f1e_sd15_tile.pth"}),
                "UPSCALE_IMAGE": ("BOOLEAN", {"default": True}),
                "UPSCALE_MODEL": (["Default"] + available_upscalers, {"default": "4x_NMKD-Siax_200k.pth"}),
                "OUTPUT_LONGEST_SIDE": (
                    "INT",
                    {
                        "default": 3200,
                        "min": 64,
                        "step": 64,
                        "max": 99968,
                        "display_name": "Output Size (longest)",
                    },
                ),
            },
        }

    # Single ready IMAGE output after internal sampling & decode
    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("image", "latent")
    FUNCTION = "process"
    CATEGORY = "⭐StarNodes/Upscale"

    def _apply_lora_if_needed(self, model, clip, lora_name, strength):
        """Apply a single LoRA if name is not None/"None"."""
        if lora_name is None or lora_name == "None":
            return model, clip

        loader = LoraLoader()
        model, clip = loader.load_lora(model, clip, lora_name, strength, strength)
        return model, clip

    def process(
        self,
        checkpoint_name,
        lora_1_name,
        lora_1_strength,
        lora_2_name,
        lora_2_strength,
        lora_3_name,
        lora_3_strength,
        positive_prompt,
        negative_prompt,
        seed,
        refine_steps,
        refine_denoise,
        IMAGE=None,
        controlnet_name="None",
        UPSCALE_IMAGE=True,
        UPSCALE_MODEL="Default",
        OUTPUT_LONGEST_SIDE=3200,
    ):
        """Simple refinement setup node.

        Graph inside (conceptually):
        - CheckpointLoaderSimple
          -> 3x LoraLoader
          -> CLIPTextEncode (pos/neg)
        """

        # 1) Load checkpoint (MODEL, CLIP, VAE)
        ckpt_loader = CheckpointLoaderSimple()
        model, clip, vae = ckpt_loader.load_checkpoint(checkpoint_name)

        # 2) Apply up to three LoRAs in sequence
        model, clip = self._apply_lora_if_needed(model, clip, lora_1_name, lora_1_strength)
        model, clip = self._apply_lora_if_needed(model, clip, lora_2_name, lora_2_strength)
        model, clip = self._apply_lora_if_needed(model, clip, lora_3_name, lora_3_strength)

        # 3) Encode prompts
        clip_encoder = CLIPTextEncode()
        positive_cond = clip_encoder.encode(clip, positive_prompt)[0]
        negative_cond = clip_encoder.encode(clip, negative_prompt)[0]

        # 4) Optional simple image upscale path
        upscaled_image = None
        if IMAGE is not None:
            image = IMAGE

            if UPSCALE_IMAGE:
                # Optional model-based upscaling
                upscale_model = None
                if UPSCALE_MODEL != "Default":
                    upscale_model = UpscaleModelLoader().load_model(UPSCALE_MODEL)[0]

                if upscale_model is not None:
                    image = ImageUpscaleWithModel().upscale(upscale_model, image)[0]

                # Resize the image based on longest side (logic mirrored from Starupscale)
                assert isinstance(image, torch.Tensor)
                assert isinstance(OUTPUT_LONGEST_SIDE, int)

                # Ensure OUTPUT_LONGEST_SIDE is divisible by 64 (nearest multiple)
                OUTPUT_LONGEST_SIDE = round(OUTPUT_LONGEST_SIDE / 64) * 64
                OUTPUT_LONGEST_SIDE = max(64, OUTPUT_LONGEST_SIDE)

                interpolation_mode = InterpolationMode.BICUBIC

                _, h, w, _ = image.shape
                if h >= w:
                    new_h = OUTPUT_LONGEST_SIDE
                    new_w = round(w * new_h / h)
                    new_w = round(new_w / 64) * 64
                    new_w = max(64, new_w)
                else:
                    new_w = OUTPUT_LONGEST_SIDE
                    new_h = round(h * new_w / w)
                    new_h = round(new_h / 64) * 64
                    new_h = max(64, new_h)

                # Resize using torchvision (expects NCHW)
                image = image.permute(0, 3, 1, 2)
                image = F.resize(
                    image,
                    (new_h, new_w),
                    interpolation=interpolation_mode,
                    antialias=True,
                )
                image = image.permute(0, 2, 3, 1)

                upscaled_image = image

        # 5) Optional ControlNet apply using hardcoded settings
        #    Only applied if a controlnet_name is selected and an image is provided.
        controlnet = None
        if controlnet_name is not None and controlnet_name != "None":
            cn_loader = ControlNetLoader()
            controlnet = cn_loader.load_controlnet(controlnet_name)[0]

        if controlnet is not None and (upscaled_image is not None or IMAGE is not None):
            controlnet_image = upscaled_image if upscaled_image is not None else IMAGE

            cn_apply = ControlNetApplyAdvanced()
            # Hardcoded parameters based on refinerstep_2.json:
            # widgets_values: [0.5, 0, 1] -> strength, start_percent, end_percent
            strength = 0.5
            start_percent = 0.0
            end_percent = 1.0

            positive_cond, negative_cond = cn_apply.apply_controlnet(
                positive_cond,
                negative_cond,
                controlnet,
                controlnet_image,
                vae,
                strength,
                start_percent,
                end_percent,
            )
        model = _apply_freeu_v2(model, 1.05, 1.08, 0.95, 0.8)
        model = _apply_pag(model, 1.0)
        model = _apply_auto_cfg(model)
        model = _apply_tiled_diffusion(model, 1024, 1024, 128, 4)

        latent = None
        pixels_for_latent = None
        if upscaled_image is not None:
            pixels_for_latent = upscaled_image
        elif IMAGE is not None:
            pixels_for_latent = IMAGE

        if pixels_for_latent is not None:
            tile_size = 1024
            overlap = 64
            temporal_size = 64
            temporal_overlap = 8
            t = vae.encode_tiled(
                pixels_for_latent[:, :, :, :3],
                tile_x=tile_size,
                tile_y=tile_size,
                overlap=overlap,
                tile_t=temporal_size,
                overlap_t=temporal_overlap,
            )
            latent = {"samples": t}

        if latent is None:
            return (upscaled_image if upscaled_image is not None else IMAGE, None)

        if any(x is None for x in (KSamplerSelect, SamplerCustom, AlignYourStepsScheduler)):
            # Dependencies missing: fall back to returning the upscaled (or original) image
            return (upscaled_image if upscaled_image is not None else IMAGE, latent)

        scheduler = AlignYourStepsScheduler()
        (sigmas,) = scheduler.get_sigmas(
            "SD1",
            int(refine_steps),
            float(refine_denoise),
        )

        sampler_selector = KSamplerSelect()
        (sampler_obj,) = sampler_selector.get_sampler("dpmpp_3m_sde_gpu")

        sampler_custom = SamplerCustom()

        arg_names = inspect.getfullargspec(sampler_custom.sample).args[1:]
        kwargs = {}
        for name in arg_names:
            if name in ("model",):
                kwargs[name] = model
            elif name in ("positive", "pos_cond", "pos"):
                kwargs[name] = positive_cond
            elif name in ("negative", "neg_cond", "neg"):
                kwargs[name] = negative_cond
            elif name in ("latent", "latent_image", "latent_in"):
                kwargs[name] = latent
            elif name in ("sampler", "sampler_name", "sampler_obj"):
                kwargs[name] = sampler_obj
            elif name in ("sigmas",):
                kwargs[name] = sigmas
            elif name in ("add_noise",):
                kwargs[name] = True
            elif name in ("noise_seed", "seed"):
                kwargs[name] = int(seed)
            elif name in ("noise_type",):
                kwargs[name] = "fixed"
            elif name in ("cfg", "cfg_scale", "cfg_value"):
                kwargs[name] = float(8.0)

        out = sampler_custom.sample(**kwargs)
        if isinstance(out, tuple) and len(out) >= 1:
            out_latent = out[0]
        else:
            out_latent = out

        # Unwrap comfy_api latest IO container to raw latent dict if needed.
        # Newer APIs may wrap node outputs in a tuple-like object where the
        # first element is the actual {"samples": tensor, ...} dictionary.
        try:
            if not isinstance(out_latent, dict):
                candidate = out_latent[0]
                if isinstance(candidate, dict) and "samples" in candidate:
                    out_latent = candidate
        except Exception:
            # If anything goes wrong, fall back to original value; a later
            # decode/usage error will make debugging easier than swallowing it.
            pass

        image = None
        if VAEDecodeTiled is not None:
            vdt = VAEDecodeTiled()
            func_name = getattr(VAEDecodeTiled, "FUNCTION", "decode")
            decode_fn = getattr(vdt, func_name, None)
            if decode_fn is None:
                images = vae.decode(out_latent["samples"])
            else:
                arg_names = inspect.getfullargspec(decode_fn).args[1:]
                dkwargs = {}
                for name in arg_names:
                    if name in ("samples", "latent", "latent_image"):
                        dkwargs[name] = out_latent
                    elif name in ("vae",):
                        dkwargs[name] = vae
                    elif name in ("tile_size", "tile_width"):
                        dkwargs[name] = int(1024)
                    elif name in ("overlap", "tile_overlap"):
                        dkwargs[name] = int(64)
                    elif name in ("fastest_tile_size", "fast_tile_size"):
                        dkwargs[name] = int(64)
                    elif name in ("fast_decode", "tile_batch_size"):
                        dkwargs[name] = int(8)

                images = decode_fn(**dkwargs)

            if isinstance(images, tuple) and len(images) >= 1:
                image = images[0]
            else:
                image = images
        else:
            images = vae.decode(out_latent["samples"])
            if len(images.shape) == 5:
                images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
            image = images

        return (image, out_latent)


class StarSDUpscaleRefinerAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + folder_paths.get_filename_list("loras")
        available_upscalers = folder_paths.get_filename_list("upscale_models")
        controlnets = ["None"] + folder_paths.get_filename_list("controlnet")

        sampler_list = getattr(comfy_samplers.KSampler, "SAMPLERS", [])
        scheduler_list = getattr(comfy_samplers.KSampler, "SCHEDULERS", [])

        if not sampler_list:
            sampler_list = ["dpmpp_3m_sde_gpu"]
        if not scheduler_list:
            scheduler_list = ["karras"]

        return {
            "required": {
                "model": ("MODEL", {}),
                "clip": ("CLIP", {}),
                "vae": ("VAE", {}),

                "lora_1_name": (loras, {
                    "default": "None",
                }),
                "lora_1_strength": ("FLOAT", {
                    "default": 0.25,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.01,
                }),
                "lora_2_name": (loras, {
                    "default": "None",
                }),
                "lora_2_strength": ("FLOAT", {
                    "default": 0.1,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.01,
                }),
                "lora_3_name": (loras, {
                    "default": "None",
                }),
                "lora_3_strength": ("FLOAT", {
                    "default": 0.1,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.01,
                }),

                "positive_prompt": ("STRING", {
                    "default": "masterpiece, best quality, highres",
                    "multiline": True,
                }),
                "negative_prompt": ("STRING", {
                    "default": "(worst quality, low quality, normal quality:1.5)",
                    "multiline": True,
                }),
                "use_negative": (["No", "Yes"], {
                    "default": "No",
                }),
                "seed": ("INT", {
                    "default": 5398475983,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                }),
                "refine_steps": ("INT", {
                    "default": 25,
                    "min": 1,
                    "max": 500,
                }),
                "refine_denoise": ("FLOAT", {
                    "default": 0.30,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "tile_size": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "step": 64,
                    "max": 16384,
                }),
                "sampler_name": (sampler_list, {
                    "default": "dpmpp_3m_sde_gpu" if "dpmpp_3m_sde_gpu" in sampler_list else sampler_list[0],
                }),
                "scheduler_name": (scheduler_list, {
                    "default": "karras" if "karras" in scheduler_list else scheduler_list[0],
                }),
            },
            "optional": {
                "IMAGE": ("IMAGE",),
                "controlnet_name": (controlnets, {"default": "control_v11f1e_sd15_tile.pth"}),
                "UPSCALE_IMAGE": ("BOOLEAN", {"default": True}),
                "UPSCALE_MODEL": (["Default"] + available_upscalers, {"default": "4x_NMKD-Siax_200k.pth"}),
                "OUTPUT_LONGEST_SIDE": (
                    "INT",
                    {
                        "default": 3200,
                        "min": 64,
                        "step": 64,
                        "max": 99968,
                    },
                ),
                # Optional FlowMatch/StarSampler-style sigma schedule override
                # (e.g. from ⭐ Star FlowMatch Option)
                "options": ("SIGMAS", {}),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("image", "latent")
    FUNCTION = "process"
    CATEGORY = "⭐StarNodes/Upscale"

    def _apply_lora_if_needed(self, model, clip, lora_name, strength):
        if lora_name is None or lora_name == "None":
            return model, clip

        loader = LoraLoader()
        model, clip = loader.load_lora(model, clip, lora_name, strength, strength)
        return model, clip

    def process(
        self,
        model,
        clip,
        vae,
        lora_1_name,
        lora_1_strength,
        lora_2_name,
        lora_2_strength,
        lora_3_name,
        lora_3_strength,
        positive_prompt,
        negative_prompt,
        use_negative,
        seed,
        refine_steps,
        refine_denoise,
        tile_size,
        sampler_name,
        scheduler_name,
        IMAGE=None,
        controlnet_name="None",
        UPSCALE_IMAGE=True,
        UPSCALE_MODEL="Default",
        OUTPUT_LONGEST_SIDE=3200,
        options=None,
    ):
        model, clip = self._apply_lora_if_needed(model, clip, lora_1_name, lora_1_strength)
        model, clip = self._apply_lora_if_needed(model, clip, lora_2_name, lora_2_strength)
        model, clip = self._apply_lora_if_needed(model, clip, lora_3_name, lora_3_strength)

        clip_encoder = CLIPTextEncode()
        positive_cond = clip_encoder.encode(clip, positive_prompt)[0]
        negative_cond = clip_encoder.encode(clip, negative_prompt)[0]

        upscaled_image = None
        if IMAGE is not None:
            image = IMAGE

            if UPSCALE_IMAGE:
                upscale_model = None
                if UPSCALE_MODEL != "Default":
                    upscale_model = UpscaleModelLoader().load_model(UPSCALE_MODEL)[0]

                if upscale_model is not None:
                    image = ImageUpscaleWithModel().upscale(upscale_model, image)[0]

                assert isinstance(image, torch.Tensor)
                assert isinstance(OUTPUT_LONGEST_SIDE, int)

                OUTPUT_LONGEST_SIDE = round(OUTPUT_LONGEST_SIDE / 64) * 64
                OUTPUT_LONGEST_SIDE = max(64, OUTPUT_LONGEST_SIDE)

                interpolation_mode = InterpolationMode.BICUBIC

                _, h, w, _ = image.shape
                if h >= w:
                    new_h = OUTPUT_LONGEST_SIDE
                    new_w = round(w * new_h / h)
                    new_w = round(new_w / 64) * 64
                    new_w = max(64, new_w)
                else:
                    new_w = OUTPUT_LONGEST_SIDE
                    new_h = round(h * new_w / w)
                    new_h = round(new_h / 64) * 64
                    new_h = max(64, new_h)

                image = image.permute(0, 3, 1, 2)
                image = F.resize(
                    image,
                    (new_h, new_w),
                    interpolation=interpolation_mode,
                    antialias=True,
                )
                image = image.permute(0, 2, 3, 1)

                upscaled_image = image

        controlnet = None
        if controlnet_name is not None and controlnet_name != "None":
            cn_loader = ControlNetLoader()
            controlnet = cn_loader.load_controlnet(controlnet_name)[0]

        if controlnet is not None and (upscaled_image is not None or IMAGE is not None):
            controlnet_image = upscaled_image if upscaled_image is not None else IMAGE

            cn_apply = ControlNetApplyAdvanced()
            strength = 0.5
            start_percent = 0.0
            end_percent = 1.0

            positive_cond, negative_cond = cn_apply.apply_controlnet(
                positive_cond,
                negative_cond,
                controlnet,
                controlnet_image,
                vae,
                strength,
                start_percent,
                end_percent,
            )

        model = _apply_freeu_v2(model, 1.05, 1.08, 0.95, 0.8)
        model = _apply_pag(model, 1.0)
        model = _apply_auto_cfg(model)
        model = _apply_tiled_diffusion(model, int(tile_size), int(tile_size), 128, 4)

        latent = None
        pixels_for_latent = None
        if upscaled_image is not None:
            pixels_for_latent = upscaled_image
        elif IMAGE is not None:
            pixels_for_latent = IMAGE

        if pixels_for_latent is not None:
            tile_size_enc = int(tile_size)
            overlap = 64
            temporal_size = 64
            temporal_overlap = 8
            t = vae.encode_tiled(
                pixels_for_latent[:, :, :, :3],
                tile_x=tile_size_enc,
                tile_y=tile_size_enc,
                overlap=overlap,
                tile_t=temporal_size,
                overlap_t=temporal_overlap,
            )
            latent = {"samples": t}

        if latent is None:
            return (upscaled_image if upscaled_image is not None else IMAGE, None)

        if any(x is None for x in (KSamplerSelect, SamplerCustom, AlignYourStepsScheduler)):
            return (upscaled_image if upscaled_image is not None else IMAGE, latent)

        # Build sigma schedule: use external options (e.g. Star FlowMatch Option)
        # if provided, otherwise fall back to AlignYourStepsScheduler.
        if options is not None:
            sigmas = options
            # Honor refine_denoise by shortening the sigma schedule proportionally,
            # similar in spirit to how ComfyUI schedulers treat denoise.
            try:
                if hasattr(sigmas, "shape") and len(sigmas.shape) > 0 and sigmas.shape[0] > 1:
                    total_steps = sigmas.shape[0] - 1
                    denoise = float(refine_denoise)
                    denoise = max(0.0, min(1.0, denoise))
                    effective_steps = max(1, int(round(total_steps * denoise)))
                    start_index = max(0, sigmas.shape[0] - 1 - effective_steps)
                    sigmas = sigmas[start_index:]
            except Exception:
                # If anything goes wrong, fall back to the original sigmas tensor
                pass
        else:
            scheduler = AlignYourStepsScheduler()
            get_sigmas = scheduler.get_sigmas
            s_arg_names = inspect.getfullargspec(get_sigmas).args[1:]
            s_kwargs = {}
            for name in s_arg_names:
                if name in ("model_type", "model"):
                    s_kwargs[name] = "SD1"
                elif name in ("steps",):
                    s_kwargs[name] = int(refine_steps)
                elif name in ("denoise", "denoise_strength"):
                    s_kwargs[name] = float(refine_denoise)
                elif name in ("scheduler", "scheduler_name"):
                    s_kwargs[name] = scheduler_name
            (sigmas,) = get_sigmas(**s_kwargs)

        sampler_selector = KSamplerSelect()
        get_sampler = sampler_selector.get_sampler
        g_arg_names = inspect.getfullargspec(get_sampler).args[1:]
        g_kwargs = {}
        for name in g_arg_names:
            if name in ("sampler", "sampler_name"):
                g_kwargs[name] = sampler_name
        (sampler_obj,) = get_sampler(**g_kwargs)

        sampler_custom = SamplerCustom()

        if use_negative == "No":
            # Build an "empty" negative CONDITIONING matching ComfyUI's structure
            # positive_cond is typically a list like [[tensor, dict], ...]
            if isinstance(positive_cond, list) and len(positive_cond) > 0:
                first = positive_cond[0]
                if isinstance(first, (list, tuple)) and len(first) >= 2:
                    empty_tensor = torch.zeros_like(first[0])
                    meta = first[1].copy()
                    negative_cond = [[empty_tensor, meta]]

        arg_names = inspect.getfullargspec(sampler_custom.sample).args[1:]
        kwargs = {}
        for name in arg_names:
            if name in ("model",):
                kwargs[name] = model
            elif name in ("positive", "pos_cond", "pos"):
                kwargs[name] = positive_cond
            elif name in ("negative", "neg_cond", "neg"):
                kwargs[name] = negative_cond
            elif name in ("latent", "latent_image", "latent_in"):
                kwargs[name] = latent
            elif name in ("sampler", "sampler_name", "sampler_obj"):
                kwargs[name] = sampler_obj
            elif name in ("sigmas",):
                kwargs[name] = sigmas
            elif name in ("add_noise",):
                kwargs[name] = True
            elif name in ("noise_seed", "seed"):
                kwargs[name] = int(seed)
            elif name in ("noise_type",):
                kwargs[name] = "fixed"
            elif name in ("cfg", "cfg_scale", "cfg_value"):
                kwargs[name] = float(8.0)

        out = sampler_custom.sample(**kwargs)
        if isinstance(out, tuple) and len(out) >= 1:
            out_latent = out[0]
        else:
            out_latent = out

        try:
            if not isinstance(out_latent, dict):
                candidate = out_latent[0]
                if isinstance(candidate, dict) and "samples" in candidate:
                    out_latent = candidate
        except Exception:
            pass

        image = None
        if VAEDecodeTiled is not None:
            vdt = VAEDecodeTiled()
            func_name = getattr(VAEDecodeTiled, "FUNCTION", "decode")
            decode_fn = getattr(vdt, func_name, None)
            if decode_fn is None:
                images = vae.decode(out_latent["samples"])
            else:
                d_arg_names = inspect.getfullargspec(decode_fn).args[1:]
                dkwargs = {}
                for name in d_arg_names:
                    if name in ("samples", "latent", "latent_image"):
                        dkwargs[name] = out_latent
                    elif name in ("vae",):
                        dkwargs[name] = vae
                    elif name in ("tile_size", "tile_width"):
                        dkwargs[name] = int(tile_size)
                    elif name in ("overlap", "tile_overlap"):
                        dkwargs[name] = int(64)
                    elif name in ("fastest_tile_size", "fast_tile_size"):
                        dkwargs[name] = int(64)
                    elif name in ("fast_decode", "tile_batch_size"):
                        dkwargs[name] = int(8)

                images = decode_fn(**dkwargs)

            if isinstance(images, tuple) and len(images) >= 1:
                image = images[0]
            else:
                image = images
        else:
            images = vae.decode(out_latent["samples"])
            if len(images.shape) == 5:
                images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
            image = images

        return (image, out_latent)


NODE_CLASS_MAPPINGS = {
    "StarSDUpscaleRefiner": StarSDUpscaleRefiner,
    "StarSDUpscaleRefinerAdvanced": StarSDUpscaleRefinerAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarSDUpscaleRefiner": "⭐ Star SD Upscale Refiner",
    "StarSDUpscaleRefinerAdvanced": "⭐ Star SD Upscale Refiner Advanced",
}

