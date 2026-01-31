import torch
import folder_paths
import comfy
import comfy.utils
import comfy.ops
import comfy.model_management
import comfy.model_patcher
import comfy.ldm.common_dit
import comfy.latent_formats
import comfy.ldm.lumina.controlnet


class StarFP8ModelPatchLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"name": (folder_paths.get_filename_list("model_patches"),)}}

    RETURN_TYPES = ("MODEL_PATCH",)
    FUNCTION = "load_model_patch"
    EXPERIMENTAL = False

    CATEGORY = "\u2b50StarNodes/Loaders"

    def _count_nonzero_fp8_safe(self, t: torch.Tensor) -> int:
        if t is None:
            return 0

        fp8_dtypes = []
        fp8_e4m3fn = getattr(torch, "float8_e4m3fn", None)
        fp8_e5m2 = getattr(torch, "float8_e5m2", None)
        if fp8_e4m3fn is not None:
            fp8_dtypes.append(fp8_e4m3fn)
        if fp8_e5m2 is not None:
            fp8_dtypes.append(fp8_e5m2)

        if t.device.type == "cpu" and len(fp8_dtypes) > 0 and t.dtype in fp8_dtypes:
            return int(torch.count_nonzero(t.to(torch.float16)))

        return int(torch.count_nonzero(t))

    def load_model_patch(self, name):
        model_patch_path = folder_paths.get_full_path_or_raise("model_patches", name)
        sd = comfy.utils.load_torch_file(model_patch_path, safe_load=True)
        dtype = comfy.utils.weight_dtype(sd)

        model = None

        if 'controlnet_blocks.0.y_rms.weight' in sd:
            additional_in_dim = sd["img_in.weight"].shape[1] - 64
            model = comfy_extras_nodes_model_patch_QwenImageBlockWiseControlNet(additional_in_dim=additional_in_dim, device=comfy.model_management.unet_offload_device(), dtype=dtype, operations=comfy.ops.manual_cast)
        elif 'feature_embedder.mid_layer_norm.bias' in sd:
            sd = comfy.utils.state_dict_prefix_replace(sd, {"feature_embedder.": ""}, filter_keys=True)
            model = comfy_extras_nodes_model_patch_SigLIPMultiFeatProjModel(device=comfy.model_management.unet_offload_device(), dtype=dtype, operations=comfy.ops.manual_cast)
        elif 'control_all_x_embedder.2-1.weight' in sd:
            sd = comfy_extras_nodes_model_patch_z_image_convert(sd)
            config = {}
            if 'control_layers.14.adaLN_modulation.0.weight' in sd:
                config['n_control_layers'] = 15
                config['additional_in_dim'] = 17
                config['refiner_control'] = True
                ref_weight = sd.get("control_noise_refiner.0.after_proj.weight", None)
                if ref_weight is not None:
                    if self._count_nonzero_fp8_safe(ref_weight) == 0:
                        config['broken'] = True
            model = comfy.ldm.lumina.controlnet.ZImage_Control(device=comfy.model_management.unet_offload_device(), dtype=dtype, operations=comfy.ops.manual_cast, **config)

        if model is None:
            raise RuntimeError(f"StarFP8ModelPatchLoader: Unrecognized model patch format for '{name}'.")

        model.load_state_dict(sd)
        model = comfy.model_patcher.ModelPatcher(model, load_device=comfy.model_management.get_torch_device(), offload_device=comfy.model_management.unet_offload_device())
        return (model,)


def comfy_extras_nodes_model_patch_z_image_convert(sd):
    replace_keys = {".attention.to_out.0.bias": ".attention.out.bias",
                    ".attention.norm_k.weight": ".attention.k_norm.weight",
                    ".attention.norm_q.weight": ".attention.q_norm.weight",
                    ".attention.to_out.0.weight": ".attention.out.weight"
                    }

    out_sd = {}
    for k in sorted(sd.keys()):
        w = sd[k]

        k_out = k
        if k_out.endswith(".attention.to_k.weight"):
            cc = [w]
            continue
        if k_out.endswith(".attention.to_q.weight"):
            cc = [w] + cc
            continue
        if k_out.endswith(".attention.to_v.weight"):
            cc = cc + [w]
            w = torch.cat(cc, dim=0)
            k_out = k_out.replace(".attention.to_v.weight", ".attention.qkv.weight")

        for r, rr in replace_keys.items():
            k_out = k_out.replace(r, rr)
        out_sd[k_out] = w

    return out_sd


class comfy_extras_nodes_model_patch_BlockWiseControlBlock(torch.nn.Module):
    def __init__(self, dim: int = 3072, device=None, dtype=None, operations=None):
        super().__init__()
        self.x_rms = operations.RMSNorm(dim, eps=1e-6)
        self.y_rms = operations.RMSNorm(dim, eps=1e-6)
        self.input_proj = operations.Linear(dim, dim)
        self.act = torch.nn.GELU()
        self.output_proj = operations.Linear(dim, dim)

    def forward(self, x, y):
        x, y = self.x_rms(x), self.y_rms(y)
        x = self.input_proj(x + y)
        x = self.act(x)
        x = self.output_proj(x)
        return x


class comfy_extras_nodes_model_patch_QwenImageBlockWiseControlNet(torch.nn.Module):
    def __init__(
        self,
        num_layers: int = 60,
        in_dim: int = 64,
        additional_in_dim: int = 0,
        dim: int = 3072,
        device=None, dtype=None, operations=None
    ):
        super().__init__()
        self.additional_in_dim = additional_in_dim
        self.img_in = operations.Linear(in_dim + additional_in_dim, dim, device=device, dtype=dtype)
        self.controlnet_blocks = torch.nn.ModuleList(
            [
                comfy_extras_nodes_model_patch_BlockWiseControlBlock(dim, device=device, dtype=dtype, operations=operations)
                for _ in range(num_layers)
            ]
        )

    def process_input_latent_image(self, latent_image):
        latent_image[:, :16] = comfy.latent_formats.Wan21().process_in(latent_image[:, :16])
        patch_size = 2
        hidden_states = comfy.ldm.common_dit.pad_to_patch_size(latent_image, (1, patch_size, patch_size))
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(orig_shape[0], orig_shape[1], orig_shape[-2] // 2, 2, orig_shape[-1] // 2, 2)
        hidden_states = hidden_states.permute(0, 2, 4, 1, 3, 5)
        hidden_states = hidden_states.reshape(orig_shape[0], (orig_shape[-2] // 2) * (orig_shape[-1] // 2), orig_shape[1] * 4)
        return self.img_in(hidden_states)

    def control_block(self, img, controlnet_conditioning, block_id):
        return self.controlnet_blocks[block_id](img, controlnet_conditioning)


class comfy_extras_nodes_model_patch_SigLIPMultiFeatProjModel(torch.nn.Module):
    def __init__(
        self,
        siglip_token_nums: int = 729,
        style_token_nums: int = 64,
        siglip_token_dims: int = 1152,
        hidden_size: int = 3072,
        context_layer_norm: bool = True,
        device=None, dtype=None, operations=None
    ):
        super().__init__()

        self.high_embedding_linear = torch.nn.Sequential(
            operations.Linear(siglip_token_nums, style_token_nums),
            torch.nn.SiLU()
        )
        self.high_layer_norm = (
            operations.LayerNorm(siglip_token_dims) if context_layer_norm else torch.nn.Identity()
        )
        self.high_projection = operations.Linear(siglip_token_dims, hidden_size, bias=True)

        self.mid_embedding_linear = torch.nn.Sequential(
            operations.Linear(siglip_token_nums, style_token_nums),
            torch.nn.SiLU()
        )
        self.mid_layer_norm = (
            operations.LayerNorm(siglip_token_dims) if context_layer_norm else torch.nn.Identity()
        )
        self.mid_projection = operations.Linear(siglip_token_dims, hidden_size, bias=True)

        self.low_embedding_linear = torch.nn.Sequential(
            operations.Linear(siglip_token_nums, style_token_nums),
            torch.nn.SiLU()
        )
        self.low_layer_norm = (
            operations.LayerNorm(siglip_token_dims) if context_layer_norm else torch.nn.Identity()
        )
        self.low_projection = operations.Linear(siglip_token_dims, hidden_size, bias=True)

    def forward(self, siglip_outputs):
        dtype = next(self.high_embedding_linear.parameters()).dtype

        high_embedding = self._process_layer_features(
            siglip_outputs[2],
            self.high_embedding_linear,
            self.high_layer_norm,
            self.high_projection,
            dtype
        )

        mid_embedding = self._process_layer_features(
            siglip_outputs[1],
            self.mid_embedding_linear,
            self.mid_layer_norm,
            self.mid_projection,
            dtype
        )

        low_embedding = self._process_layer_features(
            siglip_outputs[0],
            self.low_embedding_linear,
            self.low_layer_norm,
            self.low_projection,
            dtype
        )

        return torch.cat((high_embedding, mid_embedding, low_embedding), dim=1)

    def _process_layer_features(
        self,
        hidden_states: torch.Tensor,
        embedding_linear: torch.nn.Module,
        layer_norm: torch.nn.Module,
        projection: torch.nn.Module,
        dtype: torch.dtype
    ) -> torch.Tensor:
        embedding = embedding_linear(
            hidden_states.to(dtype).transpose(1, 2)
        ).transpose(1, 2)

        embedding = layer_norm(embedding)
        embedding = projection(embedding)

        return embedding


NODE_CLASS_MAPPINGS = {
    "StarFP8ModelPatchLoader": StarFP8ModelPatchLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarFP8ModelPatchLoader": "\u2b50 Star FP8 Model Patch Loader",
}
