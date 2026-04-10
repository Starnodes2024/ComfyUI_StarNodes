import folder_paths
import comfy.utils
import os
import torch
import json
from safetensors.torch import save_file

class StarVAE_LTXV_Save:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint": (folder_paths.get_filename_list("checkpoints"),),
                "save_name": ("STRING", {"default": "ltxv_vae"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "save_vae"
    CATEGORY = "StarVAE"
    OUTPUT_NODE = True

    def save_vae(self, checkpoint, save_name):
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", checkpoint)
        sd, metadata = comfy.utils.load_torch_file(ckpt_path, return_metadata=True)
        
        audio_vae_keys = [
            "audio_vae.encoder.conv_in.weight",
            "audio_vae.encoder.conv_in.bias",
            "audio_vae.encoder.down.0.block.0.norm1.weight",
            "audio_vae.encoder.down.0.block.0.norm1.bias",
            "audio_vae.encoder.down.0.block.0.conv1.weight",
            "audio_vae.encoder.down.0.block.0.conv1.bias",
            "audio_vae.encoder.down.0.block.0.norm2.weight",
            "audio_vae.encoder.down.0.block.0.norm2.bias",
            "audio_vae.encoder.down.0.block.0.conv2.weight",
            "audio_vae.encoder.down.0.block.0.conv2.bias",
            "audio_vae.encoder.down.0.block.1.norm1.weight",
            "audio_vae.encoder.down.0.block.1.norm1.bias",
            "audio_vae.encoder.down.0.block.1.conv1.weight",
            "audio_vae.encoder.down.0.block.1.conv1.bias",
            "audio_vae.encoder.down.0.block.1.norm2.weight",
            "audio_vae.encoder.down.0.block.1.norm2.bias",
            "audio_vae.encoder.down.0.block.1.conv2.weight",
            "audio_vae.encoder.down.0.block.1.conv2.bias",
            "audio_vae.encoder.down.0.downsample.conv.weight",
            "audio_vae.encoder.down.0.downsample.conv.bias",
            "audio_vae.encoder.down.1.block.0.norm1.weight",
            "audio_vae.encoder.down.1.block.0.norm1.bias",
            "audio_vae.encoder.down.1.block.0.conv1.weight",
            "audio_vae.encoder.down.1.block.0.conv1.bias",
            "audio_vae.encoder.down.1.block.0.norm2.weight",
            "audio_vae.encoder.down.1.block.0.norm2.bias",
            "audio_vae.encoder.down.1.block.0.conv2.weight",
            "audio_vae.encoder.down.1.block.0.conv2.bias",
            "audio_vae.encoder.down.1.block.1.norm1.weight",
            "audio_vae.encoder.down.1.block.1.norm1.bias",
            "audio_vae.encoder.down.1.block.1.conv1.weight",
            "audio_vae.encoder.down.1.block.1.conv1.bias",
            "audio_vae.encoder.down.1.block.1.norm2.weight",
            "audio_vae.encoder.down.1.block.1.norm2.bias",
            "audio_vae.encoder.down.1.block.1.conv2.weight",
            "audio_vae.encoder.down.1.block.1.conv2.bias",
            "audio_vae.encoder.down.1.downsample.conv.weight",
            "audio_vae.encoder.down.1.downsample.conv.bias",
            "audio_vae.encoder.down.2.block.0.norm1.weight",
            "audio_vae.encoder.down.2.block.0.norm1.bias",
            "audio_vae.encoder.down.2.block.0.conv1.weight",
            "audio_vae.encoder.down.2.block.0.conv1.bias",
            "audio_vae.encoder.down.2.block.0.norm2.weight",
            "audio_vae.encoder.down.2.block.0.norm2.bias",
            "audio_vae.encoder.down.2.block.0.conv2.weight",
            "audio_vae.encoder.down.2.block.0.conv2.bias",
            "audio_vae.encoder.down.2.block.1.norm1.weight",
            "audio_vae.encoder.down.2.block.1.norm1.bias",
            "audio_vae.encoder.down.2.block.1.conv1.weight",
            "audio_vae.encoder.down.2.block.1.conv1.bias",
            "audio_vae.encoder.down.2.block.1.norm2.weight",
            "audio_vae.encoder.down.2.block.1.norm2.bias",
            "audio_vae.encoder.down.2.block.1.conv2.weight",
            "audio_vae.encoder.down.2.block.1.conv2.bias",
            "audio_vae.encoder.down.2.downsample.conv.weight",
            "audio_vae.encoder.down.2.downsample.conv.bias",
            "audio_vae.encoder.down.3.block.0.norm1.weight",
            "audio_vae.encoder.down.3.block.0.norm1.bias",
            "audio_vae.encoder.down.3.block.0.conv1.weight",
            "audio_vae.encoder.down.3.block.0.conv1.bias",
            "audio_vae.encoder.down.3.block.0.norm2.weight",
            "audio_vae.encoder.down.3.block.0.norm2.bias",
            "audio_vae.encoder.down.3.block.0.conv2.weight",
            "audio_vae.encoder.down.3.block.0.conv2.bias",
            "audio_vae.encoder.down.3.block.1.norm1.weight",
            "audio_vae.encoder.down.3.block.1.norm1.bias",
            "audio_vae.encoder.down.3.block.1.conv1.weight",
            "audio_vae.encoder.down.3.block.1.conv1.bias",
            "audio_vae.encoder.down.3.block.1.norm2.weight",
            "audio_vae.encoder.down.3.block.1.norm2.bias",
            "audio_vae.encoder.down.3.block.1.conv2.weight",
            "audio_vae.encoder.down.3.block.1.conv2.bias",
            "audio_vae.encoder.mid.block_1.norm1.weight",
            "audio_vae.encoder.mid.block_1.norm1.bias",
            "audio_vae.encoder.mid.block_1.conv1.weight",
            "audio_vae.encoder.mid.block_1.conv1.bias",
            "audio_vae.encoder.mid.block_1.norm2.weight",
            "audio_vae.encoder.mid.block_1.norm2.bias",
            "audio_vae.encoder.mid.block_1.conv2.weight",
            "audio_vae.encoder.mid.block_1.conv2.bias",
            "audio_vae.encoder.mid.attn_1.norm.weight",
            "audio_vae.encoder.mid.attn_1.norm.bias",
            "audio_vae.encoder.mid.attn_1.q.weight",
            "audio_vae.encoder.mid.attn_1.q.bias",
            "audio_vae.encoder.mid.attn_1.k.weight",
            "audio_vae.encoder.mid.attn_1.k.bias",
            "audio_vae.encoder.mid.attn_1.v.weight",
            "audio_vae.encoder.mid.attn_1.v.bias",
            "audio_vae.encoder.mid.attn_1.proj_out.weight",
            "audio_vae.encoder.mid.attn_1.proj_out.bias",
            "audio_vae.encoder.mid.block_2.norm1.weight",
            "audio_vae.encoder.mid.block_2.norm1.bias",
            "audio_vae.encoder.mid.block_2.conv1.weight",
            "audio_vae.encoder.mid.block_2.conv1.bias",
            "audio_vae.encoder.mid.block_2.norm2.weight",
            "audio_vae.encoder.mid.block_2.norm2.bias",
            "audio_vae.encoder.mid.block_2.conv2.weight",
            "audio_vae.encoder.mid.block_2.conv2.bias",
            "audio_vae.encoder.norm_out.weight",
            "audio_vae.encoder.norm_out.bias",
            "audio_vae.encoder.conv_out.weight",
            "audio_vae.encoder.conv_out.bias",
            "audio_vae.decoder.conv_in.weight",
            "audio_vae.decoder.conv_in.bias",
            "audio_vae.decoder.mid.block_1.norm1.weight",
            "audio_vae.decoder.mid.block_1.norm1.bias",
            "audio_vae.decoder.mid.block_1.conv1.weight",
            "audio_vae.decoder.mid.block_1.conv1.bias",
            "audio_vae.decoder.mid.block_1.norm2.weight",
            "audio_vae.decoder.mid.block_1.norm2.bias",
            "audio_vae.decoder.mid.block_1.conv2.weight",
            "audio_vae.decoder.mid.block_1.conv2.bias",
            "audio_vae.decoder.mid.attn_1.norm.weight",
            "audio_vae.decoder.mid.attn_1.norm.bias",
            "audio_vae.decoder.mid.attn_1.q.weight",
            "audio_vae.decoder.mid.attn_1.q.bias",
            "audio_vae.decoder.mid.attn_1.k.weight",
            "audio_vae.decoder.mid.attn_1.k.bias",
            "audio_vae.decoder.mid.attn_1.v.weight",
            "audio_vae.decoder.mid.attn_1.v.bias",
            "audio_vae.decoder.mid.attn_1.proj_out.weight",
            "audio_vae.decoder.mid.attn_1.proj_out.bias",
            "audio_vae.decoder.mid.block_2.norm1.weight",
            "audio_vae.decoder.mid.block_2.norm1.bias",
            "audio_vae.decoder.mid.block_2.conv1.weight",
            "audio_vae.decoder.mid.block_2.conv1.bias",
            "audio_vae.decoder.mid.block_2.norm2.weight",
            "audio_vae.decoder.mid.block_2.norm2.bias",
            "audio_vae.decoder.mid.block_2.conv2.weight",
            "audio_vae.decoder.mid.block_2.conv2.bias",
            "audio_vae.decoder.up.0.block.0.norm1.weight",
            "audio_vae.decoder.up.0.block.0.norm1.bias",
            "audio_vae.decoder.up.0.block.0.conv1.weight",
            "audio_vae.decoder.up.0.block.0.conv1.bias",
            "audio_vae.decoder.up.0.block.0.norm2.weight",
            "audio_vae.decoder.up.0.block.0.norm2.bias",
            "audio_vae.decoder.up.0.block.0.conv2.weight",
            "audio_vae.decoder.up.0.block.0.conv2.bias",
            "audio_vae.decoder.up.0.block.1.norm1.weight",
            "audio_vae.decoder.up.0.block.1.norm1.bias",
            "audio_vae.decoder.up.0.block.1.conv1.weight",
            "audio_vae.decoder.up.0.block.1.conv1.bias",
            "audio_vae.decoder.up.0.block.1.norm2.weight",
            "audio_vae.decoder.up.0.block.1.norm2.bias",
            "audio_vae.decoder.up.0.block.1.conv2.weight",
            "audio_vae.decoder.up.0.block.1.conv2.bias",
            "audio_vae.decoder.up.0.block.2.norm1.weight",
            "audio_vae.decoder.up.0.block.2.norm1.bias",
            "audio_vae.decoder.up.0.block.2.conv1.weight",
            "audio_vae.decoder.up.0.block.2.conv1.bias",
            "audio_vae.decoder.up.0.block.2.norm2.weight",
            "audio_vae.decoder.up.0.block.2.norm2.bias",
            "audio_vae.decoder.up.0.block.2.conv2.weight",
            "audio_vae.decoder.up.0.block.2.conv2.bias",
            "audio_vae.decoder.up.0.upsample.conv.weight",
            "audio_vae.decoder.up.0.upsample.conv.bias",
            "audio_vae.decoder.up.1.block.0.norm1.weight",
            "audio_vae.decoder.up.1.block.0.norm1.bias",
            "audio_vae.decoder.up.1.block.0.conv1.weight",
            "audio_vae.decoder.up.1.block.0.conv1.bias",
            "audio_vae.decoder.up.1.block.0.norm2.weight",
            "audio_vae.decoder.up.1.block.0.norm2.bias",
            "audio_vae.decoder.up.1.block.0.conv2.weight",
            "audio_vae.decoder.up.1.block.0.conv2.bias",
            "audio_vae.decoder.up.1.block.1.norm1.weight",
            "audio_vae.decoder.up.1.block.1.norm1.bias",
            "audio_vae.decoder.up.1.block.1.conv1.weight",
            "audio_vae.decoder.up.1.block.1.conv1.bias",
            "audio_vae.decoder.up.1.block.1.norm2.weight",
            "audio_vae.decoder.up.1.block.1.norm2.bias",
            "audio_vae.decoder.up.1.block.1.conv2.weight",
            "audio_vae.decoder.up.1.block.1.conv2.bias",
            "audio_vae.decoder.up.1.block.2.norm1.weight",
            "audio_vae.decoder.up.1.block.2.norm1.bias",
            "audio_vae.decoder.up.1.block.2.conv1.weight",
            "audio_vae.decoder.up.1.block.2.conv1.bias",
            "audio_vae.decoder.up.1.block.2.norm2.weight",
            "audio_vae.decoder.up.1.block.2.norm2.bias",
            "audio_vae.decoder.up.1.block.2.conv2.weight",
            "audio_vae.decoder.up.1.block.2.conv2.bias",
            "audio_vae.decoder.up.1.upsample.conv.weight",
            "audio_vae.decoder.up.1.upsample.conv.bias",
            "audio_vae.decoder.up.2.block.0.norm1.weight",
            "audio_vae.decoder.up.2.block.0.norm1.bias",
            "audio_vae.decoder.up.2.block.0.conv1.weight",
            "audio_vae.decoder.up.2.block.0.conv1.bias",
            "audio_vae.decoder.up.2.block.0.norm2.weight",
            "audio_vae.decoder.up.2.block.0.norm2.bias",
            "audio_vae.decoder.up.2.block.0.conv2.weight",
            "audio_vae.decoder.up.2.block.0.conv2.bias",
            "audio_vae.decoder.up.2.block.1.norm1.weight",
            "audio_vae.decoder.up.2.block.1.norm1.bias",
            "audio_vae.decoder.up.2.block.1.conv1.weight",
            "audio_vae.decoder.up.2.block.1.conv1.bias",
            "audio_vae.decoder.up.2.block.1.norm2.weight",
            "audio_vae.decoder.up.2.block.1.norm2.bias",
            "audio_vae.decoder.up.2.block.1.conv2.weight",
            "audio_vae.decoder.up.2.block.1.conv2.bias",
            "audio_vae.decoder.up.2.block.2.norm1.weight",
            "audio_vae.decoder.up.2.block.2.norm1.bias",
            "audio_vae.decoder.up.2.block.2.conv1.weight",
            "audio_vae.decoder.up.2.block.2.conv1.bias",
            "audio_vae.decoder.up.2.block.2.norm2.weight",
            "audio_vae.decoder.up.2.block.2.norm2.bias",
            "audio_vae.decoder.up.2.block.2.conv2.weight",
            "audio_vae.decoder.up.2.block.2.conv2.bias",
            "audio_vae.decoder.up.2.upsample.conv.weight",
            "audio_vae.decoder.up.2.upsample.conv.bias",
            "audio_vae.decoder.up.3.block.0.norm1.weight",
            "audio_vae.decoder.up.3.block.0.norm1.bias",
            "audio_vae.decoder.up.3.block.0.conv1.weight",
            "audio_vae.decoder.up.3.block.0.conv1.bias",
            "audio_vae.decoder.up.3.block.0.norm2.weight",
            "audio_vae.decoder.up.3.block.0.norm2.bias",
            "audio_vae.decoder.up.3.block.0.conv2.weight",
            "audio_vae.decoder.up.3.block.0.conv2.bias",
            "audio_vae.decoder.up.3.block.1.norm1.weight",
            "audio_vae.decoder.up.3.block.1.norm1.bias",
            "audio_vae.decoder.up.3.block.1.conv1.weight",
            "audio_vae.decoder.up.3.block.1.conv1.bias",
            "audio_vae.decoder.up.3.block.1.norm2.weight",
            "audio_vae.decoder.up.3.block.1.norm2.bias",
            "audio_vae.decoder.up.3.block.1.conv2.weight",
            "audio_vae.decoder.up.3.block.1.conv2.bias",
            "audio_vae.decoder.up.3.block.2.norm1.weight",
            "audio_vae.decoder.up.3.block.2.norm1.bias",
            "audio_vae.decoder.up.3.block.2.conv1.weight",
            "audio_vae.decoder.up.3.block.2.conv1.bias",
            "audio_vae.decoder.up.3.block.2.norm2.weight",
            "audio_vae.decoder.up.3.block.2.norm2.bias",
            "audio_vae.decoder.up.3.block.2.conv2.weight",
            "audio_vae.decoder.up.3.block.2.conv2.bias",
            "audio_vae.decoder.norm_out.weight",
            "audio_vae.decoder.norm_out.bias",
            "audio_vae.decoder.conv_out.weight",
            "audio_vae.decoder.conv_out.bias",
            "audio_vae.quant_conv.weight",
            "audio_vae.quant_conv.bias",
            "audio_vae.post_quant_conv.weight",
            "audio_vae.post_quant_conv.bias",
        ]
        
        audio_vae_dict = {}
        video_vae_dict = {}
        
        for key in sd.keys():
            if key in audio_vae_keys:
                audio_vae_dict[key] = sd[key]
            elif key.startswith("vae."):
                new_key = key[4:]
                video_vae_dict[new_key] = sd[key]
        
        models_dir = os.path.join(folder_paths.models_dir, "vae")
        os.makedirs(models_dir, exist_ok=True)
        
        audio_vae_path = os.path.join(models_dir, f"{save_name}_audio.safetensors")
        video_vae_path = os.path.join(models_dir, f"{save_name}_video.safetensors")
        
        metadata_str = {}
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (dict, list)):
                    metadata_str[key] = json.dumps(value)
                else:
                    metadata_str[key] = str(value)
        
        save_file(audio_vae_dict, audio_vae_path, metadata=metadata_str)
        save_file(video_vae_dict, video_vae_path, metadata=metadata_str)
        
        info = f"Extracted and saved VAE files:\n"
        info += f"Audio VAE: {len(audio_vae_dict)} tensors -> {audio_vae_path}\n"
        info += f"Video VAE: {len(video_vae_dict)} tensors -> {video_vae_path}\n"
        
        print(info)
        
        return (info,)

NODE_CLASS_MAPPINGS = {
    "StarVAE_LTXV_Save": StarVAE_LTXV_Save
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarVAE_LTXV_Save": "StarVAE LTXV Save"
}
