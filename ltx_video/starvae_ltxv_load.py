import folder_paths
import comfy.utils
import comfy.sd
from comfy.ldm.lightricks.vae.audio_vae import AudioVAE

class StarVAE_LTXV_Load:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_vae": (folder_paths.get_filename_list("vae"),),
                "audio_vae": (folder_paths.get_filename_list("vae"),),
            }
        }
    
    RETURN_TYPES = ("VAE", "VAE")
    RETURN_NAMES = ("video_vae", "audio_vae")
    FUNCTION = "load_vae"
    CATEGORY = "StarVAE"

    def load_vae(self, video_vae, audio_vae):
        video_vae_path = folder_paths.get_full_path_or_raise("vae", video_vae)
        audio_vae_path = folder_paths.get_full_path_or_raise("vae", audio_vae)
        
        video_vae_sd, video_metadata = comfy.utils.load_torch_file(video_vae_path, return_metadata=True)
        video_vae_obj = comfy.sd.VAE(video_vae_sd, metadata=video_metadata)
        
        audio_vae_sd, audio_metadata = comfy.utils.load_torch_file(audio_vae_path, return_metadata=True)
        audio_vae_obj = AudioVAE(audio_vae_sd, audio_metadata)
        
        return (video_vae_obj, audio_vae_obj)

NODE_CLASS_MAPPINGS = {
    "StarVAE_LTXV_Load": StarVAE_LTXV_Load
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarVAE_LTXV_Load": "StarVAE LTXV Load"
}
