import torch
import torch.nn.functional as F

class StarVideoJoiner:
    BGCOLOR = "#3d124d"
    COLOR = "#19124d"
    CATEGORY = '⭐StarNodes/Video'
    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "join_video"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
            },
            "optional": {
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "audio_1": ("AUDIO",),
                "audio_2": ("AUDIO",),
                "audio_3": ("AUDIO",),
                "audio_4": ("AUDIO",),
                "audio_5": ("AUDIO",),
            }
        }

    def join_video(self, image_1, image_2=None, image_3=None, image_4=None, image_5=None,
                   audio_1=None, audio_2=None, audio_3=None, audio_4=None, audio_5=None):
        
        images_list = []
        audio_list = []
        
        if image_1 is not None:
            images_list.append(image_1)
            reference_height = image_1.shape[1]
            reference_width = image_1.shape[2]
        
        for img in [image_2, image_3, image_4, image_5]:
            if img is not None:
                if img.shape[1] != reference_height or img.shape[2] != reference_width:
                    img = F.interpolate(
                        img.permute(0, 3, 1, 2),
                        size=(reference_height, reference_width),
                        mode='bilinear',
                        align_corners=False
                    ).permute(0, 2, 3, 1)
                images_list.append(img)
        
        combined_images = torch.cat(images_list, dim=0) if images_list else None
        
        for audio in [audio_1, audio_2, audio_3, audio_4, audio_5]:
            if audio is not None:
                audio_list.append(audio)
        
        if audio_list:
            if len(audio_list) == 1:
                combined_audio = audio_list[0]
            else:
                waveforms = []
                sample_rate = None
                
                for audio_dict in audio_list:
                    if isinstance(audio_dict, dict):
                        waveform = audio_dict.get('waveform')
                        if sample_rate is None:
                            sample_rate = audio_dict.get('sample_rate', 44100)
                        
                        if waveform is not None:
                            waveforms.append(waveform)
                    else:
                        waveforms.append(audio_dict)
                
                if waveforms:
                    combined_waveform = torch.cat(waveforms, dim=-1)
                    
                    if isinstance(audio_list[0], dict):
                        combined_audio = {
                            'waveform': combined_waveform,
                            'sample_rate': sample_rate
                        }
                    else:
                        combined_audio = combined_waveform
                else:
                    combined_audio = None
        else:
            combined_audio = None
        
        return (combined_images, combined_audio)

NODE_CLASS_MAPPINGS = {
    "StarVideoJoiner": StarVideoJoiner
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarVideoJoiner": "⭐ Star Video Joiner"
}
