import folder_paths
import os
import json
from PIL import Image
import numpy as np
import torch


class StarPanoramaViewer:
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "layout": (["Mono", "SBS", "Top/Bottom"], {"default": "Top/Bottom"}),
            },
            "optional": {
                "depth_map": ("IMAGE",),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "view_panorama"
    OUTPUT_NODE = True
    CATEGORY = "⭐StarNodes/Image And Latent"
    DESCRIPTION = "Interactive 360-degree panorama viewer with parallax effect for stereoscopic images (SBS/Top-Bottom)."

    def view_panorama(self, image, layout="Top/Bottom", depth_map=None):
        results = []
        
        for batch_number, img_tensor in enumerate(image):
            i = 255. * img_tensor.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            filename = f"star_pano_temp_{batch_number}.jpg"
            file_path = os.path.join(self.output_dir, filename)
            img.save(file_path, quality=90)
            
            results.append({
                "filename": filename,
                "type": self.type,
                "layout": layout,
                "subfolder": ""
            })
        
        if depth_map is not None:
            for batch_number, depth_tensor in enumerate(depth_map):
                d = 255. * depth_tensor.cpu().numpy()
                depth_img = Image.fromarray(np.clip(d, 0, 255).astype(np.uint8))
                
                depth_filename = f"star_pano_depth_{batch_number}.jpg"
                depth_path = os.path.join(self.output_dir, depth_filename)
                depth_img.save(depth_path, quality=90)
                
                if batch_number < len(results):
                    results[batch_number]["depth_filename"] = depth_filename

        return {"ui": {"panoramas": results}}


NODE_CLASS_MAPPINGS = {
    "StarPanoramaViewer": StarPanoramaViewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarPanoramaViewer": "⭐ Star 360 Parallax Viewer",
}
