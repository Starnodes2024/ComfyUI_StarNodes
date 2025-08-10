import os
import json
import numpy as np
from PIL import Image
import io
import folder_paths

XMP_TEMPLATE = '''<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description
   xmlns:xmp="http://ns.adobe.com/xap/1.0/"
   xmlns:GPano="http://ns.google.com/photos/1.0/panorama/"
   xmp:CreatorTool="StarSavePanoramaJPEG"
   GPano:ProjectionType="{projection_type}"
   GPano:CroppedAreaLeftPixels="0"
   GPano:CroppedAreaTopPixels="-701"
   GPano:CroppedAreaImageWidthPixels="{width}"
   GPano:CroppedAreaImageHeightPixels="{height}"
   GPano:FullPanoWidthPixels="{width}"
   GPano:UsePanoramaViewer="True"/>
 </rdf:RDF>
</x:xmpmeta>'''

class StarSavePanoramaJPEG:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.quality = 95

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the file to save."}),
                "projection_type": (["cylindrical", "equirectangular"], {"default": "cylindrical"}),
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "⭐StarNodes/Image And Latent"
    DESCRIPTION = "Saves images as JPEG with panorama XMP metadata."

    def save_images(self, images, filename_prefix="ComfyUI", projection_type="cylindrical", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            width, height = img.size
            # Prepare XMP metadata
            xmp = XMP_TEMPLATE.format(
                projection_type=projection_type,
                width=width,
                height=height
            )
            xmp_bytes = xmp.encode("utf-8")
            xmp_tag = b"http://ns.adobe.com/xap/1.0/\x00" + xmp_bytes

            # Save image to bytes
            out_bytes = io.BytesIO()
            img.save(out_bytes, format="JPEG", quality=self.quality)
            jpeg_bytes = out_bytes.getvalue()
            # Inject XMP into JPEG (APP1 segment)
            soi = jpeg_bytes[:2]
            rest = jpeg_bytes[2:]
            app1_marker = b'\xFF\xE1' + (len(xmp_tag)+2).to_bytes(2, 'big') + xmp_tag
            new_jpeg = soi + app1_marker + rest

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.jpg"
            with open(os.path.join(full_output_folder, file), "wb") as f:
                f.write(new_jpeg)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1
        return { "ui": { "images": results } }

NODE_CLASS_MAPPINGS = {
    "StarSavePanoramaJPEG": StarSavePanoramaJPEG,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "StarSavePanoramaJPEG": "\u2b50 Star Save Panorama JPEG",
}
