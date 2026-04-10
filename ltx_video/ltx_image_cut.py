class LTXImageCut:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "cut_first_frames": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1}),
                "cut_last_frames": ("INT", {"default": 10, "min": 0, "max": 9999, "step": 1}),
                "export_frame_number": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1}),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("images", "first_image", "last_image", "selected_frame")
    FUNCTION = "cut_frames"
    CATEGORY = "⭐StarNodes/LTX Video"
    
    def cut_frames(self, images, cut_first_frames, cut_last_frames, export_frame_number):
        import torch
        
        if images is None or len(images) == 0:
            return (images, None, None, None)
        
        total_frames = len(images)
        
        # Ensure cut values don't exceed total frames
        cut_first = min(cut_first_frames, total_frames - 1)
        cut_last = min(cut_last_frames, total_frames - 1)
        
        # Calculate the range of actual content frames
        start_idx = cut_first
        end_idx = total_frames - cut_last
        
        # Ensure we have at least one content frame
        if start_idx >= end_idx:
            # If cutting would remove all frames, use middle frame
            middle_idx = total_frames // 2
            start_idx = middle_idx
            end_idx = middle_idx + 1
        
        # Get the actual content frames
        content_frames = images[start_idx:end_idx]
        
        # Get first and last frames of the content
        first_content_frame = content_frames[0:1]
        last_content_frame = content_frames[-1:]
        
        # Build the output batch maintaining original frame count
        result_frames = []
        
        # Fill cut first frames with copies of the first content frame
        if cut_first > 0:
            for _ in range(cut_first):
                result_frames.append(first_content_frame)
        
        # Add the actual content frames
        result_frames.append(content_frames)
        
        # Fill cut last frames with copies of the last content frame
        if cut_last > 0:
            for _ in range(cut_last):
                result_frames.append(last_content_frame)
        
        # Concatenate all frames
        cut_images = torch.cat(result_frames, dim=0)
        
        # Get selected frame from original batch
        if export_frame_number < total_frames:
            selected_frame = images[export_frame_number:export_frame_number+1]
        else:
            # If frame number is out of range, use last frame
            selected_frame = images[-1:]
        
        return (cut_images, first_content_frame, last_content_frame, selected_frame)


NODE_CLASS_MAPPINGS = {
    "LTXImageCut": LTXImageCut,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXImageCut": "⭐ LTX Image Cut",
}
