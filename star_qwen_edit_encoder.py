import node_helpers
import comfy.utils
import torch
import time

PREFERRED_QWENIMAGE_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]


class StarQwenEditEncoder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", ),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
            "optional": {
                "vae": ("VAE", ),
                "image": ("IMAGE", ),
                # If provided, skip VAE encode and use directly
                "reference_latent": ("LATENT", ),
                # Performance/quality controls (defaults preserve original behavior)
                "resize_mode": ( ["lanczos", "bicubic"], {"default": "lanczos"} ),
                "skip_upscale_if_match": ("BOOLEAN", {"default": True}),
                # Allow skipping resample if aspect ratio already close to target
                "ar_skip_epsilon": ("FLOAT", {"default": 0.002, "min": 0.0, "max": 0.5, "step": 0.0005}),
                # Cache prompt-only encodings (no image/reference_latent) keyed by clip id + prompt + cache_bust
                "cache_tokens": ("BOOLEAN", {"default": False}),
                "cache_bust": ("STRING", {"default": ""}),
                # Debug
                "debug_timing": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    # Keep similar category but under StarNodes
    CATEGORY = "⭐StarNodes/Conditioning"

    def encode(self, clip, prompt, vae=None, image=None, reference_latent=None,
               resize_mode="lanczos", skip_upscale_if_match=True, ar_skip_epsilon=0.002,
               cache_tokens=False, cache_bust="", debug_timing=False):
        with torch.inference_mode():
            def _sync():
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
            def _now():
                _sync()
                return time.perf_counter()

            t_start = _now() if debug_timing else None
            # init simple cache store on instance
            if not hasattr(self, "_cache"):
                self._cache = {}

            # Try cache for prompt-only case (fast path)
            if cache_tokens and image is None and reference_latent is None:
                cache_key = (id(clip), prompt, cache_bust)
                cached = self._cache.get(cache_key)
                if cached is not None:
                    return (cached["conditioning"], )

            ref_latent = None
            if reference_latent is not None:
                # Use provided latent directly
                ref_latent = reference_latent
                images = [] if image is None else [image]
            else:
                if image is None:
                    images = []
                else:
                    images = [image]
                    if vae is not None:
                        width = image.shape[2]
                        height = image.shape[1]
                        aspect_ratio = width / height
                        _, tgt_w, tgt_h = min(
                            (abs(aspect_ratio - w / h), w, h) for w, h in PREFERRED_QWENIMAGE_RESOLUTIONS
                        )
                        # Skip resample if (a) exact size match OR (b) AR close enough
                        do_skip = False
                        if skip_upscale_if_match:
                            if width == tgt_w and height == tgt_h:
                                do_skip = True
                            else:
                                tgt_ar = float(tgt_w) / float(tgt_h)
                                img_ar = float(width) / float(height)
                                if abs(img_ar - tgt_ar) <= float(ar_skip_epsilon):
                                    do_skip = True

                        if do_skip:
                            image_rs = image
                        else:
                            t_rs0 = _now() if debug_timing else None
                            image_rs = comfy.utils.common_upscale(
                                image.movedim(-1, 1), tgt_w, tgt_h, resize_mode, "center"
                            ).movedim(1, -1)
                            if debug_timing:
                                t_rs1 = _now()
                                print(f"[StarQwenEditEncoder] resize ({resize_mode}) to {tgt_w}x{tgt_h} took {(t_rs1 - t_rs0):.3f}s;")
                        t_ve0 = _now() if debug_timing else None
                        ref_latent = vae.encode(image_rs[:, :, :, :3])
                        if debug_timing:
                            t_ve1 = _now()
                            dev = str(image.device) if hasattr(image, 'device') else 'unknown'
                            print(f"[StarQwenEditEncoder] VAE.encode on device={dev} took {(t_ve1 - t_ve0):.3f}s;")

            t_tok0 = _now() if debug_timing else None
            tokens = clip.tokenize(prompt, images=images)
            if debug_timing:
                t_tok1 = _now()
                print(f"[StarQwenEditEncoder] clip.tokenize took {(t_tok1 - t_tok0):.3f}s; images={len(images)}")
            t_enc0 = _now() if debug_timing else None
            conditioning = clip.encode_from_tokens_scheduled(tokens)
            if debug_timing:
                t_enc1 = _now()
                print(f"[StarQwenEditEncoder] clip.encode_from_tokens_scheduled took {(t_enc1 - t_enc0):.3f}s")
            if ref_latent is not None:
                conditioning = node_helpers.conditioning_set_values(
                    conditioning, {"reference_latents": [ref_latent]}, append=True
                )

            # store cache for prompt-only requests
            if cache_tokens and image is None and reference_latent is None:
                cache_key = (id(clip), prompt, cache_bust)
                self._cache[cache_key] = {"conditioning": conditioning}
            if debug_timing and t_start is not None:
                t_end = _now()
                print(f"[StarQwenEditEncoder] total encode took {(t_end - t_start):.3f}s")
            return (conditioning, )


NODE_CLASS_MAPPINGS = {
    "StarQwenEditEncoder": StarQwenEditEncoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarQwenEditEncoder": "⭐ Star Qwen Edit Encoder",
}
