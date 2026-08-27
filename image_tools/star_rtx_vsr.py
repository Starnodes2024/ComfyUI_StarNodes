"""Star Advanced RTX VSR - NVIDIA RTX Video Super Resolution upscaler.

Vendors the VSR logic from the comfyui_nvidia_rtx_nodes pack
(RTXVideoSuperResolution) so this node stands alone. By default every frame
is super-resolved with the NVIDIA VFX SDK at 4K (8K renders directly), then
resized down when a lower output size is chosen - the 4K pass keeps the VSR
detail at any output size. Disable "render_at_4k" to let VSR upscale
directly to the chosen output size instead (lower VRAM, faster, less
detail). Requires the nvidia-vfx pip package, an RTX GPU and Windows/Linux
with the NVIDIA VFX runtime.
"""

import torch
import nvvfx

import comfy.utils
from comfy_api.latest import io

OUTPUT_SIZES = {
    "8K": (7680, 4320),
    "4K": (3840, 2160),
    "2K": (2560, 1440),
    "Full HD": (1920, 1080),
}
VSR_TARGET = OUTPUT_SIZES["4K"]   # default VSR render target (8K renders directly)

QUALITY_LEVELS = {
    "LOW": nvvfx.effects.QualityLevel.LOW,
    "MEDIUM": nvvfx.effects.QualityLevel.MEDIUM,
    "HIGH": nvvfx.effects.QualityLevel.HIGH,
    "ULTRA": nvvfx.effects.QualityLevel.ULTRA,
}

# downscale filters offered when render_at_4k downsizes 4K/8K -> 2K/Full HD
DOWNSCALE_FILTERS = ["lanczos", "bicubic", "area", "nearest"]


def _fit_inside(w, h, max_w, max_h):
    """Aspect-preserving fit inside max_w x max_h, snapped to a multiple of 8."""
    scale = min(max_w / w, max_h / h)
    return (max(8, round(w * scale / 8) * 8),
            max(8, round(h * scale / 8) * 8))


class StarAdvancedRTXVSR(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="StarAdvancedRTXVSR",
            display_name="⭐ Star Advanced RTX VSR",
            category="⭐StarNodes/Upscale",
            description=(
                "NVIDIA RTX Video Super Resolution: every frame is AI-upscaled "
                "with the NVIDIA VFX SDK. By default 4K and below render at 4K "
                "first and resize down (the 4K pass keeps the VSR detail at any "
                "output size); 8K renders directly. Disable render_at_4k to let "
                "VSR upscale directly to the chosen output (lower VRAM, faster). "
                "Aspect ratio is preserved; sizes snap to a multiple of 8. "
                "Requires the nvidia-vfx package and an RTX GPU."
            ),
            search_aliases=["rtx", "nvidia", "upscale", "super resolution", "vsr"],
            inputs=[
                io.Image.Input("images",
                               tooltip="Frames/images to upscale (a video batch works directly)."),
                io.Combo.Input("output_size", options=list(OUTPUT_SIZES.keys()), default="2K",
                               tooltip="Final output size. Aspect ratio is preserved; sizes snap to a multiple of 8."),
                io.Combo.Input("strength", options=list(QUALITY_LEVELS.keys()), default="HIGH",
                               tooltip="VSR quality level (higher = sharper, slower)."),
                io.Boolean.Input("render_at_4k", default=True,
                                 tooltip="On (default): render at 4K first (8K renders directly), then resize down "
                                         "to the chosen output - keeps full VSR detail at any size. Off: let VSR "
                                         "upscale directly to the chosen output - lower VRAM, faster, less detail."),
                io.Int.Input("batch_size", default=2, min=1, max=64,
                             tooltip="Frames processed per batch. Lower (1) saves VRAM on low-end GPUs; higher "
                                     "speeds up long clips on GPUs with headroom."),
                io.Combo.Input("downscale_filter", options=DOWNSCALE_FILTERS, default="area",
                               tooltip="Filter used when downscaling the 4K/8K VSR result to 2K/Full HD. "
                                       "lanczos: sharpest, preserves VSR detail best, slight ringing/halos on "
                                       "hard edges - best default for video. "
                                       "bicubic: balanced, moderate sharpening, more aliasing on diagonals. "
                                       "area: box-average, cleanest anti-aliasing for large downscale factors "
                                       "(e.g. 4K->Full HD), softest - good for smooth video. "
                                       "nearest: fastest, blocky, not recommended for video."),
            ],
            outputs=[
                io.Image.Output("upscaled_images",
                                tooltip="The upscaled frames at the chosen output size."),
            ],
        )

    @classmethod
    def execute(cls, images: torch.Tensor, output_size: str, strength: str,
                render_at_4k: bool, batch_size: int, downscale_filter: str) -> io.NodeOutput:
        if not torch.cuda.is_available():
            raise RuntimeError("Star Advanced RTX VSR: an NVIDIA RTX GPU with CUDA is required.")

        b, h, w, c = images.shape
        out_w, out_h = _fit_inside(w, h, *OUTPUT_SIZES[output_size])

        # decide VSR render target
        if output_size == "8K" or (render_at_4k and output_size != "8K"):
            # 8K always renders natively; 4K/2K/FHD render at 4K first when enabled
            vsr_w, vsr_h = (_fit_inside(w, h, *OUTPUT_SIZES["8K"])
                            if output_size == "8K" else _fit_inside(w, h, *VSR_TARGET))
        else:
            # direct mode: VSR upscales straight to the chosen output
            vsr_w, vsr_h = out_w, out_h

        batch_size = max(1, int(batch_size))
        quality = QUALITY_LEVELS.get(strength, nvvfx.effects.QualityLevel.HIGH)

        print(f"StarNodes RTX VSR: {b} frame(s) {w}x{h} -> VSR {vsr_w}x{vsr_h} "
              f"(quality={strength}, batch={batch_size}, render_at_4k={render_at_4k})")
        if (out_w, out_h) != (vsr_w, vsr_h):
            print(f"StarNodes RTX VSR: downsample target {out_w}x{out_h} ({output_size})")

        with nvvfx.VideoSuperRes(quality) as sr:
            sr.output_width = vsr_w
            sr.output_height = vsr_h
            print("StarNodes RTX VSR: loading VSR model...")
            sr.load()
            print("StarNodes RTX VSR: model loaded, running super-resolution...")

            out_tensor = torch.empty((b, vsr_h, vsr_w, c), dtype=images.dtype)
            done = 0
            bar_w = 30
            for i in range(0, b, batch_size):
                batch_cuda = images[i:i + batch_size].cuda().permute(0, 3, 1, 2).float().contiguous()
                for j in range(batch_cuda.shape[0]):
                    dlpack_out = sr.run(batch_cuda[j]).image
                    out_tensor[i + j: i + j + 1] = torch.from_dlpack(dlpack_out).movedim(0, -1).unsqueeze(0).cpu()
                done = min(i + batch_size, b)
                filled = bar_w * done // b
                bar = "█" * filled + "░" * (bar_w - filled)
                pct = 100 * done // b
                print(f"\rStarNodes RTX VSR: [{bar}] {done}/{b} frames ({pct}%)",
                      end="", flush=True)
            print()  # newline after the bar finishes

        if (out_w, out_h) != (vsr_w, vsr_h):
            print(f"StarNodes RTX VSR: resizing {vsr_w}x{vsr_h} -> {out_w}x{out_h} "
                  f"({downscale_filter})...")
            out_tensor = comfy.utils.common_upscale(
                out_tensor.movedim(-1, 1), out_w, out_h, downscale_filter, "disabled").movedim(1, -1)

        print(f"StarNodes RTX VSR: done, output {out_tensor.shape[2]}x{out_tensor.shape[1]} "
              f"for {out_tensor.shape[0]} frame(s)")
        return io.NodeOutput(out_tensor)


NODE_CLASS_MAPPINGS = {
    "StarAdvancedRTXVSR": StarAdvancedRTXVSR,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarAdvancedRTXVSR": "⭐ Star Advanced RTX VSR",
}
