"""Star Minimax Multiref Option

Timed keyframe references ("guides") for the ⭐ Star Minimax All In One node:
connect up to 9 reference images (or short clips) to the guide_image_N slots
and give each one its start time in seconds on the output timeline. This is
the multi-frame / keyframe path of the stock MiniMax H3 template (the chain
of "Add Guide for MiniMax H3" nodes plus their seconds -> frames Math
Expression nodes), collapsed into a single option node.

Inside the AIO the guides are applied in-process through the exact code path
of the core MiniMaxH3AddGuide node: single images become still anchors, image
batches of 5+ frames anchor as clips (cropped down to the valid 17k+5 clip
lengths), and the start time is converted with frame_idx = round(seconds * 24)
- the same math as the template's Math Expression nodes. Negative start times
count back from the end of the video, exactly like negative frame indices on
the core node.
"""

import logging

from comfy_api.latest import io

_MAX_GUIDES = 9


def _guide_inputs():
    return [io.Image.Input(
        f"guide_image_{i}", optional=(i > 0),
        tooltip="Reference image or short clip (frames @ 24 fps) to anchor on "
                f"the output timeline. Its start time is set with the "
                f"'start @ guide_image_{i} (s)' widget.")
        for i in range(_MAX_GUIDES)]


def _start_inputs():
    return [io.Float.Input(
        f"start_{i}", display_name=f"start @ guide_image_{i} (s)",
        default=0.0, min=-150.0, max=150.0, step=0.1,
        tooltip="Start time in seconds on the output timeline (24 fps, "
                "converted with frame = round(seconds * 24)). Only used while "
                f"guide_image_{i} is connected. Negative values count back "
                "from the end of the video.")
        for i in range(_MAX_GUIDES)]


class StarMinimaxMultirefOption(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="StarMinimaxMultirefOption",
            display_name="⭐ Star Minimax Multiref Option",
            category="⭐StarNodes/Video",
            description=(
                "Timed keyframe references for the ⭐ Star Minimax All In One node - "
                "the 'Add Guide for MiniMax H3' chain of the stock multiframe template "
                "as a single option node. Connect reference images (single frames or "
                "short clips @ 24 fps) to the guide_image slots and set each start "
                "time on the output timeline with the matching "
                "'start @ guide_image_N (s)' widget. A single image anchors a still "
                "frame, a batch of 5+ frames anchors as a clip (cropped to the valid "
                "17k+5 lengths). Guides influence the output on their own timeline "
                "position only - they are not part of the <Picture i> reference "
                "ordering. Also carried into the optional latent-upscale refine pass "
                "(resolution-matched)."),
            inputs=[
                *_guide_inputs(),
                *_start_inputs(),
            ],
            outputs=[
                io.Custom("MULTIREF_SETTINGS").Output(
                    "multiref_settings",
                    tooltip="Timed reference bundle for the 'multiref_settings' input "
                            "of ⭐ Star Minimax All In One."),
            ],
        )

    @classmethod
    def execute(cls, guide_image_0=None, guide_image_1=None, guide_image_2=None,
                guide_image_3=None, guide_image_4=None, guide_image_5=None,
                guide_image_6=None, guide_image_7=None, guide_image_8=None,
                start_0=0.0, start_1=0.0, start_2=0.0, start_3=0.0, start_4=0.0,
                start_5=0.0, start_6=0.0, start_7=0.0, start_8=0.0) -> io.NodeOutput:
        images = (guide_image_0, guide_image_1, guide_image_2, guide_image_3,
                  guide_image_4, guide_image_5, guide_image_6, guide_image_7,
                  guide_image_8)
        starts = (start_0, start_1, start_2, start_3, start_4,
                  start_5, start_6, start_7, start_8)
        guides = [{"image": image, "start_seconds": float(starts[i])}
                  for i, image in enumerate(images) if image is not None]
        if guides:
            logging.info("[Star Multiref Option] %d guide(s) @ %s", len(guides),
                         ", ".join(f"{g['start_seconds']:.2f}s" for g in guides))
        return io.NodeOutput({"guides": guides})


NODE_CLASS_MAPPINGS = {
    "StarMinimaxMultirefOption": StarMinimaxMultirefOption,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarMinimaxMultirefOption": "⭐ Star Minimax Multiref Option",
}
