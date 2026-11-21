"""Star Minimax Remake Option

Viggle-Remake mode for the ⭐ Star Minimax All In One node, without the
Viggle node pack: pick one of the Viggle-Remake LoRAs (lora/swap or
lora/edit from Viggle/Viggle-Remake, or the lora_lowrank variants - place
the .safetensors in models/loras), feed the source clip's frames and
soundtrack straight from ⭐ Star Video Loader, and connect one to four
reference images ("painted frames").

Inside the AIO this reproduces the Viggle Remake conditioning in-process:
the source clip is presented as <Video 1> text-only (no vision tokens),
each reference as <Picture N>, pins additionally pin a reference as a
keyframe on the output timeline (released below the keyframe_until sigma,
like Viggle's RemakeKeyframeUntil), and the source soundtrack is held as
clean target audio for the whole denoise, so the original audio passes
through unchanged.

Write the prompt in MiniMax-H3's six-section format and refer to the
references as <Picture 1> .. <Picture 4> and the source as <Video 1>,
exactly like in the Viggle workflows.
"""

import logging

import folder_paths
from comfy_api.latest import io


class StarMinimaxRemakeOption(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        loras = folder_paths.get_filename_list("loras")
        return io.Schema(
            node_id="StarMinimaxRemakeOption",
            display_name="⭐ Star Minimax Remake Option",
            category="⭐StarNodes/Video",
            description=(
                "Viggle-Remake mode for ⭐ Star Minimax All In One - uses the "
                "Viggle-Remake LoRAs (swap = replace people/characters, edit = "
                "change objects/setting/weather/light) without the Viggle node "
                "pack. Feed the source clip's 24 fps frames (and optionally its "
                "soundtrack) from ⭐ Star Video Loader, connect 1-4 painted "
                "reference frames, and write the prompt in MiniMax-H3's "
                "six-section format (<Video 1>, <Picture 1>..<Picture 4>). "
                "The source soundtrack is kept in the output. The LoRA file "
                "goes into models/loras like any other LoRA."),
            inputs=[
                io.Combo.Input("lora", options=loras,
                               default=loras[0] if loras else "",
                               tooltip="The Viggle-Remake LoRA file from models/loras "
                                       "(e.g. swap or edit from Viggle/Viggle-Remake, "
                                       "or a lora_lowrank variant). Applied to the "
                                       "diffusion model for this run only."),
                io.Float.Input("lora_strength", default=1.0, min=-10.0, max=10.0,
                               step=0.05,
                               tooltip="Strength of the remake LoRA (1.0 = full)."),
                io.Image.Input("source_video",
                               tooltip="The source clip's frames at 24 fps - supplies "
                                       "motion, camera and framing. Use the images "
                                       "output of ⭐ Star Video Loader (set force_rate "
                                       "to 24) or any IMAGE batch at 24 fps."),
                io.Autogrow.Input("ref_images", optional=True,
                                  template=io.Autogrow.TemplatePrefix(
                                      input=io.Image.Input(
                                          "ref_image",
                                          tooltip="<Picture N>: a painted frame or "
                                                  "character reference (1-4, in order). "
                                                  "Usually a source frame edited in an "
                                                  "image editor, or a character/style "
                                                  "reference."),
                                      prefix="ref_image_", min=0, max=4)),
                io.Combo.Input("ref_image_size",
                               options=["none", "0.5 MP", "1.0 MP", "1.5 MP"],
                               default="1.0 MP",
                               tooltip="Cap the reference images' size before "
                                       "they are encoded (VRAM + ref tokens). "
                                       "References are scaled to the canvas "
                                       "short edge by default - 'none' keeps "
                                       "that Viggle sizing, the MP options "
                                       "bound the encoded area instead "
                                       "(aspect preserved)."),
                io.String.Input("pins", default="",
                                tooltip="Optional: one output frame per reference "
                                        "image, e.g. '12, 190'. Each reference is then "
                                        "also pinned as a keyframe at that frame for "
                                        "the first part of the denoise. Empty = no pins."),
                io.Float.Input("keyframe_until", default=0.9, min=0.0, max=1.0,
                               step=0.01,
                               tooltip="Sigma cutoff for pinned keyframes - forwards "
                                       "below this sigma drop the pins and keep only "
                                       "the references (Viggle default 0.9)."),
                io.Audio.Input("source_audio", optional=True,
                               tooltip="Optional: the source soundtrack to keep in the "
                                       "output (e.g. the audio output of ⭐ Star Video "
                                       "Loader). Held clean through the whole denoise; "
                                       "silence when not connected."),
            ],
            outputs=[
                io.Custom("STAR_MINIMAX_REMAKE").Output(
                    "remake_settings",
                    tooltip="Remake bundle for the 'remake_settings' input of "
                            "⭐ Star Minimax All In One."),
            ],
        )

    @classmethod
    def execute(cls, lora, lora_strength, source_video, ref_images=None,
                ref_image_size="1.0 MP", pins="", keyframe_until=0.9,
                source_audio=None) -> io.NodeOutput:
        refs = [v for _name, v in sorted(
            (ref_images or {}).items(),
            key=lambda kv: int(kv[0].rsplit("_", 1)[-1])) if v is not None]
        if not refs:
            raise ValueError("[Star Minimax Remake Option] connect at least one "
                             "ref_image (painted frame / character reference)")
        pin_list = [int(p) for p in pins.replace(",", " ").split()]
        if pin_list and len(pin_list) != len(refs):
            raise ValueError(
                "[Star Minimax Remake Option] give one pin per reference image "
                f"or none: {len(pin_list)} pin(s) for {len(refs)} reference(s)")
        logging.info("[Star Minimax Remake Option] %s @ %.2f | %d ref(s) | pins %s | audio %s",
                     lora, lora_strength, len(refs), pin_list or "none",
                     "yes" if source_audio is not None else "none")
        return io.NodeOutput({
            "lora": lora,
            "lora_strength": float(lora_strength),
            "source": source_video,
            "audio": source_audio,
            "refs": refs,
            "ref_image_size": ref_image_size,
            "pins": pin_list,
            "keyframe_until": float(keyframe_until),
        })


NODE_CLASS_MAPPINGS = {
    "StarMinimaxRemakeOption": StarMinimaxRemakeOption,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarMinimaxRemakeOption": "⭐ Star Minimax Remake Option",
}
