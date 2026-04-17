import torch
from comfy_api.latest import io
from .load_model import MY_MODEL_CONFIG


class ExampleImageNode(io.ComfyNode):
    """
    Template: image-in → model → image-out.
    Replace the execute body with your actual inference logic.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ExampleImageNode",
            display_name="Example Image Node",
            category="MyNodes",
            description="Processes an image using My Model.",
            inputs=[
                io.Image.Input("image", tooltip="Input image (B,H,W,C float32 0-1)."),
                MY_MODEL_CONFIG.Input("model_config"),
                # ── common optional controls ──────────────────────────────
                io.Float.Input(
                    "strength",
                    default=1.0,
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    display_mode=io.NumberDisplay.slider,
                    tooltip="Effect strength.",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="Random seed.",
                ),
                io.Mask.Input("mask", optional=True, tooltip="Optional mask."),
            ],
            outputs=[
                io.Image.Output("IMAGE"),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def validate_inputs(cls, image, model_config, strength, seed, mask=None):
        return True

    @classmethod
    def execute(cls, image, model_config, strength, seed, mask=None):
        # ── replace with real inference ───────────────────────────────────
        result = torch.clamp(image * strength, 0.0, 1.0)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            result = result * mask.unsqueeze(-1) + image * (1 - mask.unsqueeze(-1))
        # ─────────────────────────────────────────────────────────────────
        return io.NodeOutput(result)
