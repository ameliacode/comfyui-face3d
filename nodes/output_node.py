from comfy_api.latest import io, ui


class SaveMyImage(io.ComfyNode):
    """Save images to the output folder and preview them on the node."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveMyImage",
            display_name="Save My Image",
            category="MyNodes",
            is_output_node=True,
            inputs=[
                io.Image.Input("images"),
                io.String.Input(
                    "filename_prefix",
                    default="ComfyUI",
                    tooltip="Filename prefix. Supports date formatting: %date:yyyy-MM-dd%.",
                ),
            ],
            outputs=[],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
        )

    @classmethod
    def execute(cls, images, filename_prefix):
        saved = ui.ImageSaveHelper.get_save_images_ui(images, filename_prefix, cls=cls)
        return io.NodeOutput(ui=saved)
