from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

WEB_DIRECTORY = "./js"


class ComfyuiFlameExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        try:
            from .nodes import NODE_CLASSES
        except ImportError:
            from nodes import NODE_CLASSES
        return NODE_CLASSES

async def comfy_entrypoint() -> ComfyuiFlameExtension:
    return ComfyuiFlameExtension()
