from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

WEB_DIRECTORY = "./js"


class MyExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        from .nodes import NODE_CLASSES
        return NODE_CLASSES

    @override
    async def on_load(self):
        pass  # optional: initialization logic on load


async def comfy_entrypoint() -> MyExtension:
    return MyExtension()
