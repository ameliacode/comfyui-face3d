# CLAUDE.md

Guidance for AI assistants and maintainers working on this template repository.

## Purpose

This repository is a template for future ComfyUI custom node projects.

- [`README.md`](README.md) documents the template repository itself.
- [`README.md.example`](README.md.example) is the project-facing README template that should become the real `README.md` in repos created from this template.
- [`CLAUDE.md.example`](CLAUDE.md.example) is the project-facing maintainer guidance template that should become the real `CLAUDE.md` in repos created from this template.

Do not mix those two roles.

## Documentation Rules

- Keep `README.md` focused on the template repository.
- Keep `README.md.example` focused on end users of the future generated project.
- Keep `CLAUDE.md` focused on this template repository.
- Keep `CLAUDE.md.example` focused on maintainers of the future generated project.
- Do not leave template-maintainer instructions inside `README.md.example`.
- If maintainer-only guidance is needed, put it here or in internal docs.

## Template Structure

```text
ComfyUI-MyNodes/
├── __init__.py
├── install.py
├── prestartup_script.py
├── pyproject.toml
├── requirements.txt
├── comfy-env-root.toml
├── comfy-test.toml
├── LICENSE
├── README.md
├── README.md.example
├── CLAUDE.md
├── CLAUDE.md.example
├── nodes/
│   ├── __init__.py
│   ├── load_model.py
│   ├── example_node.py
│   ├── output_node.py
│   └── utils.py
├── js/
│   └── extension.js
├── workflows/
├── assets/
├── docs/
└── .github/
    └── workflows/
```

## Integration Notes

### comfy-test

`pr-gate.yml` and `run-tests.yml` delegate to PozzettiAndrea's reusable `comfy-test` workflow. Test levels in `comfy-test.toml` are expected to cover:

`syntax` -> `install` -> `registration` -> `instantiation` -> `static_capture` -> `validation` -> `execution`

Put example workflows in `workflows/` when using that setup.

### comfy_env

This template supports `comfy_env`, but it must remain optional.

- `install.py` may use `comfy_env.install()` when available
- `prestartup_script.py` may use `setup_env()` and `copy_files()` when available
- both files must preserve a local fallback path for fresh clones without `comfy_env`

## Registration Pattern

Use V3 `ComfyExtension` registration in `__init__.py`:

```python
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
        pass


async def comfy_entrypoint() -> MyExtension:
    return MyExtension()
```

## Node Pattern

Nodes should follow the standard `define_schema()` plus `execute()` structure:

```python
from comfy_api.latest import io


class MyNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MyNode",
            display_name="My Node",
            category="MyNodes",
            description="One-line tooltip.",
            inputs=[
                io.Image.Input("image"),
                io.Float.Input(
                    "strength",
                    default=1.0,
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    display_mode=io.NumberDisplay.slider,
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                ),
                io.Mask.Input("mask", optional=True),
            ],
            outputs=[io.Image.Output("IMAGE")],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def validate_inputs(cls, image, strength, seed, mask=None):
        return True

    @classmethod
    def execute(cls, image, strength, seed, mask=None):
        import torch
        result = torch.clamp(image * strength, 0.0, 1.0)
        return io.NodeOutput(result)
```

Keep schema defaults and validation rules aligned.

## Custom Type Pattern

Use custom opaque types for values passed between nodes:

```python
MY_MODEL_CONFIG = io.Custom("MY_MODEL_CONFIG")
```

Then expose it with:

```python
outputs=[MY_MODEL_CONFIG.Output(display_name="model_config")]
inputs=[MY_MODEL_CONFIG.Input("model_config")]
```

## Model Loading Pattern

`nodes/load_model.py` is the template for first-run model download and dtype selection:

```python
checkpoint = Path(folder_paths.models_dir) / "your_model" / "model.safetensors"

if not checkpoint.exists():
    hf_hub_download(
        repo_id="org/repo",
        filename="model.safetensors",
        local_dir=str(checkpoint.parent),
    )
```

Prefer JSON-safe config values across boundaries, such as storing dtype as `"bf16"`, `"fp16"`, or `"fp32"` instead of raw torch dtypes.

## Output Node Pattern

`nodes/output_node.py` should remain a simple save-and-preview example using `ui.ImageSaveHelper`.

## Input Reference

| Need | Code |
|---|---|
| Image tensor | `io.Image.Input("image")` |
| Mask | `io.Mask.Input("mask", optional=True)` |
| Float slider | `io.Float.Input("v", default=1.0, min=0.0, max=2.0, display_mode=io.NumberDisplay.slider)` |
| Integer | `io.Int.Input("steps", default=20, min=1, max=150)` |
| Seed | `io.Int.Input("seed", default=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True)` |
| Dropdown | `io.Combo.Input("mode", options=["a", "b", "c"], default="a")` |
| Checkbox | `io.Boolean.Input("flag", default=True)` |
| Multiline text | `io.String.Input("prompt", multiline=True)` |
| Custom type | `MY_TYPE = io.Custom("MY_TYPE")` |
| Any type | `io.AnyType.Input("anything")` |

## Metadata Pattern

`pyproject.toml` should keep package metadata and the `[tool.comfy]` registry block aligned:

```toml
[project]
name = "comfyui-my-nodes"
version = "0.1.0"
description = "My custom nodes for ComfyUI."
license = { file = "LICENSE" }
requires-python = ">=3.10"

[project.urls]
Repository = "https://github.com/YOUR_USERNAME/ComfyUI-MyNodes"

[tool.comfy]
PublisherId = "your_publisher_id"
DisplayName = "My Nodes"
Icon = ""
```

## When Updating The Template

- Keep placeholders consistent across `pyproject.toml`, workflows, `README.md.example`, and `CLAUDE.md.example`.
- Prefer safe defaults over private-tooling assumptions.
- Preserve a clean fresh-clone experience.

## Checklist

- Replace placeholder names, URLs, and display strings consistently.
- Keep `install.py` and `prestartup_script.py` working without `comfy_env`.
- Keep example nodes runnable out of the box.
- Keep example inputs, defaults, and validators consistent.
- Keep publish and test workflows aligned with the actual package behavior.
