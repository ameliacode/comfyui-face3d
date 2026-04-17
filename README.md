# ComfyUI Custom Node Template

Template repository for building ComfyUI custom node packages with the V3 API.

Use [README.md.example](README.md.example) as the starting point for the real project README.

## Included

- V3 `ComfyExtension` entrypoint in [`__init__.py`](__init__.py)
- example nodes in [`nodes/`](nodes)
- install hooks with optional `comfy_env` support in [`install.py`](install.py) and [`prestartup_script.py`](prestartup_script.py)
- package metadata in [`pyproject.toml`](pyproject.toml)
- project README and maintainer templates in [`README.md.example`](README.md.example) and [`CLAUDE.md.example`](CLAUDE.md.example)
- GitHub workflows in [`.github/workflows/`](.github/workflows)

## Usage

1. Copy this template into a new repository.
2. Replace placeholders in `pyproject.toml`, the node files, and workflow metadata.
3. Rename or copy `README.md.example` to your project `README.md`.
4. Rename or copy `CLAUDE.md.example` to your project `CLAUDE.md` if you want maintainer guidance.
5. Replace the example node logic with your actual model or processing code.

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgements

- [PozzettiAndrea](https://github.com/PozzettiAndrea)'s ComfyUI tooling and workflow conventions were used as references for parts of this template, especially around `comfy_env`, `comfy-test`, and maintainer guidance structure.
- [jtydhr88/comfyui-custom-node-skills](https://github.com/jtydhr88/comfyui-custom-node-skills) was also used as a reference.
