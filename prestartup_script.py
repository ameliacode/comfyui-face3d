from pathlib import Path
import shutil

try:
    from comfy_env import setup_env, copy_files
except ImportError:
    setup_env = None
    copy_files = None

if setup_env is not None:
    setup_env()

SCRIPT_DIR   = Path(__file__).parent
COMFYUI_DIR  = SCRIPT_DIR.parent.parent

# Copy any static assets (images, default files, etc.) into ComfyUI/input
assets_dir = SCRIPT_DIR / "assets"
input_dir = COMFYUI_DIR / "input"
if copy_files is not None:
    copy_files(assets_dir, input_dir)
elif assets_dir.exists():
    input_dir.mkdir(parents=True, exist_ok=True)
    for src in assets_dir.rglob("*"):
        if not src.is_file():
            continue
        rel_path = src.relative_to(assets_dir)
        dest = input_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
