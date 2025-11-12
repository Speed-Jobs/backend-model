from __future__ import annotations

from pathlib import Path
from typing import Optional


def backend_root() -> Path:
    """Return the backend-model root directory irrespective of CWD.

    This file lives at app/services/__init__.py, so backend root is three
    parents up from here.
    """
    return Path(__file__).resolve().parents[2]


def get_output_dir() -> Path:
    return backend_root() / "data" / "output"


def get_img_dir() -> Path:
    return backend_root() / "img"


def resolve_dir(arg: Optional[Path | str], default_dir: Path) -> Path:
    """Resolve a directory path.

    - If arg is None or empty, return default_dir.
    - If arg is an absolute path, return it as Path.
    - If arg is a relative path, resolve it relative to backend root.
    Ensures the directory exists.
    """
    if arg is None or (isinstance(arg, str) and not arg.strip()):
        return default_dir

    p = Path(arg)
    if not p.is_absolute():
        p = (backend_root() / p).resolve()

    # If resolved path escapes backend_root, clamp to default_dir
    try:
        _ = p.relative_to(backend_root())
    except Exception:
        return default_dir

    return p