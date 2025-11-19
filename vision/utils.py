"""Utility helpers for the VisionMate vision package."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if it does not yet exist."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def join_strings(parts: Iterable[str], fallback: str = "") -> str:
    """Join non-empty strings with a comma separator, or return fallback."""
    cleaned = [part.strip() for part in parts if part and part.strip()]
    if not cleaned:
        return fallback
    return ", ".join(cleaned)


def optional_camera_index(value: Optional[int], default: int) -> int:
    """Normalize a camera index value by applying a default when None."""
    return value if value is not None else default
