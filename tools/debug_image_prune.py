"""Keep at most N newest debug PNGs under a directory tree."""

from __future__ import annotations

from pathlib import Path

DEFAULT_MAX_DEBUG_IMAGES = 20


def prune_debug_images(root: Path, max_images: int = DEFAULT_MAX_DEBUG_IMAGES) -> int:
    """Delete oldest PNGs until at most max_images remain. Returns number deleted."""
    if max_images <= 0:
        return 0
    paths = [p for p in root.rglob("*.png") if p.is_file()]
    if len(paths) <= max_images:
        return 0
    paths.sort(key=lambda p: p.stat().st_mtime)
    deleted = 0
    while len(paths) > max_images:
        paths.pop(0).unlink(missing_ok=True)
        deleted += 1
    return deleted
