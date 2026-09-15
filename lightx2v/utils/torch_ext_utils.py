"""Helpers for PyTorch JIT extension build cache."""

from __future__ import annotations

import os
import time
from pathlib import Path


def clear_stale_torch_extension_locks(max_age_s: float = 120.0) -> int:
    """Remove orphaned lock files from interrupted torch JIT builds.

    PyTorch's FileBaton uses a ``lock`` file under each extension build dir.
    If a build is killed mid-flight, the lock remains and later imports block
    forever in ``time.sleep()`` waiting for it.
    """
    cache_root = Path.home() / ".cache" / "torch_extensions"
    if not cache_root.is_dir():
        return 0

    now = time.time()
    removed = 0
    for lock_path in cache_root.glob("**/lock"):
        try:
            if not lock_path.is_file():
                continue
            build_dir = lock_path.parent
            age_s = now - lock_path.stat().st_mtime
            has_built_so = any(build_dir.glob("*.so"))
            # Drop stale locks when the build dir already has a .so, or the lock
            # has been sitting untouched for longer than max_age_s.
            if has_built_so or age_s >= max_age_s:
                lock_path.unlink(missing_ok=True)
                removed += 1
            ninja_lock = build_dir / ".ninja_lock"
            if ninja_lock.is_file() and (has_built_so or age_s >= max_age_s):
                ninja_lock.unlink(missing_ok=True)
        except OSError:
            continue
    return removed


def ensure_torch_extension_cache_ready() -> None:
    """Best-effort cleanup before any extension JIT import."""
    clear_stale_torch_extension_locks()
