"""Resolve recorded evidence paths after a checkout moves between machines.

This module changes only where validation reads evidence. Original metadata,
hash keys and numerical archives are never rewritten.
"""
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath


def project_path(value, root):
    """Prefer the current checkout for recognized project paths on either OS.

    PurePath roots are also supported for checking Windows/POSIX semantics
    without requiring the other operating system or touching its filesystem.
    An unrelated foreign absolute path has no unambiguous local equivalent.
    """
    root = root if isinstance(root, PurePath) else Path(root)
    text = str(value).replace("\\", "/")
    parts = text.split("/")
    markers = {root.name.casefold(), "2026a-herbal-drying"}
    positions = [i for i, part in enumerate(parts) if part.casefold() in markers]
    if positions:
        # Do this before any existence test: an older checkout can still exist.
        # Missing evidence in this copy must fail, never fall back to that copy.
        return root.joinpath(*parts[positions[-1] + 1:])
    path = type(root)(text)
    if path.is_absolute():
        return path
    if PureWindowsPath(text).is_absolute() or PurePosixPath(text).is_absolute():
        raise ValueError(f"Cannot relocate a foreign absolute path outside the project: {value}")
    return root / path
