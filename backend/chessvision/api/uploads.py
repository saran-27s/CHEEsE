from __future__ import annotations

import hashlib
from pathlib import Path

from fastapi import UploadFile

from ..media.images import MediaError

CHUNK = 1 << 20


async def read_limited(upload: UploadFile, max_bytes: int) -> bytes:
    buf = bytearray()
    while True:
        chunk = await upload.read(CHUNK)
        if not chunk:
            break
        buf.extend(chunk)
        if len(buf) > max_bytes:
            raise MediaError(f"upload exceeds the {max_bytes / 1e6:.0f} MB limit", 413)
    return bytes(buf)


async def stream_to_file(upload: UploadFile, dest: Path, max_bytes: int) -> tuple[int, str]:
    """Write an upload to disk chunk by chunk; delete it and raise if it exceeds the limit."""
    size = 0
    digest = hashlib.sha256()
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with dest.open("wb") as f:
            while True:
                chunk = await upload.read(CHUNK)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise MediaError(f"upload exceeds the {max_bytes / 1e6:.0f} MB limit", 413)
                digest.update(chunk)
                f.write(chunk)
        if size == 0:
            raise MediaError("empty upload")
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    return size, digest.hexdigest()
