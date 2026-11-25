"""Image upload validation and decoding."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMAGE_TYPES = {
    ".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", ".webp": "webp",
}


class MediaError(ValueError):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def sniff_image_type(data: bytes) -> str | None:
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def validate_image_bytes(data: bytes, filename: str, max_bytes: int) -> np.ndarray:
    """Return the decoded BGR image or raise MediaError with a specific reason."""
    if not data:
        raise MediaError("empty upload")
    if len(data) > max_bytes:
        raise MediaError(f"image is {len(data) / 1e6:.1f} MB; limit is {max_bytes / 1e6:.0f} MB", 413)
    ext = Path(filename or "").suffix.lower()
    if ext not in IMAGE_TYPES:
        raise MediaError(f"unsupported image extension {ext or '(none)'}; use JPEG, PNG or WebP", 415)
    kind = sniff_image_type(data)
    if kind is None:
        raise MediaError("file content is not a JPEG, PNG or WebP image", 415)
    if kind != IMAGE_TYPES[ext]:
        raise MediaError(f"file extension {ext} does not match its content ({kind})", 415)
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise MediaError("image data could not be decoded", 422)
    return image


def load_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise MediaError(f"stored image unreadable: {path.name}", 500)
    return image


def save_jpeg(image: np.ndarray, path: Path, max_side: int | None = None, quality: int = 88) -> Path:
    if max_side and max(image.shape[:2]) > max_side:
        scale = max_side / max(image.shape[:2])
        image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)),
                           interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise MediaError("could not encode JPEG", 500)
    path.parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(path))
    return path
