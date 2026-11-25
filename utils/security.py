"""
Security and privacy controls:
- Validates file extensions, magic bytes, and size limits before decoding.
- Computes SHA-256 digests so biometric images never need to be stored or logged.
"""
import hashlib
from pathlib import Path
from typing import Tuple
import config


def compute_sha256(data: bytes) -> str:
    """Return a non-reversible SHA-256 hash of the input bytes for audit tracking."""
    return hashlib.sha256(data).hexdigest()


def validate_uploaded_file(
    filename: str, file_bytes: bytes, expected_kind: str
) -> Tuple[bool, str]:
    """
    Validate file extension, size limits, and basic magic-byte signatures.
    expected_kind: 'image' or 'video'
    """
    if not filename or not file_bytes:
        return False, "Empty file uploaded."

    ext = Path(filename).suffix.lower()
    size_mb = len(file_bytes) / (1024 * 1024)

    if expected_kind == "image":
        if ext not in config.ALLOWED_IMAGE_EXTENSIONS:
            return False, f"Unsupported image format '{ext}'. Allowed: {sorted(config.ALLOWED_IMAGE_EXTENSIONS)}"
        if size_mb > config.MAX_IMAGE_SIZE_MB:
            return False, f"Image exceeds {config.MAX_IMAGE_SIZE_MB} MB limit ({size_mb:.2f} MB)."
        # Basic magic-byte check (JPEG, PNG, WEBP)
        is_jpeg = file_bytes[:3] == b"\xff\xd8\xff"
        is_png = file_bytes[:8] == b"\x89PNG\r\n\x1a\n"
        is_webp = file_bytes[:4] == b"RIFF" and file_bytes[8:12] == b"WEBP"
        if not (is_jpeg or is_png or is_webp):
            return False, "File header signature does not match a valid JPEG, PNG, or WEBP image."
        return True, "Valid image."

    if expected_kind == "video":
        if ext not in config.ALLOWED_VIDEO_EXTENSIONS:
            return False, f"Unsupported video format '{ext}'. Allowed: {sorted(config.ALLOWED_VIDEO_EXTENSIONS)}"
        if size_mb > config.MAX_VIDEO_SIZE_MB:
            return False, f"Video exceeds {config.MAX_VIDEO_SIZE_MB} MB limit ({size_mb:.2f} MB)."
        return True, "Valid video."

    return False, "Invalid expected media type."
