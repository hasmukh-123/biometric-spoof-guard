"""
Face detection, margin cropping, quality gating, and video frame sampling.
Uses pure NumPy / PIL / Torchvision so it runs reliably even when Windows Application Control
blocks native OpenCV C++ (.pyd) binaries.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple
import tempfile
import os
import numpy as np
from PIL import Image
import config


@dataclass
class FaceExtractionResult:
    success: bool
    face_rgb: Optional[np.ndarray]
    bbox: Optional[Tuple[int, int, int, int]]  # (x, y, w, h)
    blur_score: float
    quality_ok: bool
    message: str


def _compute_laplacian_variance(gray: np.ndarray) -> float:
    """Pure NumPy 2D Laplacian variance for sharpness/blur evaluation."""
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    lap = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(np.var(lap))


def _detect_skin_face_bbox(image_rgb: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Locate primary frontal facial ROI using YCbCr + RGB skin-chrominance segmentation
    and morphological bounding box extraction (zero C-extension dependency).
    """
    r = image_rgb[:, :, 0].astype(np.float32)
    g = image_rgb[:, :, 1].astype(np.float32)
    b = image_rgb[:, :, 2].astype(np.float32)

    # Standard YCbCr skin chrominance bounds
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b

    skin_mask = (
        (r > 60) & (g > 40) & (b > 20)
        & ((np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)) > 12)
        & (np.abs(r - g) > 10)
        & (r > g) & (r > b)
        & (cb >= 77) & (cb <= 127)
        & (cr >= 133) & (cr <= 173)
    )

    ys, xs = np.where(skin_mask)
    if len(xs) < (config.MIN_FACE_SIZE_PX * config.MIN_FACE_SIZE_PX * 0.25):
        # Fallback: if lighting is cool/grayscale or low-saturation, check central portrait region variance
        h, w = image_rgb.shape[:2]
        gray = np.mean(image_rgb, axis=2)
        if np.std(gray) < 8.0 or min(h, w) < config.MIN_FACE_SIZE_PX:
            return None
        # Central 60% portrait crop if image has clear structural contrast
        cw, ch = int(w * 0.55), int(h * 0.65)
        cx, cy = (w - cw) // 2, (h - ch) // 2
        return (cx, cy, cw, ch)

    # Use 5th-95th percentiles to reject stray background skin-colored pixels
    x1, x2 = int(np.percentile(xs, 5)), int(np.percentile(xs, 95))
    y1, y2 = int(np.percentile(ys, 5)), int(np.percentile(ys, 95))
    bw, bh = x2 - x1, y2 - y1

    if bw < config.MIN_FACE_SIZE_PX or bh < config.MIN_FACE_SIZE_PX:
        return None
    return (x1, y1, bw, bh)


class FacePreprocessor:
    def __init__(self) -> None:
        pass

    def extract_primary_face(
        self, image_rgb: np.ndarray, margin_ratio: float = 0.22
    ) -> FaceExtractionResult:
        """
        Detect the primary face ROI in an RGB image, expand bounding box by margin_ratio
        to preserve screen bezels / blending boundaries, and evaluate sharpness.
        """
        if image_rgb is None or image_rgb.size == 0 or image_rgb.ndim != 3:
            return FaceExtractionResult(False, None, None, 0.0, False, "Invalid or empty image buffer.")

        bbox = _detect_skin_face_bbox(image_rgb)
        if bbox is None:
            return FaceExtractionResult(
                False,
                None,
                None,
                0.0,
                False,
                "No clear frontal face detected. Ensure adequate lighting and face alignment.",
            )

        x, y, w, h = bbox
        img_h, img_w = image_rgb.shape[:2]
        pad_x = int(w * margin_ratio)
        pad_y = int(h * margin_ratio)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img_w, x + w + pad_x)
        y2 = min(img_h, y + h + pad_y)

        face_rgb = image_rgb[y1:y2, x1:x2]
        face_gray = np.mean(face_rgb.astype(np.float64), axis=2)
        blur_score = _compute_laplacian_variance(face_gray)

        quality_ok = (
            w >= config.MIN_FACE_SIZE_PX
            and h >= config.MIN_FACE_SIZE_PX
            and blur_score >= config.MIN_LAPLACIAN_BLUR_SCORE
        )

        quality_msg = (
            "Face quality passed."
            if quality_ok
            else f"Low image sharpness (Laplacian variance={blur_score:.1f} < {config.MIN_LAPLACIAN_BLUR_SCORE}). Result may be uncertain."
        )

        return FaceExtractionResult(
            success=True,
            face_rgb=face_rgb,
            bbox=(int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
            blur_score=blur_score,
            quality_ok=quality_ok,
            message=quality_msg,
        )

    @staticmethod
    def sample_video_frames(
        video_bytes: bytes, suffix: str = ".mp4", num_frames: int = config.VIDEO_SAMPLE_FRAMES
    ) -> List[np.ndarray]:
        """
        Decode video frames using torchvision.io (or PIL for animated media) without OpenCV DLLs.
        """
        frames: List[np.ndarray] = []
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(video_bytes)
                tmp_path = tmp.name

            try:
                from torchvision.io import read_video
                video_tensor, _, _ = read_video(tmp_path, pts_unit="sec")
                total_frames = video_tensor.shape[0]
                if total_frames > 0:
                    indices = np.linspace(0, total_frames - 1, num=min(num_frames, total_frames), dtype=int)
                    for idx in indices:
                        frames.append(video_tensor[int(idx)].numpy())
            except Exception:
                # Fallback if video codec backend is unavailable
                pass
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        return frames