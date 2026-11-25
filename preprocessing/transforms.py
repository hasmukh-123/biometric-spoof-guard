"""
Image tensor transforms and forensic cue extractors (pure PIL + NumPy + PyTorch).
"""
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import config


def _resize_gray_float(face_rgb: np.ndarray, size: tuple) -> np.ndarray:
    pil_img = Image.fromarray(face_rgb.astype(np.uint8)).resize(size, Image.Resampling.BILINEAR)
    gray = np.asarray(pil_img.convert("L"), dtype=np.float32)
    return gray


def compute_lbp_map(face_rgb: np.ndarray, size: tuple = config.PAD_INPUT_SIZE) -> np.ndarray:
    """Compute normalized 8-neighbor Local Binary Pattern (LBP) micro-texture map."""
    gray = _resize_gray_float(face_rgb, size)
    padded = np.pad(gray, 1, mode="edge")
    center = padded[1:-1, 1:-1]

    lbp = np.zeros_like(center, dtype=np.uint8)
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, 1), (1, 1), (1, 0),
        (1, -1), (0, -1),
    ]
    for bit, (dy, dx) in enumerate(neighbors):
        shifted = padded[1 + dy : padded.shape[0] - 1 + dy, 1 + dx : padded.shape[1] - 1 + dx]
        lbp |= ((shifted >= center).astype(np.uint8) << bit)

    return lbp.astype(np.float32) / 255.0


def compute_fft_spectrum(face_rgb: np.ndarray, size: tuple = config.DEEPFAKE_INPUT_SIZE) -> np.ndarray:
    """Compute centered 2D FFT log-magnitude spectrum."""
    gray = _resize_gray_float(face_rgb, size) / 255.0
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.log1p(np.abs(f_shift))
    max_val = np.max(magnitude)
    if max_val > 0:
        magnitude = magnitude / max_val
    return magnitude.astype(np.float32)


def get_pad_transform(is_train: bool = False) -> transforms.Compose:
    ops = [transforms.ToPILImage(), transforms.Resize(config.PAD_INPUT_SIZE)]
    if is_train:
        ops.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
        ])
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transforms.Compose(ops)


def get_deepfake_transform(is_train: bool = False) -> transforms.Compose:
    ops = [transforms.ToPILImage(), transforms.Resize(config.DEEPFAKE_INPUT_SIZE)]
    if is_train:
        ops.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
        ])
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return transforms.Compose(ops)


def prepare_pad_tensors(face_rgb: np.ndarray, is_train: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    rgb_tensor = get_pad_transform(is_train=is_train)(face_rgb)
    lbp_map = compute_lbp_map(face_rgb, config.PAD_INPUT_SIZE)
    lbp_tensor = torch.from_numpy(lbp_map).unsqueeze(0)
    return rgb_tensor, lbp_tensor


def prepare_deepfake_tensors(face_rgb: np.ndarray, is_train: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    rgb_tensor = get_deepfake_transform(is_train=is_train)(face_rgb)
    fft_map = compute_fft_spectrum(face_rgb, config.DEEPFAKE_INPUT_SIZE)
    fft_tensor = torch.from_numpy(fft_map).unsqueeze(0)
    return rgb_tensor, fft_tensor