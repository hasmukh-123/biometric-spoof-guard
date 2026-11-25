"""
Central configuration for the Biometric Spoofing Detection System.
Enforces security constraints, model hyperparameters, and decision thresholds.
"""
from pathlib import Path

# Base Directories
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models" / "weights"
DATASETS_DIR = BASE_DIR / "datasets" / "data"
REPORTS_DIR = BASE_DIR / "evaluation" / "reports"
DB_PATH = BASE_DIR / "audit_logs.db"

# Ensure critical directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Model Checkpoint Paths
PAD_MODEL_PATH = MODELS_DIR / "physical_pad_efficientnet_b0.pth"
DEEPFAKE_MODEL_PATH = MODELS_DIR / "digital_deepfake_xception.pth"

# Security & Input Validation Constraints
MAX_IMAGE_SIZE_MB = 10
MAX_VIDEO_SIZE_MB = 50
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
VIDEO_SAMPLE_FRAMES = 12  # Uniformly sampled frames per video

# Face Quality & Preprocessing Thresholds
MIN_FACE_SIZE_PX = 64
MIN_LAPLACIAN_BLUR_SCORE = 35.0  # Below this = overly blurry / uncertain input
PAD_INPUT_SIZE = (224, 224)
DEEPFAKE_INPUT_SIZE = (299, 299)

# Decision Engine Thresholds
PHYSICAL_SPOOF_THRESHOLD = 0.55
DIGITAL_DEEPFAKE_THRESHOLD = 0.55
UNCERTAIN_MARGIN = 0.12  # Band around threshold flagged as uncertain

# Reproducibility
RANDOM_SEED = 42
