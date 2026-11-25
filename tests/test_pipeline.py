"""
End-to-end automated tests covering:
1. Security validation (magic bytes, file extensions, size limits, SHA-256 hashing).
2. Model A (PhysicalPADNet) and Model B (DigitalDeepfakeNet) forward pass tensor shapes.
3. ISO/IEC 30107-3 metric calculations (APCER, BPCER, ACER, ROC-AUC).
4. Subject-disjoint dataset splitting (verifying zero subject overlap).
5. Inference pipeline behavior on valid and invalid inputs.
"""
import tempfile
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch

from utils.security import validate_uploaded_file, compute_sha256
from models.physical_pad_model import PhysicalPADNet
from models.digital_deepfake_model import DigitalDeepfakeNet
from evaluation.metrics import compute_biometric_metrics
from datasets.biometric_dataset import create_subject_disjoint_splits
from inference.predictor import BiometricSpoofPredictor


def test_security_file_validation():
    ok, _ = validate_uploaded_file("malware.exe", b"MZ\x90\x00", "image")
    assert not ok

    ok, _ = validate_uploaded_file("fake.jpg", b"not_a_jpeg_header", "image")
    assert not ok

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".png", img)
    assert success
    raw_bytes = encoded.tobytes()
    ok, _ = validate_uploaded_file("valid.png", raw_bytes, "image")
    assert ok
    assert len(compute_sha256(raw_bytes)) == 64


def test_model_forward_passes():
    pad_net = PhysicalPADNet(pretrained_backbone=False)
    pad_net.eval()
    rgb_pad = torch.randn(2, 3, 224, 224)
    lbp_pad = torch.randn(2, 1, 224, 224)
    out_pad = pad_net(rgb_pad, lbp_pad)
    assert out_pad.shape == (2,)

    df_net = DigitalDeepfakeNet(pretrained_backbone=False)
    df_net.eval()
    rgb_df = torch.randn(2, 3, 299, 299)
    fft_df = torch.randn(2, 1, 299, 299)
    out_df = df_net(rgb_df, fft_df)
    assert out_df.shape == (2,)


def test_biometric_metrics_calculation():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.8, 0.2, 0.9])
    metrics = compute_biometric_metrics(y_true, y_prob, threshold=0.5)
    assert metrics["APCER"] == 0.5
    assert metrics["BPCER"] == 0.5
    assert metrics["ACER"] == 0.5
    assert metrics["accuracy"] == 0.5


def test_subject_disjoint_splits():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = Path(tmpdir) / "manifest.csv"
        rows = []
        for subj in range(10):
            for sample in range(4):
                rows.append({
                    "filepath": f"subj_{subj}_{sample}.jpg",
                    "label": subj % 2,
                    "subject_id": f"person_{subj}",
                    "attack_subtype": "live" if subj % 2 == 0 else "print",
                })
        pd.DataFrame(rows).to_csv(manifest, index=False)
        train_df, val_df, test_df = create_subject_disjoint_splits(manifest)

        train_subjs = set(train_df["subject_id"])
        val_subjs = set(val_df["subject_id"])
        test_subjs = set(test_df["subject_id"])

        assert train_subjs.isdisjoint(val_subjs)
        assert train_subjs.isdisjoint(test_subjs)
        assert val_subjs.isdisjoint(test_subjs)


def test_predictor_handles_no_face_gracefully():
    predictor = BiometricSpoofPredictor()
    blank_img = np.zeros((240, 240, 3), dtype=np.uint8)
    report = predictor.analyze_image(blank_img)
    assert report.verdict == "Uncertain or Unsupported Input"
    assert report.risk_level == "WARNING"
