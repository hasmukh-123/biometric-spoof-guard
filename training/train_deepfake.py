"""
Training script for Model B: Digital Deepfake Detection (Xception + 2D FFT Spectrum).
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from datasets.biometric_dataset import BiometricSpoofDataset, create_subject_disjoint_splits
from models.digital_deepfake_model import DigitalDeepfakeNet
from evaluation.metrics import compute_biometric_metrics


def set_seed(seed: int = config.RANDOM_SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_deepfake_model(
    manifest_csv: Path,
    epochs: int = 12,
    batch_size: int = 24,
    lr: float = 8e-5,
    output_path: Path = config.DEEPFAKE_MODEL_PATH,
) -> None:
    set_seed()
    if not manifest_csv.exists():
        raise FileNotFoundError(
            f"Deepfake manifest not found: {manifest_csv}\n"
            "Please download FaceForensics++ / Celeb-DF / DFDC and generate manifest.csv."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, val_df, _ = create_subject_disjoint_splits(manifest_csv, random_state=config.RANDOM_SEED)

    train_ds = BiometricSpoofDataset(train_df, manifest_csv.parent, task="deepfake", is_train=True)
    val_ds = BiometricSpoofDataset(val_df, manifest_csv.parent, task="deepfake", is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    num_pos = max(1, int((train_df["label"] == 1).sum()))
    num_neg = max(1, int((train_df["label"] == 0).sum()))
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32, device=device)

    model = DigitalDeepfakeNet(pretrained_backbone=True).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_auc = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for rgb, fft_map, labels in train_loader:
            rgb, fft_map, labels = rgb.to(device), fft_map.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(rgb, fft_map)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * rgb.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_ds)

        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for rgb, fft_map, labels in val_loader:
                rgb, fft_map = rgb.to(device), fft_map.to(device)
                probs = torch.sigmoid(model(rgb, fft_map)).cpu().numpy()
                val_probs.extend(probs.tolist())
                val_labels.extend(labels.numpy().tolist())

        metrics = compute_biometric_metrics(
            np.array(val_labels), np.array(val_probs), threshold=config.DIGITAL_DEEPFAKE_THRESHOLD
        )
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] Loss: {train_loss:.4f} | "
            f"Val AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1_score']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}"
        )

        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_metrics": metrics,
                    "architecture": "DigitalDeepfakeNet (Xception + 2D FFT)",
                },
                output_path,
            )
            print(f"  --> Saved improved checkpoint to {output_path} (ROC-AUC={best_auc:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Digital Deepfake Model (Model B)")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to Deepfake manifest CSV")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=8e-5)
    args = parser.parse_args()
    train_deepfake_model(args.manifest, args.epochs, args.batch_size, args.lr)
