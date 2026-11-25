"""
Training script for Model A: Physical Presentation Attack Detection (EfficientNet-B0 + LBP).
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from datasets.biometric_dataset import BiometricSpoofDataset, create_subject_disjoint_splits
from models.physical_pad_model import PhysicalPADNet
from evaluation.metrics import compute_biometric_metrics


def set_seed(seed: int = config.RANDOM_SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_pad_model(
    manifest_csv: Path,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-4,
    output_path: Path = config.PAD_MODEL_PATH,
) -> None:
    set_seed()
    if not manifest_csv.exists():
        raise FileNotFoundError(
            f"Physical PAD manifest not found: {manifest_csv}\n"
            "Please download CelebA-Spoof / OULU-NPU / Replay-Attack and generate manifest.csv."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, val_df, _ = create_subject_disjoint_splits(manifest_csv, random_state=config.RANDOM_SEED)

    train_ds = BiometricSpoofDataset(train_df, manifest_csv.parent, task="pad", is_train=True)
    val_ds = BiometricSpoofDataset(val_df, manifest_csv.parent, task="pad", is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    num_pos = max(1, int((train_df["label"] == 1).sum()))
    num_neg = max(1, int((train_df["label"] == 0).sum()))
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32, device=device)

    model = PhysicalPADNet(pretrained_backbone=True).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acer = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for rgb, lbp, labels in train_loader:
            rgb, lbp, labels = rgb.to(device), lbp.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(rgb, lbp)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * rgb.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_ds)

        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for rgb, lbp, labels in val_loader:
                rgb, lbp = rgb.to(device), lbp.to(device)
                probs = torch.sigmoid(model(rgb, lbp)).cpu().numpy()
                val_probs.extend(probs.tolist())
                val_labels.extend(labels.numpy().tolist())

        metrics = compute_biometric_metrics(
            np.array(val_labels), np.array(val_probs), threshold=config.PHYSICAL_SPOOF_THRESHOLD
        )
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] Loss: {train_loss:.4f} | "
            f"Val ACER: {metrics['ACER']:.4f} (APCER: {metrics['APCER']:.4f}, BPCER: {metrics['BPCER']:.4f}) | "
            f"AUC: {metrics['roc_auc']:.4f}"
        )

        if metrics["ACER"] < best_acer:
            best_acer = metrics["ACER"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_metrics": metrics,
                    "architecture": "PhysicalPADNet (EfficientNet-B0 + LBP)",
                },
                output_path,
            )
            print(f"  --> Saved improved checkpoint to {output_path} (ACER={best_acer:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Physical PAD Model (Model A)")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to PAD manifest CSV")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    train_pad_model(args.manifest, args.epochs, args.batch_size, args.lr)
