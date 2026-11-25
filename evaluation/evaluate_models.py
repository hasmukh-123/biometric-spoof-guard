"""
Standalone test-set evaluation script for Model A (Physical PAD) and Model B (Digital Deepfake).
Generates JSON metrics reports and PNG confusion matrices in evaluation/reports/.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

import config
from datasets.biometric_dataset import BiometricSpoofDataset, create_subject_disjoint_splits
from models.physical_pad_model import PhysicalPADNet
from models.digital_deepfake_model import DigitalDeepfakeNet
from evaluation.metrics import compute_biometric_metrics


def evaluate_checkpoint(
    task: str, manifest_csv: Path, weights_path: Path, batch_size: int = 32
) -> dict:
    if not manifest_csv.exists():
        raise FileNotFoundError(
            f"Dataset manifest not found at {manifest_csv}. "
            "Prepare the CSV manifest per README.md before running evaluation."
        )
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Trained weights not found at {weights_path}. "
            f"Run training/train_{task}.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_df = create_subject_disjoint_splits(manifest_csv, random_state=config.RANDOM_SEED)
    test_ds = BiometricSpoofDataset(test_df, manifest_csv.parent, task=task, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    if task == "pad":
        model = PhysicalPADNet(pretrained_backbone=False).to(device)
        threshold = config.PHYSICAL_SPOOF_THRESHOLD
    else:
        model = DigitalDeepfakeNet(pretrained_backbone=False).to(device)
        threshold = config.DIGITAL_DEEPFAKE_THRESHOLD

    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for rgb, aux, labels in test_loader:
            rgb, aux = rgb.to(device), aux.to(device)
            logits = model(rgb, aux)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())

    metrics = compute_biometric_metrics(np.array(all_labels), np.array(all_probs), threshold=threshold)
    metrics["task"] = task
    metrics["test_samples"] = len(test_df)
    metrics["test_subjects"] = int(test_df["subject_id"].nunique())

    report_json_path = config.REPORTS_DIR / f"{task}_evaluation_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    cm_dict = metrics["confusion_matrix"]
    cm_arr = np.array([
        [cm_dict["TN_bona_fide_accepted"], cm_dict["FP_bona_fide_rejected"]],
        [cm_dict["FN_attack_accepted"], cm_dict["TP_attack_rejected"]],
    ])
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred: Genuine", "Pred: Attack"],
        yticklabels=["Actual: Genuine", "Actual: Attack"],
    )
    plt.title(f"{task.upper()} Confusion Matrix (ACER={metrics['ACER']:.4f})")
    plt.tight_layout()
    plt.savefig(config.REPORTS_DIR / f"{task}_confusion_matrix.png", dpi=150)
    plt.close()

    print(f"[INFO] Saved evaluation report to {report_json_path}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Biometric Spoofing Detection Models")
    parser.add_argument("--task", choices=["pad", "deepfake"], required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=None)
    args = parser.parse_args()

    default_weights = config.PAD_MODEL_PATH if args.task == "pad" else config.DEEPFAKE_MODEL_PATH
    evaluate_checkpoint(args.task, args.manifest, args.weights or default_weights)
