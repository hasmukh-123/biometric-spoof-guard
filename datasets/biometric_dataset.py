"""
Subject-disjoint dataset loader for Physical PAD and Digital Deepfake Detection.
Expected CSV manifest columns:
    filepath,label,subject_id,attack_subtype
"""
from pathlib import Path
from typing import Tuple
import cv2
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import torch
from torch.utils.data import Dataset
from preprocessing.transforms import prepare_pad_tensors, prepare_deepfake_tensors


class BiometricSpoofDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        root_dir: Path,
        task: str = "pad",
        is_train: bool = False,
    ) -> None:
        assert task in {"pad", "deepfake"}, "task must be 'pad' or 'deepfake'"
        self.df = df.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.task = task
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = Path(row["filepath"])
        if not img_path.is_absolute():
            img_path = self.root_dir / img_path

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise FileNotFoundError(f"Unable to read dataset image: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if self.task == "pad":
            rgb_t, aux_t = prepare_pad_tensors(rgb, is_train=self.is_train)
        else:
            rgb_t, aux_t = prepare_deepfake_tensors(rgb, is_train=self.is_train)

        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return rgb_t, aux_t, label


def create_subject_disjoint_splits(
    manifest_csv: Path,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train, validation, and test sets strictly grouped by `subject_id`
    to guarantee zero identity or video-clip leakage.
    """
    df = pd.read_csv(manifest_csv)
    required_cols = {"filepath", "label", "subject_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Manifest CSV missing required columns: {missing}")

    if df["subject_id"].nunique() < 3:
        raise ValueError("Need at least 3 distinct subject_ids for train/val/test group splits.")

    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss_test.split(df, df["label"], groups=df["subject_id"]))
    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    relative_val_size = val_size / (1.0 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=random_state)
    train_idx, val_idx = next(
        gss_val.split(train_val_df, train_val_df["label"], groups=train_val_df["subject_id"])
    )
    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    return train_df, val_df, test_df
