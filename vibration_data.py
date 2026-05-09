"""Stratified train/val datasets for saved .npy vibration clips (index.csv)."""

from __future__ import annotations

import csv
import math
from typing import Callable, Iterable, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

TransformFn = Callable[[torch.Tensor], torch.Tensor]
NumpyTransformFn = Callable[[np.ndarray], np.ndarray]


class SplitVibrationDataset(Dataset):
    """
    One dataset class supporting two behaviors:
      - split="train": random fixed-length crops from each clip
      - split="val": deterministic contiguous fixed-length segments per clip
    """

    def __init__(
        self,
        records: Iterable[dict],
        split: Literal["train", "val"],
        sample_rate: int = 7400,
        window_seconds: float = 5.0,
        normalize: str | None = "zscore",
        transform_before_normalize: NumpyTransformFn | None = None,
        transform: TransformFn | None = None,
        dtype: np.dtype | type = np.float32,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")

        self.records = list(records)
        self.split = split
        self.window_n = int(round(window_seconds * sample_rate))
        if self.window_n <= 0:
            raise ValueError("window_seconds must be positive")

        self.normalize = normalize
        self.transform_before_normalize = transform_before_normalize
        self.transform = transform
        self.dtype = dtype

        if len(self.records) == 0:
            raise RuntimeError(f"No records found for split '{split}'")

        self.segment_rows: list[tuple[dict, int]] = []
        if self.split == "val":
            for row in self.records:
                clip_len = int(np.load(row["file_path"], mmap_mode="r").shape[0])
                n_segments = max(1, int(math.ceil(clip_len / self.window_n)))
                for seg_idx in range(n_segments):
                    start = seg_idx * self.window_n
                    self.segment_rows.append((row, start))

    def __len__(self) -> int:
        if self.split == "train":
            return len(self.records)
        return len(self.segment_rows)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if self.normalize == "zscore":
            mu = float(np.mean(x))
            sigma = float(np.std(x))
            x = (x - mu) / (sigma + 1e-8)
        return x

    def _crop_or_pad_train(self, x: np.ndarray) -> np.ndarray:
        if len(x) > self.window_n:
            max_start = len(x) - self.window_n
            start = np.random.randint(0, max_start + 1)
            x = x[start : start + self.window_n]
        elif len(x) < self.window_n:
            x = np.pad(x, (0, self.window_n - len(x)), mode="constant")
        return x

    def _segment_or_pad_val(self, x: np.ndarray, start: int) -> np.ndarray:
        x = x[start : start + self.window_n]
        if len(x) < self.window_n:
            x = np.pad(x, (0, self.window_n - len(x)), mode="constant")
        return x

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.split == "train":
            row = self.records[idx]
            x_arr = np.load(row["file_path"]).astype(self.dtype)
            y = int(row["label_id"])
            x_arr = self._crop_or_pad_train(x_arr)
        else:
            row, start = self.segment_rows[idx]
            x_full = np.load(row["file_path"]).astype(self.dtype)
            y = int(row["label_id"])
            x_arr = self._segment_or_pad_val(x_full, start)

        if self.transform_before_normalize is not None:
            x_arr = self.transform_before_normalize(x_arr)

        x_arr = self._normalize(x_arr)
        x_tensor = torch.from_numpy(x_arr).unsqueeze(0)
        y_tensor = torch.tensor(y, dtype=torch.long)

        if self.transform is not None:
            x_tensor = self.transform(x_tensor)

        return x_tensor, y_tensor


def make_train_val_datasets(
    index_csv: str,
    train_ratio: float = 0.8,
    seed: int = 42,
    sample_rate: int = 7400,
    window_seconds: float = 5.0,
    normalize: str | None = "zscore",
    transform_before_normalize: NumpyTransformFn | None = None,
    transform: TransformFn | None = None,
    dtype: np.dtype | type = np.float32,
) -> tuple[SplitVibrationDataset, SplitVibrationDataset]:
    """Stratified split by label, then create train/val dataset objects."""
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be between 0 and 1")

    records: list[dict] = []
    with open(index_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    if len(records) < 2:
        raise RuntimeError("Need at least 2 clips to make a train/val split")

    by_label: dict[int, list[dict]] = {}
    for row in records:
        label = int(row["label_id"])
        by_label.setdefault(label, []).append(row)

    rng = np.random.default_rng(seed)
    train_records: list[dict] = []
    val_records: list[dict] = []

    for label, group in by_label.items():
        if len(group) < 2:
            raise RuntimeError(
                f"Label {label} has only {len(group)} clip(s); need at least 2 for stratified split"
            )

        perm = rng.permutation(len(group))
        group_train_count = int(round(len(group) * train_ratio))
        group_train_count = max(1, min(group_train_count, len(group) - 1))

        train_idx = perm[:group_train_count]
        val_idx = perm[group_train_count:]

        train_records.extend(group[i] for i in train_idx)
        val_records.extend(group[i] for i in val_idx)

    train_records = [train_records[i] for i in rng.permutation(len(train_records))]
    val_records = [val_records[i] for i in rng.permutation(len(val_records))]

    kwargs = dict(
        sample_rate=sample_rate,
        window_seconds=window_seconds,
        normalize=normalize,
        transform_before_normalize=transform_before_normalize,
        transform=transform,
        dtype=dtype,
    )
    train_ds = SplitVibrationDataset(train_records, split="train", **kwargs)
    val_ds = SplitVibrationDataset(val_records, split="val", **kwargs)
    return train_ds, val_ds
