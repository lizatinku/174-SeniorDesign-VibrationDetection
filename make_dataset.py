import os
import glob
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# CONFIG
NPY_DIR = "triple_05_03/triple_05_03/port_3331" # change path according to device
CSV_PATH = "timestamps.csv"
OUT_DIR = "classified_dataset2"

SAMPLE_RATE = 7500


def load_full_vibration_signal(npy_dir):
    npy_files = sorted(glob.glob(os.path.join(npy_dir, "*.npy")))

    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {npy_dir}")

    arrays = []
    for file in npy_files:
        arr = np.load(file)
        arr = np.asarray(arr).squeeze()
        arrays.append(arr)

    return np.concatenate(arrays)


def chop_dataset():
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    full_signal = load_full_vibration_signal(NPY_DIR)
    df = pd.read_csv(CSV_PATH)

    total_samples = len(full_signal)
    recorded_duration_s = total_samples / SAMPLE_RATE

    print(f"Total vibration samples: {total_samples}")
    print(f"Recorded duration: {recorded_duration_s:.2f} seconds")
    print(f"Recorded duration: {recorded_duration_s / 60:.2f} minutes")

    metadata = []
    skipped = []

    valid = 0
    partial = 0
    invalid = 0

    for _, row in df.iterrows():
        class_name = row["class_name"]
        class_label = int(row["class_label"])
        filename = row["filename"]

        start_s = float(row["clip_start_s"])
        end_s = float(row["clip_end_s"])

        if end_s <= recorded_duration_s:
            status = "valid"
            valid += 1
        elif start_s < recorded_duration_s < end_s:
            status = "partial"
            partial += 1
        else:
            status = "invalid"
            invalid += 1

        # ONLY save fully valid clips
        if status != "valid":
            skipped.append(filename)
            continue

        start_idx = int(start_s * SAMPLE_RATE)
        end_idx = int(end_s * SAMPLE_RATE)

        clip = full_signal[start_idx:end_idx]

        if len(clip) == 0:
            skipped.append(filename)
            continue

        class_dir = os.path.join(OUT_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)

        out_filename = filename.replace(".wav", ".npy")
        out_path = os.path.join(class_dir, out_filename)

        np.save(out_path, clip)

        metadata.append({
            "path": out_path,
            "filename": out_filename,
            "class_name": class_name,
            "class_label": class_label,
            "start_s": start_s,
            "end_s": end_s,
            "duration_s": end_s - start_s,
            "num_samples": len(clip)
        })

    metadata_df = pd.DataFrame(metadata)
    metadata_path = os.path.join(OUT_DIR, "metadata.csv")
    metadata_df.to_csv(metadata_path, index=False)

    print(f"\nSaved metadata to: {metadata_path}")
    print("\nFrom actual recording (.npy):")
    print(f"Fully valid clips: {valid}")
    print(f"Partially recorded clips: {partial}")
    print(f"Not recorded at all: {invalid}")
    
    metadata_df = pd.DataFrame(metadata)
    metadata_path = os.path.join(OUT_DIR, "metadata.csv")
    metadata_df.to_csv(metadata_path, index=False)

    print(f"\nSaved metadata to: {metadata_path}")

    if len(metadata_df) > 0:
        print("\nClass counts:")
        print(metadata_df["class_name"].value_counts())

        print("\nClip length stats:")
        print(metadata_df["num_samples"].describe())

        skipped_path = os.path.join(OUT_DIR, "skipped_clips.txt")
        with open(skipped_path, "w") as f:
            for item in skipped:
                f.write(item + "\n")

        print("\nSkipped clip list saved to:", skipped_path)


class VibrationDataset(Dataset):
    def __init__(self, metadata_csv, fixed_length=None):
        self.df = pd.read_csv(metadata_csv)
        self.fixed_length = fixed_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        x = np.load(row["path"]).astype(np.float32)
        y = int(row["class_label"])

        if np.std(x) > 0:
            x = (x - np.mean(x)) / np.std(x)

        if self.fixed_length is not None:
            if len(x) > self.fixed_length:
                x = x[:self.fixed_length]
            else:
                padding = self.fixed_length - len(x)
                x = np.pad(x, (0, padding))

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(y, dtype=torch.long)

        return x, y


if __name__ == "__main__":
    chop_dataset()

    metadata_csv = os.path.join(OUT_DIR, "metadata.csv")
    dataset = VibrationDataset(metadata_csv=metadata_csv, fixed_length=337500)
    
    if len(dataset) > 0:
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        x_batch, y_batch = next(iter(dataloader))

        print("\nBatch X shape:", x_batch.shape)
        print("Batch y shape:", y_batch.shape)
        print("Labels:", y_batch)
    else:
        print("\nNo valid clips found.")