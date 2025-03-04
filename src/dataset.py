"""
dataset.py
──────────
Data loading and label parsing for the PASCAL Heartbeat Sounds dataset.
https://www.kaggle.com/datasets/kinguistics/heartbeat-sounds
"""

import os
import numpy as np
import pandas as pd
import librosa
from pathlib import Path


# Classes that map to ABNORMAL in binary mode
ABNORMAL_CLASSES = {"murmur", "extrastole", "extrahls", "artifact"}

# Audio constants
SR       = 16000   # 16 kHz — required by Wav2Vec 2.0
DURATION = 4       # seconds — all clips padded/trimmed to this


def find_dataset(base: Path) -> dict:
    """
    Auto-detect set_a/, set_b/, set_a.csv, set_b.csv
    anywhere up to 4 levels deep from base directory.
    """
    paths = {}
    for root, dirs, files in os.walk(base):
        root = Path(root)
        if len(root.relative_to(base).parts) > 4:
            continue
        for name in files:
            if name == "set_a.csv" and "set_a_csv" not in paths:
                paths["set_a_csv"] = root / name
            if name == "set_b.csv" and "set_b_csv" not in paths:
                paths["set_b_csv"] = root / name
        for d in dirs:
            if d == "set_a" and "set_a_dir" not in paths:
                paths["set_a_dir"] = root / d
            if d == "set_b" and "set_b_dir" not in paths:
                paths["set_b_dir"] = root / d

    missing = [k for k in ("set_a_csv", "set_b_csv", "set_a_dir", "set_b_dir")
               if k not in paths]
    if missing:
        raise FileNotFoundError(f"Could not find dataset components: {missing}")

    return paths


def load_metadata(paths: dict, mode: str = "multiclass") -> pd.DataFrame:
    """
    Load and merge set_a (CSV labels) and set_b (filename labels).

    Args:
        paths  : dict from find_dataset()
        mode   : 'multiclass' or 'binary'

    Returns:
        DataFrame with columns: fname, label, original_label, dataset
    """
    # set_a — labels from CSV
    df_a = pd.read_csv(paths["set_a_csv"])
    df_a["dataset"] = "A"
    df_a.columns    = [c.lower().strip() for c in df_a.columns]

    # set_b — labels embedded in filename (normal__, murmur__, etc.)
    # The Btraining_ files were never publicly released (PASCAL CHSC2011)
    valid = {"normal", "murmur", "extrastole", "artifact", "extrahls"}
    rows  = []
    for f in paths["set_b_dir"].iterdir():
        if f.suffix != ".wav":
            continue
        prefix = f.name.split("_")[0].lower()
        if prefix in valid:
            rows.append({"fname": f.name, "label": prefix, "dataset": "B"})

    df_b = pd.DataFrame(rows)
    df   = pd.concat([df_a, df_b], ignore_index=True)
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df = df[~df["label"].isin(["nan", "unlabeled", ""])]
    df["original_label"] = df["label"].copy()

    if mode == "binary":
        df["label"] = df["label"].apply(
            lambda x: "abnormal" if x in ABNORMAL_CLASSES else "normal")

    return df


def load_audio(fname: str, dataset: str, paths: dict) -> np.ndarray | None:
    """
    Load a .wav file, resample to SR, pad/trim to DURATION seconds.

    Returns numpy array of shape (SR*DURATION,) or None if file missing.
    """
    folder = paths["set_a_dir"] if dataset == "A" else paths["set_b_dir"]
    path   = folder / Path(fname).name

    if not path.exists():
        return None
    try:
        y, _ = librosa.load(path, sr=SR, duration=DURATION)
        target = SR * DURATION
        y = np.pad(y, (0, max(0, target - len(y))))[:target]
        return y
    except Exception as e:
        print(f"  Warning: could not load {path.name}: {e}")
        return None
