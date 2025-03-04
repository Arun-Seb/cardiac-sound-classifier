"""
features.py
───────────
Three feature extraction strategies for heartbeat audio:
  1. Handcrafted  — librosa-based acoustic features (604-dim)
  2. ComParE 2016 — openSMILE clinical gold standard (6,373-dim)
  3. Wav2Vec 2.0  — frozen transformer embeddings (1,536-dim)
"""

import numpy as np
import librosa
import torch
import sys
sys.modules.setdefault("torchvision", None)   # prevent torchvision conflict
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import opensmile

SR         = 16000
HOP_LENGTH = 512
N_MFCC     = 40


# ── Handcrafted ──────────────────────────────────────────────
def extract_handcrafted(y: np.ndarray) -> np.ndarray:
    """
    Extract 604-dimensional handcrafted feature vector using librosa.

    Features:
        MFCCs (40) + 1st delta + 2nd delta  → mean/std/Q25/Q75
        Chroma, Spectral Contrast, Tonnetz
        ZCR, RMS, Mel-spectrogram stats
        Spectral Rolloff, Centroid, Bandwidth
    """
    def stats(x):
        return np.hstack([x.mean(axis=-1), x.std(axis=-1),
                          np.percentile(x, 25, axis=-1),
                          np.percentile(x, 75, axis=-1)])

    mfcc      = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    mfcc_d    = librosa.feature.delta(mfcc)
    mfcc_d2   = librosa.feature.delta(mfcc, order=2)
    chroma    = librosa.feature.chroma_stft(y=y, sr=SR, hop_length=HOP_LENGTH)
    contrast  = librosa.feature.spectral_contrast(y=y, sr=SR, hop_length=HOP_LENGTH)
    tonnetz   = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=SR)
    zcr       = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
    rms       = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
    mel       = librosa.power_to_db(
                    librosa.feature.melspectrogram(y=y, sr=SR, hop_length=HOP_LENGTH),
                    ref=np.max)
    rolloff   = librosa.feature.spectral_rolloff(y=y, sr=SR, hop_length=HOP_LENGTH)
    centroid  = librosa.feature.spectral_centroid(y=y, sr=SR, hop_length=HOP_LENGTH)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=SR, hop_length=HOP_LENGTH)

    return np.hstack([
        stats(mfcc), stats(mfcc_d), stats(mfcc_d2),
        stats(chroma), stats(contrast), stats(tonnetz),
        stats(zcr), stats(rms),
        np.array([mel.mean(), mel.std(),
                  np.percentile(mel, 25), np.percentile(mel, 75)]),
        stats(rolloff), stats(centroid), stats(bandwidth),
    ]).astype(np.float32)


# ── ComParE 2016 ─────────────────────────────────────────────
class ComParEExtractor:
    """
    Wrapper around openSMILE's ComParE 2016 feature set.
    6,373 acoustic functionals covering energy, spectral,
    voicing, MFCC, and delta features.

    Usage:
        extractor = ComParEExtractor()
        feats = extractor.extract(y)   # (6373,)
    """
    def __init__(self):
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        self.n_features = len(self.smile.feature_names)

    def extract(self, y: np.ndarray) -> np.ndarray:
        feats = self.smile.process_signal(y, SR)
        arr   = feats.values.flatten().astype(np.float32)
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return arr


# ── Wav2Vec 2.0 ──────────────────────────────────────────────
class Wav2VecExtractor:
    """
    Frozen Wav2Vec 2.0 feature extractor.
    Uses facebook/wav2vec2-base pretrained on 960h LibriSpeech.
    Mean + Std pooling of last hidden states → 1,536-dim embedding.

    Usage:
        extractor = Wav2VecExtractor()
        feats = extractor.extract(y)   # (1536,)
    """
    MODEL_NAME = "facebook/wav2vec2-base"

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.MODEL_NAME)
        self.model     = Wav2Vec2Model.from_pretrained(self.MODEL_NAME).to(self.device)
        self.model.eval()

    def extract(self, y: np.ndarray) -> np.ndarray:
        y = y / (np.abs(y).max() + 1e-8)
        inputs = self.processor(y, sampling_rate=SR,
                                return_tensors="pt", padding=True)
        with torch.no_grad():
            out = self.model(inputs.input_values.to(self.device))
        hidden = out.last_hidden_state.squeeze(0)            # (T, 768)
        emb    = torch.cat([hidden.mean(0), hidden.std(0)]).cpu().numpy()
        return emb.astype(np.float32)


# ── Build full feature matrix ────────────────────────────────
def build_feature_matrix(df, paths: dict,
                         compare_extractor: ComParEExtractor,
                         wav2vec_extractor: Wav2VecExtractor):
    """
    Extract all three feature sets for every sample in df.

    Returns:
        X_hc  : (N, 604)   handcrafted
        X_cmp : (N, 6373)  ComParE
        X_w2v : (N, 1536)  Wav2Vec
        labels: (N,)       string labels
    """
    from src.dataset import load_audio

    X_hc, X_cmp, X_w2v, labels = [], [], [], []
    skipped = 0
    total   = len(df)

    for i, (_, row) in enumerate(df.iterrows(), 1):
        if i % 10 == 0 or i == total:
            pct = i / total * 100
            bar = "█" * int(pct/5) + "░" * (20 - int(pct/5))
            print(f"  [{bar}] {pct:.0f}%  ({i}/{total})", end="\r")

        audio = load_audio(row["fname"], row["dataset"], paths)
        if audio is None:
            skipped += 1
            continue
        try:
            X_hc.append(extract_handcrafted(audio))
            X_cmp.append(compare_extractor.extract(audio))
            X_w2v.append(wav2vec_extractor.extract(audio))
            labels.append(row["label"])
        except Exception as e:
            print(f"\n  Warning: feature error on {row['fname']}: {e}")
            skipped += 1

    print(f"\n✅ Extracted {len(labels)} samples  |  skipped {skipped}")

    X_hc  = np.array(X_hc,  dtype=np.float32)
    X_cmp = np.array(X_cmp, dtype=np.float32)
    X_w2v = np.array(X_w2v, dtype=np.float32)

    for X in [X_hc, X_cmp, X_w2v]:
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    return X_hc, X_cmp, X_w2v, np.array(labels)
