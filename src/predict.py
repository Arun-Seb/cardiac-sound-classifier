"""
predict.py
──────────
Single-file inference for trained cardiac classifiers.
"""

import numpy as np
import librosa
from pathlib import Path

SR       = 16000
DURATION = 4


def predict(filepath: str,
            model,
            compare_extractor,
            wav2vec_extractor,
            feature_set: str = "Combined",
            mode: str = "binary",
            threshold: float = 0.5,
            class_names: list = None) -> dict:
    """
    Predict cardiac condition from a .wav file.

    Args:
        filepath          : path to .wav file
        model             : trained sklearn Pipeline
        compare_extractor : ComParEExtractor instance
        wav2vec_extractor : Wav2VecExtractor instance
        feature_set       : 'Handcrafted' | 'ComParE' | 'Wav2Vec 2.0' | 'Combined'
        mode              : 'binary' or 'multiclass'
        threshold         : decision threshold (binary only, default=0.5)
        class_names       : list of class names (multiclass)

    Returns:
        dict with prediction and probabilities
    """
    from src.features import extract_handcrafted

    # Load audio
    y, _ = librosa.load(filepath, sr=SR, duration=DURATION)
    target = SR * DURATION
    y = np.pad(y, (0, max(0, target - len(y))))[:target]
    np.nan_to_num(y, copy=False)

    # Extract features
    hc  = extract_handcrafted(y)
    cmp = compare_extractor.extract(y)
    w2v = wav2vec_extractor.extract(y)

    feat_map = {
        "Handcrafted" : hc,
        "ComParE"     : cmp,
        "Wav2Vec 2.0" : w2v,
        "Combined"    : np.hstack([hc, cmp, w2v]),
    }

    feats = feat_map[feature_set].reshape(1, -1)
    np.nan_to_num(feats, copy=False)

    proba = model.predict_proba(feats)[0]

    # ── Binary mode ──────────────────────────────────────────
    if mode == "binary":
        prob_abnormal = proba[1]
        prob_normal   = proba[0]
        result        = "ABNORMAL ⚠" if prob_abnormal >= threshold else "NORMAL ✅"

        _print_binary(filepath, result, prob_normal, prob_abnormal, threshold)
        return {
            "result"       : result,
            "prob_normal"  : float(prob_normal),
            "prob_abnormal": float(prob_abnormal),
            "threshold"    : threshold,
        }

    # ── Multiclass mode ──────────────────────────────────────
    pred = int(np.argmax(proba))
    label = class_names[pred] if class_names else str(pred)
    _print_multiclass(filepath, label, class_names or list(range(len(proba))), proba)
    return {
        "result"       : label,
        "probabilities": dict(zip(class_names or range(len(proba)),
                                  proba.round(3).tolist())),
    }


def _print_binary(filepath, result, prob_normal, prob_abnormal, threshold):
    print(f"\n{'='*50}")
    print(f"  🩺  Heartbeat Classification (Binary)")
    print(f"{'='*50}")
    print(f"  File      : {Path(filepath).name}")
    print(f"  Threshold : {threshold:.2f}")
    print(f"{'─'*50}")
    print(f"  Result    : {result}")
    print(f"{'─'*50}")
    print(f"  Normal    {'█' * int(prob_normal   * 40)} {prob_normal:.1%}")
    print(f"  Abnormal  {'█' * int(prob_abnormal * 40)} {prob_abnormal:.1%}")
    print(f"{'='*50}")
    if prob_abnormal >= threshold:
        print("  ⚕  Refer for further cardiac evaluation.")
    else:
        print("  ✔  No abnormality detected.")


def _print_multiclass(filepath, label, class_names, proba):
    print(f"\n{'='*50}")
    print(f"  🩺  Heartbeat Classification (Multiclass)")
    print(f"{'='*50}")
    print(f"  File   : {Path(filepath).name}")
    print(f"{'─'*50}")
    print(f"  Result : {label.upper()}")
    print(f"{'─'*50}")
    for cls, p in sorted(zip(class_names, proba), key=lambda x: -x[1]):
        bar = "█" * int(p * 40)
        print(f"  {cls:<14} {bar} {p:.1%}")
    print(f"{'='*50}")
