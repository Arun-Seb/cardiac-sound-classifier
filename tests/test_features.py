"""
test_features.py
────────────────
Unit tests for feature extraction functions.
Run with: pytest tests/
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.features import extract_handcrafted

SR       = 16000
DURATION = 4
N        = SR * DURATION   # 64000 samples


def make_sine(freq=440.0, sr=SR, duration=DURATION) -> np.ndarray:
    """Generate a sine wave for testing."""
    t = np.linspace(0, duration, sr * duration, endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def make_noise(sr=SR, duration=DURATION) -> np.ndarray:
    """Generate white noise for testing."""
    return np.random.randn(sr * duration).astype(np.float32)


# ── Handcrafted feature tests ────────────────────────────────
class TestHandcraftedFeatures:

    def test_output_shape_sine(self):
        y    = make_sine()
        feat = extract_handcrafted(y)
        assert feat.ndim == 1, "Features should be 1D"
        assert feat.shape[0] == 604, f"Expected 604 features, got {feat.shape[0]}"

    def test_output_shape_noise(self):
        y    = make_noise()
        feat = extract_handcrafted(y)
        assert feat.shape[0] == 604

    def test_output_dtype(self):
        y    = make_sine()
        feat = extract_handcrafted(y)
        assert feat.dtype == np.float32

    def test_no_nan(self):
        y    = make_sine()
        feat = extract_handcrafted(y)
        assert not np.any(np.isnan(feat)), "Features contain NaN"

    def test_no_inf(self):
        y    = make_sine()
        feat = extract_handcrafted(y)
        assert not np.any(np.isinf(feat)), "Features contain Inf"

    def test_silence(self):
        y    = np.zeros(N, dtype=np.float32)
        feat = extract_handcrafted(y)
        assert feat.shape[0] == 604, "Silence should still return 604 features"

    def test_different_frequencies_differ(self):
        """Features from 440Hz and 880Hz should not be identical."""
        f1 = extract_handcrafted(make_sine(freq=440))
        f2 = extract_handcrafted(make_sine(freq=880))
        assert not np.allclose(f1, f2), "Different signals should have different features"

    def test_reproducible(self):
        """Same input should always give same output."""
        y  = make_sine()
        f1 = extract_handcrafted(y)
        f2 = extract_handcrafted(y)
        np.testing.assert_array_equal(f1, f2)

    def test_short_audio_padded(self):
        """Audio shorter than DURATION should still work after padding."""
        y_short = make_sine(duration=1)  # 1 second
        target  = SR * DURATION
        y_padded = np.pad(y_short, (0, target - len(y_short)))
        feat = extract_handcrafted(y_padded)
        assert feat.shape[0] == 604


# ── Dataset tests ────────────────────────────────────────────
class TestLabelBinarisation:

    def test_binary_mapping(self):
        from src.dataset import ABNORMAL_CLASSES
        normal_labels   = {"normal"}
        abnormal_labels = {"murmur", "extrastole", "extrahls", "artifact"}

        for lbl in normal_labels:
            assert lbl not in ABNORMAL_CLASSES

        for lbl in abnormal_labels:
            assert lbl in ABNORMAL_CLASSES

    def test_all_classes_covered(self):
        from src.dataset import ABNORMAL_CLASSES
        all_classes = {"normal", "murmur", "extrastole", "extrahls", "artifact"}
        for cls in all_classes:
            assert cls == "normal" or cls in ABNORMAL_CLASSES, \
                f"Class '{cls}' not mapped"
