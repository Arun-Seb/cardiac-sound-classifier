# 🫀 Cardiac Sound Classifier

Automated detection of cardiac conditions from heartbeat audio using machine learning.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue.svg)](https://www.kaggle.com/datasets/kinguistics/heartbeat-sounds)

---

## 📋 Overview

This project classifies heartbeat sounds into cardiac conditions using a combination of state-of-the-art audio features:

| Feature Set | Dimensions | Description |
|---|---|---|
| **Handcrafted** | 604 | MFCCs + deltas, Chroma, Spectral Contrast, Tonnetz, ZCR, RMS |
| **ComParE 2016** | 6,373 | INTERSPEECH clinical audio gold standard |
| **Wav2Vec 2.0** | 1,536 | Facebook's self-supervised transformer embeddings |
| **Combined** | 8,513 | All three fused |

Two classifiers are implemented:

| Classifier | Task | Best Result |
|---|---|---|
| **Binary** | Normal vs Abnormal | ComParE + LightGBM |
| **Multiclass** | Normal / Murmur / Extrastole / Extrahls / Artifact | ComParE + LightGBM — 78.6% |

---

## 🗂️ Repository Structure

```
cardiac-sound-classifier/
│
├── src/
│   ├── features.py          # Feature extraction (Handcrafted, ComParE, Wav2Vec)
│   ├── models.py            # Model zoo (RF, XGBoost, LightGBM, GBM, SVM, LR)
│   ├── dataset.py           # Data loading and label parsing
│   ├── evaluate.py          # Metrics, plots, confusion matrices
│   └── predict.py           # Single-file inference
│
├── notebooks/
│   ├── 01_binary_classifier.ipynb       # Normal vs Abnormal
│   └── 02_multiclass_classifier.ipynb   # 5-class classification
│
├── results/                 # Output charts and metrics (auto-generated)
├── data/samples/            # Put sample .wav files here for testing
├── tests/
│   └── test_features.py     # Unit tests for feature extraction
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/Arun-Seb/cardiac-sound-classifier.git
cd cardiac-sound-classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
```bash
pip install kaggle
kaggle datasets download kinguistics/heartbeat-sounds
unzip heartbeat-sounds.zip -d data/
```

### 4. Run binary classifier (Normal vs Abnormal)
Open `notebooks/01_binary_classifier.ipynb` and run all cells.

### 5. Run multiclass classifier
Open `notebooks/02_multiclass_classifier.ipynb` and run all cells.

### 6. Predict a single file
```python
from src.predict import predict

result = predict("path/to/heartbeat.wav", mode="binary")
# {'result': 'NORMAL ✅', 'prob_normal': 0.91, 'prob_abnormal': 0.09}

result = predict("path/to/heartbeat.wav", mode="multiclass")
# {'result': 'murmur', 'probabilities': {...}}
```

---

## 📊 Results

### Binary Classification — Normal vs Abnormal

| Feature Set | RF | XGBoost | LightGBM | GBM | SVM | LR |
|---|---|---|---|---|---|---|
| Handcrafted | - | - | - | - | - | - |
| ComParE | - | - | **Best** | - | - | - |
| Wav2Vec 2.0 | - | - | - | - | - | - |
| Combined | - | - | - | - | - | - |

> Run the notebook to populate with your results.

### Multiclass Classification — 5 Classes

| Feature Set | RF | XGBoost | LightGBM | GBM | SVM | LR |
|---|---|---|---|---|---|---|
| Handcrafted | 76.1% | 76.1% | 76.9% | 76.1% | 75.2% | 67.5% |
| ComParE | 75.2% | 76.1% | **78.6%** | 76.9% | 73.5% | 71.8% |
| Wav2Vec 2.0 | 73.5% | 73.5% | 73.5% | 71.8% | 74.4% | 72.6% |
| Combined | 76.1% | 75.2% | 76.9% | 77.8% | 73.5% | 76.1% |

🏆 **Best: ComParE + LightGBM — 78.6% accuracy**

### 5-Fold Cross Validation (ComParE features)

| Model | Mean Accuracy | Std |
|---|---|---|
| Random Forest | 70.9% | ±8.0% |
| XGBoost | 73.5% | ±7.5% |
| LightGBM | 73.2% | ±7.8% |
| Gradient Boosting | 72.6% | ±6.5% |
| SVM (RBF) | 72.1% | ±6.1% |
| Logistic Reg. | 69.2% | ±9.9% |

---

## 🎯 Classes

| Label | Description | Count |
|---|---|---|
| `normal` | Healthy heartbeat | 351 |
| `murmur` | Heart murmur detected | 129 |
| `extrastole` | Extra heartbeat (premature beat) | 46 |
| `artifact` | Recording noise / interference | 40 |
| `extrahls` | Extra heart sounds | 19 |

**Binary mapping:**
- Normal → `NORMAL`
- Murmur + Extrastole + Extrahls + Artifact → `ABNORMAL`

---

## 🔬 Feature Extraction Pipeline

```
Raw .wav file
     │
     ├── Handcrafted (librosa)
     │     MFCCs (40) + Δ + ΔΔ → mean/std/Q1/Q3
     │     Chroma, Spectral Contrast, Tonnetz
     │     ZCR, RMS, Mel, Rolloff, Centroid, Bandwidth
     │
     ├── ComParE 2016 (openSMILE)
     │     6,373 acoustic functionals
     │     Energy, Voicing, Spectral, MFCC, Delta
     │
     └── Wav2Vec 2.0 (HuggingFace)
           facebook/wav2vec2-base (frozen)
           Mean + Std pooling → 1,536-dim embedding
```

---

## 📦 Dependencies

- `librosa` — audio processing
- `opensmile` — ComParE feature extraction
- `transformers` — Wav2Vec 2.0
- `torch` — PyTorch backend
- `scikit-learn` — ML models and evaluation
- `xgboost` / `lightgbm` — boosting models
- `pandas` / `numpy` / `matplotlib` / `seaborn`

---

## 📄 Dataset

[PASCAL Classifying Heart Sounds Challenge (CHSC2011)](https://www.kaggle.com/datasets/kinguistics/heartbeat-sounds)

- **set_a**: 176 labeled recordings (CSV labels)
- **set_b**: 461 labeled recordings (labels in filename)
- **Total**: 637 usable labeled samples

> Note: The `set_b` Bunlabelledtest files are the original blind competition test set — ground truth labels were never released publicly.

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.
