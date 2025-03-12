# ============================================================
#  01_binary_classifier.ipynb
#  BINARY CARDIAC CLASSIFIER — Normal vs Abnormal
#  Commit 1: Initial binary classifier implementation
# ============================================================

# ── Cell 1: Install & imports ────────────────────────────────
import sys, subprocess
subprocess.run([sys.executable, "-m", "pip", "install",
                "librosa", "scikit-learn", "pandas", "matplotlib",
                "seaborn", "numpy", "soundfile", "torch",
                "transformers==4.35.2", "opensmile",
                "xgboost", "lightgbm", "-q"])

import sys
sys.modules.setdefault("torchvision", None)

import os, numpy as np, pandas as pd
import librosa, torch, opensmile
import matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             classification_report)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import warnings; warnings.filterwarnings("ignore")
print("✅ Imports done")

# ── Cell 2: Config ───────────────────────────────────────────
DATASET_DIR     = Path(r"D:\Heart Beat Sound")
SR, DURATION    = 16000, 4
N_MFCC          = 40
HOP_LENGTH      = 512
ABNORMAL        = {"murmur", "extrastole", "extrahls", "artifact"}
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# ── Cell 3: Find dataset ─────────────────────────────────────
def find_dataset(base):
    paths = {}
    for root, dirs, files in os.walk(base):
        root = Path(root)
        if len(root.relative_to(base).parts) > 4: continue
        for name in files:
            if name == "set_a.csv" and "set_a_csv" not in paths: paths["set_a_csv"] = root/name
            if name == "set_b.csv" and "set_b_csv" not in paths: paths["set_b_csv"] = root/name
        for d in dirs:
            if d == "set_a" and "set_a_dir" not in paths: paths["set_a_dir"] = root/d
            if d == "set_b" and "set_b_dir" not in paths: paths["set_b_dir"] = root/d
    return paths

PATHS = find_dataset(DATASET_DIR)
print("✅ Dataset:", {k: str(v) for k,v in PATHS.items()})

# ── Cell 4: Load metadata with binary labels ─────────────────
def load_metadata():
    df_a = pd.read_csv(PATHS["set_a_csv"])
    df_a["dataset"] = "A"
    df_a.columns = [c.lower().strip() for c in df_a.columns]

    rows = []
    for f in PATHS["set_b_dir"].iterdir():
        if f.suffix != ".wav": continue
        prefix = f.name.split("_")[0].lower()
        if prefix in {"normal","murmur","extrastole","artifact","extrahls"}:
            rows.append({"fname": f.name, "label": prefix, "dataset": "B"})

    df = pd.concat([df_a, pd.DataFrame(rows)], ignore_index=True)
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df = df[~df["label"].isin(["nan","unlabeled",""])]
    df["original_label"] = df["label"]
    df["label"] = df["label"].apply(lambda x: "abnormal" if x in ABNORMAL else "normal")

    print("📊 Binary distribution:")
    print(df["label"].value_counts())
    return df

df = load_metadata()

# ── Cell 5: Feature extraction functions ─────────────────────
def load_audio(fname, dataset):
    folder = PATHS["set_a_dir"] if dataset=="A" else PATHS["set_b_dir"]
    path = folder / Path(fname).name
    if not path.exists(): return None
    y, _ = librosa.load(path, sr=SR, duration=DURATION)
    target = SR * DURATION
    return np.pad(y, (0, max(0, target-len(y))))[:target]

def extract_handcrafted(y):
    def stats(x): return np.hstack([x.mean(axis=-1), x.std(axis=-1),
                                     np.percentile(x,25,axis=-1), np.percentile(x,75,axis=-1)])
    mfcc     = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    mfcc_d   = librosa.feature.delta(mfcc)
    mfcc_d2  = librosa.feature.delta(mfcc, order=2)
    chroma   = librosa.feature.chroma_stft(y=y, sr=SR, hop_length=HOP_LENGTH)
    contrast = librosa.feature.spectral_contrast(y=y, sr=SR, hop_length=HOP_LENGTH)
    tonnetz  = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=SR)
    zcr      = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
    rms      = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
    mel      = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=SR, hop_length=HOP_LENGTH), ref=np.max)
    rolloff  = librosa.feature.spectral_rolloff(y=y, sr=SR, hop_length=HOP_LENGTH)
    centroid = librosa.feature.spectral_centroid(y=y, sr=SR, hop_length=HOP_LENGTH)
    bw       = librosa.feature.spectral_bandwidth(y=y, sr=SR, hop_length=HOP_LENGTH)
    return np.hstack([stats(mfcc), stats(mfcc_d), stats(mfcc_d2), stats(chroma),
                      stats(contrast), stats(tonnetz), stats(zcr), stats(rms),
                      [mel.mean(), mel.std(), np.percentile(mel,25), np.percentile(mel,75)],
                      stats(rolloff), stats(centroid), stats(bw)]).astype(np.float32)

print("⏳ Initialising ComParE...")
smile = opensmile.Smile(feature_set=opensmile.FeatureSet.ComParE_2016,
                        feature_level=opensmile.FeatureLevel.Functionals)
print(f"✅ ComParE: {len(smile.feature_names)} features")

def extract_compare(y):
    arr = smile.process_signal(y, SR).values.flatten().astype(np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

print("⏳ Loading Wav2Vec 2.0...")
w2v_proc  = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
w2v_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(DEVICE)
w2v_model.eval()
print("✅ Wav2Vec ready")

def extract_wav2vec(y):
    y = y / (np.abs(y).max() + 1e-8)
    inp = w2v_proc(y, sampling_rate=SR, return_tensors="pt", padding=True)
    with torch.no_grad(): out = w2v_model(inp.input_values.to(DEVICE))
    h = out.last_hidden_state.squeeze(0)
    return torch.cat([h.mean(0), h.std(0)]).cpu().numpy().astype(np.float32)

# ── Cell 6: Build feature matrices ───────────────────────────
print("\n⏳ Extracting features (~15-25 min)...")
X_hc, X_cmp, X_w2v, y_labels = [], [], [], []
skipped = 0

for i, (_, row) in enumerate(df.iterrows(), 1):
    if i % 10 == 0 or i == len(df):
        pct = i/len(df)*100
        print(f"  [{'█'*int(pct/5)+'░'*(20-int(pct/5))}] {pct:.0f}%", end="\r")
    audio = load_audio(row["fname"], row["dataset"])
    if audio is None: skipped += 1; continue
    try:
        X_hc.append(extract_handcrafted(audio))
        X_cmp.append(extract_compare(audio))
        X_w2v.append(extract_wav2vec(audio))
        y_labels.append(row["label"])
    except: skipped += 1

X_hc = np.array(X_hc, dtype=np.float32)
X_cmp = np.array(X_cmp, dtype=np.float32)
X_w2v = np.array(X_w2v, dtype=np.float32)
for X in [X_hc, X_cmp, X_w2v]: np.nan_to_num(X, copy=False)
X_combined = np.hstack([X_hc, X_cmp, X_w2v])
print(f"\n✅ {len(y_labels)} samples | skipped {skipped}")

# ── Cell 7: Binary encode + split ────────────────────────────
y_bin = np.array([0 if l=="normal" else 1 for l in y_labels])
print(f"Normal: {(y_bin==0).sum()} | Abnormal: {(y_bin==1).sum()}")

FEATURE_SETS = {"Handcrafted":X_hc, "ComParE":X_cmp,
                "Wav2Vec 2.0":X_w2v, "Combined":X_combined}
splits = {name: train_test_split(X, y_bin, test_size=0.2,
                                  random_state=42, stratify=y_bin)
          for name, X in FEATURE_SETS.items()}

# ── Cell 8: Model zoo ─────────────────────────────────────────
pos_w = (y_bin==0).sum() / (y_bin==1).sum()
def make_models():
    return {
        "Random Forest"    : Pipeline([("s",StandardScaler()),("c",RandomForestClassifier(300,class_weight="balanced",random_state=42,n_jobs=-1))]),
        "XGBoost"          : Pipeline([("s",StandardScaler()),("c",XGBClassifier(300,learning_rate=0.05,max_depth=6,scale_pos_weight=pos_w,eval_metric="logloss",random_state=42,n_jobs=-1,verbosity=0))]),
        "LightGBM"         : Pipeline([("s",StandardScaler()),("c",LGBMClassifier(300,learning_rate=0.05,num_leaves=63,class_weight="balanced",random_state=42,n_jobs=-1,verbose=-1))]),
        "Gradient Boosting": Pipeline([("s",StandardScaler()),("c",GradientBoostingClassifier(200,learning_rate=0.05,max_depth=5,random_state=42))]),
        "SVM (RBF)"        : Pipeline([("s",StandardScaler()),("c",SVC(kernel="rbf",C=10,gamma="scale",probability=True,class_weight="balanced",random_state=42))]),
        "Logistic Reg."    : Pipeline([("s",StandardScaler()),("c",LogisticRegression(max_iter=1000,C=1.0,class_weight="balanced",random_state=42,n_jobs=-1))]),
    }

# ── Cell 9: Full benchmark ────────────────────────────────────
print("\n🏋 Binary benchmark (4 feature sets × 6 models)...\n")
records, roc_data = [], {}
best = {"auc":0, "model":None, "feat":"", "label":"", "fpr":None, "tpr":None, "Xte":None, "yte":None}

for feat_name, (Xtr,Xte,ytr,yte) in splits.items():
    print(f"\n  ── {feat_name} ──")
    for model_name, model in make_models().items():
        model.fit(Xtr, ytr)
        y_pred  = model.predict(Xte)
        y_proba = model.predict_proba(Xte)[:,1]
        tn,fp,fn,tp = confusion_matrix(yte,y_pred).ravel()
        acc  = (tp+tn)/(tp+tn+fp+fn)
        sens = tp/(tp+fn)
        spec = tn/(tn+fp)
        prec = tp/(tp+fp) if (tp+fp)>0 else 0
        f1   = 2*prec*sens/(prec+sens+1e-8)
        fpr,tpr,_ = roc_curve(yte, y_proba)
        auc_score = auc(fpr,tpr)
        roc_data[f"{feat_name}\n{model_name}"] = (fpr,tpr,auc_score)
        records.append({"Feature Set":feat_name,"Model":model_name,
                        "Accuracy":acc,"Sensitivity":sens,"Specificity":spec,
                        "Precision":prec,"F1":f1,"AUC-ROC":auc_score})
        flag = " ⭐" if auc_score > best["auc"] else ""
        print(f"    {model_name:<20} Acc:{acc:.1%}  Sens:{sens:.1%}  Spec:{spec:.1%}  AUC:{auc_score:.3f}{flag}")
        if auc_score > best["auc"]:
            best.update({"auc":auc_score,"model":model,"feat":feat_name,
                         "label":f"{feat_name} + {model_name}",
                         "fpr":fpr,"tpr":tpr,"Xte":Xte,"yte":yte})

results_df = pd.DataFrame(records)
print(f"\n🏆 Best: {best['label']} → AUC {best['auc']:.3f}")

# ── Cell 10: Heatmaps ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
metrics = [("Accuracy","YlGnBu",50,100), ("Sensitivity","OrRd",30,100), ("AUC-ROC","PuRd",0.5,1.0)]
for ax, (metric, cmap, vmin, vmax) in zip(axes, metrics):
    pivot = results_df.pivot(index="Feature Set", columns="Model", values=metric)
    data  = pivot*100 if metric!="AUC-ROC" else pivot
    sns.heatmap(data, annot=data.round(1).astype(str), fmt="", cmap=cmap,
                vmin=vmin, vmax=vmax, ax=ax, linewidths=0.5, linecolor="white")
    ax.set_title(metric, fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
fig.suptitle("Binary Classifier — Normal vs Abnormal Heartbeat", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(DATASET_DIR/"binary_benchmark_heatmaps.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Cell 11: ROC curves ──────────────────────────────────────
# Best model per feature set
best_per_feat = {}
for rec in records:
    k = rec["Feature Set"]
    if k not in best_per_feat or rec["AUC-ROC"] > best_per_feat[k]["AUC-ROC"]:
        best_per_feat[k] = rec

colors = ["#4C72B0","#DD8452","#55A868","#C44E52"]
fig, ax = plt.subplots(figsize=(8,7))
for (feat_name, rec), color in zip(best_per_feat.items(), colors):
    key = f"{feat_name}\n{rec['Model']}"
    if key in roc_data:
        fpr,tpr,auc_s = roc_data[key]
        ax.plot(fpr, tpr, lw=2, color=color,
                label=f"{feat_name} + {rec['Model']} (AUC={auc_s:.3f})")
ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.5)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate (Sensitivity)")
ax.set_title("ROC Curves — Normal vs Abnormal")
ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(DATASET_DIR/"binary_roc_curves.png", dpi=150)
plt.show()

# ── Cell 12: Best model confusion matrix ─────────────────────
y_pred_best = best["model"].predict(best["Xte"])
cm = confusion_matrix(best["yte"], y_pred_best)
fig, axes = plt.subplots(1,2,figsize=(13,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal","Abnormal"], yticklabels=["Normal","Abnormal"],
            ax=axes[0], annot_kws={"size":14})
axes[0].set_title(f"Counts\n{best['label']}")
cm_norm = cm.astype(float)/cm.sum(axis=1,keepdims=True)*100
sns.heatmap(cm_norm, annot=True, fmt=".1f", cmap="Blues",
            xticklabels=["Normal","Abnormal"], yticklabels=["Normal","Abnormal"],
            ax=axes[1], annot_kws={"size":14})
axes[1].set_title(f"Normalised (%)\n{best['label']}")
for ax in axes: ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(DATASET_DIR/"binary_confusion_matrix.png", dpi=150); plt.show()
tn,fp,fn,tp = cm.ravel()
print(f"\n  True Negatives  (Normal   → Normal)   : {tn}")
print(f"  False Positives (Normal   → Abnormal) : {fp}  ← healthy flagged")
print(f"  False Negatives (Abnormal → Normal)   : {fn}  ← MISSED ⚠")
print(f"  True Positives  (Abnormal → Abnormal) : {tp}")

# ── Cell 13: Threshold analysis ──────────────────────────────
y_proba = best["model"].predict_proba(best["Xte"])[:,1]
fpr,tpr,thresholds = roc_curve(best["yte"], y_proba)
spec = 1-fpr; balanced = (tpr+spec)/2
best_thresh = thresholds[np.argmax(balanced)]

fig, ax = plt.subplots(figsize=(9,5))
ax.plot(thresholds, tpr,      color="#C44E52", lw=2, label="Sensitivity")
ax.plot(thresholds, spec,     color="#4C72B0", lw=2, label="Specificity")
ax.plot(thresholds, balanced, color="#55A868", lw=1.5, linestyle="--", label="Balanced Average")
ax.axvline(best_thresh, color="gray", linestyle=":", lw=1.5,
           label=f"Optimal threshold = {best_thresh:.2f}")
ax.set_xlabel("Decision Threshold"); ax.set_ylabel("Score")
ax.set_title("Sensitivity vs Specificity Trade-off")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(DATASET_DIR/"binary_threshold_tradeoff.png", dpi=150); plt.show()
print(f"\nOptimal threshold: {best_thresh:.2f}")

# ── Cell 14: Predict a single file ───────────────────────────
def predict_binary(filepath, threshold=best_thresh):
    y, _ = librosa.load(filepath, sr=SR, duration=DURATION)
    y = np.pad(y, (0, max(0, SR*DURATION-len(y))))[:SR*DURATION]
    hc = extract_handcrafted(y); cmp = extract_compare(y); w2v = extract_wav2vec(y)
    feat_map = {"Handcrafted":hc,"ComParE":cmp,"Wav2Vec 2.0":w2v,
                "Combined":np.hstack([hc,cmp,w2v])}
    feats = feat_map[best["feat"]].reshape(1,-1)
    prob_abn = best["model"].predict_proba(feats)[0][1]
    result   = "ABNORMAL ⚠" if prob_abn >= threshold else "NORMAL ✅"
    print(f"\n🩺  {Path(filepath).name}")
    print(f"    Result : {result}")
    print(f"    Normal   {'█'*int((1-prob_abn)*40)} {(1-prob_abn):.1%}")
    print(f"    Abnormal {'█'*int(prob_abn*40)} {prob_abn:.1%}")
    return result

# Demo
for orig_cls in ["normal","murmur","artifact"]:
    row = df[df["original_label"]==orig_cls].iloc[0]
    p   = (PATHS["set_a_dir"] if row["dataset"]=="A" else PATHS["set_b_dir"]) / Path(row["fname"]).name
    predict_binary(str(p))

print(f"\n✅ Binary classifier complete!")
print(f"🏆 Best: {best['label']} → AUC {best['auc']:.3f}")
