# ============================================================
#  02_multiclass_classifier.ipynb
#  MULTICLASS CARDIAC CLASSIFIER — 5 conditions
#  Normal | Murmur | Extrastole | Extrahls | Artifact
#  Commit 2: Added multiclass classification + full benchmark
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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
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
DATASET_DIR = Path(r"D:\Heart Beat Sound")
SR, DURATION, N_MFCC, HOP_LENGTH = 16000, 4, 40, 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
print("✅ Dataset found")

# ── Cell 4: Load metadata (multiclass labels) ─────────────────
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
    print("📊 Class distribution:"); print(df["label"].value_counts())
    return df

df = load_metadata()

# ── Cell 5: Feature extraction ────────────────────────────────
def load_audio(fname, dataset):
    folder = PATHS["set_a_dir"] if dataset=="A" else PATHS["set_b_dir"]
    path = folder / Path(fname).name
    if not path.exists(): return None
    y, _ = librosa.load(path, sr=SR, duration=DURATION)
    target = SR*DURATION
    return np.pad(y,(0,max(0,target-len(y))))[:target]

def extract_handcrafted(y):
    def stats(x): return np.hstack([x.mean(axis=-1),x.std(axis=-1),
                                     np.percentile(x,25,axis=-1),np.percentile(x,75,axis=-1)])
    mfcc=librosa.feature.mfcc(y=y,sr=SR,n_mfcc=N_MFCC,hop_length=HOP_LENGTH)
    return np.hstack([
        stats(mfcc), stats(librosa.feature.delta(mfcc)), stats(librosa.feature.delta(mfcc,order=2)),
        stats(librosa.feature.chroma_stft(y=y,sr=SR,hop_length=HOP_LENGTH)),
        stats(librosa.feature.spectral_contrast(y=y,sr=SR,hop_length=HOP_LENGTH)),
        stats(librosa.feature.tonnetz(y=librosa.effects.harmonic(y),sr=SR)),
        stats(librosa.feature.zero_crossing_rate(y,hop_length=HOP_LENGTH)),
        stats(librosa.feature.rms(y=y,hop_length=HOP_LENGTH)),
        [m:=librosa.power_to_db(librosa.feature.melspectrogram(y=y,sr=SR,hop_length=HOP_LENGTH),ref=np.max),
         m.mean(), m.std(), np.percentile(m,25), np.percentile(m,75)][-4:],
        stats(librosa.feature.spectral_rolloff(y=y,sr=SR,hop_length=HOP_LENGTH)),
        stats(librosa.feature.spectral_centroid(y=y,sr=SR,hop_length=HOP_LENGTH)),
        stats(librosa.feature.spectral_bandwidth(y=y,sr=SR,hop_length=HOP_LENGTH)),
    ]).astype(np.float32)

print("⏳ Initialising ComParE..."); 
smile = opensmile.Smile(feature_set=opensmile.FeatureSet.ComParE_2016,
                        feature_level=opensmile.FeatureLevel.Functionals)
print(f"✅ ComParE: {len(smile.feature_names)} features")

def extract_compare(y):
    return np.nan_to_num(smile.process_signal(y,SR).values.flatten().astype(np.float32))

print("⏳ Loading Wav2Vec 2.0...")
w2v_proc  = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
w2v_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(DEVICE)
w2v_model.eval()
print("✅ Wav2Vec ready")

def extract_wav2vec(y):
    y = y/(np.abs(y).max()+1e-8)
    inp = w2v_proc(y,sampling_rate=SR,return_tensors="pt",padding=True)
    with torch.no_grad(): out = w2v_model(inp.input_values.to(DEVICE))
    h = out.last_hidden_state.squeeze(0)
    return torch.cat([h.mean(0),h.std(0)]).cpu().numpy().astype(np.float32)

# ── Cell 6: Extract features ─────────────────────────────────
print("\n⏳ Extracting all features (~15-25 min)...")
X_hc, X_cmp, X_w2v, y_labels = [], [], [], []
skipped = 0
for i,(_, row) in enumerate(df.iterrows(),1):
    if i%10==0 or i==len(df):
        pct=i/len(df)*100
        print(f"  [{'█'*int(pct/5)+'░'*(20-int(pct/5))}] {pct:.0f}%",end="\r")
    audio = load_audio(row["fname"], row["dataset"])
    if audio is None: skipped+=1; continue
    try:
        X_hc.append(extract_handcrafted(audio))
        X_cmp.append(extract_compare(audio))
        X_w2v.append(extract_wav2vec(audio))
        y_labels.append(row["label"])
    except: skipped+=1

X_hc=np.array(X_hc,dtype=np.float32); X_cmp=np.array(X_cmp,dtype=np.float32)
X_w2v=np.array(X_w2v,dtype=np.float32)
for X in [X_hc,X_cmp,X_w2v]: np.nan_to_num(X,copy=False)
X_combined=np.hstack([X_hc,X_cmp,X_w2v])
print(f"\n✅ {len(y_labels)} samples | skipped {skipped}")
print(f"Dims — HC:{X_hc.shape[1]} | ComParE:{X_cmp.shape[1]} | W2V:{X_w2v.shape[1]} | Combined:{X_combined.shape[1]}")

# ── Cell 7: Encode labels & split ────────────────────────────
le = LabelEncoder()
y  = le.fit_transform(np.array(y_labels))
print("Classes:", le.classes_)

FEATURE_SETS = {"Handcrafted":X_hc,"ComParE":X_cmp,"Wav2Vec 2.0":X_w2v,"Combined":X_combined}
splits = {name: train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
          for name,X in FEATURE_SETS.items()}
print(f"Train:{len(splits['Combined'][0])} | Test:{len(splits['Combined'][1])}")

# ── Cell 8: Model zoo ─────────────────────────────────────────
def make_models():
    return {
        "Random Forest"    : Pipeline([("s",StandardScaler()),("c",RandomForestClassifier(300,class_weight="balanced",random_state=42,n_jobs=-1))]),
        "XGBoost"          : Pipeline([("s",StandardScaler()),("c",XGBClassifier(300,learning_rate=0.05,max_depth=6,eval_metric="mlogloss",random_state=42,n_jobs=-1,verbosity=0))]),
        "LightGBM"         : Pipeline([("s",StandardScaler()),("c",LGBMClassifier(300,learning_rate=0.05,num_leaves=63,class_weight="balanced",random_state=42,n_jobs=-1,verbose=-1))]),
        "Gradient Boosting": Pipeline([("s",StandardScaler()),("c",GradientBoostingClassifier(200,learning_rate=0.05,max_depth=5,random_state=42))]),
        "SVM (RBF)"        : Pipeline([("s",StandardScaler()),("c",SVC(kernel="rbf",C=10,gamma="scale",probability=True,class_weight="balanced",random_state=42))]),
        "Logistic Reg."    : Pipeline([("s",StandardScaler()),("c",LogisticRegression(max_iter=1000,C=1.0,class_weight="balanced",random_state=42,n_jobs=-1))]),
    }

# ── Cell 9: Full benchmark ────────────────────────────────────
print("\n🏋 Multiclass benchmark (4 × 6 = 24 experiments)...\n")
records = []
best = {"acc":0,"model":None,"feat":"","label":"","Xte":None,"yte":None}

for feat_name,(Xtr,Xte,ytr,yte) in splits.items():
    print(f"\n  ── {feat_name} ({Xtr.shape[1]} features) ──")
    for model_name,model in make_models().items():
        model.fit(Xtr,ytr)
        y_pred = model.predict(Xte)
        acc = (y_pred==yte).mean()
        records.append({"Feature Set":feat_name,"Model":model_name,"Accuracy":acc})
        flag=" ⭐" if acc>best["acc"] else ""
        print(f"    {model_name:<20} {acc:.1%}{flag}")
        if acc>best["acc"]:
            best.update({"acc":acc,"model":model,"feat":feat_name,
                         "label":f"{feat_name} + {model_name}","Xte":Xte,"yte":yte})

results_df = pd.DataFrame(records)
pivot = results_df.pivot(index="Feature Set",columns="Model",values="Accuracy")
print(f"\n🏆 Best: {best['label']} → {best['acc']:.1%}")
print("\n📊 Full results (%):\n", (pivot*100).round(1).to_string())

# ── Cell 10: Heatmap ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13,4))
annot = (pivot*100).round(1).astype(str)
sns.heatmap(pivot*100, annot=annot, fmt="", cmap="YlGnBu",
            vmin=50, vmax=100, ax=ax, linewidths=0.5, linecolor="white")
# Highlight best cell
best_row = list(pivot.index).index(best["feat"])
best_col = list(pivot.columns).index(best["label"].split(" + ")[1])
ax.add_patch(plt.Rectangle((best_col,best_row),1,1,fill=False,edgecolor="red",lw=2.5))
ax.set_title("Multiclass Benchmark — 5 Cardiac Conditions\nAccuracy (%) by Feature Set × Model",
             fontsize=12, fontweight="bold")
ax.tick_params(axis="x",rotation=30)
plt.tight_layout()
plt.savefig(DATASET_DIR/"multiclass_benchmark_heatmap.png",dpi=150,bbox_inches="tight"); plt.show()

# ── Cell 11: Grouped bar chart ────────────────────────────────
MODEL_NAMES   = list(make_models().keys())
FEATURE_NAMES = list(FEATURE_SETS.keys())
fig, ax = plt.subplots(figsize=(14,6))
x=np.arange(len(FEATURE_NAMES)); width=0.13
colors=["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2","#937860"]
for i,(model_name,color) in enumerate(zip(MODEL_NAMES,colors)):
    vals=[results_df[(results_df["Feature Set"]==f)&(results_df["Model"]==model_name)]["Accuracy"].values[0]*100
          for f in FEATURE_NAMES]
    bars=ax.bar(x+(i-len(MODEL_NAMES)/2)*width+width/2,vals,width,label=model_name,color=color,alpha=0.88)
ax.set_xticks(x); ax.set_xticklabels(FEATURE_NAMES,fontsize=11)
ax.set_ylabel("Accuracy (%)"); ax.set_title("Multiclass — Model vs Feature Set")
ax.legend(loc="lower right",fontsize=9); ax.set_ylim(0,100)
plt.tight_layout()
plt.savefig(DATASET_DIR/"multiclass_benchmark_bars.png",dpi=150); plt.show()

# ── Cell 12: Best model confusion matrix ─────────────────────
y_pred_best = best["model"].predict(best["Xte"])
cm = confusion_matrix(best["yte"], y_pred_best)
fig, axes = plt.subplots(1,2,figsize=(14,5))
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",
            xticklabels=le.classes_,yticklabels=le.classes_,ax=axes[0])
axes[0].set_title(f"Counts — {best['label']}")
cm_norm=cm.astype(float)/cm.sum(axis=1,keepdims=True)*100
sns.heatmap(cm_norm,annot=True,fmt=".1f",cmap="Blues",
            xticklabels=le.classes_,yticklabels=le.classes_,ax=axes[1])
axes[1].set_title(f"Normalised (%) — {best['label']}")
for ax in axes: ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(DATASET_DIR/"multiclass_confusion_matrix.png",dpi=150); plt.show()
print(classification_report(best["yte"],y_pred_best,target_names=le.classes_))

# ── Cell 13: Per-class F1 by feature set ─────────────────────
all_reports = {}
for feat_name,(Xtr,Xte,ytr,yte) in splits.items():
    for model_name,model in make_models().items():
        model.fit(Xtr,ytr); y_pred=model.predict(Xte)
        all_reports[f"{feat_name}+{model_name}"] = classification_report(
            yte,y_pred,target_names=le.classes_,output_dict=True)

fig, axes = plt.subplots(1,len(le.classes_),figsize=(14,5),sharey=True)
for ci,cls in enumerate(le.classes_):
    f1s=[max(all_reports[f"{f}+{m}"][cls]["f1-score"]
             for m in MODEL_NAMES)*100 for f in FEATURE_NAMES]
    bars=axes[ci].bar(FEATURE_NAMES,f1s,color=["#4C72B0","#55A868","#DD8452","#C44E52"],alpha=0.85)
    axes[ci].set_title(cls.upper(),fontsize=10,fontweight="bold")
    axes[ci].set_ylim(0,110); axes[ci].tick_params(axis="x",rotation=45,labelsize=8)
    for bar,v in zip(bars,f1s): axes[ci].text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,
                                               f"{v:.0f}",ha="center",fontsize=8)
    if ci==0: axes[ci].set_ylabel("Best F1-Score (%)")
fig.suptitle("Best F1-Score per Class by Feature Set",fontsize=13)
plt.tight_layout(); plt.savefig(DATASET_DIR/"multiclass_per_class_f1.png",dpi=150,bbox_inches="tight"); plt.show()

# ── Cell 14: 5-Fold Cross Validation ─────────────────────────
print(f"\n⏳ 5-fold CV on best feature set ({best['feat']})...")
X_best  = FEATURE_SETS[best["feat"]]
X_sc    = StandardScaler().fit_transform(X_best)
cv_results = {}
for model_name,model in make_models().items():
    scores = cross_val_score(model.named_steps["c"],X_sc,y,cv=5,scoring="accuracy",n_jobs=-1)
    cv_results[model_name]=scores
    print(f"  {model_name:<20} {scores.mean():.1%} ± {scores.std():.1%}")

fig, ax = plt.subplots(figsize=(10,5))
ax.boxplot(cv_results.values(),labels=cv_results.keys(),patch_artist=True,
           boxprops=dict(facecolor="#4C72B0",alpha=0.6),medianprops=dict(color="red",lw=2))
ax.set_ylabel("5-Fold CV Accuracy"); ax.tick_params(axis="x",rotation=20)
ax.set_title(f"Cross-Validation — {best['feat']} features"); ax.set_ylim(0,1.05)
plt.tight_layout(); plt.savefig(DATASET_DIR/"multiclass_cross_validation.png",dpi=150); plt.show()

# ── Cell 15: Predict a single file ───────────────────────────
def predict_multiclass(filepath):
    y, _ = librosa.load(filepath, sr=SR, duration=DURATION)
    y = np.pad(y,(0,max(0,SR*DURATION-len(y))))[:SR*DURATION]
    hc=extract_handcrafted(y); cmp=extract_compare(y); w2v=extract_wav2vec(y)
    feat_map={"Handcrafted":hc,"ComParE":cmp,"Wav2Vec 2.0":w2v,"Combined":np.hstack([hc,cmp,w2v])}
    feats=feat_map[best["feat"]].reshape(1,-1); np.nan_to_num(feats,copy=False)
    proba=best["model"].predict_proba(feats)[0]
    pred=le.inverse_transform([np.argmax(proba)])[0]
    print(f"\n🩺  {Path(filepath).name}  →  {pred.upper()}")
    for cls,p in sorted(zip(le.classes_,proba),key=lambda x:-x[1]):
        print(f"    {cls:<14} {'█'*int(p*40)} {p:.1%}")
    return pred

# Demo
for cls in ["normal","murmur","extrastole"]:
    row=df[df["label"]==cls].iloc[0]
    p=(PATHS["set_a_dir"] if row["dataset"]=="A" else PATHS["set_b_dir"])/Path(row["fname"]).name
    predict_multiclass(str(p))

print(f"\n✅ Multiclass classifier complete!")
print(f"🏆 Best: {best['label']} → {best['acc']:.1%}")
print(f"\nOutput files saved to: {DATASET_DIR}")
