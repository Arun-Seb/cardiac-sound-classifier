"""
models.py
─────────
Model zoo for cardiac sound classification.
All models wrapped in sklearn Pipelines with StandardScaler.

Models:
    Random Forest, XGBoost, LightGBM,
    Gradient Boosting, SVM (RBF), Logistic Regression
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def make_models(class_weight: str = "balanced",
                pos_weight: float = 1.0) -> dict:
    """
    Build all models with sensible defaults.

    Args:
        class_weight : 'balanced' for imbalanced datasets (binary or multiclass)
        pos_weight   : XGBoost scale_pos_weight for binary tasks
                       = n_negative / n_positive

    Returns:
        dict of {model_name: sklearn Pipeline}
    """
    return {
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                class_weight=class_weight,
                random_state=42,
                n_jobs=-1,
            ))
        ]),

        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                scale_pos_weight=pos_weight,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            ))
        ]),

        "LightGBM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=63,
                class_weight=class_weight,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            ))
        ]),

        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                random_state=42,
            ))
        ]),

        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                C=10,
                gamma="scale",
                probability=True,
                class_weight=class_weight,
                random_state=42,
            ))
        ]),

        "Logistic Reg.": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                C=1.0,
                class_weight=class_weight,
                random_state=42,
                n_jobs=-1,
            ))
        ]),
    }


MODEL_NAMES = list(make_models().keys())
