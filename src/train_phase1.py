import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from ucimlrepo import fetch_ucirepo
except ImportError:
    fetch_ucirepo = None

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

for p in (RAW, PROCESSED, MODELS, REPORTS):
    p.mkdir(parents=True, exist_ok=True)

UCI_FILE = RAW / "PhiUSIIL_Phishing_URL_Dataset.csv"

# Raw identifiers / high-cardinality categorical text are intentionally excluded
# from the numeric baseline. They can be used later for dedicated engineered features.
EXCLUDE_COLUMNS = {
    "FILENAME", "URL", "Domain", "TLD", "Title", "label"
}

RANDOM_STATE = 42


def load_dataset(csv_path=None):
    if csv_path:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")
        return pd.read_csv(path, low_memory=False)

    if UCI_FILE.exists():
        return pd.read_csv(UCI_FILE, low_memory=False)

    if fetch_ucirepo is None:
        raise RuntimeError(
            "ucimlrepo is not installed. Run: pip install ucimlrepo "
            "or download the CSV from the UCI PhiUSIIL page."
        )

    print("Downloading PhiUSIIL from UCI...")
    ds = fetch_ucirepo(id=967)
    X = ds.data.features.copy()
    y = ds.data.targets.copy()

    if isinstance(y, pd.DataFrame):
        y_col = y.columns[0]
        df = X.copy()
        df[y_col] = y[y_col].values
    else:
        df = X.copy()
        df["label"] = y

    if "label" not in df.columns:
        # UCI metadata says the target is label.
        candidates = [c for c in df.columns if c.lower() == "label"]
        if not candidates:
            raise RuntimeError("Could not identify the target column 'label'.")
        df = df.rename(columns={candidates[0]: "label"})

    df.to_csv(UCI_FILE, index=False)
    print(f"Saved raw dataset to {UCI_FILE}")
    return df


def validate_dataset(df):
    required = {"URL", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    print("\n=== DATASET VALIDATION ===")
    print("Shape:", df.shape)
    print("Missing values:", int(df.isna().sum().sum()))
    print("Duplicate URLs:", int(df["URL"].duplicated().sum()))
    print("Source label counts:")
    print(df["label"].value_counts(dropna=False).sort_index())

    bad_labels = set(pd.to_numeric(df["label"], errors="coerce").dropna().unique()) - {0, 1}
    if bad_labels:
        raise ValueError(f"Unexpected labels: {bad_labels}")

    return df


def clean_dataset(df):
    df = df.copy()

    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    df = df.dropna(subset=["URL", "label"])
    df["URL"] = df["URL"].astype(str).str.strip()

    # Remove exact duplicate URLs BEFORE splitting to reduce train/test contamination.
    df = df.drop_duplicates(subset=["URL"], keep="first").reset_index(drop=True)

    # Convert source labels: UCI 1=legitimate, 0=phishing.
    # Project convention: 0=legitimate, 1=phishing.
    df["target"] = (df["label"] == 0).astype(int)

    return df


def select_numeric_features(df):
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [
        c for c in numeric
        if c not in EXCLUDE_COLUMNS and c not in {"target"}
    ]

    if not features:
        raise ValueError("No numeric predictor columns found.")

    X = df[features].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    # Convert everything to numeric where possible.
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    return X, features


def split_data(X, y):
    # 70% train, 30% temporary
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )

    # 15% validation, 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp,
        random_state=RANDOM_STATE
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_models():
    models = {
        "LogisticRegression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=400,
                max_depth=None,
                min_samples_leaf=1,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=RANDOM_STATE
            ))
        ]),
        "SVM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(
                C=2.0,
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ])
    }

    if XGBClassifier is not None:
        models["XGBoost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                n_jobs=-1,
                random_state=RANDOM_STATE,
                tree_method="hist"
            ))
        ])
    else:
        print("XGBoost is not installed; skipping XGBoost.")

    return models


def evaluate(model, X, y):
    pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1]
    else:
        prob = None

    result = {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "mcc": matthews_corrcoef(y, pred),
    }
    if prob is not None:
        result["roc_auc"] = roc_auc_score(y, prob)
    else:
        result["roc_auc"] = np.nan

    return result, pred, prob


def save_confusion_matrix(name, y, pred):
    cm = confusion_matrix(y, pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", square=True,
        xticklabels=["Legitimate", "Phishing"],
        yticklabels=["Legitimate", "Phishing"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(REPORTS / f"confusion_matrix_{name}.png", dpi=180)
    plt.close()


def save_importance(name, model, X_test, y_test, feature_names):
    fitted = model.named_steps["model"]
    X_imp = X_test.copy()

    # Use native importance when available.
    importance = None
    if hasattr(fitted, "feature_importances_"):
        importance = fitted.feature_importances_
    elif hasattr(fitted, "coef_"):
        importance = np.abs(fitted.coef_[0])
    else:
        # SVM: permutation importance is more interpretable than kernel coefficients.
        r = permutation_importance(
            model, X_imp, y_test, n_repeats=3,
            random_state=RANDOM_STATE, scoring="f1", n_jobs=-1
        )
        importance = r.importances_mean

    imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values("importance", ascending=False)

    imp.to_csv(REPORTS / f"feature_importance_{name}.csv", index=False)

    top = imp.head(20).sort_values("importance")
    plt.figure(figsize=(9, 7))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Importance")
    plt.title(f"Top Features - {name}")
    plt.tight_layout()
    plt.savefig(REPORTS / f"feature_importance_{name}.png", dpi=180)
    plt.close()


def main(csv_path=None):
    print("AI-Driven Phishing Website Detection - Phase 1")
    print("=" * 60)

    df_raw = load_dataset(csv_path)
    validate_dataset(df_raw)

    df = clean_dataset(df_raw)

    print("\n=== AFTER DEDUPLICATION ===")
    print("Rows:", len(df))
    print("Project target counts:")
    print(df["target"].value_counts().sort_index())
    print("Project convention: 0=legitimate, 1=phishing")

    cleaned_path = PROCESSED / "cleaned_dataset.csv"
    df.to_csv(cleaned_path, index=False)

    X, feature_names = select_numeric_features(df)
    y = df["target"]

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    train_df = X_train.copy()
    train_df["target"] = y_train.values
    val_df = X_val.copy()
    val_df["target"] = y_val.values
    test_df = X_test.copy()
    test_df["target"] = y_test.values

    train_df.to_csv(PROCESSED / "train.csv", index=False)
    val_df.to_csv(PROCESSED / "validation.csv", index=False)
    test_df.to_csv(PROCESSED / "test.csv", index=False)

    print("\n=== SPLIT ===")
    print(f"Train:      {len(X_train):,}")
    print(f"Validation: {len(X_val):,}")
    print(f"Test:       {len(X_test):,}")
    print(f"Features:   {len(feature_names)}")

    models = build_models()
    comparison = []
    fitted = {}

    print("\n=== VALIDATION RESULTS ===")
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        metrics, pred, prob = evaluate(model, X_val, y_val)
        row = {"model": name, **metrics}
        comparison.append(row)
        fitted[name] = model

        print(
            f"Accuracy={metrics['accuracy']:.4f} | "
            f"Precision={metrics['precision']:.4f} | "
            f"Recall={metrics['recall']:.4f} | "
            f"F1={metrics['f1']:.4f} | "
            f"ROC-AUC={metrics['roc_auc']:.4f} | "
            f"MCC={metrics['mcc']:.4f}"
        )

    comparison_df = pd.DataFrame(comparison).sort_values(
        ["f1", "roc_auc"], ascending=False
    )
    comparison_df.to_csv(REPORTS / "model_comparison.csv", index=False)

    best_name = comparison_df.iloc[0]["model"]
    best_model = fitted[best_name]

    # Refit the selected model on train + validation before the final held-out test.
    X_trainval = pd.concat([X_train, X_val], axis=0)
    y_trainval = pd.concat([y_train, y_val], axis=0)

    print(f"\n=== FINAL MODEL ===\nSelected by validation F1: {best_name}")
    best_model.fit(X_trainval, y_trainval)

    test_metrics, test_pred, test_prob = evaluate(best_model, X_test, y_test)

    print("\n=== HELD-OUT TEST RESULTS ===")
    for k, v in test_metrics.items():
        print(f"{k.upper():10s}: {v:.6f}")

    print("\nClassification report:")
    print(classification_report(
        y_test, test_pred,
        target_names=["Legitimate", "Phishing"],
        digits=4,
        zero_division=0
    ))

    save_confusion_matrix(best_name, y_test, test_pred)
    save_importance(best_name, best_model, X_test, y_test, feature_names)

    pred_df = pd.DataFrame({
        "actual": y_test.values,
        "predicted": test_pred,
        "phishing_score": test_prob if test_prob is not None else np.nan
    })
    pred_df.to_csv(REPORTS / "test_predictions.csv", index=False)

    joblib.dump(best_model, MODELS / "best_model.joblib")

    metadata = {
        "project_target_convention": {
            "0": "legitimate",
            "1": "phishing"
        },
        "source_dataset": "UCI PhiUSIIL Phishing URL (Website), dataset id 967",
        "random_state": RANDOM_STATE,
        "split": "70% train / 15% validation / 15% held-out test",
        "duplicate_policy": "exact duplicate URLs removed before split",
        "excluded_columns": sorted(EXCLUDE_COLUMNS),
        "feature_names": feature_names,
        "selected_model": best_name,
        "validation_selection_metric": "F1",
        "test_metrics": {k: float(v) for k, v in test_metrics.items()}
    }

    with open(MODELS / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved:")
    print(MODELS / "best_model.joblib")
    print(MODELS / "model_metadata.json")
    print(REPORTS / "model_comparison.csv")
    print(REPORTS / "test_predictions.csv")
    print(REPORTS / f"confusion_matrix_{best_name}.png")
    print(REPORTS / f"feature_importance_{best_name}.csv")
    print(REPORTS / f"feature_importance_{best_name}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None, help="Optional local PhiUSIIL CSV path")
    args = parser.parse_args()
    main(args.csv)
