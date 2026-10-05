from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_FILE = PROJECT_ROOT / "reports" / "behavioral_dataset.csv"

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15

TARGET = "label"

# Columns that describe the sample rather than its behavior.
NON_FEATURE_COLUMNS = {
    "input_url",
    "final_url",
    "title",
    "navigation_error",
    "sample_file",
    "label",
    "label_source",
}

# Runtime/metadata fields that are useful for inspection
# but should not be used directly as ML predictors.
# navigation_error is already excluded above.
INSPECTION_COLUMNS = {
    "input_url",
    "final_url",
    "title",
    "navigation_error",
    "sample_file",
    "label_source",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def calculate_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "F1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

    if y_prob is not None:
        try:
            metrics["ROC-AUC"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["ROC-AUC"] = np.nan
    else:
        metrics["ROC-AUC"] = np.nan

    return metrics


def get_prediction_probability(model, X):
    """
    Get probability/decision score for ROC-AUC.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        return model.decision_function(X)

    return None


def build_models():
    """
    Create the behavioral ML models.

    All models use median imputation.
    Scaling is used for Logistic Regression and SVM.
    Random Forest does not require scaling.
    """

    models = {}

    models["Logistic Regression"] = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE
            )
        ),
    ])

    models["Random Forest"] = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1
            )
        ),
    ])

    models["SVM"] = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            SVC(
                kernel="rbf",
                probability=True,
                random_state=RANDOM_STATE,
                class_weight="balanced"
            )
        ),
    ])

    return models


# ============================================================
# MAIN
# ============================================================

def main():

    print_section("BEHAVIORAL MACHINE LEARNING TRAINING")

    print(f"\nProject root:")
    print(PROJECT_ROOT)

    print(f"\nDataset:")
    print(DATASET_FILE)

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    if not DATASET_FILE.exists():
        print("\nERROR: Behavioral dataset not found.")
        print(DATASET_FILE)
        return

    df = pd.read_csv(DATASET_FILE)

    print_section("DATASET INFORMATION")

    print(f"Rows    : {len(df)}")
    print(f"Columns : {len(df.columns)}")

    if TARGET not in df.columns:
        print("\nERROR: 'label' column not found.")
        return

    print("\nClass distribution:")
    print(df[TARGET].value_counts().sort_index().to_string())

    # --------------------------------------------------------
    # BASIC CLEANING
    # --------------------------------------------------------

    print_section("DATASET CLEANING")

    # Remove rows without labels.
    before = len(df)

    df = df.dropna(subset=[TARGET]).copy()

    print(f"Rows before label filtering : {before}")
    print(f"Rows after label filtering  : {len(df)}")

    # Make label integer.
    df[TARGET] = df[TARGET].astype(int)

    # --------------------------------------------------------
    # REMOVE CONSTANT FEATURES
    # --------------------------------------------------------

    print_section("CONSTANT FEATURE REMOVAL")

    candidate_columns = [
        column
        for column in df.columns
        if column not in NON_FEATURE_COLUMNS
    ]

    constant_features = []

    for column in candidate_columns:

        if df[column].nunique(dropna=False) <= 1:
            constant_features.append(column)

    if constant_features:

        print("Removing constant features:")

        for column in constant_features:
            print(f"  - {column}")

        df = df.drop(columns=constant_features)

    else:
        print("No constant features found.")

    # --------------------------------------------------------
    # SELECT NUMERIC FEATURES
    # --------------------------------------------------------

    print_section("FEATURE SELECTION")

    feature_columns = []

    for column in df.columns:

        if column in NON_FEATURE_COLUMNS:
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            feature_columns.append(column)

    if not feature_columns:
        print("\nERROR: No numeric behavioral features found.")
        return

    print(f"Behavioral features selected: {len(feature_columns)}")

    for column in feature_columns:
        print(f"  - {column}")

    X = df[feature_columns].copy()
    y = df[TARGET].copy()

    # --------------------------------------------------------
    # REMOVE DUPLICATE URL GROUPS
    # --------------------------------------------------------

    print_section("DUPLICATE URL GROUP ANALYSIS")

    if "input_url" not in df.columns:

        print(
            "WARNING: input_url column is missing."
            "\nProceeding with normal stratified splitting."
        )

        grouped_split_available = False

    else:

        grouped_split_available = True

        url_counts = df["input_url"].value_counts()

        duplicate_urls = url_counts[url_counts > 1]

        print(f"Unique URLs    : {df['input_url'].nunique()}")
        print(f"Duplicate URLs : {len(duplicate_urls)}")

        if len(duplicate_urls) > 0:

            print("\nRepeated URLs:")

            for url, count in duplicate_urls.items():
                print(f"  {count}x  {url}")

    # --------------------------------------------------------
    # GROUP-AWARE SPLITTING
    # --------------------------------------------------------

    print_section("TRAIN / VALIDATION / TEST SPLIT")

    if grouped_split_available:

        # Create a URL-level table.
        url_groups = (
            df[["input_url", TARGET]]
            .drop_duplicates(subset=["input_url"])
            .reset_index(drop=True)
        )

        # Check whether one URL has conflicting labels.
        label_counts = (
            df.groupby("input_url")[TARGET]
            .nunique()
        )

        conflicting_urls = label_counts[label_counts > 1]

        if len(conflicting_urls) > 0:

            print(
                "\nWARNING: Some URLs have conflicting labels."
            )

            for url in conflicting_urls.index:
                print(f"  {url}")

            print(
                "\nThese URLs cannot safely be grouped."
            )

            return

        # First split: train+validation vs test.
        groups_train_val, groups_test = train_test_split(
            url_groups,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=url_groups[TARGET]
        )

        # Validation proportion relative to remaining data.
        validation_relative_size = (
            VALIDATION_SIZE /
            (1.0 - TEST_SIZE)
        )

        groups_train, groups_val = train_test_split(
            groups_train_val,
            test_size=validation_relative_size,
            random_state=RANDOM_STATE,
            stratify=groups_train_val[TARGET]
        )

        train_urls = set(groups_train["input_url"])
        val_urls = set(groups_val["input_url"])
        test_urls = set(groups_test["input_url"])

        train_mask = df["input_url"].isin(train_urls)
        val_mask = df["input_url"].isin(val_urls)
        test_mask = df["input_url"].isin(test_urls)

        X_train = df.loc[train_mask, feature_columns]
        y_train = df.loc[train_mask, TARGET]

        X_val = df.loc[val_mask, feature_columns]
        y_val = df.loc[val_mask, TARGET]

        X_test = df.loc[test_mask, feature_columns]
        y_test = df.loc[test_mask, TARGET]

    else:

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )

        validation_relative_size = (
            VALIDATION_SIZE /
            (1.0 - TEST_SIZE)
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=validation_relative_size,
            random_state=RANDOM_STATE,
            stratify=y_train_val
        )

    print(f"\nTraining samples   : {len(X_train)}")
    print(f"Validation samples : {len(X_val)}")
    print(f"Test samples       : {len(X_test)}")

    print("\nTraining class distribution:")
    print(y_train.value_counts().sort_index().to_string())

    print("\nValidation class distribution:")
    print(y_val.value_counts().sort_index().to_string())

    print("\nTest class distribution:")
    print(y_test.value_counts().sort_index().to_string())

    # --------------------------------------------------------
    # VERIFY URL SEPARATION
    # --------------------------------------------------------

    if grouped_split_available:

        train_urls = set(
            df.loc[X_train.index, "input_url"]
        )

        val_urls = set(
            df.loc[X_val.index, "input_url"]
        )

        test_urls = set(
            df.loc[X_test.index, "input_url"]
        )

        print("\nURL overlap checks:")

        print(
            "Train ∩ Validation:",
            len(train_urls & val_urls)
        )

        print(
            "Train ∩ Test:",
            len(train_urls & test_urls)
        )

        print(
            "Validation ∩ Test:",
            len(val_urls & test_urls)
        )

    # --------------------------------------------------------
    # TRAIN MODELS
    # --------------------------------------------------------

    print_section("MODEL TRAINING")

    models = build_models()

    results = []

    trained_models = {}

    for model_name, model in models.items():

        print(f"\nTraining: {model_name}")

        model.fit(X_train, y_train)

        # Validation prediction
        val_pred = model.predict(X_val)

        val_prob = get_prediction_probability(
            model,
            X_val
        )

        val_metrics = calculate_metrics(
            y_val,
            val_pred,
            val_prob
        )

        print("\nValidation metrics:")

        for metric, value in val_metrics.items():
            print(
                f"{metric:<12}: {value:.4f}"
                if not pd.isna(value)
                else f"{metric:<12}: NaN"
            )

        trained_models[model_name] = model

        results.append({
            "Model": model_name,
            "Accuracy": val_metrics["Accuracy"],
            "Precision": val_metrics["Precision"],
            "Recall": val_metrics["Recall"],
            "F1": val_metrics["F1"],
            "ROC-AUC": val_metrics["ROC-AUC"],
            "MCC": val_metrics["MCC"],
        })

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    print_section("MODEL COMPARISON")

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="F1",
        ascending=False
    ).reset_index(drop=True)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    comparison_file = (
        REPORT_DIR /
        "behavioral_model_comparison_validation.csv"
    )

    results_df.to_csv(
        comparison_file,
        index=False
    )

    print(
        f"\nSaved validation comparison:"
        f"\n{comparison_file}"
    )

    # --------------------------------------------------------
    # SELECT MODEL
    # --------------------------------------------------------

    best_model_name = results_df.iloc[0]["Model"]

    print_section("MODEL SELECTION")

    print(
        f"Selected model based on validation F1: "
        f"{best_model_name}"
    )

    best_model = trained_models[best_model_name]

    # --------------------------------------------------------
    # FINAL TEST EVALUATION
    # --------------------------------------------------------

    print_section("FINAL HELD-OUT TEST")

    test_pred = best_model.predict(X_test)

    test_prob = get_prediction_probability(
        best_model,
        X_test
    )

    test_metrics = calculate_metrics(
        y_test,
        test_pred,
        test_prob
    )

    for metric, value in test_metrics.items():

        if pd.isna(value):
            print(f"{metric:<12}: NaN")
        else:
            print(f"{metric:<12}: {value:.4f}")

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        test_pred,
        labels=[0, 1]
    )

    print("\nConfusion Matrix")
    print(
        "                 Predicted"
    )
    print(
        "                 Legit  Phish"
    )
    print(
        f"Actual Legit     {cm[0,0]:5d}  {cm[0,1]:5d}"
    )
    print(
        f"Actual Phish     {cm[1,0]:5d}  {cm[1,1]:5d}"
    )

    # --------------------------------------------------------
    # SAVE TEST PREDICTIONS
    # --------------------------------------------------------

    test_output = df.loc[
        X_test.index,
        [
            column
            for column in [
                "input_url",
                "final_url",
                TARGET
            ]
            if column in df.columns
        ]
    ].copy()

    test_output["predicted_label"] = test_pred

    if test_prob is not None:
        test_output["phishing_probability"] = test_prob

    test_predictions_file = (
        REPORT_DIR /
        "behavioral_test_predictions.csv"
    )

    test_output.to_csv(
        test_predictions_file,
        index=False
    )

    print(
        f"\nSaved test predictions:"
        f"\n{test_predictions_file}"
    )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    model_file = (
        MODEL_DIR /
        "behavioral_best_model.joblib"
    )

    joblib.dump(
        best_model,
        model_file
    )

    print(
        f"\nSaved behavioral model:"
        f"\n{model_file}"
    )

    # --------------------------------------------------------
    # SAVE METADATA
    # --------------------------------------------------------

    metadata = {
        "project": "AI-Driven Behavioral Analysis of Phishing Websites Using Machine Learning",
        "model_type": best_model_name,
        "target": TARGET,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "validation_size": VALIDATION_SIZE,
        "dataset_rows": int(len(df)),
        "dataset_columns": int(len(df.columns)),
        "training_samples": int(len(X_train)),
        "validation_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "feature_count": int(len(feature_columns)),
        "features": feature_columns,
        "removed_constant_features": constant_features,
        "test_metrics": {
            key: (
                None
                if pd.isna(value)
                else float(value)
            )
            for key, value in test_metrics.items()
        },
    }

    metadata_file = (
        MODEL_DIR /
        "behavioral_model_metadata.json"
    )

    with open(
        metadata_file,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4
        )

    print(
        f"\nSaved model metadata:"
        f"\n{metadata_file}"
    )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    print_section("FEATURE IMPORTANCE")

    classifier = best_model.named_steps["classifier"]

    if hasattr(classifier, "feature_importances_"):

        importance_values = classifier.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": feature_columns,
            "Importance": importance_values
        })

        importance_df = importance_df.sort_values(
            by="Importance",
            ascending=False
        )

        print(
            importance_df.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}"
            )
        )

        importance_file = (
            REPORT_DIR /
            "behavioral_feature_importance.csv"
        )

        importance_df.to_csv(
            importance_file,
            index=False
        )

        print(
            f"\nSaved feature importance:"
            f"\n{importance_file}"
        )

    elif hasattr(classifier, "coef_"):

        coefficients = classifier.coef_[0]

        importance_df = pd.DataFrame({
            "Feature": feature_columns,
            "Coefficient": coefficients,
            "AbsoluteCoefficient": np.abs(coefficients)
        })

        importance_df = importance_df.sort_values(
            by="AbsoluteCoefficient",
            ascending=False
        )

        print(
            importance_df.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}"
            )
        )

    else:

        print(
            "Feature importance is not directly available "
            "for the selected model."
        )

    # --------------------------------------------------------
    # FINISH
    # --------------------------------------------------------

    print_section("BEHAVIORAL ML TRAINING COMPLETED")

    print("\nFiles generated:")

    print(
        f"  Model      : {model_file}"
    )

    print(
        f"  Metadata   : {metadata_file}"
    )

    print(
        f"  Comparison : {comparison_file}"
    )

    print(
        f"  Predictions: {test_predictions_file}"
    )

    print("\nNext stage:")
    print(
        "Use the saved behavioral model with the live "
        "feature extractor."
    )


if __name__ == "__main__":
    main()