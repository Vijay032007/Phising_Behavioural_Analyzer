from pathlib import Path
import json
import warnings

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_FILE = (
    PROJECT_ROOT
    / "reports"
    / "behavioral_dataset.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "label"

RANDOM_STATE = 42

NON_FEATURE_COLUMNS = {
    "input_url",
    "final_url",
    "title",
    "navigation_error",
    "sample_file",
    "label",
    "label_source",
}


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FINAL BEHAVIORAL RANDOM FOREST TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    print("\nDataset:")
    print(DATASET_FILE)

    if not DATASET_FILE.exists():

        print("\nERROR: Behavioral dataset not found.")
        return

    df = pd.read_csv(DATASET_FILE)

    print(f"\nRows    : {len(df)}")
    print(f"Columns : {len(df.columns)}")

    # --------------------------------------------------------
    # LABEL CHECK
    # --------------------------------------------------------

    if TARGET not in df.columns:

        print("\nERROR: 'label' column not found.")
        return

    df = df.dropna(subset=[TARGET]).copy()

    df[TARGET] = df[TARGET].astype(int)

    print("\nClass distribution:")
    print(
        df[TARGET]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # CONSTANT FEATURES
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CONSTANT FEATURE REMOVAL")
    print("=" * 70)

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

        print("\nRemoving constant features:")

        for column in constant_features:
            print(f"  - {column}")

        df = df.drop(columns=constant_features)

    else:

        print("\nNo constant features found.")

    # --------------------------------------------------------
    # FEATURE SELECTION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL FEATURE SELECTION")
    print("=" * 70)

    feature_columns = []

    for column in df.columns:

        if column in NON_FEATURE_COLUMNS:
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            feature_columns.append(column)

    print(
        f"\nFinal behavioral feature count: "
        f"{len(feature_columns)}"
    )

    for index, column in enumerate(
        feature_columns,
        start=1
    ):
        print(
            f"{index:2d}. {column}"
        )

    X = df[feature_columns].copy()
    y = df[TARGET].copy()

    # --------------------------------------------------------
    # VERIFY DATASET
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATASET VERIFICATION")
    print("=" * 70)

    print(f"\nSamples       : {len(X)}")
    print(f"Features      : {len(feature_columns)}")
    print(f"Legitimate    : {(y == 0).sum()}")
    print(f"Phishing      : {(y == 1).sum()}")

    # --------------------------------------------------------
    # BUILD FINAL RANDOM FOREST
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING FINAL RANDOM FOREST")
    print("=" * 70)

    model = Pipeline([
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

    print("\nTraining on all labeled behavioral samples...")

    model.fit(X, y)

    print("Training completed successfully.")

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    model_file = (
        MODEL_DIR
        / "behavioral_random_forest.joblib"
    )

    joblib.dump(
        model,
        model_file
    )

    print("\nSaved final behavioral model:")
    print(model_file)

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)

    classifier = model.named_steps["classifier"]

    importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": classifier.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    ).reset_index(drop=True)

    print(
        importance_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    importance_file = (
        REPORT_DIR
        / "final_behavioral_feature_importance.csv"
    )

    importance_df.to_csv(
        importance_file,
        index=False
    )

    print("\nSaved feature importance:")
    print(importance_file)

    # --------------------------------------------------------
    # SAVE FEATURE LIST
    # --------------------------------------------------------

    feature_list_file = (
        MODEL_DIR
        / "behavioral_feature_list.json"
    )

    with open(
        feature_list_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            feature_columns,
            file,
            indent=4
        )

    print("\nSaved feature list:")
    print(feature_list_file)

    # --------------------------------------------------------
    # SAVE MODEL METADATA
    # --------------------------------------------------------

    metadata = {

        "project":
            "AI-Driven Behavioral Analysis of Phishing Websites Using Machine Learning",

        "model":
            "Random Forest",

        "model_file":
            str(model_file),

        "dataset":
            str(DATASET_FILE),

        "training_samples":
            int(len(X)),

        "legitimate_samples":
            int((y == 0).sum()),

        "phishing_samples":
            int((y == 1).sum()),

        "feature_count":
            int(len(feature_columns)),

        "features":
            feature_columns,

        "removed_constant_features":
            constant_features,

        "random_state":
            RANDOM_STATE,

        "n_estimators":
            300,

        "class_weight":
            "balanced",

        "cross_validation_reference":
            {
                "folds": 5,
                "mean_accuracy": 0.9584,
                "mean_precision": 0.9618,
                "mean_recall": 0.9600,
                "mean_f1": 0.9599,
                "mean_roc_auc": 0.9913,
                "mean_mcc": 0.9185
            }
    }

    metadata_file = (
        MODEL_DIR
        / "final_behavioral_model_metadata.json"
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

    print("\nSaved model metadata:")
    print(metadata_file)

    # --------------------------------------------------------
    # FINISH
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL BEHAVIORAL MODEL TRAINING COMPLETED")
    print("=" * 70)

    print("\nFinal model:")
    print("Random Forest")

    print("\nTraining data:")
    print("96 behavioral samples")

    print("\nBehavioral features:")
    print(f"{len(feature_columns)}")

    print("\nCross-validation reference:")
    print("5-fold mean F1 = 0.9599")
    print("5-fold mean ROC-AUC = 0.9913")

    print("\nNext stage:")
    print(
        "Connect the live feature extractor to "
        "behavioral_random_forest.joblib."
    )


if __name__ == "__main__":
    main()