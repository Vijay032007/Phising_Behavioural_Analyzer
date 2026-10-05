from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

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

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
N_SPLITS = 5

TARGET = "label"

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
    print("BEHAVIORAL MODEL 5-FOLD CROSS-VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    if not DATASET_FILE.exists():

        print("\nERROR: Dataset not found:")
        print(DATASET_FILE)
        return

    df = pd.read_csv(DATASET_FILE)

    print("\nDataset:")
    print(DATASET_FILE)

    print(f"\nRows    : {len(df)}")
    print(f"Columns : {len(df.columns)}")

    # --------------------------------------------------------
    # LABEL CHECK
    # --------------------------------------------------------

    if TARGET not in df.columns:

        print("\nERROR: label column not found.")
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
    # REMOVE CONSTANT FEATURES
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # SELECT NUMERIC FEATURES
    # --------------------------------------------------------

    feature_columns = []

    for column in df.columns:

        if column in NON_FEATURE_COLUMNS:
            continue

        if pd.api.types.is_numeric_dtype(df[column]):

            feature_columns.append(column)

    print(
        f"\nBehavioral features used: "
        f"{len(feature_columns)}"
    )

    X = df[feature_columns].copy()
    y = df[TARGET].copy()

    # --------------------------------------------------------
    # CROSS-VALIDATION
    # --------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    # --------------------------------------------------------
    # MODELS
    # --------------------------------------------------------

    models = {

        "Logistic Regression": Pipeline([
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
        ]),

        "Random Forest": Pipeline([
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
        ]),

        "SVM": Pipeline([
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
        ]),
    }

    # --------------------------------------------------------
    # SCORING
    # --------------------------------------------------------

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "mcc": "matthews_corrcoef",
    }

    all_results = []

    # --------------------------------------------------------
    # RUN CROSS-VALIDATION
    # --------------------------------------------------------

    for model_name, model in models.items():

        print("\n" + "-" * 70)
        print(f"MODEL: {model_name}")
        print("-" * 70)

        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1
        )

        result = {
            "Model": model_name,

            "Accuracy_Mean":
                scores["test_accuracy"].mean(),

            "Accuracy_STD":
                scores["test_accuracy"].std(),

            "Precision_Mean":
                scores["test_precision"].mean(),

            "Precision_STD":
                scores["test_precision"].std(),

            "Recall_Mean":
                scores["test_recall"].mean(),

            "Recall_STD":
                scores["test_recall"].std(),

            "F1_Mean":
                scores["test_f1"].mean(),

            "F1_STD":
                scores["test_f1"].std(),

            "ROC_AUC_Mean":
                scores["test_roc_auc"].mean(),

            "ROC_AUC_STD":
                scores["test_roc_auc"].std(),

            "MCC_Mean":
                scores["test_mcc"].mean(),

            "MCC_STD":
                scores["test_mcc"].std(),
        }

        all_results.append(result)

        print(
            f"Accuracy : "
            f"{result['Accuracy_Mean']:.4f} "
            f"+/- {result['Accuracy_STD']:.4f}"
        )

        print(
            f"Precision: "
            f"{result['Precision_Mean']:.4f} "
            f"+/- {result['Precision_STD']:.4f}"
        )

        print(
            f"Recall   : "
            f"{result['Recall_Mean']:.4f} "
            f"+/- {result['Recall_STD']:.4f}"
        )

        print(
            f"F1       : "
            f"{result['F1_Mean']:.4f} "
            f"+/- {result['F1_STD']:.4f}"
        )

        print(
            f"ROC-AUC  : "
            f"{result['ROC_AUC_Mean']:.4f} "
            f"+/- {result['ROC_AUC_STD']:.4f}"
        )

        print(
            f"MCC      : "
            f"{result['MCC_Mean']:.4f} "
            f"+/- {result['MCC_STD']:.4f}"
        )

    # --------------------------------------------------------
    # COMPARISON TABLE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("5-FOLD CROSS-VALIDATION COMPARISON")
    print("=" * 70)

    results_df = pd.DataFrame(all_results)

    display_columns = [
        "Model",
        "Accuracy_Mean",
        "Precision_Mean",
        "Recall_Mean",
        "F1_Mean",
        "ROC_AUC_Mean",
        "MCC_Mean",
    ]

    print(
        results_df[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    output_file = (
        REPORT_DIR
        / "behavioral_5fold_cv_results.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print("\nSaved cross-validation results:")
    print(output_file)

    # --------------------------------------------------------
    # BEST MODEL BY MEAN F1
    # --------------------------------------------------------

    best_index = results_df["F1_Mean"].idxmax()

    best_model = results_df.loc[
        best_index,
        "Model"
    ]

    best_f1 = results_df.loc[
        best_index,
        "F1_Mean"
    ]

    print("\nModel with highest mean F1:")
    print(best_model)

    print(
        f"Mean F1: {best_f1:.4f}"
    )

    # --------------------------------------------------------
    # FINISH
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("5-FOLD CROSS-VALIDATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()