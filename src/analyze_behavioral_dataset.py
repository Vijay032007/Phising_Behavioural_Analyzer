from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_FILE = (
    PROJECT_ROOT
    / "reports"
    / "behavioral_dataset.csv"
)


def main():

    print("=" * 70)
    print("BEHAVIORAL DATASET QUALITY ANALYSIS")
    print("=" * 70)

    if not DATASET_FILE.exists():

        print(
            "\nERROR: Dataset not found:"
        )

        print(DATASET_FILE)

        return

    df = pd.read_csv(
        DATASET_FILE
    )

    print("\n--- DATASET SIZE ---")

    print(
        f"Rows    : {len(df)}"
    )

    print(
        f"Columns : {len(df.columns)}"
    )

    # ------------------------------------------------------------
    # CLASS DISTRIBUTION
    # ------------------------------------------------------------

    print("\n--- CLASS DISTRIBUTION ---")

    if "label" in df.columns:

        print(
            df["label"]
            .value_counts(dropna=False)
            .sort_index()
            .to_string()
        )

    # ------------------------------------------------------------
    # DUPLICATE URL CHECK
    # ------------------------------------------------------------

    print("\n--- DUPLICATE URL CHECK ---")

    if "input_url" in df.columns:

        duplicate_urls = (
            df["input_url"]
            .duplicated()
            .sum()
        )

        unique_urls = (
            df["input_url"]
            .nunique()
        )

        print(
            f"Unique URLs    : {unique_urls}"
        )

        print(
            f"Duplicate URLs : {duplicate_urls}"
        )

    # ------------------------------------------------------------
    # DUPLICATE FEATURE ROWS
    # ------------------------------------------------------------

    print("\n--- DUPLICATE ROW CHECK ---")

    duplicate_rows = (
        df.duplicated()
        .sum()
    )

    print(
        f"Duplicate complete rows: "
        f"{duplicate_rows}"
    )

    # ------------------------------------------------------------
    # MISSING VALUES
    # ------------------------------------------------------------

    print("\n--- MISSING VALUES ---")

    missing = (
        df.isna()
        .sum()
    )

    missing = missing[
        missing > 0
    ].sort_values(
        ascending=False
    )

    if missing.empty:

        print(
            "No missing values."
        )

    else:

        for column, count in missing.items():

            print(
                f"{column:<40} {count}"
            )

    # ------------------------------------------------------------
    # NUMERIC FEATURES
    # ------------------------------------------------------------

    numeric = df.select_dtypes(
        include=["number"]
    )

    print("\n--- NUMERIC FEATURES ---")

    print(
        f"Numeric columns: "
        f"{len(numeric.columns)}"
    )

    # ------------------------------------------------------------
    # CONSTANT FEATURES
    # ------------------------------------------------------------

    print("\n--- CONSTANT FEATURES ---")

    constant_features = []

    for column in numeric.columns:

        if column == "label":
            continue

        if numeric[column].nunique(
            dropna=False
        ) <= 1:

            constant_features.append(
                column
            )

    if constant_features:

        for column in constant_features:

            print(
                f"  {column}"
            )

    else:

        print(
            "No constant numeric features."
        )

    # ------------------------------------------------------------
    # FEATURE RANGES
    # ------------------------------------------------------------

    print("\n--- FEATURE RANGES ---")

    feature_columns = [
        column
        for column in numeric.columns
        if column != "label"
    ]

    for column in feature_columns:

        series = numeric[column]

        print(
            f"{column:<40} "
            f"min={series.min():.4f} "
            f"max={series.max():.4f} "
            f"unique={series.nunique()}"
        )

    # ------------------------------------------------------------
    # CORRELATION WITH LABEL
    # ------------------------------------------------------------

    print(
        "\n--- CORRELATION WITH LABEL ---"
    )

    if "label" in numeric.columns:

        correlations = (
            numeric[
                feature_columns + ["label"]
            ]
            .corr()["label"]
            .drop("label")
            .abs()
            .sort_values(
                ascending=False
            )
        )

        for column, value in correlations.items():

            print(
                f"{column:<40} "
                f"{value:.4f}"
            )

    # ------------------------------------------------------------
    # POTENTIAL LABEL LEAKAGE CHECK
    # ------------------------------------------------------------

    print(
        "\n--- POTENTIAL LABEL LEAKAGE CHECK ---"
    )

    if "label" in numeric.columns:

        suspicious = []

        for column in feature_columns:

            values = numeric[
                column
            ].dropna()

            if values.empty:
                continue

            unique_values = set(
                values.unique()
            )

            if unique_values.issubset(
                {0, 1}
            ):

                suspicious.append(
                    column
                )

        if suspicious:

            print(
                "Binary features requiring review:"
            )

            for column in suspicious:

                print(
                    f"  {column}"
                )

        else:

            print(
                "No obvious binary "
                "label-like features found."
            )

    # ------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "DATASET QUALITY ANALYSIS COMPLETED"
    )

    print("=" * 70)


if __name__ == "__main__":

    main()