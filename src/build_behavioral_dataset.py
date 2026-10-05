import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SAMPLE_DIR = (
    PROJECT_ROOT
    / "reports"
    / "behavioral_samples"
)

LEGIT_LABEL_FILES = [
    PROJECT_ROOT
    / "reports"
    / "behavioral_labels.csv",

    PROJECT_ROOT
    / "reports"
    / "legitimate_behavioral_labels.csv"
]

PHISH_LABEL_FILE = (
    PROJECT_ROOT
    / "reports"
    / "phishing_behavioral_labels.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "behavioral_dataset.csv"
)


def load_label_file(path, source_name):

    if not path.exists():

        print(
            f"[WARNING] Label file not found: {path}"
        )

        return pd.DataFrame(
            columns=[
                "sample_file",
                "label",
                "label_source"
            ]
        )

    df = pd.read_csv(path)

    required = {
        "sample_file",
        "label"
    }

    missing = required - set(df.columns)

    if missing:

        print(
            f"[WARNING] {source_name} is missing columns: "
            f"{missing}"
        )

        return pd.DataFrame(
            columns=[
                "sample_file",
                "label",
                "label_source"
            ]
        )

    keep = [
        "sample_file",
        "label"
    ]

    if "label_source" in df.columns:
        keep.append("label_source")

    df = df[keep].copy()

    if "label_source" not in df.columns:

        df["label_source"] = source_name

    df["label"] = pd.to_numeric(
        df["label"],
        errors="coerce"
    )

    return df


def flatten_sample(data):

    row = {}

    features = data.get(
        "features",
        {}
    )

    runtime = data.get(
        "runtime",
        {}
    )

    browser = data.get(
        "browser_observations",
        {}
    )

    for key, value in features.items():

        row[f"feature_{key}"] = value

    for key, value in runtime.items():

        row[f"runtime_{key}"] = value

    for key, value in browser.items():

        if isinstance(value, (dict, list)):
            continue

        row[f"browser_{key}"] = value

    row["input_url"] = data.get(
        "input_url"
    )

    row["final_url"] = data.get(
        "final_url"
    )

    row["title"] = data.get(
        "title"
    )

    row["navigation_error"] = data.get(
        "navigation_error"
    )

    return row


def main():

    print("=" * 70)
    print("LABELED BEHAVIORAL DATASET BUILDER")
    print("=" * 70)

    if not SAMPLE_DIR.exists():

        print(
            "\nERROR: Sample directory not found:"
        )

        print(SAMPLE_DIR)

        return

    sample_files = sorted(
        SAMPLE_DIR.glob("sample_*.json")
    )

    print(
        f"\nSamples found : {len(sample_files)}"
    )

    rows = []
    invalid = []

    for sample_file in sample_files:

        try:

            with open(
                sample_file,
                "r",
                encoding="utf-8"
            ) as file:

                data = json.load(file)

            row = flatten_sample(data)

            row["sample_file"] = (
                sample_file.name
            )

            rows.append(row)

        except Exception as error:

            invalid.append(
                (
                    sample_file.name,
                    str(error)
                )
            )

    dataset = pd.DataFrame(rows)

    print(
        f"Valid samples: {len(dataset)}"
    )

    if invalid:

        print(
            f"Invalid samples: {len(invalid)}"
        )

        for filename, error in invalid:

            print(
                f"  [INVALID] {filename}: {error}"
            )

    if dataset.empty:

        print(
            "\nERROR: No valid samples."
        )

        return

    # ------------------------------------------------------------
    # LOAD LABELS
    # ------------------------------------------------------------

    legit_label_tables = []

    for label_file in LEGIT_LABEL_FILES:

      legit_label_tables.append(
        load_label_file(
            label_file,
            "Legitimate sample labels"
        )
    )

    legit_labels = pd.concat(
    legit_label_tables,
    ignore_index=True
)

    phish_labels = load_label_file(
    PHISH_LABEL_FILE,
    "PhishTank verified online"
)

    labels = pd.concat(
    [
        legit_labels,
        phish_labels
    ],
    ignore_index=True
)

    # Remove duplicate sample labels.
    labels = labels.drop_duplicates(
        subset=["sample_file"],
        keep="last"
    )

    # ------------------------------------------------------------
    # MERGE LABELS
    # ------------------------------------------------------------

    dataset = dataset.merge(
        labels,
        on="sample_file",
        how="left"
    )

    # ------------------------------------------------------------
    # LABEL STATUS
    # ------------------------------------------------------------

    print("\n--- LABEL STATUS ---")

    missing_labels = dataset[
        dataset["label"].isna()
    ]

    print(
        f"Labeled samples : "
        f"{len(dataset) - len(missing_labels)}"
    )

    print(
        f"Unlabeled samples: "
        f"{len(missing_labels)}"
    )

    if not missing_labels.empty:

        print(
            "\nUnlabeled files:"
        )

        for filename in missing_labels[
            "sample_file"
        ]:

            print(
                f"  {filename}"
            )

    # ------------------------------------------------------------
    # CLASS DISTRIBUTION
    # ------------------------------------------------------------

    labeled = dataset[
        dataset["label"].notna()
    ].copy()

    if not labeled.empty:

        labeled["label"] = labeled[
            "label"
        ].astype(int)

        legitimate = (
            labeled["label"] == 0
        ).sum()

        phishing = (
            labeled["label"] == 1
        ).sum()

        print("\n--- CLASS DISTRIBUTION ---")

        print(
            f"Legitimate : {legitimate}"
        )

        print(
            f"Phishing   : {phishing}"
        )

    # ------------------------------------------------------------
    # MISSING VALUES
    # ------------------------------------------------------------

    missing = dataset.isna().sum()

    missing = missing[
        missing > 0
    ].sort_values(
        ascending=False
    )

    print("\n--- MISSING VALUES ---")

    if missing.empty:

        print("No missing values.")

    else:

        for column, count in missing.items():

            print(
                f"{column:<40} {count}"
            )

    # ------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    dataset.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print("\n" + "=" * 70)

    print(
        "DATASET BUILD COMPLETED"
    )

    print("=" * 70)

    print(
        f"Rows    : {len(dataset)}"
    )

    print(
        f"Columns : {len(dataset.columns)}"
    )

    print(
        f"Output  : {OUTPUT_FILE}"
    )


if __name__ == "__main__":

    main()