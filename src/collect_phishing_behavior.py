import json
from pathlib import Path

import pandas as pd

from live_feature_extractor import extract_features


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MANIFEST_FILE = (
    PROJECT_ROOT
    / "reports"
    / "phishing_sample_manifest.csv"
)

SAMPLE_DIR = (
    PROJECT_ROOT
    / "reports"
    / "behavioral_samples"
)

LABEL_FILE = (
    PROJECT_ROOT
    / "reports"
    / "phishing_behavioral_labels.csv"
)


def get_next_sample_number():

    existing = list(
        SAMPLE_DIR.glob("sample_*.json")
    )

    numbers = []

    for file in existing:

        try:

            number = int(
                file.stem.split("_")[1]
            )

            numbers.append(number)

        except Exception:

            continue

    if not numbers:
        return 1

    return max(numbers) + 1


def main():

    print("=" * 70)
    print("PHISHING BEHAVIORAL COLLECTION")
    print("=" * 70)

    if not MANIFEST_FILE.exists():

        print(
            "\nERROR: Phishing manifest not found:"
        )

        print(MANIFEST_FILE)

        return

    manifest = pd.read_csv(
        MANIFEST_FILE
    )

    if manifest.empty:

        print("\nNo phishing URLs found.")

        return

    SAMPLE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    next_number = get_next_sample_number()

    label_rows = []

    print(
        f"\nPhishing URLs to process: "
        f"{len(manifest)}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "The extractor is passive."
    )

    print(
        "No clicks, form submissions, "
        "credentials, or downloads are performed."
    )

    print()

    for index, row in manifest.iterrows():

        phish_id = row["phish_id"]
        url = row["url"]

        print("-" * 70)

        print(
            f"Processing {index + 1}/"
            f"{len(manifest)}"
        )

        print(
            f"PhishTank ID : {phish_id}"
        )

        print(
            f"URL          : {url}"
        )

        try:

            result = extract_features(
                url
            )

            sample_file = (
                SAMPLE_DIR
                / f"sample_{next_number:06d}.json"
            )

            with open(
                sample_file,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    result,
                    file,
                    indent=4,
                    ensure_ascii=False
                )

            label_rows.append(
                {
                    "sample_file":
                        sample_file.name,

                    "phish_id":
                        phish_id,

                    "input_url":
                        url,

                    "label":
                        1,

                    "label_source":
                        "PhishTank verified online"
                }
            )

            print(
                f"Saved: {sample_file.name}"
            )

            next_number += 1

        except Exception as error:

            print(
                "\nERROR processing URL:"
            )

            print(
                f"{type(error).__name__}: "
                f"{error}"
            )

            print(
                "Continuing with next URL."
            )

    if label_rows:

        new_labels = pd.DataFrame(
            label_rows
        )

        if LABEL_FILE.exists():

            old_labels = pd.read_csv(
                LABEL_FILE
            )

            combined = pd.concat(
                [
                    old_labels,
                    new_labels
                ],
                ignore_index=True
            )

            combined = combined.drop_duplicates(
                subset=["sample_file"]
            )

        else:

            combined = new_labels

        combined.to_csv(
            LABEL_FILE,
            index=False,
            encoding="utf-8"
        )

    print("\n" + "=" * 70)

    print(
        "PHISHING COLLECTION COMPLETED"
    )

    print("=" * 70)

    print(
        f"Successful samples : "
        f"{len(label_rows)}"
    )

    print(
        f"Label file         : "
        f"{LABEL_FILE}"
    )


if __name__ == "__main__":

    main()