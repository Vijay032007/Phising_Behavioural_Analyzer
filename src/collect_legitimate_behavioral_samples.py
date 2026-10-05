import json
from pathlib import Path

from live_feature_extractor import extract_features


PROJECT_ROOT = Path(__file__).resolve().parent.parent

URL_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "legitimate_urls.txt"
)

SAMPLE_DIR = (
    PROJECT_ROOT
    / "reports"
    / "behavioral_samples"
)

LABEL_FILE = (
    PROJECT_ROOT
    / "reports"
    / "legitimate_behavioral_labels.csv"
)


def get_next_sample_number():

    existing = list(
        SAMPLE_DIR.glob("sample_*.json")
    )

    numbers = []

    for file in existing:

        try:
            numbers.append(
                int(file.stem.split("_")[1])
            )
        except Exception:
            pass

    if not numbers:
        return 1

    return max(numbers) + 1


def main():

    print("=" * 70)
    print("LEGITIMATE BEHAVIORAL COLLECTION")
    print("=" * 70)

    if not URL_FILE.exists():

        print(
            f"\nERROR: URL list not found:\n{URL_FILE}"
        )

        return

    with open(
        URL_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        urls = [
            line.strip()
            for line in file
            if line.strip()
            and not line.startswith("#")
        ]

    if not urls:

        print("\nNo URLs found.")
        return

    SAMPLE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    next_number = get_next_sample_number()

    label_rows = []

    print(
        f"\nURLs to process: {len(urls)}"
    )

    print(
        "\nPassive mode:"
    )

    print(
        "No clicks, form submissions, credentials, "
        "or downloads."
    )

    print()

    for index, url in enumerate(
        urls,
        start=1
    ):

        print("-" * 70)

        print(
            f"Processing {index}/{len(urls)}"
        )

        print(
            f"URL: {url}"
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

                    "input_url":
                        url,

                    "label":
                        0,

                    "label_source":
                        "Known legitimate website list"
                }
            )

            print(
                f"Saved: {sample_file.name}"
            )

            next_number += 1

        except Exception as error:

            print(
                f"[ERROR] {type(error).__name__}: "
                f"{error}"
            )

            print(
                "Continuing..."
            )

    if label_rows:

        import pandas as pd

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
            index=False
        )

    print("\n" + "=" * 70)

    print(
        "LEGITIMATE COLLECTION COMPLETED"
    )

    print("=" * 70)

    print(
        f"Successful samples: "
        f"{len(label_rows)}"
    )

    print(
        f"Label file: {LABEL_FILE}"
    )


if __name__ == "__main__":
    main()