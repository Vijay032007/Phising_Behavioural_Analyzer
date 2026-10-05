import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SAMPLE_DIR = (
    PROJECT_ROOT
    / "reports"
    / "behavioral_samples"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "behavioral_labels.csv"
)


def main():

    print("=" * 70)
    print("BEHAVIORAL SAMPLE LABELING")
    print("=" * 70)

    sample_files = sorted(
        SAMPLE_DIR.glob("sample_*.json")
    )

    if not sample_files:

        print(
            "\nNo behavioral samples found."
        )

        return

    rows = []

    for sample_file in sample_files:

        try:

            with open(
                sample_file,
                "r",
                encoding="utf-8"
            ) as file:

                sample = json.load(file)

            url = sample.get(
                "input_url",
                ""
            )

            print("\n" + "-" * 70)

            print(
                f"Sample : {sample_file.name}"
            )

            print(
                f"URL    : {url}"
            )

            print()

            print(
                "Enter label:"
            )

            print(
                "0 = Legitimate"
            )

            print(
                "1 = Phishing"
            )

            print(
                "s = Skip"
            )

            while True:

                label = input(
                    "Label: "
                ).strip().lower()

                if label in (
                    "0",
                    "1",
                    "s"
                ):

                    break

                print(
                    "Please enter 0, 1, or s."
                )

            if label == "s":

                print(
                    "Skipped."
                )

                continue

            if label == "1":

                source = input(
                    "Phishing label source "
                    "(example: PhishTank): "
                ).strip()

            else:

                source = input(
                    "Legitimate label source "
                    "(example: Official website): "
                ).strip()

            rows.append(
                {
                    "sample_file":
                        sample_file.name,

                    "input_url":
                        url,

                    "label":
                        int(label),

                    "label_source":
                        source
                }
            )

        except Exception as error:

            print(
                f"\nERROR reading "
                f"{sample_file.name}:"
            )

            print(
                f"{type(error).__name__}: "
                f"{error}"
            )

    if not rows:

        print(
            "\nNo samples were labeled."
        )

        return

    df = pd.DataFrame(rows)

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print("\n" + "=" * 70)

    print(
        "LABELING COMPLETED"
    )

    print("=" * 70)

    print(
        f"Samples labeled : {len(df)}"
    )

    print(
        f"Legitimate      : "
        f"{(df['label'] == 0).sum()}"
    )

    print(
        f"Phishing        : "
        f"{(df['label'] == 1).sum()}"
    )

    print(
        f"Output          : {OUTPUT_FILE}"
    )


if __name__ == "__main__":

    main()