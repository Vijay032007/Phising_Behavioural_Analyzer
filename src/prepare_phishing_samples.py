import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "phishtank_online_valid.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "phishing_sample_manifest.csv"
)


def main():

    print("=" * 70)
    print("PHISHTANK PHISHING SAMPLE PREPARATION")
    print("=" * 70)

    if not INPUT_FILE.exists():

        print("\nERROR: PhishTank CSV not found:")
        print(INPUT_FILE)
        return

    df = pd.read_csv(INPUT_FILE)

    print(
        f"\nTotal PhishTank records : {len(df)}"
    )

    # Keep only records explicitly marked
    # verified and online.
    filtered = df[
        (df["verified"] == "yes")
        &
        (df["online"] == "yes")
    ].copy()

    # Remove duplicate URLs.
    filtered = filtered.drop_duplicates(
        subset=["url"]
    )

    # Take a small initial batch.
    samples = filtered.head(10).copy()

    manifest = pd.DataFrame(
        {
            "phish_id": samples["phish_id"],
            "url": samples["url"],
            "phish_detail_url":
                samples["phish_detail_url"],
            "submission_time":
                samples["submission_time"],
            "verification_time":
                samples["verification_time"],
            "target": samples["target"],
            "label": 1,
            "label_source":
                "PhishTank verified online"
        }
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    manifest.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print(
        f"\nVerified online records : "
        f"{len(filtered)}"
    )

    print(
        f"Initial samples selected : "
        f"{len(manifest)}"
    )

    print("\nManifest saved to:")
    print(OUTPUT_FILE)

    print(
        "\nThese URLs have NOT been opened."
    )


if __name__ == "__main__":
    main()