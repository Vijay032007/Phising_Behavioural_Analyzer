import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PHISHTANK_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "phishtank_online_valid.csv"
)

EXISTING_MANIFEST = (
    PROJECT_ROOT
    / "reports"
    / "phishing_sample_manifest.csv"
)

OUTPUT_MANIFEST = (
    PROJECT_ROOT
    / "reports"
    / "phishing_more_sample_manifest.csv"
)

NUMBER_TO_SELECT = 40


def main():

    print("=" * 70)
    print("PREPARE ADDITIONAL PHISHTANK SAMPLES")
    print("=" * 70)

    if not PHISHTANK_FILE.exists():

        print(
            f"\nERROR: PhishTank file not found:\n"
            f"{PHISHTANK_FILE}"
        )

        return

    df = pd.read_csv(
        PHISHTANK_FILE,
        low_memory=False
    )

    print(
        f"\nTotal PhishTank records: {len(df)}"
    )

    # Keep verified and currently online entries.
    df = df[
        (df["verified"].astype(str).str.lower() == "yes")
        &
        (df["online"].astype(str).str.lower() == "yes")
    ].copy()

    print(
        f"Verified + online records: {len(df)}"
    )

    # Remove URLs already used in the first 10 samples.
    existing_urls = set()

    if EXISTING_MANIFEST.exists():

        existing = pd.read_csv(
            EXISTING_MANIFEST
        )

        if "url" in existing.columns:

            existing_urls = set(
                existing["url"]
                .dropna()
                .astype(str)
                .str.strip()
            )

    print(
        f"Previously used URLs: "
        f"{len(existing_urls)}"
    )

    df["url"] = (
        df["url"]
        .astype(str)
        .str.strip()
    )

    df = df[
        ~df["url"].isin(existing_urls)
    ].copy()

    # Remove duplicate URLs.
    df = df.drop_duplicates(
        subset=["url"]
    )

    # Keep a reasonably varied set by using
    # different registrable domains where possible.
    df["domain"] = (
        df["url"]
        .str.extract(
            r"https?://([^/]+)",
            expand=False
        )
        .str.lower()
    )

    df = df.drop_duplicates(
        subset=["domain"]
    )

    selected = df.head(
        NUMBER_TO_SELECT
    ).copy()

    selected["label"] = 1

    selected["label_source"] = (
        "PhishTank verified online"
    )

    print(
        f"Additional samples selected: "
        f"{len(selected)}"
    )

    columns = [
        "phish_id",
        "url",
        "target",
        "label",
        "label_source"
    ]

    available_columns = [
        column
        for column in columns
        if column in selected.columns
    ]

    selected[
        available_columns
    ].to_csv(
        OUTPUT_MANIFEST,
        index=False,
        encoding="utf-8"
    )

    print(
        f"\nOutput:"
    )

    print(
        OUTPUT_MANIFEST
    )

    print("\nFirst selected URLs:")

    for url in selected["url"].head(10):

        print(
            f"  {url}"
        )

    print("\n" + "=" * 70)
    print("MANIFEST PREPARATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()