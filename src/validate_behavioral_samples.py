import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = PROJECT_ROOT / "reports" / "behavioral_samples"


def main():
    print("=" * 70)
    print("BEHAVIORAL SAMPLE VALIDATOR")
    print("=" * 70)

    if not SAMPLE_DIR.exists():
        print(f"ERROR: Directory not found:\n{SAMPLE_DIR}")
        return

    files = sorted(SAMPLE_DIR.glob("sample_*.json"))

    if not files:
        print("No sample_*.json files found.")
        return

    required_sections = [
        "input_url",
        "features",
        "runtime_features",
        "browser_observations",
    ]

    valid = 0

    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                sample = json.load(file)

            missing_sections = [
                section
                for section in required_sections
                if section not in sample
            ]

            if missing_sections:
                print(
                    f"[INVALID] {file_path.name} "
                    f"missing: {missing_sections}"
                )
                continue

            feature_count = len(sample.get("features", {}))
            runtime_count = len(
                sample.get("runtime_features", {})
            )

            print(
                f"[OK] {file_path.name:25} "
                f"features={feature_count:2d} "
                f"runtime={runtime_count:2d} "
                f"url={sample.get('input_url')}"
            )

            valid += 1

        except Exception as error:
            print(
                f"[ERROR] {file_path.name}: "
                f"{type(error).__name__}: {error}"
            )

    print("\n" + "=" * 70)
    print(f"Total samples : {len(files)}")
    print(f"Valid samples : {valid}")
    print(f"Invalid       : {len(files) - valid}")


if __name__ == "__main__":
    main()
