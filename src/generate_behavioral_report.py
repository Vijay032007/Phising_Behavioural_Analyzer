"""
Behavioral Security Report Generator

Reads:
    reports/behavioral_prediction.json

Uses:
    src/behavioral_explainer.py

Creates:
    reports/behavioral_security_report.json
    reports/behavioral_security_report.txt
"""

import json
from pathlib import Path

from behavioral_explainer import generate_explanation


BASE_DIR = Path(__file__).resolve().parent.parent

PREDICTION_FILE = BASE_DIR / "reports" / "behavioral_prediction.json"
JSON_REPORT_FILE = BASE_DIR / "reports" / "behavioral_security_report.json"
TEXT_REPORT_FILE = BASE_DIR / "reports" / "behavioral_security_report.txt"


def load_prediction():
    """Load the latest prediction result."""

    if not PREDICTION_FILE.exists():
        raise FileNotFoundError(
            f"Prediction file not found:\n{PREDICTION_FILE}\n\n"
            "Run predict_behavioral_url.py first."
        )

    with open(PREDICTION_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def create_text_report(prediction, explanation):
    """Create a grouped, human-readable security report."""

    summary = explanation["summary"]
    statistics = explanation["statistics"]
    observations = explanation["observations"]

    prediction_label = prediction.get("prediction", "UNKNOWN")

    legitimate_probability = prediction.get(
        "legitimate_probability"
    )

    phishing_probability = prediction.get(
        "phishing_probability"
    )

    model_name = prediction.get(
        "model",
        "Machine Learning Model"
    )

    # ---------------------------------------------------------
    # Category order
    # ---------------------------------------------------------

    category_order = [
        "URL Analysis",
        "Domain Analysis",
        "Connection",
        "Obfuscation",
        "Page Structure",
        "Form Analysis",
        "Resources",
        "References",
        "Runtime",
    ]

    grouped = {}

    for item in observations:

        category = item.get(
            "category",
            "Other"
        )

        if category not in grouped:
            grouped[category] = []

        grouped[category].append(item)

    lines = []

    # ---------------------------------------------------------
    # Header
    # ---------------------------------------------------------

    lines.append("=" * 70)
    lines.append(
        "AI-DRIVEN PHISHING WEBSITE SECURITY REPORT"
    )
    lines.append("=" * 70)

    # ---------------------------------------------------------
    # Project
    # ---------------------------------------------------------

    lines.append("")
    lines.append("PROJECT")
    lines.append("-" * 70)

    lines.append(
        "AI-Driven Behavioral Analysis of Phishing Websites "
        "Using Machine Learning"
    )

    # ---------------------------------------------------------
    # Classification
    # ---------------------------------------------------------

    lines.append("")
    lines.append("CLASSIFICATION")
    lines.append("-" * 70)

    lines.append(
        f"Prediction : {prediction_label}"
    )

    lines.append(
        f"Model      : {model_name}"
    )

    if legitimate_probability is not None:

        lines.append(
            "Legitimate model probability : "
            f"{legitimate_probability * 100:.2f}%"
        )

    if phishing_probability is not None:

        lines.append(
            "Phishing model probability   : "
            f"{phishing_probability * 100:.2f}%"
        )

    # ---------------------------------------------------------
    # URL information
    # ---------------------------------------------------------

    lines.append("")
    lines.append("URL INFORMATION")
    lines.append("-" * 70)

    lines.append(
        f"Input URL   : {summary.get('input_url')}"
    )

    lines.append(
        f"Final URL   : {summary.get('final_url')}"
    )

    lines.append(
        f"Domain      : {summary.get('final_domain')}"
    )

    lines.append(
        f"HTTP Status : {summary.get('http_status')}"
    )

    lines.append(
        f"Page Title  : {summary.get('page_title')}"
    )

    # ---------------------------------------------------------
    # Behavioral analysis
    # ---------------------------------------------------------

    lines.append("")
    lines.append("OBSERVED SECURITY ANALYSIS")
    lines.append("=" * 70)

    for category in category_order:

        if category not in grouped:
            continue

        lines.append("")
        lines.append(category.upper())
        lines.append("-" * 70)

        for item in grouped[category]:

            severity = item.get(
                "severity",
                "INFO"
            )

            name = item.get(
                "name",
                "Unknown"
            )

            description = item.get(
                "description",
                ""
            )

            lines.append(
                f"[{severity}] {name}: {description}"
            )

    # ---------------------------------------------------------
    # Other categories, if any
    # ---------------------------------------------------------

    known_categories = set(category_order)

    for category, items in grouped.items():

        if category in known_categories:
            continue

        lines.append("")
        lines.append(category.upper())
        lines.append("-" * 70)

        for item in items:

            severity = item.get(
                "severity",
                "INFO"
            )

            name = item.get(
                "name",
                "Unknown"
            )

            description = item.get(
                "description",
                ""
            )

            lines.append(
                f"[{severity}] {name}: {description}"
            )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    lines.append("")
    lines.append("ANALYSIS SUMMARY")
    lines.append("-" * 70)

    lines.append(
        f"Total observations : "
        f"{statistics['total_observations']}"
    )

    lines.append(
        f"HIGH                : "
        f"{statistics['high']}"
    )

    lines.append(
        f"MEDIUM              : "
        f"{statistics['medium']}"
    )

    lines.append(
        f"LOW                 : "
        f"{statistics['low']}"
    )

    lines.append(
        f"INFO                : "
        f"{statistics['info']}"
    )

    # ---------------------------------------------------------
    # Interpretation
    # ---------------------------------------------------------

    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("-" * 70)

    if prediction_label.upper() == "PHISHING":

        lines.append(
            "The machine-learning model classified the "
            "analyzed URL as PHISHING."
        )

        lines.append(
            "The observations describe the URL structure, "
            "page structure and runtime behavior observed "
            "during the analysis."
        )

    elif prediction_label.upper() == "LEGITIMATE":

        lines.append(
            "The machine-learning model classified the "
            "analyzed URL as LEGITIMATE."
        )

        lines.append(
            "The observations describe the URL structure, "
            "page structure and runtime behavior observed "
            "during the analysis."
        )

    else:

        lines.append(
            "The machine-learning model returned an "
            "unknown classification."
        )

    # ---------------------------------------------------------
    # Important limitations
    # ---------------------------------------------------------

    lines.append("")
    lines.append("IMPORTANT NOTES")
    lines.append("-" * 70)

    lines.append(
        "The classification is a machine-learning prediction "
        "based on the extracted features and behavioral "
        "observations."
    )

    lines.append(
        "Individual observations should not be interpreted "
        "as proof of malicious activity by themselves."
    )

    lines.append(
        "The model probability is the model's estimated "
        "class probability and should not be interpreted "
        "as a guaranteed real-world probability."
    )

    if summary.get("http_status") == 403:

        lines.append(
            "The analyzed page returned HTTP 403. The observed "
            "content may represent an access-control or "
            "anti-bot/interstitial response rather than the "
            "underlying website content."
        )

    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)

    return "\n".join(lines)

def main():

    print("=" * 70)
    print("BEHAVIORAL SECURITY REPORT GENERATOR")
    print("=" * 70)

    print("\n[1] Loading prediction...")
    prediction = load_prediction()

    print("[2] Generating behavioral explanation...")
    explanation = generate_explanation(prediction)

    report = {
        "prediction": prediction,
        "explanation": explanation,
    }

    print("[3] Saving JSON report...")

    with open(
        JSON_REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(f"    Saved: {JSON_REPORT_FILE}")

    print("[4] Creating human-readable report...")

    text_report = create_text_report(
        prediction,
        explanation,
    )

    with open(
        TEXT_REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(text_report)

    print(f"    Saved: {TEXT_REPORT_FILE}")

    print("\n" + "=" * 70)
    print("REPORT GENERATED SUCCESSFULLY")
    print("=" * 70)

    print("\nFinding summary:")

    statistics = explanation["statistics"]

    print(f"  Total : {statistics['total_observations']}")
    print(f"  HIGH  : {statistics['high']}")
    print(f"  MEDIUM: {statistics['medium']}")
    print(f"  LOW   : {statistics['low']}")
    print(f"  INFO  : {statistics['info']}")

    print("\nFiles created:")
    print(f"  {JSON_REPORT_FILE}")
    print(f"  {TEXT_REPORT_FILE}")


if __name__ == "__main__":
    main()