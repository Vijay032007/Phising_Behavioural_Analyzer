from pathlib import Path
import sys
import json

import joblib
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "behavioral_random_forest.joblib"
)

FEATURE_LIST_FILE = (
    PROJECT_ROOT
    / "models"
    / "behavioral_feature_list.json"
)

REPORT_DIR = PROJECT_ROOT / "reports"

PREDICTION_FILE = (
    REPORT_DIR
    / "behavioral_prediction.json"
)


# ============================================================
# IMPORT LIVE FEATURE EXTRACTOR
# ============================================================

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent)
)

from live_feature_extractor import extract_features


# ============================================================
# BUILD MODEL FEATURE VECTOR
# ============================================================

def build_model_features(result, feature_columns):

    webpage_features = result.get(
        "features",
        {}
    )

    runtime_features = result.get(
        "runtime_features",
        {}

    )

    browser_observations = result.get(
        "browser_observations",
        {}
    )

    model_features = {}

    # --------------------------------------------------------
    # WEBPAGE / URL FEATURES
    # --------------------------------------------------------

    for feature_name, value in webpage_features.items():

        model_name = (
            "feature_"
            + feature_name
        )

        model_features[model_name] = value

    # --------------------------------------------------------
    # RUNTIME FEATURES
    # --------------------------------------------------------

    runtime_mapping = {

        "TotalRequests":
            "browser_total_requests",

        "HTMLSize":
            "browser_html_size",

        "DOMElements":
            "browser_dom_elements",
    }

    for runtime_name, model_name in runtime_mapping.items():

        if runtime_name in runtime_features:

            model_features[model_name] = (
                runtime_features[runtime_name]
            )

    # --------------------------------------------------------
    # FALLBACK TO BROWSER OBSERVATIONS
    # --------------------------------------------------------

    if (
        "browser_total_requests"
        not in model_features
    ):

        model_features[
            "browser_total_requests"
        ] = browser_observations.get(
            "total_requests",
            0
        )

    if (
        "browser_html_size"
        not in model_features
    ):

        model_features[
            "browser_html_size"
        ] = browser_observations.get(
            "html_size",
            0
        )

    if (
        "browser_dom_elements"
        not in model_features
    ):

        model_features[
            "browser_dom_elements"
        ] = browser_observations.get(
            "dom_elements",
            0
        )

    # --------------------------------------------------------
    # CHECK REQUIRED FEATURES
    # --------------------------------------------------------

    missing_features = []

    for feature_name in feature_columns:

        if feature_name not in model_features:

            missing_features.append(
                feature_name
            )

    return model_features, missing_features


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BEHAVIORAL PHISHING WEBSITE DETECTION")
    print("=" * 70)

    # --------------------------------------------------------
    # COMMAND-LINE URL
    # --------------------------------------------------------

    if len(sys.argv) < 2:

        print("\nUsage:")

        print(
            "python src/predict_behavioral_url.py "
            "https://example.com"
        )

        return

    url = sys.argv[1]

    print("\nInput URL:")
    print(url)

    # --------------------------------------------------------
    # CHECK URL FORMAT
    # --------------------------------------------------------

    if (
        url.startswith("[")
        and "](" in url
        and url.endswith(")")
    ):

        print("\nERROR: Markdown-formatted URL detected.")

        print(
            "Please provide the plain URL, for example:"
        )

        print(
            "https://example.com"
        )

        return

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    if not MODEL_FILE.exists():

        print("\nERROR: Model not found:")
        print(MODEL_FILE)

        return

    if not FEATURE_LIST_FILE.exists():

        print("\nERROR: Feature list not found:")
        print(FEATURE_LIST_FILE)

        return

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print("\nLoading behavioral model...")

    model = joblib.load(
        MODEL_FILE
    )

    print(
        "Model loaded successfully."
    )

    # --------------------------------------------------------
    # LOAD FEATURE LIST
    # --------------------------------------------------------

    with open(
        FEATURE_LIST_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        feature_columns = json.load(
            file
        )

    print(
        f"Expected behavioral features: "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # LIVE ANALYSIS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LIVE BEHAVIORAL ANALYSIS")
    print("=" * 70)

    print(
        "\nStarting Playwright analysis..."
    )

    print(
        "This is passive analysis."
    )

    print(
        "No forms will be submitted and "
        "no credentials will be entered."
    )

    try:

        result = extract_features(
            url
        )

    except Exception as error:

        print(
            "\nERROR during live feature extraction:"
        )

        print(error)

        return

    if not isinstance(
        result,
        dict
    ):

        print(
            "\nERROR: Extractor did not "
            "return a dictionary."
        )

        return

    print(
        "\nLive analysis completed."
    )

    # --------------------------------------------------------
    # BUILD FEATURES
    # --------------------------------------------------------

    model_features, missing_features = (
        build_model_features(
            result,
            feature_columns
        )
    )

    # --------------------------------------------------------
    # FEATURE CHECK
    # --------------------------------------------------------

    if missing_features:

        print(
            "\nERROR: Feature mismatch detected."
        )

        print(
            "\nMissing features:"
        )

        for feature in missing_features:

            print(
                f"  - {feature}"
            )

        return

    print(
        "\nAll required behavioral features "
        "are available."
    )

    # --------------------------------------------------------
    # CREATE DATAFRAME
    # --------------------------------------------------------

    X_live = pd.DataFrame(
        [[
            model_features[
                feature
            ]
            for feature in feature_columns
        ]],
        columns=feature_columns
    )

    X_live = X_live.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # --------------------------------------------------------
    # MISSING VALUE CHECK
    # --------------------------------------------------------

    missing_count = int(
        X_live.isna().sum().sum()
    )

    if missing_count > 0:

        print(
            f"\nWarning: "
            f"{missing_count} feature value(s) "
            f"are NaN."
        )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MACHINE LEARNING PREDICTION")
    print("=" * 70)

    prediction = model.predict(
        X_live
    )[0]

    probabilities = model.predict_proba(
        X_live
    )[0]

    legitimate_probability = float(
        probabilities[0]
    )

    phishing_probability = float(
        probabilities[1]
    )

    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    if prediction == 1:

        classification = "PHISHING"

        confidence = (
            phishing_probability
        )

    else:

        classification = "LEGITIMATE"

        confidence = (
            legitimate_probability
        )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    print("\nPrediction:")
    print(
        f"  {classification}"
    )

    print(
        "\nLegitimate probability:"
    )

    print(
        f"  {legitimate_probability * 100:.2f}%"
    )

    print(
        "\nPhishing probability:"
    )

    print(
        f"  {phishing_probability * 100:.2f}%"
    )

    print(
        "\nModel confidence:"
    )

    print(
        f"  {confidence * 100:.2f}%"
    )

    # --------------------------------------------------------
    # PAGE INFORMATION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PAGE / RUNTIME INFORMATION")
    print("=" * 70)

    print(
        f"\nFinal URL:"
        f"\n  {result.get('final_url')}"
    )

    print(
        f"\nFinal domain:"
        f"\n  {result.get('final_domain')}"
    )

    print(
        f"\nHTTP status:"
        f"\n  {result.get('http_status')}"
    )

    print(
        f"\nPage title:"
        f"\n  {result.get('page_title')}"
    )

    print(
        f"\nNavigation error:"
        f"\n  {result.get('navigation_error')}"
    )

    # --------------------------------------------------------
    # BEHAVIORAL OBSERVATIONS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BEHAVIORAL OBSERVATIONS")
    print("=" * 70)

    important_features = [

        "feature_URLLength",

        "feature_DomainLength",

        "feature_NoOfSubDomain",

        "feature_NoOfDegitsInURL",

        "feature_DegitRatioInURL",

        "feature_SpacialCharRatioInURL",

        "feature_HasObfuscation",

        "feature_NoOfiFrame",

        "feature_HasExternalFormSubmit",

        "feature_HasSubmitButton",

        "feature_HasHiddenFields",

        "feature_NoOfExternalRef",

        "feature_NoOfURLRedirect",

        "feature_NoOfSelfRedirect",

        "browser_total_requests",

        "browser_html_size",

        "browser_dom_elements",
    ]

    for feature in important_features:

        if feature in model_features:

            print(
                f"{feature:<40}: "
                f"{model_features[feature]}"
            )

    # --------------------------------------------------------
    # SAVE PREDICTION
    # --------------------------------------------------------

    prediction_data = {

        "input_url":
            url,

        "final_url":
            result.get(
                "final_url"
            ),

        "final_domain":
            result.get(
                "final_domain"
            ),

        "http_status":
            result.get(
                "http_status"
            ),

        "page_title":
            result.get(
                "page_title"
            ),

        "navigation_error":
            result.get(
                "navigation_error"
            ),

        "prediction":
            classification,

        "prediction_label":
            int(prediction),

        "legitimate_probability":
            legitimate_probability,

        "phishing_probability":
            phishing_probability,

        "confidence":
            float(confidence),

        "model":
            "Random Forest",

        "feature_count":
            len(feature_columns),

        "features":
            model_features,
    }

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        PREDICTION_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            prediction_data,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # FINISH
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PREDICTION SAVED")
    print("=" * 70)

    print(
        PREDICTION_FILE
    )

    print("\n" + "=" * 70)
    print("BEHAVIORAL DETECTION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":

    main()