import json
from pathlib import Path

import streamlit as st
import joblib

from src.live_feature_extractor import extract_features
from src.behavioral_explainer import generate_explanation


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "behavioral_random_forest.joblib"
FEATURE_LIST_PATH = BASE_DIR / "models" / "behavioral_feature_list.json"

REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)

    with open(FEATURE_LIST_PATH, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    return model, feature_columns


# ============================================================
# BUILD MODEL FEATURES
# ============================================================

def build_model_features(result, feature_columns):

    features = result.get("features", {})
    runtime = result.get("runtime_features", {})
    browser = result.get("browser_observations", {})

    values = {}

    # Dataset/model features
    for key, value in features.items():
        values[f"feature_{key}"] = value

    # Runtime mappings
    values["browser_total_requests"] = runtime.get(
        "TotalRequests",
        browser.get("total_requests", 0)
    )

    values["browser_html_size"] = runtime.get(
        "HTMLSize",
        browser.get("html_size", 0)
    )

    values["browser_dom_elements"] = runtime.get(
        "DOMElements",
        browser.get("dom_elements", 0)
    )

    # Build vector in exactly the same order used during training
    missing = []

    model_values = []

    for column in feature_columns:

        if column not in values:
            missing.append(column)
        else:
            model_values.append(values[column])

    if missing:
        raise ValueError(
            "Missing model features: " + ", ".join(missing)
        )

    return model_values


# ============================================================
# SAVE PREDICTION REPORT
# ============================================================

def save_prediction_report(prediction_data):

    output_path = REPORTS_DIR / "behavioral_prediction.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            prediction_data,
            f,
            indent=2,
            ensure_ascii=False
        )

    return output_path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Phishing Website Detector",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("🛡️ AI-Driven Phishing Website Detector")

st.write(
    "Behavioral analysis of websites using Machine Learning"
)

st.divider()


# ============================================================
# URL INPUT
# ============================================================

url = st.text_input(
    "Enter Website URL",
    placeholder="https://example.com"
)


analyze = st.button(
    "🔍 Analyze Website",
    type="primary"
)


# ============================================================
# ANALYSIS
# ============================================================

if analyze:

    if not url.strip():

        st.warning("Please enter a website URL.")

    else:

        url = url.strip()

        try:

            # ------------------------------------------------
            # Load model
            # ------------------------------------------------

            model, feature_columns = load_model()


            # ------------------------------------------------
            # Website behavioral analysis
            # ------------------------------------------------

            url = url.strip()

            if url and not url.startswith(("http://", "https://")):
                url = "https://" + url

            try:
                with st.spinner(
                    "Analyzing website behavior... Please wait."
                ):
                    result = extract_features(url)

            except ValueError as e:
                st.error(f"Unable to analyze URL: {e}")
                st.stop()

            except Exception:
                st.error(
                    "Unable to analyze this website. "
                    "Please check the URL and try again."
                )
                st.stop()


            # ------------------------------------------------
            # Build ML feature vector
            # ------------------------------------------------

            model_features = build_model_features(
                result,
                feature_columns
            )

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            prediction = int(
                model.predict([model_features])[0]
            )

            probabilities = model.predict_proba(
                [model_features]
            )[0]

            legitimate_probability = float(
                probabilities[0]
            )

            phishing_probability = float(
                probabilities[1]
            )

            if prediction == 1:
                classification = "PHISHING"
            else:
                classification = "LEGITIMATE"

            # ------------------------------------------------
            # Prepare prediction data for explainer
            # ------------------------------------------------

            prediction_data = {
                "input_url": result.get("input_url"),
                "final_url": result.get("final_url"),
                "final_domain": result.get("final_domain"),
                "http_status": result.get("http_status"),
                "page_title": result.get("page_title"),
                "navigation_error": result.get(
                    "navigation_error"
                ),

                "prediction": classification,

                "model": "Random Forest",

                "legitimate_probability":
                    legitimate_probability,

                "phishing_probability":
                    phishing_probability,

                "features": {
                    f"feature_{key}": value
                    for key, value
                    in result.get("features", {}).items()
                },

                "runtime_features":
                    result.get(
                        "runtime_features", {}
                    ),

                "browser_observations":
                    result.get(
                        "browser_observations", {}
                    )
            }

            # ------------------------------------------------
            # Generate explanation
            # ------------------------------------------------

            explanation = generate_explanation(
                prediction_data
            )

            prediction_data["explanation"] = explanation

            # ------------------------------------------------
            # Save report
            # ------------------------------------------------

            save_prediction_report(
                prediction_data
            )

            # =================================================
            # RESULT
            # =================================================

            st.divider()

            st.subheader("Analysis Result")

            col1, col2, col3 = st.columns(3)

            with col1:

                if classification == "PHISHING":

                    st.error(
                        "🚨 PHISHING WEBSITE"
                    )

                else:

                    st.success(
                        "✅ LEGITIMATE WEBSITE"
                    )

            with col2:

                st.metric(
                    "Legitimate Probability",
                    f"{legitimate_probability * 100:.2f}%"
                )

            with col3:

                st.metric(
                    "Phishing Probability",
                    f"{phishing_probability * 100:.2f}%"
                )


            # =================================================
            # WEBSITE INFORMATION
            # =================================================

            st.subheader("Website Information")

            info_col1, info_col2 = st.columns(2)

            with info_col1:

                st.write(
                    "**Input URL:**",
                    result.get("input_url")
                )

                st.write(
                    "**Final URL:**",
                    result.get("final_url")
                )

                st.write(
                    "**Domain:**",
                    result.get("final_domain")
                )

            with info_col2:

                st.write(
                    "**HTTP Status:**",
                    result.get("http_status")
                )

                st.write(
                    "**Page Title:**",
                    result.get("page_title")
                )

                st.write(
                    "**Navigation Error:**",
                    result.get("navigation_error")
                )


            # =================================================
            # BEHAVIORAL FEATURES
            # =================================================

            st.subheader("Behavioral Analysis")

            features = result.get(
                "features",
                {}
            )

            runtime = result.get(
                "runtime_features",
                {}
            )

            feature_col1, feature_col2 = st.columns(2)

            with feature_col1:

                st.write(
                    f"**URL Length:** "
                    f"{features.get('URLLength', 0)}"
                )

                st.write(
                    f"**Domain Length:** "
                    f"{features.get('DomainLength', 0)}"
                )

                st.write(
                    f"**Subdomains:** "
                    f"{features.get('NoOfSubDomain', 0)}"
                )

                st.write(
                    f"**Digits in URL:** "
                    f"{features.get('NoOfDegitsInURL', 0)}"
                )

                st.write(
                    f"**Digit Ratio:** "
                    f"{features.get('DegitRatioInURL', 0):.3f}"
                )

                st.write(
                    f"**Special Character Ratio:** "
                    f"{features.get('SpacialCharRatioInURL', 0):.3f}"
                )

            with feature_col2:

                st.write(
                    f"**Obfuscation:** "
                    f"{features.get('HasObfuscation', 0)}"
                )

                st.write(
                    f"**IFrames:** "
                    f"{features.get('NoOfiFrame', 0)}"
                )

                st.write(
                    f"**Hidden Fields:** "
                    f"{features.get('HasHiddenFields', 0)}"
                )

                st.write(
                    f"**External Form Submit:** "
                    f"{features.get('HasExternalFormSubmit', 0)}"
                )

                st.write(
                    f"**External References:** "
                    f"{features.get('NoOfExternalRef', 0)}"
                )

                st.write(
                    f"**URL Redirects:** "
                    f"{features.get('NoOfURLRedirect', 0)}"
                )


            # =================================================
            # RUNTIME INFORMATION
            # =================================================

            st.subheader("Runtime Analysis")

            runtime_col1, runtime_col2, runtime_col3 = st.columns(3)

            with runtime_col1:

                st.metric(
                    "Network Requests",
                    runtime.get(
                        "TotalRequests",
                        0
                    )
                )

            with runtime_col2:

                st.metric(
                    "HTML Size",
                    f"{runtime.get('HTMLSize', 0)} bytes"
                )

            with runtime_col3:

                st.metric(
                    "DOM Elements",
                    runtime.get(
                        "DOMElements",
                        0
                    )
                )


            # =================================================
            # SECURITY OBSERVATIONS
            # =================================================

            st.subheader(
                "Security Observations"
            )

            observations = explanation.get(
                "observations",
                []
            )

            if not observations:

                st.info(
                    "No security observations were generated."
                )

            else:

                for observation in observations:

                    severity = observation.get(
                        "severity",
                        "INFO"
                    )

                    name = observation.get(
                        "name",
                        "Observation"
                    )

                    description = observation.get(
                        "description",
                        ""
                    )

                    if severity == "HIGH":

                        st.error(
                            f"🔴 **{name}** — "
                            f"{description}"
                        )

                    elif severity == "MEDIUM":

                        st.warning(
                            f"🟠 **{name}** — "
                            f"{description}"
                        )

                    elif severity == "LOW":

                        st.info(
                            f"🟡 **{name}** — "
                            f"{description}"
                        )

                    else:

                        st.write(
                            f"ℹ️ **{name}** — "
                            f"{description}"
                        )


            # =================================================
            # IMPORTANT NOTE
            # =================================================

            st.divider()

            st.caption(
                "This analyzer performs passive behavioral "
                "analysis. It does not submit credentials, "
                "click website forms, or download files. "
                "Model probabilities are model outputs and "
                "should not be interpreted as guaranteed "
                "real-world probabilities."
            )

        except Exception as e:

            st.error(
                "An error occurred during analysis."
            )

            st.exception(e)