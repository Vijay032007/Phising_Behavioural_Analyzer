"""
Behavioral Explanation Module

AI-Driven Behavioral Analysis of Phishing Websites Using Machine Learning

This module reads the feature structure produced by
predict_behavioral_url.py and converts it into human-readable
security observations.
"""

from typing import Dict, Any


def get_feature(result: Dict[str, Any], name: str, default=None):
    """
    Read a feature from the prediction JSON.

    Prediction JSON stores model features as:
        feature_URLLength
        feature_NoOfJS
        etc.
    """

    features = result.get("features", {})

    # Model feature format
    key = f"feature_{name}"

    if key in features:
        return features[key]

    # Runtime/browser feature format
    runtime_key_map = {
        "TotalRequests": "browser_total_requests",
        "HTMLSize": "browser_html_size",
        "DOMElements": "browser_dom_elements",
    }

    runtime_key = runtime_key_map.get(name)

    if runtime_key and runtime_key in features:
        return features[runtime_key]

    return default


def observation(
    category: str,
    name: str,
    value: Any,
    description: str,
    severity: str = "INFO",
):
    """Create a structured observation."""

    return {
        "category": category,
        "name": name,
        "value": value,
        "severity": severity,
        "description": description,
    }


def analyze_url_structure(result):

    observations = []

    url_length = get_feature(result, "URLLength")
    letters = get_feature(result, "NoOfLettersInURL")
    digit_count = get_feature(result, "NoOfDegitsInURL")
    digit_ratio = get_feature(result, "DegitRatioInURL")

    equals = get_feature(result, "NoOfEqualsInURL")
    question_marks = get_feature(result, "NoOfQMarkInURL")
    ampersand = get_feature(result, "NoOfAmpersandInURL")
    special_ratio = get_feature(
        result,
        "SpacialCharRatioInURL",
    )

    domain_length = get_feature(result, "DomainLength")
    subdomains = get_feature(result, "NoOfSubDomain")
    is_ip = get_feature(result, "IsDomainIP")
    https = get_feature(result, "IsHTTPS")

    if url_length is not None:

        severity = (
            "MEDIUM"
            if url_length >= 100
            else "INFO"
        )

        observations.append(
            observation(
                "URL Analysis",
                "URL Length",
                url_length,
                f"The URL contains {url_length} characters.",
                severity,
            )
        )

    if letters is not None:

        observations.append(
            observation(
                "URL Analysis",
                "Letters",
                letters,
                f"The URL contains {letters} alphabetic characters.",
            )
        )

    if digit_count is not None:

        severity = (
            "MEDIUM"
            if digit_count >= 8
            else "INFO"
        )

        observations.append(
            observation(
                "URL Analysis",
                "Digits",
                digit_count,
                f"The URL contains {digit_count} digit(s).",
                severity,
            )
        )

    if digit_ratio is not None:

        severity = (
            "MEDIUM"
            if digit_ratio >= 0.20
            else "INFO"
        )

        observations.append(
            observation(
                "URL Analysis",
                "Digit Ratio",
                digit_ratio,
                f"The digit ratio is {digit_ratio:.3f}.",
                severity,
            )
        )

    if equals is not None:

        observations.append(
            observation(
                "URL Analysis",
                "Equals Signs",
                equals,
                f"The URL contains {equals} '=' character(s).",
            )
        )

    if question_marks is not None:

        observations.append(
            observation(
                "URL Analysis",
                "Question Marks",
                question_marks,
                f"The URL contains {question_marks} question mark(s).",
            )
        )

    if ampersand is not None:

        observations.append(
            observation(
                "URL Analysis",
                "Ampersands",
                ampersand,
                f"The URL contains {ampersand} ampersand character(s).",
            )
        )

    if special_ratio is not None:

        severity = (
            "MEDIUM"
            if special_ratio >= 0.30
            else "INFO"
        )

        observations.append(
            observation(
                "URL Analysis",
                "Special Character Ratio",
                special_ratio,
                f"The special-character ratio is {special_ratio:.3f}.",
                severity,
            )
        )

    if domain_length is not None:

        observations.append(
            observation(
                "Domain Analysis",
                "Domain Length",
                domain_length,
                f"The domain contains {domain_length} characters.",
            )
        )

    if subdomains is not None:

        severity = (
            "MEDIUM"
            if subdomains >= 3
            else "INFO"
        )

        observations.append(
            observation(
                "Domain Analysis",
                "Subdomains",
                subdomains,
                f"The domain contains {subdomains} subdomain level(s).",
                severity,
            )
        )

    if is_ip is not None:

        if is_ip == 1:

            observations.append(
                observation(
                    "Domain Analysis",
                    "IP Address",
                    True,
                    "The URL uses an IP address instead of a conventional domain name.",
                    "HIGH",
                )
            )

        else:

            observations.append(
                observation(
                    "Domain Analysis",
                    "IP Address",
                    False,
                    "The URL uses a conventional domain name.",
                )
            )

    if https is not None:

        if https == 1:

            observations.append(
                observation(
                    "Connection",
                    "HTTPS",
                    True,
                    "HTTPS is enabled for the analyzed URL.",
                )
            )

        else:

            observations.append(
                observation(
                    "Connection",
                    "HTTPS",
                    False,
                    "HTTPS was not detected for the analyzed URL.",
                    "MEDIUM",
                )
            )

    return observations


def analyze_obfuscation(result):

    observations = []

    has_obfuscation = get_feature(
        result,
        "HasObfuscation",
    )

    obfuscated_chars = get_feature(
        result,
        "NoOfObfuscatedChar",
    )

    ratio = get_feature(
        result,
        "ObfuscationRatio",
    )

    if has_obfuscation is not None:

        if has_obfuscation == 1:

            observations.append(
                observation(
                    "Obfuscation",
                    "URL Obfuscation",
                    True,
                    "Potential URL character obfuscation was detected.",
                    "HIGH",
                )
            )

        else:

            observations.append(
                observation(
                    "Obfuscation",
                    "URL Obfuscation",
                    False,
                    "No URL obfuscation was detected by the analyzer.",
                )
            )

    if obfuscated_chars is not None:

        observations.append(
            observation(
                "Obfuscation",
                "Obfuscated Characters",
                obfuscated_chars,
                f"{obfuscated_chars} potentially obfuscated character(s) were detected.",
            )
        )

    if ratio is not None:

        observations.append(
            observation(
                "Obfuscation",
                "Obfuscation Ratio",
                ratio,
                f"The observed obfuscation ratio is {ratio:.3f}.",
            )
        )

    return observations


def analyze_page_structure(result):

    observations = []

    title = result.get("page_title")

    has_title = get_feature(
        result,
        "HasTitle",
    )

    has_description = get_feature(
        result,
        "HasDescription",
    )

    iframe_count = get_feature(
        result,
        "NoOfiFrame",
    )

    line_of_code = get_feature(
        result,
        "LineOfCode",
    )

    largest_line = get_feature(
        result,
        "LargestLineLength",
    )

    if title:

        observations.append(
            observation(
                "Page Structure",
                "Page Title",
                title,
                f"The page title is '{title}'.",
            )
        )

    elif has_title == 0:

        observations.append(
            observation(
                "Page Structure",
                "Page Title",
                None,
                "No page title was detected.",
                "MEDIUM",
            )
        )

    if has_description is not None:

        observations.append(
            observation(
                "Page Structure",
                "Description",
                bool(has_description),
                (
                    "A page description was detected."
                    if has_description
                    else "No page description was detected."
                ),
            )
        )

    if iframe_count is not None:

        severity = (
            "MEDIUM"
            if iframe_count > 0
            else "INFO"
        )

        observations.append(
            observation(
                "Page Structure",
                "Iframes",
                iframe_count,
                f"{iframe_count} iframe(s) were detected.",
                severity,
            )
        )

    if line_of_code is not None:

        observations.append(
            observation(
                "Page Structure",
                "Lines of Code",
                line_of_code,
                f"The analyzed HTML contains approximately {line_of_code} line(s).",
            )
        )

    if largest_line is not None:

        observations.append(
            observation(
                "Page Structure",
                "Largest Line",
                largest_line,
                f"The largest HTML line contains approximately {largest_line} characters.",
            )
        )

    return observations


def analyze_forms(result):

    observations = []

    hidden = get_feature(
        result,
        "HasHiddenFields",
    )

    password = get_feature(
        result,
        "HasPasswordField",
    )

    submit = get_feature(
        result,
        "HasSubmitButton",
    )

    external_submit = get_feature(
        result,
        "HasExternalFormSubmit",
    )

    if hidden is not None:

        observations.append(
            observation(
                "Form Analysis",
                "Hidden Fields",
                bool(hidden),
                (
                    "Hidden form fields were detected."
                    if hidden
                    else "No hidden form fields were detected."
                ),
                "MEDIUM" if hidden else "INFO",
            )
        )

    if password is not None:

        observations.append(
            observation(
                "Form Analysis",
                "Password Field",
                bool(password),
                (
                    "A password input field was detected."
                    if password
                    else "No password input field was detected."
                ),
                "MEDIUM" if password else "INFO",
            )
        )

    if submit is not None:

        observations.append(
            observation(
                "Form Analysis",
                "Submit Button",
                bool(submit),
                (
                    "A form submission button was detected."
                    if submit
                    else "No form submission button was detected."
                ),
            )
        )

    if external_submit is not None:

        observations.append(
            observation(
                "Form Analysis",
                "External Form Submission",
                bool(external_submit),
                (
                    "A form was observed submitting to an external destination."
                    if external_submit
                    else "No external form submission was detected."
                ),
                "HIGH" if external_submit else "INFO",
            )
        )

    return observations


def analyze_resources(result):

    observations = []

    js = get_feature(
        result,
        "NoOfJS",
    )

    css = get_feature(
        result,
        "NoOfCSS",
    )

    images = get_feature(
        result,
        "NoOfImage",
    )

    self_refs = get_feature(
        result,
        "NoOfSelfRef",
    )

    external_refs = get_feature(
        result,
        "NoOfExternalRef",
    )

    empty_refs = get_feature(
        result,
        "NoOfEmptyRef",
    )

    if js is not None:

        observations.append(
            observation(
                "Resources",
                "JavaScript",
                js,
                f"{js} JavaScript reference(s) were detected.",
            )
        )

    if css is not None:

        observations.append(
            observation(
                "Resources",
                "CSS",
                css,
                f"{css} CSS reference(s) were detected.",
            )
        )

    if images is not None:

        observations.append(
            observation(
                "Resources",
                "Images",
                images,
                f"{images} image reference(s) were detected.",
            )
        )

    if self_refs is not None:

        observations.append(
            observation(
                "References",
                "Self References",
                self_refs,
                f"{self_refs} self-reference(s) were detected.",
            )
        )

    if external_refs is not None:

        severity = (
            "MEDIUM"
            if external_refs >= 20
            else "INFO"
        )

        observations.append(
            observation(
                "References",
                "External References",
                external_refs,
                f"{external_refs} external reference(s) were detected.",
                severity,
            )
        )

    if empty_refs is not None:

        observations.append(
            observation(
                "References",
                "Empty References",
                empty_refs,
                f"{empty_refs} empty reference(s) were detected.",
                "LOW" if empty_refs > 0 else "INFO",
            )
        )

    return observations


def analyze_runtime(result):

    observations = []

    status = result.get("http_status")

    total_requests = get_feature(
        result,
        "TotalRequests",
    )

    html_size = get_feature(
        result,
        "HTMLSize",
    )

    dom_elements = get_feature(
        result,
        "DOMElements",
    )

    navigation_error = result.get(
        "navigation_error"
    )

    if status is not None:

        observations.append(
            observation(
                "Runtime",
                "HTTP Status",
                status,
                f"The page returned HTTP status {status}.",
                "MEDIUM" if status >= 400 else "INFO",
            )
        )

    if navigation_error:

        observations.append(
            observation(
                "Runtime",
                "Navigation Error",
                navigation_error,
                "A navigation error occurred while loading the page.",
                "MEDIUM",
            )
        )

    else:

        observations.append(
            observation(
                "Runtime",
                "Navigation",
                "Success",
                "No navigation error was reported.",
            )
        )

    if total_requests is not None:

        observations.append(
            observation(
                "Runtime",
                "Network Requests",
                total_requests,
                f"{total_requests} network request(s) were observed.",
            )
        )

    if html_size is not None:

        observations.append(
            observation(
                "Runtime",
                "HTML Size",
                html_size,
                f"The downloaded HTML size was {html_size} bytes.",
            )
        )

    if dom_elements is not None:

        observations.append(
            observation(
                "Runtime",
                "DOM Elements",
                dom_elements,
                f"The page contained approximately {dom_elements} DOM element(s).",
            )
        )

    return observations


def generate_explanation(
    result: Dict[str, Any]
) -> Dict[str, Any]:

    observations = []

    observations.extend(
        analyze_url_structure(result)
    )

    observations.extend(
        analyze_obfuscation(result)
    )

    observations.extend(
        analyze_page_structure(result)
    )

    observations.extend(
        analyze_forms(result)
    )

    observations.extend(
        analyze_resources(result)
    )

    observations.extend(
        analyze_runtime(result)
    )

    severity_order = {
        "HIGH": 0,
        "MEDIUM": 1,
        "LOW": 2,
        "INFO": 3,
    }

    observations.sort(
        key=lambda x: severity_order.get(
            x["severity"],
            4,
        )
    )

    high = sum(
        1
        for x in observations
        if x["severity"] == "HIGH"
    )

    medium = sum(
        1
        for x in observations
        if x["severity"] == "MEDIUM"
    )

    low = sum(
        1
        for x in observations
        if x["severity"] == "LOW"
    )

    info = sum(
        1
        for x in observations
        if x["severity"] == "INFO"
    )

    return {
        "summary": {
            "input_url": result.get("input_url"),
            "final_url": result.get("final_url"),
            "final_domain": result.get("final_domain"),
            "http_status": result.get("http_status"),
            "page_title": result.get("page_title"),
        },

        "observations": observations,

        "statistics": {
            "total_observations": len(observations),
            "high": high,
            "medium": medium,
            "low": low,
            "info": info,
        },
    }


if __name__ == "__main__":

    print(
        "Behavioral Explainer Module"
    )

    print(
        "Import generate_explanation() "
        "to use this module."
    )