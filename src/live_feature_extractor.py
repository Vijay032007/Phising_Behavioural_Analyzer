import json

import re

from pathlib import Path

from urllib.parse import urlparse, urljoin



from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError





# ============================================================

# Configuration

# ============================================================



NAVIGATION_TIMEOUT = 30_000

WAIT_TIME = 2_000



PROJECT_ROOT = Path(__file__).resolve().parent.parent

REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)



SAMPLE_DIR = REPORT_DIR / "behavioral_samples"

SAMPLE_DIR.mkdir(parents=True, exist_ok=True)



OUTPUT_FILE = REPORT_DIR / "live_features.json"





# ============================================================

# Utility functions

# ============================================================



def get_domain(url):

    try:

        return urlparse(url).netloc.lower()

    except Exception:

        return ""





def is_ip_address(domain):

    return bool(

        re.fullmatch(

            r"\d{1,3}(\\.\d{1,3}){3}",

            domain

        )

    )





def count_letters(text):

    return sum(c.isalpha() for c in text)





def count_digits(text):

    return sum(c.isdigit() for c in text)





def count_special_characters(text):

    return sum(not c.isalnum() for c in text)





def safe_ratio(a, b):

    if b == 0:

        return 0.0

    return a / b





def count_subdomains(domain):

    parts = domain.split(".")



    if len(parts) <= 2:

        return 0



    return max(0, len(parts) - 2)





# ============================================================

# URL feature extraction

# ============================================================



def calculate_url_features(url):



    url_length = len(url)



    letters = count_letters(url)

    digits = count_digits(url)

    special = count_special_characters(url)



    other_special = sum(

        1

        for c in url

        if not c.isalnum()

        and c not in ".-/?=&:_#%"

    )



    return {

        "URLLength": url_length,



        "NoOfLettersInURL": letters,



        "LetterRatioInURL": safe_ratio(

            letters,

            url_length

        ),



        "NoOfDegitsInURL": digits,



        "DegitRatioInURL": safe_ratio(

            digits,

            url_length

        ),



        "NoOfEqualsInURL": url.count("="),



        "NoOfQMarkInURL": url.count("?"),



        "NoOfAmpersandInURL": url.count("&"),



        "NoOfOtherSpecialCharsInURL":

            other_special,



        "SpacialCharRatioInURL":

            safe_ratio(

                special,

                url_length

            )

    }





def calculate_obfuscation_features(url):



    encoded_chars = re.findall(

        r"%[0-9a-fA-F]{2}",

        url

    )



    hex_sequences = re.findall(

        r"[0-9a-fA-F]{8,}",

        url

    )



    at_count = url.count("@")



    obfuscated_count = (

        len(encoded_chars)

        + at_count

        + len(hex_sequences)

    )



    return {

        "HasObfuscation":

            int(obfuscated_count > 0),



        "NoOfObfuscatedChar":

            obfuscated_count,



        "ObfuscationRatio":

            safe_ratio(

                obfuscated_count,

                len(url)

            )

    }





# ============================================================

# Main extractor

# ============================================================



def extract_features(url):



    parsed = urlparse(url)



    if parsed.scheme not in ("http", "https"):

        raise ValueError(

            "Only HTTP and HTTPS URLs are supported."

        )



    if not parsed.netloc:

        raise ValueError(

            "Invalid URL."

        )



    original_domain = parsed.netloc.lower()



    features = {}



    features.update(

        calculate_url_features(url)

    )



    features["DomainLength"] = len(

        original_domain

    )



    features["IsDomainIP"] = int(

        is_ip_address(original_domain)

    )



    features["NoOfSubDomain"] = (

        count_subdomains(original_domain)

    )



    features["IsHTTPS"] = int(

        parsed.scheme == "https"

    )



    features.update(

        calculate_obfuscation_features(url)

    )



    # ========================================================

    # Runtime information

    # ========================================================



    requests = []

    failed_requests = []

    redirects = []

    unique_domains = set()



    final_url = url

    final_domain = original_domain



    html = ""

    title = ""



    http_status = None



    navigation_success = False

    navigation_timeout = False

    navigation_error = None



    # ========================================================

    # DOM variables

    # ========================================================



    dom_elements = 0

    forms = 0

    password_fields = 0

    hidden_elements = 0

    iframes = 0

    images = 0

    css_files = 0

    js_files = 0

    inline_scripts = 0



    line_count = 0

    largest_line_length = 0



    description = 0

    favicon = 0



    has_social = False

    has_copyright = False



    external_form_submit = 0

    submit_button = 0



    form_details = []



    self_refs = 0

    empty_refs = 0

    external_refs = 0



    # ========================================================

    # Browser

    # ========================================================



    with sync_playwright() as p:



        browser = None

        context = None



        try:



            browser = p.chromium.launch(

                headless=True

            )



            context = browser.new_context(

                ignore_https_errors=True,

                accept_downloads=False,

                service_workers="block"

            )



            page = context.new_page()



            page.set_default_timeout(

                10_000

            )



            # ------------------------------------------------

            # Request monitoring

            # ------------------------------------------------



            def handle_request(request):



                request_url = request.url



                requests.append(

                    {

                        "url": request_url,

                        "method": request.method,

                        "resource_type":

                            request.resource_type

                    }

                )



                domain = get_domain(

                    request_url

                )



                if domain:

                    unique_domains.add(

                        domain

                    )



            # ------------------------------------------------

            # Failed requests

            # ------------------------------------------------



            def handle_request_failed(request):



                failed_requests.append(

                    {

                        "url": request.url,

                        "method": request.method,

                        "resource_type":

                            request.resource_type,

                        "failure":

                            request.failure

                    }

                )



            # ------------------------------------------------

            # Response / redirect monitoring

            # ------------------------------------------------



            def handle_response(response):



                request = response.request



                redirected_from = (

                    request.redirected_from

                )



                if redirected_from:



                    redirects.append(

                        {

                            "from":

                                redirected_from.url,



                            "to":

                                response.url,



                            "status":

                                response.status

                        }

                    )



            page.on(

                "request",

                handle_request

            )



            page.on(

                "requestfailed",

                handle_request_failed

            )



            page.on(

                "response",

                handle_response

            )



            # ------------------------------------------------

            # Navigation

            # ------------------------------------------------



            try:



                response = page.goto(

                    url,

                    wait_until="domcontentloaded",

                    timeout=NAVIGATION_TIMEOUT

                )



                navigation_success = True



                if response:

                    http_status = response.status



            except PlaywrightTimeoutError:



                navigation_timeout = True



                navigation_error = (

                    "Navigation timed out."

                )



            except Exception as error:



                navigation_error = (

                    f"{type(error).__name__}: "

                    f"{error}"

                )



            # Allow short JavaScript execution time.

            try:



                page.wait_for_timeout(

                    WAIT_TIME

                )



            except Exception:

                pass



            # ------------------------------------------------

            # Final URL

            # ------------------------------------------------



            final_url = page.url



            final_domain = get_domain(

                final_url

            )



            # ------------------------------------------------

            # HTML

            # ------------------------------------------------



            try:

                html = page.content()

            except Exception:

                html = ""



            try:

                title = page.title()

            except Exception:

                title = ""



            # ------------------------------------------------

            # DOM analysis

            # ------------------------------------------------



            try:

                dom_elements = page.locator(

                    "*"

                ).count()

            except Exception:

                pass



            try:

                forms = page.locator(

                    "form"

                ).count()

            except Exception:

                pass



            try:

                password_fields = page.locator(

                    "input[type='password']"

                ).count()

            except Exception:

                pass



            try:



                hidden_selector = (

                    "[hidden], "

                    "[style*='display:none'], "

                    "[style*='display: none'], "

                    "[style*='visibility:hidden'], "

                    "[style*='visibility: hidden']"

                )



                hidden_elements = page.locator(

                    hidden_selector

                ).count()



            except Exception:

                pass



            try:

                iframes = page.locator(

                    "iframe"

                ).count()

            except Exception:

                pass



            try:

                images = page.locator(

                    "img"

                ).count()

            except Exception:

                pass



            try:

                css_files = page.locator(

                    "link[rel='stylesheet']"

                ).count()

            except Exception:

                pass



            try:

                js_files = page.locator(

                    "script"

                ).count()

            except Exception:

                pass



            try:

                inline_scripts = page.locator(

                    "script:not([src])"

                ).count()

            except Exception:

                pass



            # ------------------------------------------------

            # HTML statistics

            # ------------------------------------------------



            lines = html.splitlines()



            line_count = len(lines)



            if lines:



                largest_line_length = max(

                    len(line)

                    for line in lines

                )



            # ------------------------------------------------

            # Meta description

            # ------------------------------------------------



            try:



                description = page.locator(

                    "meta[name='description']"

                ).count()



            except Exception:

                pass



            # ------------------------------------------------

            # Favicon

            # ------------------------------------------------



            try:



                favicon = page.locator(

                    "link[rel\~='icon']"

                ).count()



            except Exception:

                pass



            # ------------------------------------------------

            # Forms

            # ------------------------------------------------



            for i in range(forms):



                try:



                    form = page.locator(

                        "form"

                    ).nth(i)



                    action = form.get_attribute(

                        "action"

                    )



                    method = (

                        form.get_attribute(

                            "method"

                        )

                        or "GET"

                    )



                    action = (

                        action

                        or final_url

                    )



                    absolute_action = urljoin(

                        final_url,

                        action

                    )



                    action_domain = get_domain(

                        absolute_action

                    )



                    external = (

                        bool(action_domain)

                        and

                        action_domain != final_domain

                    )



                    if external:

                        external_form_submit = 1



                    form_details.append(

                        {

                            "method":

                                method.upper(),



                            "action":

                                absolute_action,



                            "external":

                                external

                        }

                    )



                except Exception:

                    continue



            # ------------------------------------------------

            # Submit buttons

            # ------------------------------------------------



            try:



                submit_button = page.locator(

                    "button[type='submit'], "

                    "input[type='submit']"

                ).count()



            except Exception:

                pass



            # ------------------------------------------------

            # Social indicators

            # ------------------------------------------------



            page_text = html.lower()



            social_patterns = [

                "facebook.com",

                "instagram.com",

                "twitter.com",

                "x.com",

                "linkedin.com",

                "youtube.com"

            ]



            has_social = any(

                pattern in page_text

                for pattern in social_patterns

            )



            # ------------------------------------------------

            # Copyright

            # ------------------------------------------------



            has_copyright = (

                "copyright" in page_text

                or "©" in html

            )



            # ------------------------------------------------

            # Link analysis

            # ------------------------------------------------



            try:



                links = page.locator(

                    "a[href]"

                )



                link_count = links.count()



                for i in range(

                    link_count

                ):



                    try:



                        href = links.nth(

                            i

                        ).get_attribute(

                            "href"

                        )



                        if not href:



                            empty_refs += 1

                            continue



                        absolute = urljoin(

                            final_url,

                            href

                        )



                        link_domain = get_domain(

                            absolute

                        )



                        if not link_domain:

                            continue



                        if (

                            link_domain

                            == final_domain

                        ):



                            self_refs += 1



                        else:



                            external_refs += 1



                    except Exception:

                        continue



            except Exception:

                pass



        finally:



            if context:



                try:

                    context.close()

                except Exception:

                    pass



            if browser:



                try:

                    browser.close()

                except Exception:

                    pass



    # ========================================================

    # Feature vector

    # ========================================================



    features["LineOfCode"] = line_count



    features["LargestLineLength"] = (

        largest_line_length

    )



    features["HasTitle"] = int(

        bool(title.strip())

    )



    features["HasFavicon"] = int(

        favicon > 0

    )



    features["HasDescription"] = int(

        description > 0

    )



    features["NoOfiFrame"] = iframes



    features["HasExternalFormSubmit"] = (

        external_form_submit

    )



    features["HasSubmitButton"] = int(

        submit_button > 0

    )



    features["HasHiddenFields"] = int(

        hidden_elements > 0

    )



    features["HasPasswordField"] = int(

        password_fields > 0

    )



    features["HasSocialNet"] = int(

        has_social

    )



    features["HasCopyrightInfo"] = int(

        has_copyright

    )



    features["NoOfImage"] = images



    features["NoOfCSS"] = css_files



    features["NoOfJS"] = js_files



    features["NoOfSelfRef"] = self_refs



    features["NoOfEmptyRef"] = empty_refs



    features["NoOfExternalRef"] = external_refs



    features["NoOfURLRedirect"] = len(

        redirects

    )



    self_redirects = 0



    for redirect in redirects:



        from_domain = get_domain(

            redirect["from"]

        )



        to_domain = get_domain(

            redirect["to"]

        )



        if (

            from_domain

            and to_domain

            and from_domain == to_domain

        ):



            self_redirects += 1



    features["NoOfSelfRedirect"] = (

        self_redirects

    )



    # ========================================================

    # Runtime behavioral features

    # ========================================================



    runtime_features = {



        "PageLoadSuccess":

            int(navigation_success),



        "NavigationTimeout":

            int(navigation_timeout),



        "HTTPStatus":

            http_status,



        "TotalRequests":

            len(requests),



        "FailedRequests":

            len(failed_requests),



        "UniqueDomains":

            len(unique_domains),



        "ExternalDomainCount":

            len(

                [

                    domain

                    for domain in unique_domains

                    if domain != final_domain

                ]

            ),



        "CrossDomainRedirect":

            int(

                original_domain

                != final_domain

            ),



        "RedirectCount":

            len(redirects),



        "HTMLSize":

            len(html),



        "DOMElements":

            dom_elements,



        "InlineScripts":

            inline_scripts

    }



    # ========================================================

    # Possible challenge/interstitial detection

    # ========================================================



    lower_html = html.lower()

    lower_title = title.lower()



    challenge_terms = [

        "checking your browser",

        "verify you are human",

        "verify you’re human",

        "captcha",

        "access denied",

        "security check",

        "checking your connection",

        "robot check",

        "automated access"

    ]



    challenge_detected = any(

        term in lower_html

        or term in lower_title

        for term in challenge_terms

    )



    runtime_features[

        "PossibleChallengeOrInterstitial"

    ] = int(challenge_detected)



    # ========================================================

    # Dataset-specific features

    # ========================================================



    dataset_specific_features = {



        "URLSimilarityIndex": None,



        "CharContinuationRate": None,



        "TLDLegitimateProb": None,



        "URLCharProb": None,



        "TLDLength": None,



        "DomainTitleMatchScore": None,



        "URLTitleMatchScore": None,



        "IsResponsive": None,



        "Robots": None,



        "NoOfPopup": None,



        "Bank": None,



        "Pay": None,



        "Crypto": None

    }



    # ========================================================

    # Result

    # ========================================================



    result = {



        "input_url":

            url,



        "original_domain":

            original_domain,



        "final_url":

            final_url,



        "final_domain":

            final_domain,



        "http_status":

            http_status,



        "page_title":

            title,



        "navigation_error":

            navigation_error,



        "features":

            features,



        "runtime_features":

            runtime_features,



        "dataset_specific_features":

            dataset_specific_features,



        "browser_observations": {



            "total_requests":

                len(requests),



            "failed_requests":

                failed_requests,



            "unique_domains":

                sorted(

                    unique_domains

                ),



            "external_domains":

                sorted(

                    domain

                    for domain in unique_domains

                    if domain != final_domain

                ),



            "redirects":

                redirects,



            "form_details":

                form_details,



            "html_size":

                len(html),



            "dom_elements":

                dom_elements

        },



        "prediction_ready":

            False

    }



    return result





# ============================================================

# ============================================================

# Main

# ============================================================


if __name__ == "__main__":

    print("=" * 70)
    print("LIVE FEATURE EXTRACTOR")
    print("=" * 70)

    target_url = input(
        "\nEnter website URL: "
    ).strip()

    try:

        result = extract_features(
            target_url
        )

        # --------------------------------------------------------
        # Save permanent behavioral sample
        # --------------------------------------------------------

        existing_samples = list(
            SAMPLE_DIR.glob("sample_*.json")
        )

        sample_number = len(existing_samples) + 1

        sample_file = (
            SAMPLE_DIR
            / f"sample_{sample_number:06d}.json"
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

        # Also keep the latest result here for easy inspection.

        with open(
            OUTPUT_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                result,
                file,
                indent=4,
                ensure_ascii=False
            )

        print(
            "\nPermanent sample saved to:"
        )

        print(
            sample_file
        )

        print(
            "\nLatest result saved to:"
        )

        print(
            OUTPUT_FILE
        )

        print(
            "\nFeature extraction completed."
        )

        print(
            "\nSaved feature report to:"
        )

        print(
            OUTPUT_FILE
        )

        print("\n--- RUNTIME FEATURES ---")

        for name, value in result[
            "runtime_features"
        ].items():

            print(
                f"{name:35} : {value}"
            )

        print("\n--- EXTRACTED FEATURES ---")

        for name, value in result[
            "features"
        ].items():

            print(
                f"{name:35} : {value}"
            )

        print(
            "\nPrediction Ready : "
            f"{result['prediction_ready']}"
        )

    except Exception as error:

        print("\nERROR:")

        print(
            f"{type(error).__name__}: {error}"
        )
