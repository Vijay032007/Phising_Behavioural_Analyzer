import json
from pathlib import Path
from urllib.parse import urlparse, urljoin

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

NAVIGATION_TIMEOUT = 30_000
ANALYSIS_WAIT_TIME = 2_000

REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_FILE = REPORT_DIR / "behavioral_analysis.json"


# ---------------------------------------------------------
# URL validation
# ---------------------------------------------------------

def validate_url(url):
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            "Only HTTP and HTTPS URLs are allowed."
        )

    if not parsed.netloc:
        raise ValueError(
            "Invalid URL. Please enter a complete URL."
        )

    return parsed


# ---------------------------------------------------------
# Domain helper
# ---------------------------------------------------------

def get_domain(url):
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def is_external_domain(target_url, base_domain):
    target_domain = get_domain(target_url)

    if not target_domain:
        return False

    return target_domain != base_domain


# ---------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------

def analyze_website(url):

    print("\n" + "=" * 70)
    print("        PHISHING WEBSITE BEHAVIORAL ANALYZER")
    print("=" * 70)

    parsed = validate_url(url)

    original_domain = parsed.netloc.lower()

    print("\nTarget URL:")
    print(url)

    print("\nStarting isolated browser...")

    analysis = {
        "input_url": url,
        "original_domain": original_domain,
        "final_url": "",
        "final_domain": "",
        "http_status": None,
        "page_title": "",
        "html_size": 0,

        # DOM features
        "dom_elements": 0,
        "javascript_files": 0,
        "css_files": 0,
        "images": 0,
        "iframes": 0,
        "forms": 0,
        "input_fields": 0,
        "password_fields": 0,
        "hidden_elements": 0,

        # Form behavior
        "external_form_actions": 0,
        "form_details": [],

        # Network behavior
        "total_requests": 0,
        "unique_domains": 0,
        "external_domains": [],

        # Redirect behavior
        "redirect_count": 0,
        "redirects": [],

        # JavaScript
        "inline_scripts": 0,
        "external_scripts": 0,

        # Security observations
        "uses_https": parsed.scheme == "https",
        "cross_domain_redirect": False,
        "cross_domain_form": False,

        "status": "success",
        "error": None
    }

    with sync_playwright() as p:

        browser = None
        context = None

        try:

            # -------------------------------------------------
            # Launch fresh temporary browser
            # -------------------------------------------------

            browser = p.chromium.launch(
                headless=True
            )

            context = browser.new_context(
                ignore_https_errors=True,
                accept_downloads=False,
                service_workers="block"
            )

            page = context.new_page()

            # Prevent excessively long pages from running forever
            page.set_default_timeout(10_000)

            requests = []
            redirects = []
            unique_domains = set()

            # -------------------------------------------------
            # Network request monitoring
            # -------------------------------------------------

            def handle_request(request):

                request_url = request.url

                requests.append({
                    "url": request_url,
                    "method": request.method,
                    "resource_type": request.resource_type
                })

                domain = get_domain(request_url)

                if domain:
                    unique_domains.add(domain)

            # -------------------------------------------------
            # Response / redirect monitoring
            # -------------------------------------------------

            def handle_response(response):

                request = response.request

                redirected_from = request.redirected_from

                if redirected_from:

                    redirect_data = {
                        "from": redirected_from.url,
                        "to": response.url,
                        "status": response.status
                    }

                    redirects.append(redirect_data)

            page.on("request", handle_request)
            page.on("response", handle_response)

            # -------------------------------------------------
            # Load website
            # -------------------------------------------------

            print("\nLoading website...")

            try:

                response = page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=NAVIGATION_TIMEOUT
                )

            except PlaywrightTimeoutError:

                print(
                    "\nWarning: Navigation timed out."
                )

                response = None

            # Give client-side JavaScript a short time to execute.
            # No clicks or form submissions are performed.
            try:
                page.wait_for_timeout(
                    ANALYSIS_WAIT_TIME
                )
            except Exception:
                pass

            print("Website analysis started.")

            # -------------------------------------------------
            # Basic information
            # -------------------------------------------------

            final_url = page.url
            final_domain = get_domain(final_url)

            analysis["final_url"] = final_url
            analysis["final_domain"] = final_domain

            if response:
                analysis["http_status"] = response.status

            try:
                analysis["page_title"] = page.title()
            except Exception:
                analysis["page_title"] = ""

            try:
                html = page.content()
                analysis["html_size"] = len(html)
            except Exception:
                html = ""

            print("\n--- BASIC INFORMATION ---")

            print(f"Original Domain : {original_domain}")
            print(f"Final Domain    : {final_domain}")
            print(f"HTTP Status     : {analysis['http_status']}")
            print(f"Page Title      : {analysis['page_title']}")
            print(f"HTML Size       : {analysis['html_size']:,}")

            # -------------------------------------------------
            # DOM analysis
            # -------------------------------------------------

            print("\n--- DOM ANALYSIS ---")

            try:
                analysis["dom_elements"] = page.locator("*").count()
            except Exception:
                pass

            try:
                analysis["javascript_files"] = page.locator(
                    "script"
                ).count()
            except Exception:
                pass

            try:
                analysis["css_files"] = page.locator(
                    "link[rel='stylesheet']"
                ).count()
            except Exception:
                pass

            try:
                analysis["images"] = page.locator(
                    "img"
                ).count()
            except Exception:
                pass

            try:
                analysis["iframes"] = page.locator(
                    "iframe"
                ).count()
            except Exception:
                pass

            try:
                analysis["forms"] = page.locator(
                    "form"
                ).count()
            except Exception:
                pass

            try:
                analysis["input_fields"] = page.locator(
                    "input"
                ).count()
            except Exception:
                pass

            try:
                analysis["password_fields"] = page.locator(
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

                analysis["hidden_elements"] = page.locator(
                    hidden_selector
                ).count()

            except Exception:
                pass

            print(
                f"DOM Elements    : {analysis['dom_elements']}"
            )

            print(
                f"JavaScript      : {analysis['javascript_files']}"
            )

            print(
                f"CSS Files       : {analysis['css_files']}"
            )

            print(
                f"Images          : {analysis['images']}"
            )

            print(
                f"Iframes         : {analysis['iframes']}"
            )

            print(
                f"Forms           : {analysis['forms']}"
            )

            print(
                f"Input Fields    : {analysis['input_fields']}"
            )

            print(
                f"Password Fields : {analysis['password_fields']}"
            )

            print(
                f"Hidden Elements : {analysis['hidden_elements']}"
            )

            # -------------------------------------------------
            # JavaScript analysis
            # -------------------------------------------------

            print("\n--- JAVASCRIPT ANALYSIS ---")

            try:

                inline_scripts = page.locator(
                    "script:not([src])"
                ).count()

                external_scripts = page.locator(
                    "script[src]"
                ).count()

                analysis["inline_scripts"] = inline_scripts
                analysis["external_scripts"] = external_scripts

            except Exception:
                pass

            print(
                f"Inline Scripts   : {analysis['inline_scripts']}"
            )

            print(
                f"External Scripts : {analysis['external_scripts']}"
            )

            # -------------------------------------------------
            # Form analysis
            # -------------------------------------------------

            print("\n--- FORM ANALYSIS ---")

            for i in range(analysis["forms"]):

                try:

                    form = page.locator(
                        "form"
                    ).nth(i)

                    action = form.get_attribute(
                        "action"
                    )

                    method = form.get_attribute(
                        "method"
                    )

                    action = action or final_url
                    method = method or "GET"

                    absolute_action = urljoin(
                        final_url,
                        action
                    )

                    external = is_external_domain(
                        absolute_action,
                        final_domain
                    )

                    if external:
                        analysis[
                            "external_form_actions"
                        ] += 1

                    form_info = {
                        "method": method.upper(),
                        "action": absolute_action,
                        "external_domain": external
                    }

                    analysis[
                        "form_details"
                    ].append(form_info)

                    print(f"\nForm {i + 1}")
                    print(f"Method   : {method.upper()}")
                    print(f"Action   : {absolute_action}")
                    print(f"External : {external}")

                except Exception:
                    continue

            analysis["cross_domain_form"] = (
                analysis["external_form_actions"] > 0
            )

            # -------------------------------------------------
            # Network analysis
            # -------------------------------------------------

            print("\n--- NETWORK ANALYSIS ---")

            analysis["total_requests"] = len(
                requests
            )

            analysis["unique_domains"] = len(
                unique_domains
            )

            external_domains = sorted(
                domain
                for domain in unique_domains
                if domain != final_domain
            )

            analysis["external_domains"] = (
                external_domains
            )

            print(
                f"Total Requests : {analysis['total_requests']}"
            )

            print(
                f"Unique Domains : {analysis['unique_domains']}"
            )

            print("\nExternal Domains:")

            for domain in external_domains:
                print(f"  - {domain}")

            # -------------------------------------------------
            # Redirect analysis
            # -------------------------------------------------

            print("\n--- REDIRECT ANALYSIS ---")

            analysis["redirect_count"] = len(
                redirects
            )

            analysis["redirects"] = redirects

            print(
                f"Redirect Count : {analysis['redirect_count']}"
            )

            for redirect in redirects:

                print("\nRedirect:")
                print(
                    f"FROM : {redirect['from']}"
                )
                print(
                    f"TO   : {redirect['to']}"
                )
                print(
                    f"HTTP : {redirect['status']}"
                )

            # -------------------------------------------------
            # Cross-domain redirect
            # -------------------------------------------------

            analysis["cross_domain_redirect"] = (
                original_domain != final_domain
            )

            print("\n--- SECURITY OBSERVATIONS ---")

            print(
                f"HTTPS Used             : "
                f"{analysis['uses_https']}"
            )

            print(
                f"Cross-Domain Redirect  : "
                f"{analysis['cross_domain_redirect']}"
            )

            print(
                f"Cross-Domain Form      : "
                f"{analysis['cross_domain_form']}"
            )

            # -------------------------------------------------
            # Save JSON report
            # -------------------------------------------------

            with open(
                REPORT_FILE,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    analysis,
                    file,
                    indent=4,
                    ensure_ascii=False
                )

            print("\n--- REPORT ---")

            print(
                f"Saved to: {REPORT_FILE}"
            )

            print("\n" + "=" * 70)
            print("             ANALYSIS COMPLETED")
            print("=" * 70)

        except Exception as error:

            analysis["status"] = "error"
            analysis["error"] = (
                f"{type(error).__name__}: {error}"
            )

            print("\nERROR:")
            print(
                analysis["error"]
            )

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


# ---------------------------------------------------------
# Program entry point
# ---------------------------------------------------------

if __name__ == "__main__":

    target_url = input(
        "\nEnter website URL to analyze: "
    ).strip()

    try:

        analyze_website(
            target_url
        )

    except Exception as error:

        print("\nERROR:")
        print(
            f"{type(error).__name__}: {error}"
        )