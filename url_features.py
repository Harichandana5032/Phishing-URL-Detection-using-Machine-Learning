
import re
import ipaddress
import pandas as pd
from urllib.parse import urlsplit


def extract_url_features(url):

    url = str(url).strip()

    # Add a scheme if the user has not entered one
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "https://" + url

    parsed = urlsplit(url)

    hostname = parsed.hostname or ""

    # Check whether the hostname is an IP address
    try:
        ipaddress.ip_address(hostname)
        is_ip = 1
    except ValueError:
        is_ip = 0

    # Extract TLD and subdomain count
    if hostname and not is_ip:
        domain_parts = hostname.split(".")
        tld = domain_parts[-1].lower()

        # Exclude www from the subdomain count
        subdomains = domain_parts[:-2]
        subdomains = [s for s in subdomains if s.lower() != "www"]

        no_of_subdomains = len(subdomains)

    elif is_ip:
        tld = "IP"
        no_of_subdomains = 0

    else:
        tld = "unknown"
        no_of_subdomains = 0

    url_length = len(url)

    # Character counts
    no_of_letters = sum(c.isalpha() for c in url)
    no_of_digits = sum(c.isdigit() for c in url)

    no_of_equals = url.count("=")
    no_of_qmark = url.count("?")
    no_of_ampersand = url.count("&")

    # Basic obfuscation indicators
    encoded_chars = re.findall(r"%[0-9a-fA-F]{2}", url)
    no_of_obfuscated = len(encoded_chars) + url.count("@")

    has_obfuscation = int(
        no_of_obfuscated > 0
    )

    # Count special characters excluding common URL separators
    excluded_chars = set("abcdefghijklmnopqrstuvwxyz"
                         "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                         "0123456789./?=&#:-_%")

    no_of_other_special = sum(
        1 for c in url if c not in excluded_chars
    )

    features = {
        "URLLength": url_length,
        "DomainLength": len(hostname),
        "IsDomainIP": is_ip,
        "IsHTTPS": int(parsed.scheme.lower() == "https"),
        "TLD": tld,
        "TLDLength": len(tld),
        "NoOfSubDomain": no_of_subdomains,
        "HasObfuscation": has_obfuscation,
        "NoOfObfuscatedChar": no_of_obfuscated,
        "ObfuscationRatio": no_of_obfuscated / max(url_length, 1),
        "NoOfLettersInURL": no_of_letters,
        "LetterRatioInURL": no_of_letters / max(url_length, 1),
        "NoOfDegitsInURL": no_of_digits,
        "DegitRatioInURL": no_of_digits / max(url_length, 1),
        "NoOfEqualsInURL": no_of_equals,
        "NoOfQMarkInURL": no_of_qmark,
        "NoOfAmpersandInURL": no_of_ampersand,
        "NoOfOtherSpecialCharsInURL": no_of_other_special,
        "SpacialCharRatioInURL": (
            no_of_other_special / max(url_length, 1)
        )
    }

    return features


def extract_features_dataframe(url_series):

    return pd.DataFrame(
        [extract_url_features(url) for url in url_series]
    )
