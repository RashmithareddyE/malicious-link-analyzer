"""
feature_extractor.py - URL parsing helpers and numeric feature extraction.

The same parsing code is used by the rule engine (url_analyzer.py) and by the
machine learning classifier (ml_classifier.py) so both see identical data.
Only the text of the URL is inspected - the URL is never requested.
"""
from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from typing import Dict, List, Optional
from urllib.parse import urlparse

import config

# Common two-part public suffixes (so "bbc.co.uk" -> registrable "bbc.co.uk").
# This is a small built-in list to avoid an extra dependency.
SECOND_LEVEL_SUFFIXES = {
    "co.uk", "org.uk", "ac.uk", "gov.uk", "com.au", "net.au", "org.au",
    "co.in", "net.in", "org.in", "gov.in", "ac.in", "co.jp", "com.br",
    "com.cn", "co.za", "com.mx", "co.nz", "com.sg", "com.tr", "com.pk",
}

_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")


@dataclass
class ParsedURL:
    """Structured, safe-to-inspect view of a URL string."""
    original: str
    scheme: str
    host: str                 # lower-cased hostname (no port / userinfo)
    port: Optional[int]
    path: str
    query: str
    userinfo: str             # text before '@' in the authority, if any
    registrable_domain: str   # e.g. "example.co.uk"
    domain_name: str          # registrable domain without suffix, e.g. "example"
    suffix: str               # public suffix, e.g. "co.uk" or "com"
    subdomain: str            # everything left of the registrable domain
    is_ip: bool
    valid_port: bool = True


def add_default_scheme(raw: str) -> str:
    """Add http:// if the text has no scheme, so urlparse works correctly."""
    raw = raw.strip()
    return raw if _SCHEME_RE.match(raw) else "http://" + raw


def is_ip_host(host: str) -> bool:
    """True for dotted IPv4/IPv6, plus decimal (3232235777) and hex (0xC0A80001)."""
    host = host.strip("[]")
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass
    if host.isdigit() and len(host) >= 8:                       # decimal IPv4
        return True
    if re.fullmatch(r"0x[0-9a-fA-F]{6,8}", host):               # hex IPv4
        return True
    return False


def parse_url(raw: str) -> ParsedURL:
    """Parse a URL string into a ParsedURL. Never raises for odd input."""
    text = raw.strip()
    has_scheme = bool(_SCHEME_RE.match(text))
    parsed = urlparse(text if has_scheme else "http://" + text)

    # Authority parts: userinfo@host:port
    netloc = parsed.netloc
    userinfo = ""
    if "@" in netloc:
        userinfo, _, netloc = netloc.rpartition("@")
    try:
        port = parsed.port
        valid_port = True
    except ValueError:
        port, valid_port = None, False

    host = (parsed.hostname or "").lower().rstrip(".")
    ip_flag = is_ip_host(host) if host else False

    # Split host into subdomain / registrable domain / suffix
    subdomain = registrable = domain_name = suffix = ""
    if host and not ip_flag:
        labels = host.split(".")
        if len(labels) >= 3 and ".".join(labels[-2:]) in SECOND_LEVEL_SUFFIXES:
            suffix = ".".join(labels[-2:])
            domain_name = labels[-3]
            registrable = ".".join(labels[-3:])
            subdomain = ".".join(labels[:-3])
        elif len(labels) >= 2:
            suffix = labels[-1]
            domain_name = labels[-2]
            registrable = ".".join(labels[-2:])
            subdomain = ".".join(labels[:-2])
        else:
            domain_name = labels[0]
            registrable = labels[0]

    return ParsedURL(
        original=text,
        scheme=parsed.scheme.lower() if has_scheme else "",
        host=host,
        port=port,
        path=parsed.path or "",
        query=parsed.query or "",
        userinfo=userinfo,
        registrable_domain=registrable,
        domain_name=domain_name,
        suffix=suffix,
        subdomain=subdomain,
        is_ip=ip_flag,
        valid_port=valid_port,
    )


def find_phishing_keywords(url: str) -> List[str]:
    """Return the distinct phishing keywords that appear in the URL text."""
    lowered = re.sub(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", "", url.lower())
    return sorted({kw for kw in config.PHISHING_KEYWORDS if kw in lowered})


# --------------------------------------------------------------------------
# Numeric features for the ML model
# --------------------------------------------------------------------------
FEATURE_NAMES: List[str] = [
    "url_length",
    "domain_length",
    "num_dots",
    "num_hyphens",
    "num_digits",
    "num_special_chars",
    "has_https",
    "has_ip",
    "num_suspicious_keywords",
    "has_at_symbol",
    "num_subdomains",
    "path_length",
    "num_query_params",
    "has_punycode",
    "is_shortener",
    "unusual_port",
    "digit_ratio",
    "suspicious_tld",
    "hyphens_in_domain",
]


def extract_features(url: str) -> Dict[str, float]:
    """Compute the numeric feature dictionary used by the ML classifier."""
    p = parse_url(url)
    text = url.strip()
    digits = sum(ch.isdigit() for ch in text)
    special = sum(ch in config.SPECIAL_CHARACTERS for ch in text)
    subdomain_count = len([s for s in p.subdomain.split(".") if s]) if p.subdomain else 0
    query_params = len([q for q in p.query.split("&") if q]) if p.query else 0

    return {
        "url_length": len(text),
        "domain_length": len(p.host),
        "num_dots": text.count("."),
        "num_hyphens": text.count("-"),
        "num_digits": digits,
        "num_special_chars": special,
        "has_https": int(p.scheme == "https"),
        "has_ip": int(p.is_ip),
        "num_suspicious_keywords": len(find_phishing_keywords(text)),
        "has_at_symbol": int("@" in text),
        "num_subdomains": subdomain_count,
        "path_length": len(p.path),
        "num_query_params": query_params,
        "has_punycode": int("xn--" in p.host),
        "is_shortener": int(p.registrable_domain in config.URL_SHORTENERS),
        "unusual_port": int(p.port is not None and p.port not in config.STANDARD_PORTS),
        "digit_ratio": round(digits / max(len(text), 1), 4),
        "suspicious_tld": int(p.suffix.split(".")[-1] in config.SUSPICIOUS_TLDS),
        "hyphens_in_domain": p.host.count("-"),
    }


def features_as_vector(url: str) -> List[float]:
    """Feature values in the fixed order given by FEATURE_NAMES."""
    feats = extract_features(url)
    return [feats[name] for name in FEATURE_NAMES]
