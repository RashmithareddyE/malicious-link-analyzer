"""
url_analyzer.py - URL validation and the rule-based detection engine.

Every check inspects only the TEXT of the URL. The URL is never opened,
resolved via DNS, or downloaded. Each rule that fires adds a configurable number
of points (see config.RULE_POINTS); the sum is capped at 100.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from difflib import SequenceMatcher
from typing import List, Optional, Tuple
from urllib.parse import unquote

import config
from modules.feature_extractor import (
    ParsedURL, add_default_scheme, find_phishing_keywords, parse_url,
)

# Schemes that must never be treated as web links (they can run code or leak data)
DANGEROUS_SCHEMES = ("javascript", "data", "vbscript", "file", "blob")

_HOST_LABEL_RE = re.compile(r"^[a-z0-9_]([a-z0-9_\-]{0,61}[a-z0-9_])?$", re.IGNORECASE)
_LEET_MAP = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "$": "s"})


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------
@dataclass
class RuleCheck:
    """Result of one rule (triggered or not) - shown in the UI for transparency."""
    rule_id: str
    name: str
    triggered: bool
    points: int = 0
    severity: str = "info"          # info | low | medium | high
    detail: str = ""


@dataclass
class RuleAnalysis:
    url: str
    normalized_url: str
    checks: List[RuleCheck] = field(default_factory=list)
    rule_score: int = 0

    @property
    def triggered(self) -> List[RuleCheck]:
        return [c for c in self.checks if c.triggered]

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "normalized_url": self.normalized_url,
            "rule_score": self.rule_score,
            "checks": [asdict(c) for c in self.checks],
        }


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def validate_url(content: str) -> Tuple[bool, str, str]:
    """Decide whether decoded QR text / user input is a usable web URL.

    Returns (is_valid, normalized_url, message).
    """
    if content is None:
        return False, "", "No content provided."
    text = content.strip()
    if not text:
        return False, "", "The content is empty."
    if len(text) > config.MAX_URL_LENGTH_ACCEPTED:
        return False, "", "The content is too long to be a normal URL."
    if any(ord(ch) < 32 for ch in text):
        return False, "", "The content contains control characters."
    if re.search(r"\s", text):
        return False, "", "The content contains spaces, so it is not a single URL."

    lowered = text.lower()
    for scheme in DANGEROUS_SCHEMES:
        if lowered.startswith(scheme + ":"):
            return False, "", (f"Blocked: '{scheme}:' content is not a web link and "
                               "could execute code or expose data. It was not analyzed.")

    # Non-web schemes (mailto:, tel:, WIFI:, geo:, ...) are not URLs we can score
    scheme_match = re.match(r"^([a-zA-Z][a-zA-Z0-9+.\-]*):", text)
    if scheme_match and not re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", text):
        # "host:8080/path" looks like a scheme to the regex - allow if host-like
        if not re.match(r"^[A-Za-z0-9.\-]+:\d{1,5}(/|$)", text):
            return False, "", (f"This QR code contains '{scheme_match.group(1)}:' data, "
                               "which is not a web URL.")

    candidate = add_default_scheme(text)
    parsed = parse_url(candidate)
    if parsed.scheme not in ("http", "https", "ftp"):
        return False, "", f"Unsupported URL scheme '{parsed.scheme}'. Only http/https/ftp are analyzed."
    if not parsed.host:
        return False, "", "No hostname could be found in the URL."
    if not parsed.valid_port:
        return False, "", "The URL contains an invalid port number."

    if not parsed.is_ip:
        labels = parsed.host.split(".")
        # Non-ASCII labels (IDN) are allowed here and flagged later by the rules
        for label in labels:
            if not label:
                return False, "", "The hostname contains an empty label ('..')."
            if label.isascii() and not _HOST_LABEL_RE.match(label):
                return False, "", f"The hostname label '{label[:30]}' has invalid characters."
        if len(labels) < 2 and parsed.host != "localhost":
            return False, "", "The hostname has no domain suffix (e.g. '.com')."

    msg = "Valid URL."
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", text):
        msg = "No scheme was present; 'http://' was assumed for analysis."
    return True, candidate, msg


# --------------------------------------------------------------------------
# Individual rules (each returns a RuleCheck)
# --------------------------------------------------------------------------
def _pts(rule_id: str) -> int:
    return config.RULE_POINTS[rule_id]


def _check_ip(p: ParsedURL) -> RuleCheck:
    hit = p.is_ip
    return RuleCheck("ip_address", "IP address instead of domain name", hit,
                     _pts("ip_address") if hit else 0, "high",
                     f"Host '{p.host}' is a raw IP address; legitimate services almost always use domain names."
                     if hit else "Host is a normal domain name.")


def _check_https(p: ParsedURL) -> RuleCheck:
    hit = p.scheme != "https"
    scheme = p.scheme or "none"
    return RuleCheck("no_https", "HTTP instead of HTTPS", hit,
                     _pts("no_https") if hit else 0, "medium",
                     f"Connection uses '{scheme}', so traffic is not encrypted and can be intercepted."
                     if hit else "URL uses HTTPS (this alone does not prove the site is safe).")


def _check_length(url: str) -> List[RuleCheck]:
    n = len(url)
    if n >= config.VERY_LONG_URL_THRESHOLD:
        return [RuleCheck("very_long_url", "Excessive URL length", True,
                          _pts("very_long_url"), "medium",
                          f"URL is {n} characters long (very long URLs often hide the real destination).")]
    if n >= config.LONG_URL_THRESHOLD:
        return [RuleCheck("long_url", "Long URL", True, _pts("long_url"), "low",
                          f"URL is {n} characters long (typical safe URLs are shorter).")]
    return [RuleCheck("very_long_url", "Excessive URL length", False, 0, "info",
                      f"URL length ({n}) is within the normal range.")]


def _check_special_chars(url: str) -> RuleCheck:
    specials = [ch for ch in url if ch in config.SPECIAL_CHARACTERS and ch != "@"]
    encoded = len(re.findall(r"%[0-9a-fA-F]{2}", url))
    hit = len(specials) >= 3 or encoded >= 5
    detail = (f"Found {len(specials)} special characters and {encoded} percent-encoded sequences."
              if hit else "No unusual special characters.")
    return RuleCheck("special_characters", "Suspicious special characters", hit,
                     _pts("special_characters") if hit else 0, "low", detail)


def _check_shortener(p: ParsedURL) -> RuleCheck:
    hit = p.registrable_domain in config.URL_SHORTENERS
    return RuleCheck("url_shortener", "URL shortener", hit,
                     _pts("url_shortener") if hit else 0, "medium",
                     f"'{p.registrable_domain}' is a link shortener that hides the final destination."
                     if hit else "Not a known URL shortener.")


def _check_subdomains(p: ParsedURL) -> RuleCheck:
    labels = [s for s in p.subdomain.split(".") if s]
    reasons = []
    if len(labels) > config.MAX_NORMAL_SUBDOMAINS:
        reasons.append(f"{len(labels)} subdomain levels")
    # A brand name used as a subdomain of an unrelated domain (paypal.evil.example)
    for brand in config.POPULAR_BRANDS:
        if len(brand) >= 4 and any(brand in lab for lab in labels) and brand not in p.domain_name:
            reasons.append(f"brand name '{brand}' used in a subdomain of '{p.registrable_domain}'")
            break
    sub_kw = [kw for kw in config.PHISHING_KEYWORDS if kw in p.subdomain and len(kw) >= 5]
    if sub_kw and labels:
        reasons.append(f"keywords in subdomain ({', '.join(sub_kw[:3])})")
    hit = bool(reasons)
    return RuleCheck("suspicious_subdomains", "Suspicious subdomains", hit,
                     _pts("suspicious_subdomains") if hit else 0, "medium",
                     "; ".join(reasons).capitalize() + "." if hit else "Subdomain structure looks normal.")


def _normalise_leet(text: str) -> str:
    """Map common look-alike characters back to letters (g00gle -> google)."""
    text = text.translate(_LEET_MAP)
    return text.replace("rn", "m").replace("vv", "w")


def _check_typosquatting(p: ParsedURL) -> RuleCheck:
    name = p.domain_name
    if not name or p.is_ip:
        return RuleCheck("typosquatting", "Misspelled popular domain", False, 0, "high",
                         "No domain name to compare.")
    name_clean = re.sub(r"[^a-z0-9\-]", "", name)
    tokens = [t for t in name_clean.split("-") if t]
    # Tokens with look-alike characters mapped back to letters (paypa1 -> paypal)
    leet_tokens = [_normalise_leet(t) for t in tokens]
    reason = None
    for brand in config.POPULAR_BRANDS:
        if name_clean == brand:
            continue  # exact match on the name part - could be the real brand
        leet = _normalise_leet(name_clean)
        if leet == brand and name_clean != brand:
            reason = f"'{name}' imitates '{brand}' using look-alike characters"
        elif len(brand) >= 5 and SequenceMatcher(None, name_clean, brand).ratio() >= config.TYPOSQUAT_SIMILARITY \
                and abs(len(name_clean) - len(brand)) <= 2 and name_clean != brand:
            reason = f"'{name}' is spelled very similarly to '{brand}'"
        elif len(brand) >= 4 and len(tokens) > 1 and (brand in tokens or brand in leet_tokens):
            reason = f"'{name}' embeds the brand name '{brand}' inside a hyphenated domain"
        elif len(brand) >= 6 and brand in name_clean and name_clean != brand:
            reason = f"'{name}' contains the brand name '{brand}' with extra characters"
        if reason:
            break
    hit = reason is not None
    return RuleCheck("typosquatting", "Misspelled popular domain", hit,
                     _pts("typosquatting") if hit else 0, "high",
                     (reason + ". Attackers register look-alike domains to impersonate trusted brands.")
                     if hit else "Domain does not closely imitate a known popular brand.")


def _check_keywords(url: str) -> RuleCheck:
    found = find_phishing_keywords(url)
    n = len(found)
    if n == 0:
        return RuleCheck("phishing_keywords", "Phishing-related keywords", False, 0, "low",
                         "No phishing-related keywords found.")
    points = _pts("phishing_keywords") * (2 if n >= 3 else 1)
    return RuleCheck("phishing_keywords", "Phishing-related keywords", True, points,
                     "medium" if n >= 3 else "low",
                     f"Found {n} keyword(s) often used in phishing: {', '.join(found[:8])}.")


def _check_extension(p: ParsedURL) -> RuleCheck:
    path = unquote(p.path).lower()
    last = path.rsplit("/", 1)[-1]
    ext = last.rsplit(".", 1)[-1] if "." in last else ""
    double = bool(re.search(r"\.(pdf|docx?|xlsx?|jpg|png|txt)\.(exe|scr|bat|js|vbs|apk|zip)$", last))
    hit = ext in config.SUSPICIOUS_FILE_EXTENSIONS or double
    return RuleCheck("suspicious_extension", "Suspicious file extension", hit,
                     _pts("suspicious_extension") if hit else 0, "high",
                     f"The link points to a file ending in '.{ext}', which can install malware. "
                     "Do not download it." if hit else "No executable/archive file extension in the path.")


def _check_dots(p: ParsedURL) -> RuleCheck:
    dots = p.host.count(".")
    hit = dots >= config.MAX_NORMAL_DOTS and not p.is_ip
    return RuleCheck("many_dots", "Unusual number of dots", hit,
                     _pts("many_dots") if hit else 0, "low",
                     f"The hostname contains {dots} dots (many levels can disguise the real domain)."
                     if hit else f"Hostname has {dots} dot(s), which is normal.")


def _check_at(url: str, p: ParsedURL) -> RuleCheck:
    hit = "@" in url
    detail = "No '@' symbol."
    if hit:
        detail = ("'@' found. Everything before '@' in the address is ignored by browsers, so "
                  f"the real host is '{p.host}'." if p.userinfo else
                  "'@' symbol found in the URL (path/query); this is unusual.")
    return RuleCheck("at_symbol", "'@' symbol in URL", hit,
                     _pts("at_symbol") if hit else 0, "high", detail)


def _check_credentials(p: ParsedURL) -> RuleCheck:
    hit = ":" in p.userinfo
    return RuleCheck("credentials_in_url", "Credentials embedded in URL", hit,
                     _pts("credentials_in_url") if hit else 0, "high",
                     "The URL embeds a username:password pair, a classic spoofing trick."
                     if hit else "No embedded credentials.")


def _check_redirect(p: ParsedURL) -> RuleCheck:
    q = (p.query + "&" + p.path).lower()
    hits = [ind for ind in config.REDIRECT_INDICATORS if ind in q]
    nested = bool(re.search(r"https?(://|%3a%2f%2f)", p.query + p.path, re.IGNORECASE))
    double_slash = "//" in p.path
    reasons = []
    if hits:
        reasons.append("redirect parameter(s): " + ", ".join(h.rstrip("=") for h in hits[:4]))
    if nested:
        reasons.append("another URL is embedded inside this URL")
    if double_slash:
        reasons.append("'//' inside the path")
    hit = bool(reasons)
    return RuleCheck("redirect_indicator", "URL redirection indicators", hit,
                     _pts("redirect_indicator") if hit else 0, "medium",
                     "; ".join(reasons).capitalize() + "." if hit else "No redirection indicators.")


def _check_punycode(p: ParsedURL) -> RuleCheck:
    non_ascii = any(ord(ch) > 127 for ch in p.host)
    puny = "xn--" in p.host
    hit = puny or non_ascii
    return RuleCheck("punycode", "Punycode / internationalized domain", hit,
                     _pts("punycode") if hit else 0, "high",
                     "The domain uses internationalized characters (or 'xn--' punycode). Attackers use "
                     "look-alike letters from other alphabets to imitate real sites (homograph attack)."
                     if hit else "Domain is plain ASCII.")


def _check_port(p: ParsedURL) -> RuleCheck:
    hit = p.port is not None and p.port not in config.STANDARD_PORTS
    return RuleCheck("unusual_port", "Unusual port number", hit,
                     _pts("unusual_port") if hit else 0, "medium",
                     f"Port {p.port} is not a standard web port (80/443)." if hit
                     else "No unusual port.")


def _check_hyphens(p: ParsedURL) -> RuleCheck:
    # The "--" in a punycode prefix ("xn--") is not a real hyphen
    count = p.host.replace("xn--", "").count("-")
    hit = count > config.MAX_NORMAL_HYPHENS_IN_DOMAIN
    return RuleCheck("multiple_hyphens", "Multiple hyphens in domain", hit,
                     _pts("multiple_hyphens") if hit else 0, "low",
                     f"The domain contains {count} hyphens, common in throw-away phishing domains."
                     if hit else f"Domain has {count} hyphen(s).")


def _check_tld(p: ParsedURL) -> RuleCheck:
    tld = p.suffix.split(".")[-1] if p.suffix else ""
    hit = tld in config.SUSPICIOUS_TLDS
    return RuleCheck("suspicious_tld", "Frequently abused domain ending", hit,
                     _pts("suspicious_tld") if hit else 0, "low",
                     f"'.{tld}' is a domain ending often abused for cheap throw-away sites." if hit
                     else "Domain ending is not on the frequently-abused list.")


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def analyze_url(url: str) -> RuleAnalysis:
    """Run every rule on a (validated) URL and return the rule-based analysis."""
    normalized = add_default_scheme(url)
    # parse_url() on the ORIGINAL text keeps scheme == "" when none was given,
    # so the HTTPS rule can tell "missing scheme" apart from "https".
    p = parse_url(url)

    checks: List[RuleCheck] = [
        _check_ip(p),
        _check_https(p),
        *_check_length(normalized),
        _check_special_chars(normalized),
        _check_shortener(p),
        _check_subdomains(p),
        _check_typosquatting(p),
        _check_keywords(normalized),
        _check_extension(p),
        _check_dots(p),
        _check_at(normalized, p),
        _check_credentials(p),
        _check_redirect(p),
        _check_punycode(p),
        _check_port(p),
        _check_hyphens(p),
        _check_tld(p),
    ]
    score = min(100, sum(c.points for c in checks if c.triggered))
    return RuleAnalysis(url=url.strip(), normalized_url=normalized, checks=checks, rule_score=score)
