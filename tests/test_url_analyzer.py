from modules.url_analyzer import analyze_url, validate_url


def ids(url):
    return {c.rule_id for c in analyze_url(url).triggered}


def test_safe_url_scores_zero():
    assert analyze_url("https://www.example.com/about").rule_score == 0


def test_validation():
    assert validate_url("https://example.com")[0]
    assert validate_url("example.com/path")[0]               # scheme assumed
    assert not validate_url("javascript:alert(1)")[0]        # blocked
    assert not validate_url("WIFI:S:x;T:WPA;P:y;;")[0]       # not a URL
    assert not validate_url("hello world")[0]
    assert not validate_url("")[0]


def test_ip_http_port():
    found = ids("http://203.0.113.5:8080/login")
    assert {"ip_address", "no_https", "unusual_port"} <= found


def test_typosquat_at_punycode_shortener():
    assert "typosquatting" in ids("https://paypa1.com")
    assert "typosquatting" in ids("https://g00gle.com")
    assert "at_symbol" in ids("https://google.com@evil.example.test/")
    assert "punycode" in ids("https://xn--pple-43d.example/")
    assert "url_shortener" in ids("https://bit.ly/abc")


def test_extension_redirect_hyphens_tld():
    assert "suspicious_extension" in ids("https://files.example.test/setup.exe")
    assert "redirect_indicator" in ids("https://example.com/go?url=https://other.example")
    assert "multiple_hyphens" in ids("https://a-b-c-d.example.com")
    assert "suspicious_tld" in ids("https://example.xyz")


def test_real_brand_not_flagged():
    assert "typosquatting" not in ids("https://www.paypal.com/signin")


def test_score_capped_at_100():
    assert analyze_url("http://1.2.3.4:81/login/verify/account/update.exe?redirect=http://x@y").rule_score <= 100
