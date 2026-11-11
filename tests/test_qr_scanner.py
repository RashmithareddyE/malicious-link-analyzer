import io

import pytest
import qrcode
from PIL import Image

from modules.qr_scanner import ImageValidationError, scan_qr_bytes, validate_image_bytes


def _qr_png(text: str) -> bytes:
    buf = io.BytesIO()
    qrcode.make(text).convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def test_decodes_https_qr():
    r = scan_qr_bytes(_qr_png("https://www.example.com/about"), "a.png")
    assert r.success and r.contents == ["https://www.example.com/about"]


def test_decodes_http_and_ip_qr():
    assert scan_qr_bytes(_qr_png("http://www.example.com"), "a.png").contents == ["http://www.example.com"]
    assert scan_qr_bytes(_qr_png("http://203.0.113.5/x"), "a.png").contents == ["http://203.0.113.5/x"]


def test_image_without_qr_gives_error():
    buf = io.BytesIO()
    Image.new("RGB", (300, 300), "white").save(buf, format="PNG")
    r = scan_qr_bytes(buf.getvalue(), "blank.png")
    assert not r.success and "No QR code" in r.error


def test_corrupt_file_rejected():
    r = scan_qr_bytes(b"not an image at all", "fake.png")
    assert not r.success and "not a valid" in r.error


def test_empty_and_wrong_extension_rejected():
    with pytest.raises(ImageValidationError):
        validate_image_bytes(b"", "x.png")
    with pytest.raises(ImageValidationError):
        validate_image_bytes(_qr_png("https://example.com"), "x.gif")


def test_oversized_file_rejected():
    with pytest.raises(ImageValidationError):
        validate_image_bytes(b"0" * (6 * 1024 * 1024), "big.png")
