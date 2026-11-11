"""
generate_sample_qr.py - Create harmless QR codes for testing (run from the project root).

    python generate_sample_qr.py

Uses only reserved/example domains (example.com, *.test, *.example, documentation IPs).
No real malicious site is ever encoded. Output goes to sample_qr_codes/.
"""
import sys
from pathlib import Path

import qrcode
from PIL import Image, ImageDraw

import config

SAMPLES = {
    "01_safe_https.png": "https://www.example.com/about",
    "02_http_url.png": "http://www.example.com/welcome",
    "03_ip_based.png": "http://203.0.113.5:8080/login/verify.php",          # TEST-NET-3 documentation IP
    "04_suspicious.png": "http://paypa1-secure-login.example.test/verify?redirect=http://x.example.test",
    "05_at_symbol.png": "https://www.bank.example.com@secure-update.example.test/account",
    "06_shortener.png": "https://bit.ly/example-demo",
}


def make_qr(text: str, path: Path) -> None:
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=10, border=4)
    qr.add_data(text)
    qr.make(fit=True)
    qr.make_image(fill_color="black", back_color="white").convert("RGB").save(path)


def main() -> int:
    out = config.SAMPLE_QR_DIR
    out.mkdir(parents=True, exist_ok=True)
    for name, text in SAMPLES.items():
        make_qr(text, out / name)
        print(f"[OK] {name:24s} -> {text}")

    # Multiple QR codes in one image
    make_qr("https://www.example.com/one", out / "_tmp1.png")
    make_qr("https://www.example.org/two", out / "_tmp2.png")
    canvas = Image.new("RGB", (760, 380), "white")
    for i, tmp in enumerate(("_tmp1.png", "_tmp2.png")):
        im = Image.open(out / tmp).resize((340, 340))
        canvas.paste(im, (20 + i * 380, 20))
        (out / tmp).unlink()
    canvas.save(out / "07_multiple_qr.png")
    print("[OK] 07_multiple_qr.png       -> two QR codes in one image")

    # Non-URL QR (plain text) -> should be rejected as 'not a URL'
    make_qr("Hello, this QR code contains plain text only.", out / "08_text_only.png")
    print("[OK] 08_text_only.png         -> plain text, not a URL")

    # Image with NO QR code
    img = Image.new("RGB", (400, 400), "white")
    d = ImageDraw.Draw(img)
    d.ellipse((80, 80, 320, 320), outline="navy", width=6)
    d.text((140, 190), "No QR code here", fill="black")
    img.save(out / "09_no_qr.png")
    print("[OK] 09_no_qr.png             -> image without a QR code")

    # Invalid QR image: a QR code that has been heavily damaged
    make_qr("https://www.example.com/damaged", out / "_dmg.png")
    dmg = Image.open(out / "_dmg.png")
    dd = ImageDraw.Draw(dmg)
    dd.rectangle((0, 0, dmg.width, dmg.height // 2), fill="gray")
    dd.rectangle((0, dmg.height // 2, dmg.width, dmg.height), fill="white")
    dmg.save(out / "10_invalid_damaged_qr.png")
    (out / "_dmg.png").unlink()
    print("[OK] 10_invalid_damaged_qr.png -> unreadable / damaged QR code")

    # Corrupt file that only pretends to be an image
    (out / "11_corrupt_file.png").write_bytes(b"this is not really a png image")
    print("[OK] 11_corrupt_file.png      -> fake PNG (validation should reject it)")
    print(f"\nAll sample files saved in: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
