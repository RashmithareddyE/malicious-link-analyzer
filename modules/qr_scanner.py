"""
qr_scanner.py - Safe QR code image validation and decoding.

Pipeline
--------
1. validate_image_bytes(): treat the upload as UNTRUSTED. Check size, verify the
   real image format with Pillow (not just the file extension) and limit pixels.
2. Pillow pre-processing: convert to RGB/grayscale, auto-contrast, upscale,
   add a white quiet-zone border (improves detection on cropped QR codes).
3. Decode with OpenCV's QRCodeDetector (always available) and, if the optional
   native ZBar library is installed, also with Pyzbar. Results are merged.

Nothing here ever opens, downloads or visits the decoded content.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

import config

# Pillow's own decompression-bomb guard
Image.MAX_IMAGE_PIXELS = config.MAX_IMAGE_PIXELS

# Pyzbar needs the native ZBar DLL. On Windows it may be missing, so the import
# is optional and OpenCV is used as the guaranteed fallback.
try:  # pragma: no cover - depends on the machine
    from pyzbar import pyzbar as _pyzbar
    PYZBAR_AVAILABLE = True
except Exception:  # ImportError or OSError (missing libzbar-64.dll)
    _pyzbar = None
    PYZBAR_AVAILABLE = False


class ImageValidationError(Exception):
    """Raised when an uploaded file is not an acceptable image."""


@dataclass
class QRScanResult:
    """Outcome of scanning one image."""
    contents: List[str] = field(default_factory=list)   # decoded QR payloads
    decoders_used: List[str] = field(default_factory=list)
    image_size: Optional[tuple] = None
    image_format: Optional[str] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return bool(self.contents) and self.error is None


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def validate_image_bytes(data: bytes, filename: str = "") -> Image.Image:
    """Validate raw upload bytes and return a loaded Pillow image (RGB).

    Raises ImageValidationError with a user-friendly message on any problem.
    """
    if not data:
        raise ImageValidationError("The uploaded file is empty.")

    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    if len(data) > max_bytes:
        raise ImageValidationError(
            f"File is too large ({len(data) / 1024 / 1024:.1f} MB). "
            f"Maximum allowed size is {config.MAX_UPLOAD_MB} MB."
        )

    if filename:
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in config.ALLOWED_IMAGE_EXTENSIONS:
            raise ImageValidationError(
                f"Unsupported file extension '.{ext}'. "
                f"Allowed: {', '.join(config.ALLOWED_IMAGE_EXTENSIONS).upper()}."
            )

    # First pass: verify() checks file integrity without decoding all pixels.
    try:
        probe = Image.open(io.BytesIO(data))
        fmt = (probe.format or "").upper()
        probe.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError,
            Image.DecompressionBombError) as exc:
        raise ImageValidationError(
            "The file is not a valid or readable image (it may be corrupted "
            f"or not an image at all). Details: {type(exc).__name__}"
        ) from exc

    if fmt not in config.ALLOWED_IMAGE_FORMATS:
        raise ImageValidationError(
            f"Image content is '{fmt or 'unknown'}', but only PNG, JPEG and BMP "
            "are accepted. (The file extension does not match the real format.)"
        )

    # Second pass: re-open (verify() invalidates the first handle) and load.
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ImageValidationError(
            f"The image could not be decoded: {type(exc).__name__}"
        ) from exc

    if img.width < 20 or img.height < 20:
        raise ImageValidationError("The image is too small to contain a QR code.")

    # Normalise colour modes (palette, RGBA, grayscale...) to plain RGB.
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        background.alpha_composite(img)
        img = background.convert("RGB")
    else:
        img = img.convert("RGB")
    img.info["source_format"] = fmt
    return img


# --------------------------------------------------------------------------
# Pre-processing variants (Pillow) -> numpy arrays for OpenCV
# --------------------------------------------------------------------------
def _preprocess_variants(image: Image.Image) -> List[np.ndarray]:
    """Create several enhanced versions of the image to improve decode rate."""
    variants: List[np.ndarray] = []
    gray = ImageOps.grayscale(image)

    # 1. Plain grayscale
    variants.append(np.array(gray))

    # 2. Auto-contrast + white quiet-zone border
    bordered = ImageOps.expand(ImageOps.autocontrast(gray), border=40, fill=255)
    variants.append(np.array(bordered))

    # 3. Upscaled (helps with small images) - capped to keep processing fast
    scale = 2 if max(gray.size) < 700 else 1
    if scale > 1:
        up = gray.resize((gray.width * scale, gray.height * scale), Image.LANCZOS)
        variants.append(np.array(ImageOps.expand(up, border=60, fill=255)))

    # 4. Downscaled (helps with very large images)
    if max(gray.size) > 1600:
        ratio = 1200 / max(gray.size)
        down = gray.resize((int(gray.width * ratio), int(gray.height * ratio)),
                           Image.LANCZOS)
        variants.append(np.array(ImageOps.expand(down, border=40, fill=255)))

    # 5. Binarised (Otsu) via OpenCV on the bordered version
    _, otsu = cv2.threshold(np.array(bordered), 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)

    # 6. Inverted (some QR codes are light-on-dark)
    variants.append(255 - np.array(bordered))
    return variants


# --------------------------------------------------------------------------
# Decoders
# --------------------------------------------------------------------------
def _decode_with_opencv(arrays: List[np.ndarray]) -> List[str]:
    """Decode QR codes with OpenCV; supports multiple codes per image."""
    found: List[str] = []
    detector = cv2.QRCodeDetector()
    for arr in arrays:
        arr = np.ascontiguousarray(arr)
        # Multi-QR detection (OpenCV >= 4.3)
        try:
            ok, infos, _pts, _ = detector.detectAndDecodeMulti(arr)
            if ok and infos is not None:
                found.extend(text for text in infos if text)
        except cv2.error:
            pass
        # Single-QR fallback
        try:
            text, _pts, _ = detector.detectAndDecode(arr)
            if text:
                found.append(text)
        except cv2.error:
            pass
        if found:
            break  # stop at the first variant that works
    return found


def _decode_with_pyzbar(arrays: List[np.ndarray]) -> List[str]:
    """Decode with Pyzbar (ZBar) if the native library is available."""
    if not PYZBAR_AVAILABLE:
        return []
    found: List[str] = []
    for arr in arrays:
        try:
            for obj in _pyzbar.decode(Image.fromarray(arr),
                                      symbols=[_pyzbar.ZBarSymbol.QRCODE]):
                text = obj.data.decode("utf-8", errors="replace")
                if text:
                    found.append(text)
        except Exception:
            continue
        if found:
            break
    return found


def scan_qr_image(image: Image.Image) -> QRScanResult:
    """Decode every QR code in a validated Pillow image."""
    result = QRScanResult(image_size=image.size,
                          image_format=image.info.get("source_format"))
    try:
        arrays = _preprocess_variants(image)
    except Exception as exc:
        result.error = f"Image processing failed: {type(exc).__name__}: {exc}"
        return result

    merged: List[str] = []
    cv_found = _decode_with_opencv(arrays)
    if cv_found:
        result.decoders_used.append("OpenCV QRCodeDetector")
        merged.extend(cv_found)

    zb_found = _decode_with_pyzbar(arrays)
    if zb_found:
        result.decoders_used.append("Pyzbar (ZBar)")
        merged.extend(zb_found)

    # De-duplicate while preserving order; trim whitespace; enforce length cap
    seen = set()
    for text in merged:
        text = text.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.contents.append(text[: config.MAX_URL_LENGTH_ACCEPTED])

    if not result.contents:
        result.error = ("No QR code could be detected or decoded in this image. "
                        "Try a sharper, well-lit image with the whole QR code visible.")
    return result


def scan_qr_bytes(data: bytes, filename: str = "") -> QRScanResult:
    """Validate raw bytes, then scan. Never raises - errors are in the result."""
    try:
        image = validate_image_bytes(data, filename)
    except ImageValidationError as exc:
        return QRScanResult(error=str(exc))
    return scan_qr_image(image)
