"""
image_forensics.py - Deep + Classical Image Tampering Detection
Requirement: The Fraud Detective & The Duplicate Hunter
"""
import io
import numpy as np
from PIL import Image, ImageChops
from typing import Optional
import cv2
import imagehash
import piexif

try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

def get_image_phash(file_bytes: bytes) -> str:
    """Requirement: The Duplicate Hunter. Generates visual fingerprint."""
    try:
        img = Image.open(io.BytesIO(file_bytes))
        return str(imagehash.phash(img))
    except Exception: return ""

def detect_tampering(file_bytes: bytes, filename: str) -> tuple:
    name = filename.lower()
    if name.endswith(".pdf") and PDF2IMAGE_AVAILABLE:
        try:
            pages = convert_from_bytes(file_bytes, dpi=150, first_page=1, last_page=1)
            buf = io.BytesIO(); pages[0].save(buf, format="JPEG")
            file_bytes = buf.getvalue()
        except Exception: return None, 0.0

    try:
        img = Image.open(io.BytesIO(file_bytes))
        
        # 1. Deep Metadata Scan (FIXED for clean output)
        suspicious_list = ["canva", "photoshop", "gimp", "adobe", "illustrator", "framer"]
        for key, value in img.info.items():
            if isinstance(value, (str, bytes)):
                val_str = str(value).lower()
                for tool in suspicious_list:
                    if tool in val_str:
                        # Return ONLY the tool name instead of the whole metadata block
                        return f"Tampering Signature: '{tool}' marker found in metadata", 0.93

        # 2. Raw Binary Signature Scan
        raw_data = file_bytes.lower()
        for tool in suspicious_list:
            if tool.encode() in raw_data:
                return f"Tampering Signature: '{tool}' marker found in raw file data", 0.87

        # 3. Standard EXIF Scan
        if "exif" in img.info:
            exif = piexif.load(img.info["exif"])
            software = exif.get("0th", {}).get(piexif.ImageIFD.Software, b"").decode().lower()
            for tool in suspicious_list:
                if tool in software:
                    return f"Metadata Fraud: Software signature '{tool}' detected", 0.90
    except Exception: pass

    return detect_ela(file_bytes)

def detect_ela(file_bytes: bytes, threshold: float = 30.0) -> tuple:
    """Detects inconsistencies caused by overlays and digital pastes.
    Returns (message, confidence) or (None, 0.0).
    """
    try:
        original = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        buf = io.BytesIO(); original.save(buf, "JPEG", quality=90)
        resaved = Image.open(io.BytesIO(buf.getvalue()))
        ela_img = ImageChops.difference(original, resaved)
        max_diff = max([e[1] for e in ela_img.getextrema()])
        if max_diff > threshold:
            # Confidence based on pixel inconsistency magnitude: higher diff = higher confidence
            confidence = min(0.96, 0.70 + (max_diff / 255) * 0.26)
            return "Visual Tampering: Pixel-level inconsistencies detected (ELA)", confidence
    except: pass
    return None, 0.0