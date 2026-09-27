"""
Stage 2 of the ANPR pipeline: reads text from a plate crop.

Techniques used to push accuracy toward the >90% target:
  1. Preprocessing (grayscale, contrast boost, adaptive threshold) before OCR.
  2. Multi-frame confidence voting: a vehicle is seen across several
     consecutive frames per camera pass, so we OCR each one and vote on the
     final string instead of trusting a single frame.
  3. Regex-based format correction against the Indian plate pattern, with
     auto-correction for common OCR confusions (0/O, 1/I, 8/B, 5/S).

NOTE: pytesseract (Tesseract) is used here as the working stand-in engine so
this file is fully runnable without GPU/heavy downloads. For production,
swap `_read_raw` to call a fine-tuned CRNN or PARSeq/TrOCR model instead --
the voting and correction logic around it stays the same.
"""
import re
from collections import Counter
import cv2
import numpy as np
import pytesseract

# Indian plate format: SS DD L(L) NNNN  e.g. UP53AB1234, DL8CAF5030
PLATE_REGEX = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")

CONFUSIONS = {
    "0": "O", "O": "0",
    "1": "I", "I": "1",
    "8": "B", "B": "8",
    "5": "S", "S": "5",
    "2": "Z", "Z": "2",
}

TESS_CONFIG = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def preprocess(plate_bgr):
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 9, 60, 60)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Otsu can invert depending on background; plates here are dark text on
    # light background, so ensure background (majority pixel value) is white.
    if np.mean(thresh) < 127:
        thresh = cv2.bitwise_not(thresh)
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return thresh


def _read_raw(plate_bgr):
    """Runs OCR on one frame, returns (text, mean_confidence 0-100)."""
    processed = preprocess(plate_bgr)
    data = pytesseract.image_to_data(
        processed, config=TESS_CONFIG, output_type=pytesseract.Output.DICT
    )
    chars, confs = [], []
    for i, txt in enumerate(data["text"]):
        txt = txt.strip().upper()
        conf = float(data["conf"][i])
        if txt and conf > 0:
            chars.append(re.sub(r"[^A-Z0-9]", "", txt))
            confs.append(conf)
    text = "".join(chars)
    mean_conf = sum(confs) / len(confs) if confs else 0.0
    return text, mean_conf


def correct_format(text):
    """If text is one edit away from a valid plate format, try confusion-swap correction."""
    if PLATE_REGEX.match(text):
        return text, True

    for i, ch in enumerate(text):
        if ch in CONFUSIONS:
            candidate = text[:i] + CONFUSIONS[ch] + text[i + 1:]
            if PLATE_REGEX.match(candidate):
                return candidate, True
    return text, False


def read_plate_multiframe(frame_crops):
    """
    Takes a list of plate crop images (same vehicle, consecutive frames from
    one camera pass) and returns the highest-confidence, format-corrected
    plate reading via majority vote.
    """
    reads = []
    for crop in frame_crops:
        text, conf = _read_raw(crop)
        if not text:
            continue
        corrected, was_corrected = correct_format(text)
        reads.append({"raw": text, "corrected": corrected, "confidence": conf, "format_valid": was_corrected})

    if not reads:
        return {"plate": None, "confidence": 0.0, "frames_used": 0, "votes": {}}

    # weight votes by confidence; prefer format-valid reads
    vote_scores = Counter()
    for r in reads:
        weight = r["confidence"] * (1.5 if r["format_valid"] else 1.0)
        vote_scores[r["corrected"]] += weight

    best_plate, _ = vote_scores.most_common(1)[0]
    matching_reads = [r for r in reads if r["corrected"] == best_plate]
    final_confidence = sum(r["confidence"] for r in matching_reads) / len(matching_reads)

    return {
        "plate": best_plate,
        "confidence": round(final_confidence, 1),
        "frames_used": len(reads),
        "format_valid": any(r["format_valid"] for r in matching_reads),
        "votes": dict(vote_scores),
    }


if __name__ == "__main__":
    import sys, glob
    paths = sys.argv[1:] if len(sys.argv) > 1 else []
    if not paths:
        print("Usage: python ocr_engine.py plate_frame1.png plate_frame2.png plate_frame3.png")
    else:
        crops = [cv2.imread(p) for p in paths]
        result = read_plate_multiframe(crops)
        print(result)
