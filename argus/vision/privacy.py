"""Privacy-by-design for user uploads: blur any faces with OpenCV 5's DNN face detector (YuNet).

Uploaded photos are processed in memory; only the blurred, rectified evidence
overlay is kept, and only for the lifetime of the server process.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

MODEL = Path(__file__).resolve().parents[2] / "models" / "face_detection_yunet_2023mar.onnx"
_DET = None


def _detector():
    global _DET
    if _DET is None and MODEL.exists():
        _DET = cv2.FaceDetectorYN.create(str(MODEL), "", (320, 320), 0.7, 0.3, 100)
    return _DET


def available() -> bool:
    try:
        return _detector() is not None
    except cv2.error:
        return False


def blur_faces(img: np.ndarray) -> tuple[np.ndarray, int]:
    det = _detector()
    if det is None:
        return img, -1
    h, w = img.shape[:2]
    s = 640.0 / max(h, w) if max(h, w) > 640 else 1.0
    small = cv2.resize(img, (int(w * s), int(h * s))) if s != 1.0 else img
    det.setInputSize((small.shape[1], small.shape[0]))
    _, faces = det.detect(small)
    if faces is None:
        return img, 0
    out = img.copy()
    for f in faces:
        x, y, fw, fh = (f[:4] / s).astype(int)
        x0, y0 = max(0, x - fw // 4), max(0, y - fh // 4)
        x1, y1 = min(w, x + fw + fw // 4), min(h, y + fh + fh // 4)
        roi = out[y0:y1, x0:x1]
        if roi.size:
            k = max(15, (max(roi.shape[:2]) // 3) | 1)
            out[y0:y1, x0:x1] = cv2.GaussianBlur(roi, (k, k), 0)
    return out, len(faces)
