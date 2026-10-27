"""Software image enhancement the agent can apply to a rectified dial."""
from __future__ import annotations

import cv2
import numpy as np

OPS = {
    "clahe": "local contrast equalisation (CLAHE on the L channel) for haze, dirty glass or flat light",
    "deglare": "inpaint clipped specular highlights from their surroundings (Telea); only helps partial glare",
    "denoise": "non-local-means denoising for low-light noise",
}


def clahe(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    L = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(L)
    return cv2.cvtColor(cv2.merge([L, a, b]), cv2.COLOR_LAB2BGR)


def deglare(img: np.ndarray) -> np.ndarray:
    mask = (img.max(axis=2) >= 248).astype(np.uint8) * 255
    if mask.sum() == 0:
        return img
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    return cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)


def denoise(img: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(img, None, 7, 7, 5, 15)


def apply(img: np.ndarray, ops: list) -> np.ndarray:
    out = img
    for op in ops:
        try:
            if op == "clahe":
                out = clahe(out)
            elif op == "deglare":
                out = deglare(out)
            elif op == "denoise":
                out = denoise(out)
        except (cv2.error, AttributeError):  # operation not in this OpenCV build: skip it
            continue
    return out
