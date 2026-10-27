"""OpenCV 5 perception pipeline for analog gauges.

    capture -> localize (ArUco fiducial | robot-pose prediction | Hough circle)
            -> rectify (homography to a canonical fronto-parallel dial, refined by an
               ellipse fit of the dial rim)
            -> assess (resolution, obliqueness, blur, motion blur, exposure, glare on
               the needle sweep, noise, haze, occlusion)
            -> read (polar-unwrapped needle detection + robust line fit, tick-template
               scale calibration, value + calibrated uncertainty)

Every step reports what it saw so the agent can decide what to do next.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from ..registry import GaugeSpec
from . import geometry as g
from .textures import ARUCO_DICT, tag_corners_mm

CANON = 512
RC = 200.0  # canonical dial radius in px
CX = CY = CANON / 2.0
N_ANG = 720  # polar angle bins (0.5 deg)
N_RAD = 240
POLAR_MAXR = 1.2 * RC

# Bands (fractions of dial radius) where each feature lives
NEEDLE_BAND = (0.20, 0.54)
MAJOR_BAND = (0.80, 0.86)
RIM_BAND = (0.82, 1.15)

# Uncertainty model constants (calibrated on held-out simulator seeds, see eval/calibrate.py)
SIGMA_FLOOR_DEG = 0.22
SIGMA_FWHM_K = 0.06
SIGMA_SCALE = 1.0


def _detector():
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.cornerRefinementWinSize = 4
    p.adaptiveThreshWinSizeMin = 5
    p.adaptiveThreshWinSizeMax = 45
    p.adaptiveThreshWinSizeStep = 8
    return cv2.aruco.ArucoDetector(d, p)


_DET = None


def detector():
    global _DET
    if _DET is None:
        _DET = _detector()
    return _DET


@dataclass
class Localization:
    ok: bool
    method: str  # fiducial | predicted | circle | none
    H_face_to_img: Optional[np.ndarray] = None
    marker_id: Optional[int] = None
    markers_seen: list = field(default_factory=list)
    radius_px: float = 0.0
    axis_ratio: float = 1.0
    center_px: tuple = (0.0, 0.0)
    note: str = ""


@dataclass
class Rectified:
    canon: np.ndarray  # BGR uint8 CANON x CANON
    H_canon_from_img: np.ndarray
    rim_coverage: float
    rim_residual: float
    refine_shift: float
    refine_scale: float
    hub: Optional[tuple] = None


@dataclass
class Reading:
    ok: bool
    value: float = float("nan")
    sigma: float = float("nan")
    theta: float = float("nan")
    theta_polar: float = float("nan")
    theta_line: float = float("nan")
    agreement_deg: float = float("nan")
    needle_strength: float = 0.0
    peak_ratio: float = 0.0
    fwhm_deg: float = 0.0
    tick_offset_deg: float = 0.0
    tick_score: float = 0.0
    tick_sigma_deg: float = 0.0
    scale_verified: bool = False
    confidence: float = 0.0
    flags: list = field(default_factory=list)
    line_center_dist: float = 0.0

    def to_dict(self, spec: Optional[GaugeSpec] = None) -> dict:
        d = {k: (round(v, 4) if isinstance(v, float) and math.isfinite(v) else v) for k, v in self.__dict__.items()}
        if spec is not None and self.ok:
            d["value_pct_span"] = round((self.value - spec.min_value) / spec.span * 100, 2)
        return d


# ---------------------------------------------------------------------------
# Localization


def detect_markers(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if not hasattr(cv2, "aruco"):  # build without objdetect/aruco: fall back to pose prediction / Hough
        return []
    corners, ids, _ = detector().detectMarkers(gray)
    out = []
    if ids is not None:
        for c, i in zip(corners, ids.ravel()):
            out.append((int(i), c.reshape(4, 2).astype(np.float64)))
    return out


def _geom_from_H(H: np.ndarray, R_mm: float):
    pts = np.array([g.face_point(t, R_mm) for t in np.linspace(-180, 180, 48, endpoint=False)])
    pi = g.apply_h(H, pts).astype(np.float32)
    (cx, cy), (w, h), _ = cv2.fitEllipse(pi)
    r = 0.25 * (w + h)
    return (float(cx), float(cy)), float(r), float(min(w, h) / max(w, h))


def localize(img: np.ndarray, spec: GaugeSpec, predicted_H: Optional[np.ndarray] = None,
             allow_circle: bool = True) -> Localization:
    markers = detect_markers(img)
    seen = [m[0] for m in markers]
    for mid, corners in markers:
        if mid == spec.marker_id:
            H = cv2.getPerspectiveTransform(tag_corners_mm().astype(np.float32), corners.astype(np.float32))
            H = H / H[2, 2]
            c, r, ar = _geom_from_H(H, spec.dial_radius_mm)
            return Localization(True, "fiducial", H, mid, seen, r, ar, c)
    if predicted_H is not None:
        c, r, ar = _geom_from_H(predicted_H, spec.dial_radius_mm)
        H = predicted_H.copy()
        # snap to the dial with a local Hough search (pose prediction is only ~1-3% accurate)
        hc = find_dial_circle(img, approx=(c, r))
        note = "pose prediction"
        if hc is not None:
            (hx, hy), hr = hc
            # keep the predicted perspective, correct translation and scale around the dial centre
            s = hr / r if 0.7 < hr / r < 1.4 else 1.0
            A = g.translate(hx, hy) @ g.scale(s) @ g.translate(-c[0], -c[1])
            H = A @ H
            c, r, ar = _geom_from_H(H, spec.dial_radius_mm)
            note = "pose prediction snapped to Hough circle"
        return Localization(True, "predicted", H / H[2, 2], None, seen, r, ar, c, note)
    if allow_circle:
        hc = find_dial_circle(img)
        if hc is not None:
            (hx, hy), hr = hc
            s = hr / spec.dial_radius_mm
            H = np.array([[s, 0, hx], [0, s, hy], [0, 0, 1]], np.float64)
            return Localization(True, "circle", H, None, seen, hr, 1.0, (hx, hy), "Hough circle, roll unknown")
    return Localization(False, "none", None, None, seen, note="no fiducial, no prediction, no circle")


def find_dial_circle(img: np.ndarray, approx=None):
    """Find the dial face with HOUGH_GRADIENT_ALT.  ``approx`` = ((cx, cy), r) to search locally."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape
    scale = 800.0 / max(h, w) if max(h, w) > 800 else 1.0
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1 else gray
    small = cv2.GaussianBlur(small, (0, 0), 1.2)
    if approx is not None:
        (cx, cy), r = approx
        rmin, rmax = int(r * 0.75 * scale), int(r * 1.3 * scale)
    else:
        rmin, rmax = int(min(h, w) * 0.06 * scale), int(min(h, w) * 0.48 * scale)
    circles = cv2.HoughCircles(small, cv2.HOUGH_GRADIENT_ALT, dp=1.5, minDist=max(8, rmin),
                               param1=150, param2=0.8, minRadius=max(4, rmin), maxRadius=max(rmin + 2, rmax))
    if circles is None:
        circles = cv2.HoughCircles(small, cv2.HOUGH_GRADIENT_ALT, dp=1.5, minDist=max(8, rmin),
                                   param1=100, param2=0.6, minRadius=max(4, rmin), maxRadius=max(rmin + 2, rmax))
    if circles is None:
        return None
    cs = circles.reshape(-1, 3) / scale
    if approx is not None:
        (cx, cy), r = approx
        d = np.hypot(cs[:, 0] - cx, cs[:, 1] - cy) + 0.5 * np.abs(cs[:, 2] - r)
        best = cs[int(np.argmin(d))]
        if math.hypot(best[0] - cx, best[1] - cy) > 0.6 * r:
            return None
    else:
        # prefer large, bright-faced circles
        scores = []
        for x, y, r in cs:
            m = np.zeros_like(gray)
            cv2.circle(m, (int(x), int(y)), int(r * 0.8), 255, -1)
            scores.append(r * (cv2.mean(gray, mask=m)[0] / 255.0 + 0.2))
        best = cs[int(np.argmax(scores))]
    return (float(best[0]), float(best[1])), float(best[2])


# ---------------------------------------------------------------------------
# Rectification


def canon_from_face(spec: GaugeSpec) -> np.ndarray:
    s = RC / spec.dial_radius_mm
    return np.array([[s, 0, CX], [0, s, CY], [0, 0, 1]], np.float64)


def warp_canon(img: np.ndarray, H_canon_from_img: np.ndarray) -> np.ndarray:
    return cv2.warpPerspective(img, H_canon_from_img, (CANON, CANON), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def to_polar(img: np.ndarray, center=(CX, CY), max_r=POLAR_MAXR, n_rad=N_RAD, n_ang=N_ANG) -> np.ndarray:
    return cv2.warpPolar(img, (n_rad, n_ang), center, max_r, cv2.WARP_POLAR_LINEAR | cv2.INTER_LINEAR)


def _col(fr: float, max_r=POLAR_MAXR, n_rad=N_RAD) -> int:
    return int(round(fr * RC / max_r * n_rad))


def rim_points(canon_gray: np.ndarray, band=RIM_BAND):
    """Radial search for the face->rim (bright->dark) edge.  Returns (points, strengths, radii)."""
    P = to_polar(canon_gray.astype(np.float32))
    P = cv2.GaussianBlur(P, (3, 1), 0)
    d = cv2.Sobel(P, cv2.CV_32F, 1, 0, ksize=3)
    j0, j1 = _col(band[0]), _col(band[1])
    band = d[:, j0:j1]
    idx = np.argmin(band, axis=1)
    strength = -band[np.arange(N_ANG), idx]
    # sub-pixel parabola
    jj = idx.astype(np.float64)
    for i in range(N_ANG):
        k = idx[i]
        if 0 < k < band.shape[1] - 1:
            a, b, c = band[i, k - 1], band[i, k], band[i, k + 1]
            den = a - 2 * b + c
            if abs(den) > 1e-6:
                jj[i] = k + 0.5 * (a - c) / den
    r = (jj + j0) * POLAR_MAXR / N_RAD
    phi = np.deg2rad(np.arange(N_ANG) * 360.0 / N_ANG)
    pts = np.stack([CX + r * np.cos(phi), CY + r * np.sin(phi)], axis=1)
    return pts, strength, r


def rectify(img: np.ndarray, loc: Localization, spec: GaugeSpec, refine: bool = True) -> Rectified:
    H0 = canon_from_face(spec) @ np.linalg.inv(loc.H_face_to_img)
    H0 /= H0[2, 2]
    canon = warp_canon(img, H0)
    H = H0
    cov, resid, shift, sc_change = 0.0, 0.0, 0.0, 1.0
    if refine:
        # How much we trust the rim depends on how good the starting geometry is: a fiducial homography is
        # already ~1 px accurate, so the rim may only nudge it and only if the fit is clean.
        if loc.method == "fiducial":
            max_shift, band, max_resid, max_aniso = 0.04, (0.93, 1.07), 1.2, 0.03
        elif loc.method == "predicted":
            max_shift, band, max_resid, max_aniso = 0.25, (0.84, 1.16), 2.5, 0.25
        else:
            max_shift, band, max_resid, max_aniso = 0.6, (0.80, 1.20), 3.0, 0.25
        for _ in range(2):
            gray = cv2.cvtColor(canon, cv2.COLOR_BGR2GRAY)
            pts, st, rr = rim_points(gray, band)
            thr = max(8.0, 0.35 * float(np.percentile(st, 80)))
            good = st > thr
            if good.sum() < 40:
                break
            p = pts[good].astype(np.float32)
            ell = cv2.fitEllipse(p)
            for _k in range(2):  # trim outliers against the current ellipse and refit
                (ex, ey), (ew, eh), ea = ell
                ca, sa = math.cos(math.radians(ea)), math.sin(math.radians(ea))
                dx, dy = p[:, 0] - ex, p[:, 1] - ey
                xr, yr = dx * ca + dy * sa, -dx * sa + dy * ca
                rr_n = np.sqrt((xr / (ew / 2)) ** 2 + (yr / (eh / 2)) ** 2)
                res = np.abs(rr_n - 1.0) * RC
                keep = res < max(1.5, np.percentile(res, 75) * 1.5)
                if keep.sum() < 30:
                    break
                p = p[keep]
                ell = cv2.fitEllipse(p)
            (ex, ey), (ew, eh), ea = ell
            cov = float(len(p) / N_ANG)
            ca, sa = math.cos(math.radians(ea)), math.sin(math.radians(ea))
            dx, dy = p[:, 0] - ex, p[:, 1] - ey
            xr, yr = dx * ca + dy * sa, -dx * sa + dy * ca
            resid = float(np.median(np.abs(np.sqrt((xr / (ew / 2)) ** 2 + (yr / (eh / 2)) ** 2) - 1.0)) * RC)
            shift = math.hypot(ex - CX, ey - CY) / RC
            sc_change = 0.25 * (ew + eh) / RC
            aniso = 1 - min(ew, eh) / max(ew, eh)
            if (shift > max_shift or not (0.8 < sc_change < 1.25) or aniso > max_aniso or resid > max_resid
                    or cov < 0.35):
                break  # implausible or noisy correction: keep the geometric estimate
            A = (g.translate(CX, CY) @ g.rot2d(ea) @ g.scale(2 * RC / ew, 2 * RC / eh) @ g.rot2d(-ea)
                 @ g.translate(-ex, -ey))
            H = A @ H
            H /= H[2, 2]
            canon = warp_canon(img, H)
            if shift < 0.002 and abs(sc_change - 1) < 0.002:
                break
    # Re-centre on the needle's pivot (hub cap).  A few pixels of pivot error bias the polar needle angle
    # by degrees at the inner radii, and tick calibration cannot see a translation error.
    hub = find_hub(canon)
    if hub is not None:
        hx, hy, hr = hub
        dx, dy = hx - CX, hy - CY
        dd = math.hypot(dx, dy)
        if 1.2 < dd < 0.08 * RC:
            H = g.translate(-dx, -dy) @ H
            H /= H[2, 2]
            canon = warp_canon(img, H)
    rect = Rectified(canon, H, cov, resid, shift, sc_change)
    rect.hub = hub
    return rect


def find_hub(canon: np.ndarray):
    """Locate the needle hub cap (dark disc at the pivot) near the canonical centre."""
    gray = cv2.cvtColor(canon, cv2.COLOR_BGR2GRAY)
    m = int(0.17 * RC)
    x0, y0 = int(CX) - m, int(CY) - m
    crop = cv2.GaussianBlur(gray[y0:y0 + 2 * m, x0:x0 + 2 * m], (0, 0), 1.0)
    circles = cv2.HoughCircles(crop, cv2.HOUGH_GRADIENT_ALT, dp=1, minDist=8, param1=60, param2=0.55,
                               minRadius=int(0.045 * RC), maxRadius=int(0.13 * RC))
    if circles is None:
        return None
    face = float(np.percentile(gray, 70))
    best, best_d = None, 1e9
    for x, y, r in circles.reshape(-1, 3):
        mask = np.zeros_like(crop)
        cv2.circle(mask, (int(x), int(y)), max(1, int(r * 0.85)), 255, -1)
        cv2.circle(mask, (int(x), int(y)), max(1, int(r * 0.45)), 0, -1)  # skip a lighter centre dot
        inside = cv2.mean(crop, mask=mask)[0]
        if inside > 0.6 * face:
            continue
        d = math.hypot(x + x0 - CX, y + y0 - CY)
        if d < best_d:
            best, best_d = (float(x + x0), float(y + y0), float(r)), d
    return best


# ---------------------------------------------------------------------------
# Reading


def _darkness_polar(canon: np.ndarray, spec: GaugeSpec) -> np.ndarray:
    gray = cv2.cvtColor(canon, cv2.COLOR_BGR2GRAY).astype(np.float32)
    P = to_polar(gray)
    bg = np.percentile(P, 75, axis=0, keepdims=True)
    D = np.clip((bg - P) / np.maximum(bg, 25.0), 0, 1)
    if spec.needle_color == "red":
        b, gch, r = cv2.split(canon.astype(np.float32))
        red = np.clip((r - np.maximum(gch, b)) / 120.0, 0, 1)
        D = np.maximum(D, to_polar(red))
    return D


def _circ_smooth(x: np.ndarray, sigma_bins: float) -> np.ndarray:
    n = len(x)
    k = int(max(3, round(sigma_bins * 4)))
    pad = np.concatenate([x[-k:], x, x[:k]])
    sm = cv2.GaussianBlur(pad.reshape(1, -1).astype(np.float32), (2 * k + 1, 1), sigma_bins).ravel()
    return sm[k:k + n]


def _parabolic(y: np.ndarray, i: int) -> float:
    n = len(y)
    a, b, c = y[(i - 1) % n], y[i], y[(i + 1) % n]
    den = a - 2 * b + c
    return i + (0.5 * (a - c) / den if abs(den) > 1e-9 else 0.0)


def _fwhm_bins(y: np.ndarray, i: int) -> float:
    n = len(y)
    half = y[i] / 2.0
    lo = 0
    while lo < n // 4 and y[(i - lo) % n] > half:
        lo += 1
    hi = 0
    while hi < n // 4 and y[(i + hi) % n] > half:
        hi += 1
    return float(lo + hi)


def needle_profile(D: np.ndarray):
    j0, j1 = _col(NEEDLE_BAND[0]), _col(NEEDLE_BAND[1])
    band = D[:, j0:j1]
    cover = (band > 0.33).mean(axis=1)
    mean = band.mean(axis=1)
    s = 0.6 * cover + 0.4 * np.clip(mean / 0.6, 0, 1)
    return _circ_smooth(s, 1.5)


def detect_needle(D: np.ndarray, canon: np.ndarray, spec: GaugeSpec, theta_hint: Optional[float] = None):
    S = needle_profile(D)
    if theta_hint is not None:
        i_h = int(round(g.theta_to_phi(theta_hint) / 360 * N_ANG)) % N_ANG
        w = np.exp(-0.5 * (((np.arange(N_ANG) - i_h + N_ANG / 2) % N_ANG - N_ANG / 2) / 30.0) ** 2)
        i = int(np.argmax(S * (0.4 + 0.6 * w)))
    else:
        i = int(np.argmax(S))
    peak = float(S[i])
    excl = int(10 * N_ANG / 360)
    mask = np.ones(N_ANG, bool)
    for k in range(-excl, excl + 1):
        mask[(i + k) % N_ANG] = False
    second = float(S[mask].max()) if mask.any() else 0.0
    ip = _parabolic(S, i)
    theta_p = g.phi_to_theta(ip * 360.0 / N_ANG)
    fwhm = _fwhm_bins(S, i) * 360.0 / N_ANG
    # robust line fit on dark pixels in a wedge around the polar estimate
    gray = cv2.cvtColor(canon, cv2.COLOR_BGR2GRAY)
    yy, xx = np.mgrid[0:CANON, 0:CANON]
    dx, dy = xx - CX, yy - CY
    rr = np.hypot(dx, dy)
    ang = np.degrees(np.arctan2(dx, -dy))
    dang = (ang - theta_p + 180) % 360 - 180
    wedge = (np.abs(dang) < 5.0) & (rr > 0.16 * RC) & (rr < 0.54 * RC)
    theta_l, cdist = float("nan"), float("nan")
    vals = gray[wedge]
    if vals.size > 50:
        face = float(np.percentile(gray[(rr < 0.9 * RC)], 75))
        thr, _ = cv2.threshold(vals.astype(np.uint8).reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = min(float(thr), face * 0.78)
        if spec.needle_color == "red":
            b, gch, r = cv2.split(canon.astype(np.int16))
            redm = (r - np.maximum(gch, b)) > 45
            sel = wedge & ((gray < thr) | redm)
        else:
            sel = wedge & (gray < thr)
        ys, xs = np.nonzero(sel)
        if len(xs) > 30:
            pts = np.stack([xs, ys], axis=1).astype(np.float32)
            vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            m = pts.mean(axis=0)
            if (m[0] - CX) * vx + (m[1] - CY) * vy < 0:
                vx, vy = -vx, -vy
            theta_l = g.angle_of(float(vx), float(vy))
            cdist = abs((CX - x0) * vy - (CY - y0) * vx) / RC
    return dict(theta_polar=theta_p, theta_line=theta_l, peak=peak, second=second, fwhm=fwhm,
                line_center_dist=cdist, profile=S)


def calibrate_ticks(D: np.ndarray, spec: GaugeSpec, theta_needle: Optional[float], search_deg: float = 12.0):
    """Match the expected major-tick pattern to the dial: returns (offset_deg, score, sigma_deg)."""
    j0, j1 = _col(MAJOR_BAND[0]), _col(MAJOR_BAND[1])
    T = D[:, j0:j1].mean(axis=1).astype(np.float64)
    if theta_needle is not None and math.isfinite(theta_needle):
        i_n = int(round(g.theta_to_phi(theta_needle) / 360 * N_ANG))
        med = float(np.median(T))
        for k in range(-8, 9):
            T[(i_n + k) % N_ANG] = med
    base = _circ_smooth(T, 12.0)
    hp = np.clip(T - base, 0, None)
    if hp.max() <= 1e-6:
        return 0.0, 0.0, 3.0
    hp = hp / hp.max()
    n = spec.major_ticks
    ticks = np.array([spec.theta_min + (spec.theta_max - spec.theta_min) * k / (n - 1) for k in range(n)])
    phis = (ticks - 90.0) % 360.0
    grid = np.arange(-search_deg, search_deg + 1e-9, 0.25)
    xs = np.arange(N_ANG + 1) * 360.0 / N_ANG
    hpw = np.append(hp, hp[0])
    scores = []
    for dlt in grid:
        pos = (phis + dlt) % 360.0
        scores.append(np.interp(pos, xs, hpw).mean())
    scores = np.array(scores)
    b = int(np.argmax(scores))
    off = float(grid[b])
    if 0 < b < len(grid) - 1:
        a, c0, c = scores[b - 1], scores[b], scores[b + 1]
        den = a - 2 * c0 + c
        if abs(den) > 1e-9:
            off += 0.25 * 0.5 * (a - c) / den
    # per-tick residuals -> sigma
    res, found = [], 0
    for ph in phis:
        center = (ph + off) / 360.0 * N_ANG
        lo, hi = int(math.floor(center - 4)), int(math.ceil(center + 4))
        idx = np.arange(lo, hi + 1) % N_ANG
        seg = hp[idx]
        k = int(np.argmax(seg))
        if seg[k] > 0.25:
            found += 1
            res.append((lo + k - center) * 360.0 / N_ANG)
    score = found / n
    sig = float(np.std(res) / math.sqrt(max(1, len(res)))) if len(res) >= 3 else 2.0
    return off, float(score), max(0.05, sig)


def read_canon(canon: np.ndarray, spec: GaugeSpec, unknown_roll: bool = False,
               theta_hint: Optional[float] = None, ticks: bool = True) -> Reading:
    D = _darkness_polar(canon, spec)
    nd = detect_needle(D, canon, spec, theta_hint)
    flags = []
    tp, tl = nd["theta_polar"], nd["theta_line"]
    theta = tp
    agree = float("nan")
    if math.isfinite(tl):
        # the polar estimate is primary (median error ~0.1 deg in clean frames); the line fit is an
        # independent check that the detected structure is a straight radial needle through the pivot
        agree = abs(g.wrap180(tl - tp))
        if agree > 2.5 or nd["line_center_dist"] > 0.08:
            flags.append("needle_methods_disagree")
    else:
        flags.append("needle_line_fit_failed")
    if ticks:
        off, tscore, tsig = calibrate_ticks(D, spec, theta, search_deg=180.0 if unknown_roll else 12.0)
    else:
        off, tscore, tsig = 0.0, 0.0, 1.0
    verified = tscore >= 0.55
    if verified:
        theta_c = g.wrap180(theta - off)
        if not unknown_roll and abs(off) > 6.0:
            flags.append("scale_offset_large")
    else:
        theta_c = theta
        tsig = 1.0 if not unknown_roll else 6.0
        flags.append("scale_unverified")
    frac = g.theta_to_fraction(theta_c, spec.theta_min, spec.theta_max)
    value = spec.min_value + frac * spec.span
    if frac < -0.006:
        flags.append("below_range_on_stop")
    elif frac > 1.006:
        flags.append("over_range")
    peak, second, fwhm = nd["peak"], nd["second"], nd["fwhm"]
    ratio = peak / max(second, 0.02)
    a = agree if math.isfinite(agree) else 3.0
    sig_deg = SIGMA_SCALE * math.sqrt(SIGMA_FLOOR_DEG ** 2 + (SIGMA_FWHM_K * fwhm) ** 2
                                      + (max(0.0, min(a, 8.0) - 0.8) / 3.0) ** 2 + tsig ** 2)
    sigma = sig_deg / (spec.theta_max - spec.theta_min) * spec.span
    conf = (np.clip((peak - 0.22) / 0.40, 0, 1) * np.clip((ratio - 1.15) / 1.0, 0, 1)
            * (1.0 if a < 2.0 else max(0.0, 1 - (a - 2.0) / 5.0)) * (1.0 if verified else 0.8))
    if peak < 0.35:
        flags.append("needle_weak")
    if ratio < 1.6:
        flags.append("needle_ambiguous")
    return Reading(True, float(value), float(sigma), float(theta_c), float(tp), float(tl), float(agree),
                   float(peak), float(ratio), float(fwhm), float(off), float(tscore), float(tsig), bool(verified),
                   float(conf), flags, float(nd["line_center_dist"]))


# ---------------------------------------------------------------------------
# Quality assessment


def immerkaer_noise(gray: np.ndarray) -> float:
    kern = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    h, w = gray.shape
    if h < 8 or w < 8:
        return 0.0
    conv = cv2.filter2D(gray.astype(np.float32), -1, kern)
    return float(math.sqrt(math.pi / 2) * np.abs(conv[1:-1, 1:-1]).sum() / (6.0 * (w - 2) * (h - 2)))


def _dial_roi(img, loc: Localization, pad=1.05):
    (cx, cy), r = loc.center_px, loc.radius_px
    x0, y0 = int(max(0, cx - pad * r)), int(max(0, cy - pad * r))
    x1, y1 = int(min(img.shape[1], cx + pad * r)), int(min(img.shape[0], cy + pad * r))
    return img[y0:y1, x0:x1]


def assess(img: np.ndarray, loc: Localization, rect: Rectified, spec: GaugeSpec,
           reading: Optional[Reading] = None) -> dict:
    m: dict = {"localization": loc.method, "marker_found": loc.method == "fiducial",
               "dial_radius_px": round(loc.radius_px, 1), "view_axis_ratio": round(loc.axis_ratio, 3)}
    canon = rect.canon
    gray = cv2.cvtColor(canon, cv2.COLOR_BGR2GRAY)
    yy, xx = np.mgrid[0:CANON, 0:CANON]
    rr = np.hypot(xx - CX, yy - CY)
    face = rr < 0.93 * RC
    mx = canon.max(axis=2)
    clipped = mx >= 250
    m["clipped_frac"] = round(float(clipped[face].mean()), 4)
    m["face_level"] = round(float(np.percentile(gray[face], 75)) / 255.0, 3)
    p5, p95 = np.percentile(gray[face], [5, 95])
    m["contrast"] = round(float(p95 - p5), 1)
    # glare on the needle sweep (polar clipped map)
    Pc = to_polar(clipped.astype(np.float32))
    j0, j1 = _col(0.2), _col(0.86)
    ang_clip = Pc[:, j0:j1].mean(axis=1)
    thetas = np.array([g.phi_to_theta(i * 360.0 / N_ANG) for i in range(N_ANG)])
    in_sweep = (thetas >= spec.theta_min) & (thetas <= spec.theta_max)
    glare_ang = ang_clip > 0.25
    m["glare_sweep_frac"] = round(float(glare_ang[in_sweep].mean()), 4)
    m["rim_coverage"] = round(rect.rim_coverage, 3)
    m["rim_residual_px"] = round(rect.rim_residual, 2)
    # sharpness & motion blur on the raw dial at a normalised scale
    roi = _dial_roi(img, loc)
    if roi.size > 0 and loc.radius_px > 4:
        sc = 100.0 / loc.radius_px
        roi_n = cv2.resize(roi, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR)
        rg = cv2.cvtColor(roi_n, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lap = cv2.Laplacian(rg, cv2.CV_32F, ksize=3)
        std = float(rg.std()) + 1.0
        m["sharpness"] = round(float(lap.var()) / (std ** 2) * 100.0, 2)
        gx = cv2.Sobel(rg, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(rg, cv2.CV_32F, 0, 1)
        j = np.array([[float((gx * gx).sum()), float((gx * gy).sum())], [float((gx * gy).sum()), float((gy * gy).sum())]])
        ev = np.linalg.eigvalsh(j)
        m["blur_anisotropy"] = round(float(ev[1] / max(ev[0], 1e-6)), 2)
        raw = cv2.cvtColor(_dial_roi(img, loc, 0.7), cv2.COLOR_BGR2GRAY)
        m["noise_sigma"] = round(immerkaer_noise(raw), 2)
    else:
        m.update(sharpness=0.0, blur_anisotropy=1.0, noise_sigma=0.0)
    if reading is not None and reading.ok:
        m["needle_strength"] = round(reading.needle_strength, 3)
        m["needle_peak_ratio"] = round(reading.peak_ratio, 2)
        m["tick_score"] = round(reading.tick_score, 2)
        i_n = int(round(g.theta_to_phi(reading.theta) / 360 * N_ANG)) % N_ANG
        w = int(8 * N_ANG / 360)
        near = [glare_ang[(i_n + k) % N_ANG] for k in range(-w, w + 1)]
        m["glare_near_needle"] = bool(np.mean(near) > 0.2)
    m["issues"] = issues_from(m)
    return m


# Thresholds (set from simulator calibration; see docs/EVALUATION.md)
TH = dict(min_radius_px=70, min_axis_ratio=0.72, very_oblique=0.45, min_sharpness=7.0, motion_anisotropy=2.6,
          max_clipped=0.45,
          max_glare_sweep=0.03, min_face_level=0.30, max_noise=5.5, min_contrast=70, min_rim_coverage=0.6,
          min_needle_strength=0.35, min_peak_ratio=1.6, min_tick_score=0.55)

ISSUE_TEXT = {
    "no_dial": "the dial could not be located in the frame",
    "fiducial_missing": "the gauge's fiducial tag was not detected; geometry comes from the robot pose estimate",
    "low_resolution": "the dial is too small in the frame for a precise reading",
    "oblique_view": "the dial is viewed at an angle (foreshortened)",
    "very_oblique_view": "the dial is viewed at a very steep angle; resolution across the dial is poor",
    "defocus_blur": "the image is blurred in all directions (focus)",
    "motion_blur": "the image is blurred in one direction (vibration or movement during exposure)",
    "glare_on_needle": "specular glare saturates the dial where the needle is",
    "glare_on_scale": "specular glare saturates part of the needle sweep",
    "overexposed": "most of the dial face is clipped to white (exposure too high)",
    "underexposed": "the dial is too dark",
    "noisy": "sensor noise is high (low light / high gain)",
    "low_contrast": "the dial has low contrast (haze, dirty glass, or flat lighting)",
    "occluded": "something in front of the dial hides part of the rim",
    "needle_not_clear": "the needle is weak or ambiguous in the image",
    "scale_not_verified": "the printed scale could not be matched to the expected tick pattern",
    "needle_flutter": "the needle is fluttering (process pulsation); more frames are needed to find its centre",
    "frames_inconsistent": "individual burst frames cannot be read consistently (moving needle or heavy noise)",
}


def issues_from(m: dict) -> list:
    out = []
    if m.get("localization") == "none":
        return ["no_dial"]
    if not m.get("marker_found"):
        out.append("fiducial_missing")
    if m["dial_radius_px"] < TH["min_radius_px"]:
        out.append("low_resolution")
    if m["view_axis_ratio"] < TH["very_oblique"]:
        out.append("very_oblique_view")
    elif m["view_axis_ratio"] < TH["min_axis_ratio"]:
        out.append("oblique_view")
    if m.get("sharpness", 99) < TH["min_sharpness"]:
        out.append("motion_blur" if m.get("blur_anisotropy", 1) > TH["motion_anisotropy"] else "defocus_blur")
    overexposed = m["clipped_frac"] > TH["max_clipped"]
    if overexposed:
        # uniform clipping of a white face: the dark needle and ticks usually survive
        out.append("overexposed")
    else:
        needle_weak = ("needle_strength" in m and (m["needle_strength"] < 0.6 or m["needle_peak_ratio"] < 2.2))
        if m.get("glare_near_needle") and needle_weak:
            out.append("glare_on_needle")
        elif m["glare_sweep_frac"] > TH["max_glare_sweep"]:
            out.append("glare_on_scale")
    if m["face_level"] < TH["min_face_level"]:
        out.append("underexposed")
    if m.get("noise_sigma", 0) > TH["max_noise"]:
        out.append("noisy")
    if m["contrast"] < TH["min_contrast"] and m["face_level"] >= TH["min_face_level"]:
        out.append("low_contrast")
    if m["rim_coverage"] < TH["min_rim_coverage"] and m["glare_sweep_frac"] < 0.15:
        out.append("occluded")
    if "needle_strength" in m and (m["needle_strength"] < TH["min_needle_strength"]
                                   or m["needle_peak_ratio"] < TH["min_peak_ratio"]):
        out.append("needle_not_clear")
    if "tick_score" in m and m["tick_score"] < TH["min_tick_score"]:
        out.append("scale_not_verified")
    return out


# ---------------------------------------------------------------------------
# Multi-frame helpers


def hdr_fuse(frames: list) -> np.ndarray:
    """Exposure fusion (Mertens) after median-threshold-bitmap alignment."""
    imgs = [f.copy() for f in frames]
    try:
        cv2.createAlignMTB().process(imgs, imgs)
    except (cv2.error, AttributeError):
        pass
    try:
        fused = cv2.createMergeMertens().process(imgs)
    except (cv2.error, AttributeError):  # no photo module: plain average
        return np.mean(np.stack(imgs).astype(np.float32), axis=0).astype(np.uint8)
    return np.clip(fused * 255.0, 0, 255).astype(np.uint8)


def fuse_readings(readings: list, spec: GaugeSpec) -> dict:
    """Fuse per-frame readings.

    A fluttering needle sampled at random phases follows an arcsine distribution, whose centre is
    estimated far better by the (outlier-trimmed) midrange than by the median or mean.
    """
    ok = [r for r in readings if r.ok and r.confidence > 0.25 and math.isfinite(r.value)]
    if not ok:
        return dict(ok=False, n=0)
    vals = np.array([r.value for r in ok])
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) * 1.4826
    n = len(ok)
    span = spec.span
    # drop frames that are clearly a different structure (> 6 % span from the median)
    keep = np.abs(vals - med) < 0.06 * span
    v = vals[keep] if keep.sum() >= max(2, n // 2) else vals
    n_k = len(v)
    flutter = float((v.max() - v.min()) / 2) if n_k >= 2 else 0.0
    midrange = float((v.max() + v.min()) / 2) if n_k >= 2 else med
    sig_sys = float(np.median([r.tick_sigma_deg for r in ok])) / (spec.theta_max - spec.theta_min) * span
    sig_frame = float(np.median([r.sigma for r in ok]))
    if flutter > 0.004 * span:
        centre = midrange if n_k >= 5 else med
        # residual error of the centre estimate shrinks roughly as 1/n for the midrange
        sig_centre = flutter * (1.3 / n_k if n_k >= 5 else 0.8)
        sigma = math.sqrt(sig_sys ** 2 + sig_frame ** 2 / n_k + sig_centre ** 2)
    else:
        centre = med
        sigma = math.sqrt(sig_sys ** 2 + (sig_frame ** 2 + mad ** 2) / n)
    conf = float(np.median([r.confidence for r in ok])) * min(1.0, 0.6 + 0.1 * n)
    flags = sorted({f for r in ok for f in r.flags})
    return dict(ok=True, n=n, value=centre, median=med, midrange=midrange, sigma=sigma, flutter=flutter,
                confidence=min(1.0, conf), flags=flags, frames_used=n_k, frames_total=len(readings))
