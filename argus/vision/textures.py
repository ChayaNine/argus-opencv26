"""Procedural textures for the simulated plant: gauge assemblies, needles, wall.

Everything is drawn with OpenCV at 2x supersampling and reduced with
INTER_AREA so edges are anti-aliased like a real photograph.  Textures are in
face millimetres at ``PPM`` pixels per millimetre.
"""
from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np

from ..registry import GaugeSpec
from .geometry import face_point, value_to_theta

PPM = 6.0  # texture pixels per millimetre
SS = 2  # supersampling factor
# Assembly texture covers a in [-60, 60], b in [-62, 118] mm (dial + bracket + tag plate)
ASM_A0, ASM_B0 = -60.0, -62.0
ASM_W_MM, ASM_H_MM = 120.0, 180.0
# Needle layer covers the dial square a, b in [-60, 60]
NDL_A0, NDL_B0 = -60.0, -60.0
NDL_MM = 120.0
# Fiducial tag (ArUco DICT_4X4_50) geometry in face mm: black square 36 mm, centre (0, 88)
TAG_SIDE_MM = 36.0
TAG_CENTER_MM = (0.0, 88.0)
ARUCO_DICT = cv2.aruco.DICT_4X4_50


def tag_corners_mm() -> np.ndarray:
    """ArUco corner order: TL, TR, BR, BL (face mm)."""
    cx, cy = TAG_CENTER_MM
    h = TAG_SIDE_MM / 2
    return np.array([[cx - h, cy - h], [cx + h, cy - h], [cx + h, cy + h], [cx - h, cy + h]], np.float64)


def _fmt(v: float) -> str:
    if abs(v - round(v)) < 1e-6:
        return str(int(round(v)))
    return f"{v:.1f}".rstrip("0").rstrip(".")


def _put_centered(img, text, center, scale, color, thickness, font=cv2.FONT_HERSHEY_SIMPLEX):
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    org = (int(round(center[0] - tw / 2)), int(round(center[1] + th / 2)))
    cv2.putText(img, text, org, font, scale, color, thickness, cv2.LINE_AA)
    return tw, th


def _units_text(img, units, center, scale, color, thickness):
    if units.startswith("°"):
        txt = units[1:]
        tw, th = _put_centered(img, txt, (center[0] + scale * 6, center[1]), scale, color, thickness)
        r = max(2, int(scale * 4))
        cv2.circle(img, (int(center[0] - tw / 2 - r + scale * 4), int(center[1] - th / 2 + r)), r, color,
                   max(1, thickness // 2), cv2.LINE_AA)
    else:
        _put_centered(img, units, center, scale, color, thickness)


def face_style(spec: GaugeSpec, style_seed: int) -> dict:
    rng = np.random.default_rng(style_seed)
    face = [(244, 244, 242), (236, 240, 244), (226, 236, 240), (240, 240, 236)][rng.integers(0, 4)]
    return dict(
        face=face,
        bezel=[(170, 172, 176), (120, 122, 126), (60, 62, 66)][rng.integers(0, 3)],
        ink=(28, 28, 30),
        zones=bool(rng.random() < 0.6),
        brand=["KESTREL", "NORDIA", "VALTEK", "ORMAN"][rng.integers(0, 4)],
        font=[cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX][rng.integers(0, 2)],
    )


def render_assembly(spec: GaugeSpec, style_seed: int = 0) -> np.ndarray:
    """Static gauge assembly (bezel, face, ticks, numerals, bracket, ArUco tag) as BGRA uint8."""
    st = face_style(spec, style_seed)
    k = PPM * SS
    W, H = int(ASM_W_MM * k), int(ASM_H_MM * k)
    img = np.zeros((H, W, 4), np.uint8)

    def P(a, b):  # face mm -> supersampled px
        return (int(round((a - ASM_A0) * k)), int(round((b - ASM_B0) * k)))

    def Pf(a, b):
        return ((a - ASM_A0) * k, (b - ASM_B0) * k)

    R = spec.dial_radius_mm
    c = P(0, 0)
    # bracket and tag plate
    cv2.rectangle(img, P(-12, 40), P(12, 70), (118, 122, 128, 255), -1, cv2.LINE_AA)
    cv2.rectangle(img, P(-24, 64), P(24, 112), (246, 246, 246, 255), -1, cv2.LINE_AA)
    cv2.rectangle(img, P(-24, 64), P(24, 112), (150, 150, 150, 255), max(1, int(0.4 * k)), cv2.LINE_AA)
    dic = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    side = int(round(TAG_SIDE_MM * k))
    tag = cv2.aruco.generateImageMarker(dic, int(spec.marker_id), side, borderBits=1)
    x0, y0 = P(TAG_CENTER_MM[0] - TAG_SIDE_MM / 2, TAG_CENTER_MM[1] - TAG_SIDE_MM / 2)
    img[y0:y0 + side, x0:x0 + side, :3] = np.where(tag[..., None] > 127, 246, 18)
    img[y0:y0 + side, x0:x0 + side, 3] = 255
    # bezel with a soft metallic gradient
    rb = 1.13 * R
    for i in range(24, -1, -1):
        rr = R + (rb - R) * i / 24.0
        t = i / 24.0
        shade = 0.75 + 0.35 * math.sin(t * math.pi)
        col = tuple(int(min(255, ch * shade)) for ch in st["bezel"]) + (255,)
        cv2.circle(img, c, int(rr * k), col, -1, cv2.LINE_AA)
    cv2.circle(img, c, int(rb * k), (40, 40, 44, 255), max(2, int(0.5 * k)), cv2.LINE_AA)
    # face with slight radial falloff (printed dials are not perfectly flat in colour)
    cv2.circle(img, c, int(R * k), (*st["face"], 255), -1, cv2.LINE_AA)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    rr = np.sqrt((xx - c[0]) ** 2 + (yy - c[1]) ** 2) / (R * k)
    inside = rr < 1.0
    fall = (1.0 - 0.07 * rr ** 2)
    for ch in range(3):
        plane = img[..., ch].astype(np.float32)
        plane[inside] *= fall[inside]
        img[..., ch] = np.clip(plane, 0, 255).astype(np.uint8)
    # crisp black face edge (the rim the reader locks on to)
    cv2.circle(img, c, int(R * k), (24, 24, 26, 255), max(2, int(1.1 * k)), cv2.LINE_AA)

    tmin, tmax = spec.theta_min, spec.theta_max
    ink = (*st["ink"], 255)
    # colour zones just inside the tick ring
    if st["zones"]:
        def arc(v0, v1, col):
            t0 = value_to_theta(v0, spec.min_value, spec.max_value, tmin, tmax)
            t1 = value_to_theta(v1, spec.min_value, spec.max_value, tmin, tmax)
            # cv2.ellipse angles: measured from +x, clockwise on screen -> phi = theta - 90
            cv2.ellipse(img, c, (int(0.735 * R * k), int(0.735 * R * k)), 0, t0 - 90, t1 - 90, col,
                        max(2, int(0.04 * R * k)), cv2.LINE_AA)
        if spec.high_alarm is not None:
            arc(spec.high_alarm, spec.max_value, (40, 40, 200, 255))
        if spec.low_alarm is not None:
            arc(spec.min_value, spec.low_alarm, (40, 40, 200, 255))
        if spec.low_warn is not None or spec.high_warn is not None:
            lo = spec.low_warn if spec.low_warn is not None else spec.min_value
            hi = spec.high_warn if spec.high_warn is not None else spec.max_value
            arc(lo, hi, (60, 150, 60, 255))
    # ticks
    n_major = spec.major_ticks
    n_minor = (n_major - 1) * spec.minor_per_major
    for i in range(n_minor + 1):
        th = tmin + (tmax - tmin) * i / n_minor
        major = i % spec.minor_per_major == 0
        r0 = 0.78 * R if major else 0.87 * R
        p0 = Pf(*face_point(th, r0))
        p1 = Pf(*face_point(th, 0.95 * R))
        thick = 0.028 * R if major else 0.012 * R
        cv2.line(img, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), ink, max(1, int(round(thick * k))),
                 cv2.LINE_AA)
    # numerals at the major ticks
    font = st["font"]
    sc = 0.0105 * R * k / 2.0
    for i in range(n_major):
        th = tmin + (tmax - tmin) * i / (n_major - 1)
        val = spec.min_value + spec.span * i / (n_major - 1)
        pc = Pf(*face_point(th, 0.62 * R))
        _put_centered(img, _fmt(val), pc, sc, ink, max(2, int(round(0.012 * R * k))), font)
    # brand, units, accuracy class
    _put_centered(img, st["brand"], Pf(0, -0.32 * R), sc * 0.55, (70, 70, 72, 255), max(1, int(0.006 * R * k)),
                  font)
    _units_text(img, spec.units, Pf(0, 0.38 * R), sc * 0.9, ink, max(2, int(round(0.011 * R * k))))
    _put_centered(img, "CL 1.0", Pf(0, 0.52 * R), sc * 0.5, (80, 80, 80, 255), max(1, int(0.005 * R * k)), font)
    # stop pin just below the minimum mark
    sp = P(*face_point(tmin - 6.0, 0.80 * R))
    cv2.circle(img, sp, int(0.022 * R * k), (60, 60, 60, 255), -1, cv2.LINE_AA)

    out = cv2.resize(img, (W // SS, H // SS), interpolation=cv2.INTER_AREA)
    return out


NEEDLE_COLORS = {"black": (32, 30, 30), "red": (36, 36, 190)}


def render_needle(spec: GaugeSpec, theta: float | list[float]) -> np.ndarray:
    """Needle + hub layer (BGRA uint8).  A list of angles renders a motion-blurred needle."""
    thetas = theta if isinstance(theta, (list, tuple)) else [theta]
    k = PPM
    N = int(NDL_MM * k)
    R = spec.dial_radius_mm
    col = NEEDLE_COLORS.get(spec.needle_color, NEEDLE_COLORS["black"])
    c = np.array([-NDL_A0 * k, -NDL_B0 * k])

    def draw(layer, th):
        t = math.radians(th)
        d = np.array([math.sin(t), -math.cos(t)])
        p = np.array([math.cos(t), math.sin(t)])

        def at(r, w):
            return c + d * r * R * k + p * w * R * k

        poly = np.array([at(-0.20, 0.035), at(0.0, 0.030), at(0.88, 0.006), at(0.88, -0.006), at(0.0, -0.030),
                         at(-0.20, -0.035)], np.float64)
        cv2.fillPoly(layer, [np.round(poly * 16).astype(np.int32)], (*col, 255), cv2.LINE_AA, shift=4)
        cc = (int(round(c[0] * 16)), int(round(c[1] * 16)))
        cv2.circle(layer, cc, int(0.075 * R * k * 16), (58, 58, 60, 255), -1, cv2.LINE_AA, shift=4)
        cv2.circle(layer, cc, int(0.03 * R * k * 16), (120, 120, 124, 255), -1, cv2.LINE_AA, shift=4)

    if len(thetas) == 1:
        layer = np.zeros((N, N, 4), np.uint8)
        draw(layer, thetas[0])
        return layer
    acc_c = np.zeros((N, N, 3), np.float32)
    acc_a = np.zeros((N, N), np.float32)
    for th in thetas:
        layer = np.zeros((N, N, 4), np.uint8)
        draw(layer, th)
        a = layer[..., 3].astype(np.float32) / 255.0
        acc_c += layer[..., :3].astype(np.float32) * a[..., None]
        acc_a += a
    n = float(len(thetas))
    color = acc_c / np.maximum(acc_a[..., None], 1e-4)
    out = np.dstack([color, acc_a / n * 255.0])
    return np.clip(out, 0, 255).astype(np.uint8)


def render_occluder(kind: str, seed: int) -> tuple[np.ndarray, float, float, float]:
    """A cable or hanging tag in a plane in front of the dial.  Returns (BGRA, a0, b0, ppm)."""
    rng = np.random.default_rng(seed)
    ppm = 4.0
    a0, b0, w_mm, h_mm = -90.0, -110.0, 180.0, 200.0
    img = np.zeros((int(h_mm * ppm), int(w_mm * ppm), 4), np.uint8)

    def P(a, b):
        return [int((a - a0) * ppm), int((b - b0) * ppm)]

    if kind == "cable":
        xs = np.linspace(-90, 90, 40)
        y0 = rng.uniform(-40, 10)
        slope = rng.uniform(-0.35, 0.35)
        sag = rng.uniform(10, 35)
        pts = [P(x, y0 + slope * x + sag * (1 - (x / 90) ** 2)) for x in xs]
        cv2.polylines(img, [np.array(pts, np.int32)], False, (24, 24, 26, 255), int(9 * ppm), cv2.LINE_AA)
        cv2.polylines(img, [np.array(pts, np.int32)], False, (60, 60, 64, 255), int(3 * ppm), cv2.LINE_AA)
    else:  # hanging maintenance tag on a string
        ax = rng.uniform(-30, 30)
        cv2.line(img, P(ax, -110), P(ax, -40), (200, 200, 200, 255), int(1.2 * ppm), cv2.LINE_AA)
        cv2.rectangle(img, P(ax - 22, -40), P(ax + 22, 25), (40, 180, 230, 255), -1, cv2.LINE_AA)
        cv2.putText(img, "DO NOT", tuple(P(ax - 18, -18)), cv2.FONT_HERSHEY_SIMPLEX, 0.09 * ppm * 2.2,
                    (20, 20, 20, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "OPERATE", tuple(P(ax - 20, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.08 * ppm * 2.2,
                    (20, 20, 20, 255), 2, cv2.LINE_AA)
    return img, a0, b0, ppm


@lru_cache(maxsize=8)
def _noise_field(h: int, w: int, seed: int, cells: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    small = rng.standard_normal((max(2, h // cells), max(2, w // cells))).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def render_wall(width_mm: float, height_mm: float, ppm: float, risers_x: list[float], labels: list[tuple],
                seed: int = 0, x0_mm: float = 0.0, y0_mm: float = 0.0) -> np.ndarray:
    """Painted steel wall with a pipe header, risers to each gauge and equipment labels (BGR uint8)."""
    rng = np.random.default_rng(seed)
    W, H = int(width_mm * ppm), int(height_mm * ppm)
    base = np.array([96, 104, 108], np.float32) + rng.uniform(-14, 14, 3).astype(np.float32)
    img = np.ones((H, W, 3), np.float32) * base
    img += _noise_field(H, W, seed, 120)[..., None] * 7.0
    img += _noise_field(H, W, seed + 1, 18)[..., None] * 3.0
    img += rng.standard_normal((H, W, 1)).astype(np.float32) * 2.0
    img = np.clip(img, 0, 255).astype(np.uint8)

    def X(x):
        return int((x - x0_mm) * ppm)

    def Y(y):
        return int((y - y0_mm) * ppm)

    # vertical panel seams
    for xs in np.arange(x0_mm, x0_mm + width_mm, 1200.0):
        cv2.line(img, (X(xs), 0), (X(xs), H), (70, 76, 80), max(1, int(4 * ppm)))
    # pipe header near the top (Y is negative upward from the floor)
    hy = -1950.0
    for i in range(int(170 * ppm)):
        t = i / (170 * ppm)
        shade = 0.55 + 0.6 * math.sin(t * math.pi)
        cv2.line(img, (0, Y(hy - 85) + i), (W, Y(hy - 85) + i), (70 * shade, 120 * shade, 150 * shade), 1)
    # risers to each instrument
    for rx in risers_x:
        for i in range(int(56 * ppm)):
            t = i / (56 * ppm)
            shade = 0.5 + 0.6 * math.sin(t * math.pi)
            cv2.line(img, (X(rx - 28) + i, Y(hy)), (X(rx - 28) + i, Y(-200)), (60 * shade, 105 * shade, 135 * shade),
                     1)
        for fy in (hy + 160, -620):
            cv2.rectangle(img, (X(rx - 46), Y(fy - 14)), (X(rx + 46), Y(fy + 14)), (52, 92, 120), -1)
    # floor and hazard stripe
    cv2.rectangle(img, (0, Y(-160)), (W, H), (58, 60, 62), -1)
    for xs in np.arange(x0_mm, x0_mm + width_mm, 160.0):
        pts = np.array([[X(xs), Y(-160)], [X(xs + 80), Y(-160)], [X(xs + 20), Y(-100)], [X(xs - 60), Y(-100)]])
        cv2.fillPoly(img, [pts], (30, 190, 230))
    # equipment labels
    for (lx, ly, text) in labels:
        cv2.rectangle(img, (X(lx - 130), Y(ly - 32)), (X(lx + 130), Y(ly + 32)), (235, 235, 235), -1)
        cv2.rectangle(img, (X(lx - 130), Y(ly - 32)), (X(lx + 130), Y(ly + 32)), (40, 40, 40), max(1, int(3 * ppm)))
        _put_centered(img, text, (X(lx), Y(ly)), 0.9 * ppm * 1.6, (25, 25, 25), max(1, int(4 * ppm)))
    return img
