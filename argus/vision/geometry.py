"""Geometry conventions shared by the simulator and the reader.

Gauge angle ``theta`` (degrees) is measured CLOCKWISE from 12 o'clock, as a
person reads a dial: 0 = up, 90 = right (3 o'clock), -90 = left.  A standard
270-degree pressure gauge runs from theta_min=-135 (7:30) to theta_max=+135
(4:30).

OpenCV's ``warpPolar`` measures ``phi`` from the +x axis of the image with y
pointing down, so phi also turns clockwise on screen.  The two are related by
``theta = phi + 90``.

Face coordinates (a, b) are millimetres on the dial plane: a to the right,
b downwards, origin at the needle pivot.
"""
from __future__ import annotations

import math

import numpy as np


def wrap180(deg: float) -> float:
    """Wrap an angle to (-180, 180]."""
    d = (deg + 180.0) % 360.0 - 180.0
    return 180.0 if d == -180.0 else d


def theta_to_phi(theta: float) -> float:
    return (theta - 90.0) % 360.0


def phi_to_theta(phi: float) -> float:
    return wrap180(phi + 90.0)


def face_point(theta_deg: float, r: float) -> tuple[float, float]:
    """Point on the dial (face coords) at gauge angle theta and radius r."""
    t = math.radians(theta_deg)
    return r * math.sin(t), -r * math.cos(t)


def angle_of(dx: float, dy: float) -> float:
    """Gauge angle of a direction vector in image/face coords (y down)."""
    return wrap180(math.degrees(math.atan2(dx, -dy)))


def value_to_theta(value: float, vmin: float, vmax: float, tmin: float, tmax: float) -> float:
    f = (value - vmin) / (vmax - vmin)
    return tmin + f * (tmax - tmin)


def theta_to_fraction(theta: float, tmin: float, tmax: float) -> float:
    """Fraction of the scale for a needle angle; may be <0 or >1 in the dead zone.

    The dead zone (between max and min, through 6 o'clock) is split at its middle
    so that a needle resting on the low stop reads slightly below 0 and a pegged
    needle reads slightly above 1.
    """
    sweep = tmax - tmin
    rel = (theta - tmin) % 360.0  # clockwise distance from the min mark
    dead = 360.0 - sweep
    if rel > sweep + dead / 2.0:  # just counter-clockwise of min -> below zero
        rel -= 360.0
    return rel / sweep


# --------------------------------------------------------------------------
# Homographies


def apply_h(H: np.ndarray, pts) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    ph = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return ph[:, :2] / ph[:, 2:3]


def local_scale(H: np.ndarray, pt) -> float:
    """Approximate isotropic scale (output px per input unit) of H at pt."""
    x, y = float(pt[0]), float(pt[1])
    e = 1.0
    p0 = apply_h(H, [(x, y), (x + e, y), (x, y + e)])
    jx = p0[1] - p0[0]
    jy = p0[2] - p0[0]
    det = abs(jx[0] * jy[1] - jx[1] * jy[0])
    return math.sqrt(det) / e


def translate(tx: float, ty: float) -> np.ndarray:
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)


def scale(sx: float, sy: float | None = None) -> np.ndarray:
    sy = sx if sy is None else sy
    return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)


def rot2d(deg: float) -> np.ndarray:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


# --------------------------------------------------------------------------
# 3D camera helpers (world: X right, Y down, Z into the wall)


def look_at(C: np.ndarray, target: np.ndarray) -> np.ndarray:
    """World->camera rotation (rows = camera x, y, z axes), OpenCV convention."""
    z = np.asarray(target, float) - np.asarray(C, float)
    z /= np.linalg.norm(z)
    down = np.array([0.0, 1.0, 0.0])
    x = np.cross(down, z)
    nx = np.linalg.norm(x)
    if nx < 1e-9:  # looking straight up/down; pick any x
        x = np.array([1.0, 0.0, 0.0])
    else:
        x /= nx
    y = np.cross(z, x)
    return np.stack([x, y, z])


def rot_y(deg: float) -> np.ndarray:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_x(deg: float) -> np.ndarray:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def plane_homography(K: np.ndarray, R: np.ndarray, C: np.ndarray, origin: np.ndarray,
                     u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Homography mapping plane coords (a, b) -> image pixels."""
    cols = np.column_stack([R @ u, R @ v, R @ (origin - C)])
    H = K @ cols
    return H / H[2, 2]


def intrinsics(f_px: float, width: int, height: int) -> np.ndarray:
    return np.array([[f_px, 0, width / 2.0], [0, f_px, height / 2.0], [0, 0, 1]], dtype=np.float64)
