"""End-to-end analysis of one capture (single frame, burst, or HDR bracket).

This is the core image workload that runs on COOL / Graviton and that the
benchmark times.  It is deterministic given the input frames.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from ..registry import GaugeSpec
from . import enhance as E
from . import geometry as g
from . import perception as P

# Multiplicative confidence penalties for quality issues.  A reading is only as trustworthy as the
# evidence it came from: a crisp needle peak inside a glare blob or on a mis-rectified dial is not
# a reliable measurement.
PENALTY = {
    "glare_on_needle": 0.25, "needle_not_clear": 0.3, "no_geometry_check": 0.2, "oblique_view": 0.9,
    "very_oblique_view": 0.55, "motion_blur": 0.6, "defocus_blur": 0.6, "noisy": 0.85, "scale_not_verified": 0.8,
    "overexposed": 0.9, "needle_flutter": 0.5, "frames_inconsistent": 0.4,
    "underexposed": 0.8, "low_resolution": 0.6, "occluded": 0.7, "low_contrast": 0.85, "glare_on_scale": 0.9,
}


def gated_confidence(conf: float, issues: list, reading_flags: list, loc_method: str) -> tuple[float, list]:
    reasons = []
    c = conf
    for iss in issues:
        if iss in PENALTY:
            c *= PENALTY[iss]
            reasons.append(iss)
    # Geometry was not anchored by the fiducial AND the printed scale could not confirm it:
    # the reading can be internally consistent and still be on a mis-rectified dial.
    if loc_method != "fiducial" and "scale_unverified" in reading_flags:
        c *= PENALTY["no_geometry_check"]
        reasons.append("no_geometry_check")
    if "needle_methods_disagree" in reading_flags:
        c *= 0.6
        reasons.append("needle_methods_disagree")
    return float(np.clip(c, 0, 1)), reasons


@dataclass
class Analysis:
    ok: bool
    spec_id: str
    localization: dict
    quality: dict
    reading: dict
    timings_ms: dict
    canon: Optional[np.ndarray] = None
    work_image: Optional[np.ndarray] = None
    loc: Optional[P.Localization] = None
    frame_readings: list = field(default_factory=list)

    def summary(self) -> dict:
        r = self.reading
        return {
            "ok": self.ok,
            "value": r.get("value"),
            "sigma": r.get("sigma"),
            "confidence": r.get("confidence"),
            "issues": self.quality.get("issues", []),
        }


def _mean_frames(frames: list) -> np.ndarray:
    acc = np.zeros(frames[0].shape, np.float32)
    for f in frames:
        acc += f.astype(np.float32)
    return np.clip(acc / len(frames) + 0.5, 0, 255).astype(np.uint8)


def analyze_capture(frames: list, spec: GaugeSpec, predicted_H: Optional[np.ndarray] = None, hdr: bool = False,
                    enhance_ops: Optional[list] = None, upload: bool = False, theta_hint: Optional[float] = None,
                    ticks: bool = True) -> Analysis:
    t0 = time.perf_counter()
    tm = {}
    if hdr and len(frames) > 1:
        work = P.hdr_fuse(frames)
        per_frame = [work]
    elif len(frames) > 1:
        work = _mean_frames(frames)
        per_frame = frames
    else:
        work = frames[0]
        per_frame = frames
    tm["fuse"] = (time.perf_counter() - t0) * 1000
    t = time.perf_counter()
    loc = P.localize(work, spec, predicted_H, allow_circle=True)
    tm["localize"] = (time.perf_counter() - t) * 1000
    if not loc.ok:
        q = {"localization": "none", "issues": ["no_dial"]}
        return Analysis(False, spec.id, {"method": "none", "note": loc.note, "markers_seen": loc.markers_seen},
                        q, {"ok": False, "confidence": 0.0}, tm, None, work, loc)
    t = time.perf_counter()
    rect = P.rectify(work, loc, spec)
    canon = rect.canon
    if enhance_ops:
        canon = E.apply(canon, enhance_ops)
    tm["rectify"] = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    unknown_roll = loc.method == "circle" and ticks
    rd = P.read_canon(canon, spec, unknown_roll=unknown_roll, theta_hint=theta_hint, ticks=ticks)
    frame_readings = []
    fused = None
    if len(per_frame) > 1:
        Hc = rect.H_canon_from_img
        for f in per_frame:
            cf = P.warp_canon(f, Hc)
            if enhance_ops:
                cf = E.apply(cf, enhance_ops)
            frame_readings.append(P.read_canon(cf, spec, unknown_roll=unknown_roll, theta_hint=rd.theta,
                                               ticks=ticks))
        fused = P.fuse_readings(frame_readings, spec)
    tm["read"] = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    q = P.assess(work, loc, rect, spec, rd)
    tm["assess"] = (time.perf_counter() - t) * 1000
    reading = rd.to_dict(spec)
    reading["method"] = "single"
    if fused and fused.get("ok"):
        # flutter -> use the median of per-frame readings; static -> the mean image is cleaner
        span = spec.span
        if fused["flutter"] > 0.004 * span:
            est = "midrange" if fused["frames_used"] >= 5 else "median"
            reading.update(value=round(fused["value"], 4), sigma=round(fused["sigma"], 4),
                           method=f"{est} of {fused['frames_used']} frames (needle flutter)",
                           flutter=round(fused["flutter"], 4))
            if fused["frames_used"] < 7:
                reading.setdefault("flags", []).append("flutter_needs_more_frames")
        else:
            reading.update(sigma=round(min(rd.sigma, fused["sigma"] * 1.0 + rd.sigma * 0.3), 4),
                           method=f"mean image of {len(per_frame)} frames", flutter=round(fused["flutter"], 4))
        reading["frames_used"] = fused["n"]
        reading["frames_total"] = fused["frames_total"]
        reading["confidence"] = max(rd.confidence, fused["confidence"])
    if len(per_frame) > 1 and not hdr:
        if not fused or not fused.get("ok") or fused["frames_used"] < max(2, (len(per_frame) + 1) // 2):
            # a mean image of frames that individually cannot be read (moving needle, noise) is not evidence
            reading.setdefault("flags", []).append("frames_inconsistent")
    if hdr:
        reading["method"] = "HDR exposure fusion (3 frames)"
    if "flutter_needs_more_frames" in reading.get("flags", []):
        q["issues"] = q["issues"] + ["needle_flutter"]
    if "frames_inconsistent" in reading.get("flags", []):
        q["issues"] = q["issues"] + ["frames_inconsistent"]
    elif len(per_frame) == 1 and q["issues"].count("motion_blur"):
        pass
    conf, reasons = gated_confidence(float(reading.get("confidence", 0.0)), q["issues"], rd.flags, loc.method)
    reading["raw_confidence"] = reading.get("confidence")
    reading["confidence"] = round(conf, 3)
    reading["confidence_penalties"] = reasons
    loc_d = {"method": loc.method, "marker_id": loc.marker_id, "markers_seen": loc.markers_seen,
             "radius_px": round(loc.radius_px, 1), "axis_ratio": round(loc.axis_ratio, 3), "note": loc.note,
             "rim_refine_shift": round(rect.refine_shift, 4), "rim_refine_scale": round(rect.refine_scale, 4)}
    tm["total"] = (time.perf_counter() - t0) * 1000
    tm = {k: round(v, 2) for k, v in tm.items()}
    return Analysis(True, spec.id, loc_d, q, reading, tm, canon, work, loc, frame_readings)


# ---------------------------------------------------------------------------
# Evidence overlays


def overlay(an: Analysis, spec: GaugeSpec, title: str = "") -> np.ndarray:
    """Side-by-side evidence panel: raw dial ROI with geometry, and the annotated canonical dial."""
    W = 512
    panel = np.full((W + 56, 2 * W, 3), 24, np.uint8)
    work = an.work_image
    if work is not None and an.loc is not None and an.loc.ok:
        (cx, cy), r = an.loc.center_px, an.loc.radius_px
        pad = 1.9 * r
        x0, y0 = int(max(0, cx - pad)), int(max(0, cy - pad * 0.9))
        x1, y1 = int(min(work.shape[1], cx + pad)), int(min(work.shape[0], cy + pad * 1.4))
        roi = work[y0:y1, x0:x1].copy()
        if an.loc.H_face_to_img is not None:
            pts = np.array([g.face_point(t, spec.dial_radius_mm) for t in np.linspace(-180, 180, 64)])
            pi = g.apply_h(an.loc.H_face_to_img, pts) - [x0, y0]
            cv2.polylines(roi, [np.round(pi).astype(np.int32)], True, (0, 220, 255), 2, cv2.LINE_AA)
        h, w = roi.shape[:2]
        s = min(W / max(w, 1), W / max(h, 1))
        rs = cv2.resize(roi, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        panel[(W - rs.shape[0]) // 2:(W - rs.shape[0]) // 2 + rs.shape[0],
              (W - rs.shape[1]) // 2:(W - rs.shape[1]) // 2 + rs.shape[1]] = rs
    elif work is not None:
        h, w = work.shape[:2]
        s = min(W / w, W / h)
        rs = cv2.resize(work, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        panel[(W - rs.shape[0]) // 2:(W - rs.shape[0]) // 2 + rs.shape[0], (W - rs.shape[1]) // 2:
              (W - rs.shape[1]) // 2 + rs.shape[1]] = rs
    if an.canon is not None:
        c = an.canon.copy()
        clipped = c.max(axis=2) >= 250
        tint = c.copy()
        tint[clipped] = (200, 40, 220)
        c = cv2.addWeighted(c, 0.55, tint, 0.45, 0)
        cv2.circle(c, (int(P.CX), int(P.CY)), int(P.RC), (0, 220, 255), 1, cv2.LINE_AA)
        rd = an.reading
        if rd.get("ok"):
            off = rd.get("tick_offset_deg", 0.0) if rd.get("scale_verified") else 0.0
            for k in range(spec.major_ticks):
                th = spec.theta_min + (spec.theta_max - spec.theta_min) * k / (spec.major_ticks - 1) + off
                a, b = g.face_point(th, 0.83 * P.RC)
                cv2.circle(c, (int(P.CX + a), int(P.CY + b)), 4, (60, 200, 60), -1, cv2.LINE_AA)
            th = rd["theta"] + off
            a, b = g.face_point(th, 0.9 * P.RC)
            col = (60, 200, 60) if rd.get("confidence", 0) >= 0.6 else ((0, 190, 255) if rd.get("confidence", 0) >= 0.3
                                                                        else (60, 60, 235))
            cv2.line(c, (int(P.CX), int(P.CY)), (int(P.CX + a), int(P.CY + b)), col, 3, cv2.LINE_AA)
        panel[0:W, W:2 * W] = c
    rd = an.reading
    if rd.get("ok"):
        txt = f"{spec.id}  {rd['value']:.3g} +/- {2 * rd['sigma']:.2g} {spec.units}  conf {rd.get('confidence', 0):.2f}"
    else:
        txt = f"{spec.id}  no reading"
    if title:
        txt = f"{title}  |  {txt}"
    cv2.putText(panel, txt, (14, W + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (235, 235, 235), 2, cv2.LINE_AA)
    shown = [i for i in an.quality.get("issues", []) if not (i == "fiducial_missing" and spec.marker_id < 0)]
    iss = ", ".join(i.replace("_", " ") for i in shown[:4])
    if iss:
        cv2.putText(panel, iss, (W + 14, W + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 200, 255), 1, cv2.LINE_AA)
    return panel


def encode_jpeg(img: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()
