"""Perception accuracy against simulator ground truth."""
import math

import numpy as np
import pytest

from argus.registry import GaugeSpec
from argus.vision import geometry as g
from argus.vision import perception as P
from argus.vision import sim
from argus.vision.pipeline import analyze_capture


def test_angle_conventions():
    assert g.theta_to_phi(0) == 270.0  # 12 o'clock is phi=270 in warpPolar
    assert g.phi_to_theta(0) == 90.0
    assert abs(g.theta_to_fraction(-135, -135, 135)) < 1e-9
    assert abs(g.theta_to_fraction(135, -135, 135) - 1) < 1e-9
    assert g.theta_to_fraction(-140, -135, 135) < 0  # resting on the low stop
    assert g.theta_to_fraction(140, -135, 135) > 1  # pegged high


def _captures(profile, seeds, **kw):
    for sd in seeds:
        sc = sim.make_scene(seed=sd, profile=profile)
        cam = sim.RobotCamera(sc, seed=sd)
        for sg in sc.gauges:
            cap = cam.capture(sg.spec.id, "front", **kw)
            yield sg, cap


def test_clean_reading_accuracy():
    errs = []
    for sg, cap in _captures("clean", [11, 12]):
        an = analyze_capture(cap["frames"], sg.spec, cap["predicted_H"])
        if cap["defocus"]:
            continue
        assert an.ok and an.localization["method"] == "fiducial"
        err = g.wrap180(an.reading["theta"] - cap["truth"][0]["theta"]) / 270 * 100
        errs.append(abs(err))
        assert an.reading["confidence"] >= 0.6
    assert len(errs) >= 6
    assert np.median(errs) < 0.15, errs  # % of span
    assert max(errs) < 0.6, errs


def test_glare_is_flagged_not_trusted():
    """A capture whose needle sits under saturated glare must not be confidently wrong."""
    for sg, cap in _captures("glare", [21, 22, 23]):
        an = analyze_capture(cap["frames"], sg.spec, cap["predicted_H"])
        if not an.reading.get("ok"):
            continue
        err = abs(g.wrap180(an.reading["theta"] - cap["truth"][0]["theta"])) / 270 * 100
        if err > 2.0:
            assert an.reading["confidence"] < 0.6, (sg.spec.id, err, an.reading["confidence"], an.quality["issues"])


def test_moving_viewpoint_moves_glare():
    """Specular geometry: the side view clears glare placed for the front view."""
    sc = sim.make_scene(seed=4, profile="glare")
    sg = next(s for s in sc.gauges if s.glare)
    cam = sim.RobotCamera(sc, seed=1)
    front = analyze_capture(cam.capture(sg.spec.id, "front")["frames"], sg.spec)
    sides = [analyze_capture(cam.capture(sg.spec.id, v)["frames"], sg.spec) for v in ("left", "right")]
    gf = front.quality["glare_sweep_frac"]
    assert min(s.quality["glare_sweep_frac"] for s in sides) < gf


def test_flutter_midrange_burst():
    sc = sim.make_scene(seed=504, profile="vibration")
    sg = next(s for s in sc.gauges if s.flutter_amp > 0.01 * s.spec.span)
    cam = sim.RobotCamera(sc, seed=3)
    cap = cam.capture(sg.spec.id, "front", burst=9, fast_shutter=True)
    an = analyze_capture(cap["frames"], sg.spec, cap["predicted_H"])
    assert "midrange" in an.reading["method"]
    err = abs(an.reading["value"] - sg.true_value) / sg.spec.span * 100
    assert err < 1.0, err
    assert an.reading["flutter"] > 0.3 * sg.flutter_amp


def test_stuck_needle_is_instrument_fault():
    sc = sim.make_scene(seed=3, profile="clean")
    sg = sc.gauges[0]
    sg.fault = "stuck_low"
    cam = sim.RobotCamera(sc, seed=2)
    an = analyze_capture(cam.capture(sg.spec.id, "front", burst=3)["frames"], sg.spec)
    assert "below_range_on_stop" in an.reading["flags"]


def test_photo_mode_recovers_unknown_rotation():
    """No fiducial, rotated photo: with the tick count ARGUS verifies scale+roll; without it, it refuses."""
    import cv2
    sc = sim.make_scene(seed=12, profile="clean")
    cam = sim.RobotCamera(sc, seed=1)
    sg = sc.gauges[0]
    img = cam.capture(sg.spec.id, "front", zoom=1.35)["frames"][0]
    h, w = img.shape[:2]
    img = cv2.warpAffine(img, cv2.getRotationMatrix2D((w / 2, h / 2), 10, 1.0), (w, h),
                         borderMode=cv2.BORDER_REPLICATE)[: int(h * 0.82)]
    s = sg.spec
    spec = GaugeSpec(id="PHOTO", name="p", units=s.units, min_value=s.min_value, max_value=s.max_value,
                     major_ticks=s.major_ticks, marker_id=-1)
    with_ticks = analyze_capture([img], spec, None, upload=True, ticks=True)
    assert with_ticks.localization["method"] == "circle"
    err = abs(with_ticks.reading["value"] - sg.true_value) / s.span * 100
    assert err < 1.0 and with_ticks.reading["confidence"] >= 0.6
    no_ticks = analyze_capture([img], spec, None, upload=True, ticks=False)
    assert no_ticks.reading["confidence"] < 0.6


def test_hub_recentering_fixes_pivot_offset():
    sc = sim.make_scene(seed=11, profile="clean")
    cam = sim.RobotCamera(sc, seed=1)
    sg = sc.gauges[1]
    cap = cam.capture(sg.spec.id, "front")
    img = cap["frames"][0]
    loc = P.localize(img, sg.spec)
    # corrupt the fiducial homography by ~4 px at the dial
    loc.H_face_to_img = g.translate(4.0, -3.0) @ loc.H_face_to_img
    rect = P.rectify(img, loc, sg.spec)
    rd = P.read_canon(rect.canon, sg.spec)
    err = abs(g.wrap180(rd.theta - cap["truth"][0]["theta"]))
    assert rect.hub is not None
    assert err < 0.8, err
