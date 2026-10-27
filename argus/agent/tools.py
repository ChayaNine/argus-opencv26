"""The tool surface shared by the LLM agent (Bedrock), the policy agent, the MCP server and the API.

Every tool call goes through ``call_tool`` which validates arguments, applies the
safety guard, executes, and writes tool_call / tool_result events to the trace.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict
from typing import Any

from ..registry import GaugeSpec
from ..vision.perception import ISSUE_TEXT
from ..vision.pipeline import analyze_capture, overlay
from . import guard
from .session import AUTO_ACCEPT_CONFIDENCE, Capture, ReadingRec, Review, Session, WorkOrder

TOOL_SPECS: list[dict] = [
    {
        "name": "capture",
        "description": (
            "Point the robot's PTZ camera at the current gauge and capture image(s). Moving to a different view "
            "('left'/'right' are 0.65 m along the walkway) costs ~3 s of robot time; glare is a mirror reflection, "
            "so it moves on the dial when the camera moves, and objects hanging in front of a dial shift by parallax. "
            "Returns a capture_id plus an OpenCV quality assessment (issues + metrics)."),
        "input_schema": {
            "type": "object",
            "properties": {
                "view": {"type": "string", "enum": ["front", "left", "right"],
                         "description": "Walkway position relative to the gauge."},
                "zoom": {"type": "number", "minimum": 0.6, "maximum": 1.6,
                         "description": "1.0 frames dial + fiducial tag. >1.25 may crop the tag."},
                "ev_bias": {"type": "number", "minimum": -3, "maximum": 2,
                            "description": "Exposure compensation in stops relative to auto-exposure."},
                "burst": {"type": "integer", "minimum": 1, "maximum": 9,
                          "description": "Frames at 15 fps. Averaging reduces noise; per-frame median handles flutter."},
                "hdr": {"type": "boolean", "description": "3-shot exposure bracket fused with Mertens exposure fusion."},
                "light": {"type": "boolean", "description": "Robot spotlight on/off (can itself cause glare)."},
                "fast_shutter": {"type": "boolean",
                                 "description": "Cap shutter at 4 ms (freezes vibration, adds gain/noise)."},
            },
            "required": [],
        },
    },
    {
        "name": "read_gauge",
        "description": (
            "Run the OpenCV 5 reader on a capture: fiducial homography rectification, polar needle detection, "
            "tick-scale verification. Returns value, sigma, confidence, issues and a limit check that accounts for "
            "uncertainty. Optional software enhancement: clahe (haze/low contrast), deglare (partial glare), "
            "denoise (low light)."),
        "input_schema": {
            "type": "object",
            "properties": {
                "capture_id": {"type": "string"},
                "enhance": {"type": "array", "items": {"type": "string", "enum": ["clahe", "deglare", "denoise"]}},
            },
            "required": ["capture_id"],
        },
    },
    {
        "name": "record_reading",
        "description": (
            f"Commit a reading produced by read_gauge to the round log. Only readings with confidence >= "
            f"{AUTO_ACCEPT_CONFIDENCE} whose uncertainty interval does not straddle an alarm/warning limit are accepted."),
        "input_schema": {
            "type": "object",
            "properties": {"reading_id": {"type": "string"}, "note": {"type": "string"}},
            "required": ["reading_id"],
        },
    },
    {
        "name": "request_human_review",
        "description": ("Escalate this gauge to the control-room operator with the evidence collected so far. "
                        "Use when the reading cannot be made trustworthy within budget."),
        "input_schema": {
            "type": "object",
            "properties": {"reason": {"type": "string"}, "reading_id": {"type": "string"}},
            "required": ["reason"],
        },
    },
    {
        "name": "create_work_order",
        "description": ("Draft a maintenance work order (requires human approval before dispatch). Required for "
                        "alarm and instrument_fault readings."),
        "input_schema": {
            "type": "object",
            "properties": {
                "priority": {"type": "string", "enum": ["P1", "P2", "P3", "P4"]},
                "title": {"type": "string"},
                "description": {"type": "string"},
                "reading_id": {"type": "string"},
            },
            "required": ["priority", "title", "description"],
        },
    },
    {
        "name": "finish_gauge",
        "description": "Close out the current gauge. The safety guard checks that the outcome is complete.",
        "input_schema": {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
        },
    },
]

TOOL_NAMES = [t["name"] for t in TOOL_SPECS]


class ToolError(Exception):
    pass


def _limit_check(spec: GaugeSpec, value: float, sigma: float, flags: list) -> dict:
    lo, hi = value - 2 * sigma, value + 2 * sigma
    if "below_range_on_stop" in flags:
        return {"status": "instrument_fault", "interval_2sigma": [round(lo, 4), round(hi, 4)],
                "explanation": "needle is resting on the low stop pin: gauge failed, isolated, or line depressurised"}
    if "over_range" in flags:
        return {"status": "alarm", "interval_2sigma": [round(lo, 4), round(hi, 4)],
                "explanation": "needle is pegged above full scale: process over-range and possible gauge damage"}
    s_lo, s_hi = spec.status_of(lo), spec.status_of(hi)
    s_mid = spec.status_of(value)
    rank = {"normal": 0, "warning": 1, "alarm": 2}
    nearest, margin = None, None
    for thr, name in spec.thresholds():
        d = abs(value - thr) / max(sigma, 1e-9)
        if margin is None or d < margin:
            nearest, margin = {"name": name, "value": thr}, d
    if s_lo == s_hi == s_mid:
        status = s_mid
    elif rank[s_lo] >= 2 and rank[s_hi] >= 2:
        status = "alarm"
    else:
        status = "uncertain"
    out = {"status": status, "interval_2sigma": [round(lo, 4), round(hi, 4)], "point_status": s_mid}
    if nearest is not None:
        out["nearest_limit"] = nearest
        out["margin_sigma"] = round(margin, 2)
    if status == "uncertain":
        out["explanation"] = "the 2-sigma interval straddles a limit; tighten the reading or escalate"
    return out


def _trend(session: Session, gid: str, value: float) -> dict:
    sg = session.scene.gauge(gid)
    hist = sg.history
    if not hist:
        return {}
    spec = sg.spec
    mean = sum(hist) / len(hist)
    d = (value - mean) / spec.span * 100
    out = {"previous_mean": round(mean, 3), "change_pct_span": round(d, 1)}
    if abs(d) > 15:
        out["note"] = "significant change since previous rounds"
    return out


def _quality_public(q: dict) -> dict:
    keys = ["localization", "marker_found", "dial_radius_px", "view_axis_ratio", "sharpness", "blur_anisotropy",
            "clipped_frac", "glare_sweep_frac", "face_level", "contrast", "noise_sigma", "rim_coverage"]
    return {"issues": [{"code": i, "meaning": ISSUE_TEXT.get(i, i)} for i in q.get("issues", [])],
            "metrics": {k: q.get(k) for k in keys if k in q}}


# ---------------------------------------------------------------------------
# Implementations


def tool_capture(s: Session, gid: str, a: dict) -> tuple[dict, list]:
    oc = s.outcomes[gid]
    t0 = s.camera.time_s
    cap = s.camera.capture(gid, view=a["view"], zoom=a["zoom"], ev_bias=a["ev_bias"], burst=a["burst"],
                           hdr=a["hdr"], light=a["light"], fast_shutter=a["fast_shutter"])
    cid = s.new_id("cap")
    img_id = s.store.put(cap["frames"][0], "cap", max_w=960)
    c = Capture(cid, gid, a["view"], dict(a), cap["settings"], cap["frames"], cap["predicted_H"], cap["hdr"],
                s.camera.time_s, img_id, truth=cap["truth"] if s.keep_truth else [])
    s.captures[cid] = c
    oc.captures += 1
    an = analyze_capture(c.frames, s.registry.gauges[gid], c.predicted_H, hdr=c.hdr)
    c.analyses[""] = an
    oc.compute_ms += an.timings_ms.get("total", 0.0)
    res = {
        "capture_id": cid, "view": a["view"], "frames": len(cap["frames"]), "hdr": c.hdr,
        "camera": {"exposure_ms": [st["exposure_ms"] for st in cap["settings"]], "gain_db": cap["settings"][0]["gain_db"],
                   "light_on": cap["light"]},
        "quality": _quality_public(an.quality),
        "robot_time_s": round(s.camera.time_s - t0, 1),
        "captures_used": oc.captures, "captures_left": guard.MAX_CAPTURES - oc.captures,
    }
    return res, [img_id]


def tool_read_gauge(s: Session, gid: str, a: dict) -> tuple[dict, list]:
    c = s.captures.get(a["capture_id"])
    if c is None or c.gauge_id != gid:
        raise ToolError(f"unknown capture_id {a['capture_id']} for gauge {gid}")
    spec = s.registry.gauges[gid]
    ops = a.get("enhance") or []
    key = ",".join(ops)
    an = c.analyses.get(key)
    if an is None or an.canon is None and key:
        if not c.frames:
            raise ToolError("capture frames were released; take a new capture")
        an = analyze_capture(c.frames, spec, c.predicted_H, hdr=c.hdr, enhance_ops=ops)
        c.analyses[key] = an
        s.outcomes[gid].compute_ms += an.timings_ms.get("total", 0.0)
    rd = an.reading
    ov = overlay(an, spec, title=f"{c.view}{' +' + '+'.join(ops) if ops else ''}")
    ov_id = s.store.put(ov, "ovl", max_w=1024)
    if not rd.get("ok"):
        return {"ok": False, "issues": _quality_public(an.quality)["issues"],
                "message": "no reading: the dial could not be located"}, [ov_id]
    rid = s.new_id("rd")
    check = _limit_check(spec, rd["value"], rd["sigma"], rd.get("flags", []))
    rec = ReadingRec(rid, gid, c.id, float(rd["value"]), float(rd["sigma"]), float(rd["confidence"]), spec.units,
                     rd.get("method", "single"), list(rd.get("flags", [])), list(an.quality.get("issues", [])), ov_id,
                     check, float(rd.get("flutter", 0.0) or 0.0), float(rd.get("theta", float("nan"))), ops)
    s.readings[rid] = rec
    res = {
        "reading_id": rid, "value": round(rec.value, 4), "sigma": round(rec.sigma, 4), "units": spec.units,
        "confidence": rec.confidence, "auto_accept_threshold": AUTO_ACCEPT_CONFIDENCE,
        "method": rec.method, "confidence_reduced_by": rd.get("confidence_penalties", []),
        "reader_flags": rec.flags, "issues": [i["code"] for i in _quality_public(an.quality)["issues"]],
        "scale_verified": rd.get("scale_verified"), "limit_check": check, "trend": _trend(s, gid, rec.value),
    }
    if rec.flutter > 0:
        res["needle_flutter"] = round(rec.flutter, 4)
        if rec.flutter > 0.008 * spec.span:
            res["flutter_note"] = "needle flutter indicates process pulsation (consider a snubber / pump check)"
    return res, [ov_id]


def tool_record(s: Session, gid: str, a: dict) -> tuple[dict, list]:
    rec = s.readings.get(a["reading_id"])
    if rec is None or rec.gauge_id != gid:
        raise ToolError(f"unknown reading_id {a['reading_id']} for gauge {gid}")
    guard.check_record(s, gid, rec)
    oc = s.outcomes[gid]
    oc.reading_id = rec.id
    oc.status = rec.check["status"]
    return {"recorded": True, "reading_id": rec.id, "status": oc.status,
            "work_order_required": oc.status in ("alarm", "instrument_fault")}, [rec.overlay_id]


def tool_review(s: Session, gid: str, a: dict, actor: str) -> tuple[dict, list]:
    rid = a.get("reading_id")
    ev = [c.image_id for c in s.captures.values() if c.gauge_id == gid][-3:]
    ev += [r.overlay_id for r in s.readings.values() if r.gauge_id == gid][-3:]
    if rid and rid not in s.readings:
        rid = None
    rv = Review(s.new_id("rev"), gid, a["reason"][:600], rid, ev, actor)
    s.reviews[rv.id] = rv
    oc = s.outcomes[gid]
    oc.review_id = rv.id
    if oc.reading_id is None:
        oc.status = "escalated"
    return {"review_id": rv.id, "status": "pending", "note": "the operator will confirm or correct the value"}, []


def tool_work_order(s: Session, gid: str, a: dict, actor: str) -> tuple[dict, list]:
    prio, note = guard.check_work_order(s, gid, a)
    rid = a.get("reading_id") if a.get("reading_id") in s.readings else s.outcomes[gid].reading_id
    ev = [r.overlay_id for r in s.readings.values() if r.gauge_id == gid][-2:]
    wo = WorkOrder(s.new_id("WO-"), gid, prio, a["title"][:140], a["description"][:1500], rid, ev, actor,
                   priority_note=note)
    s.work_orders[wo.id] = wo
    s.outcomes[gid].work_order_id = wo.id
    out = {"work_order_id": wo.id, "status": wo.status, "priority": prio,
           "note": "dispatch requires human approval in the control room"}
    if note:
        out["priority_adjusted"] = note
    return out, []


def tool_finish(s: Session, gid: str, a: dict) -> tuple[dict, list]:
    guard.check_finish(s, gid)
    oc = s.outcomes[gid]
    oc.finished = True
    oc.summary = a.get("summary", "")[:800]
    return {"finished": True, "gauge_id": gid, "status": oc.status}, []


def call_tool(s: Session, name: str, args: dict | None, actor: str, rationale: str = "") -> dict:
    """Validate -> guard -> execute -> trace.  Returns {'ok': bool, 'result'|'error': ...}."""
    gid = s.current_gauge
    args = dict(args or {})
    oc = s.outcomes.get(gid) if gid else None
    t0 = time.perf_counter()
    if rationale:
        s.emit(actor, "thought", text=rationale[:600])
    s.emit(actor, "tool_call", tool=name, args=args)
    images: list = []
    try:
        if gid is None:
            raise ToolError("no gauge is active")
        if name not in TOOL_NAMES:
            raise ToolError(f"unknown tool {name}")
        if oc is not None:
            oc.tool_calls += 1
            guard.check_budget(s, gid, name)
        if oc is not None and oc.finished and name != "finish_gauge":
            raise ToolError("gauge already finished")
        a = guard.normalise(name, args)
        if name == "capture":
            res, images = tool_capture(s, gid, a)
        elif name == "read_gauge":
            res, images = tool_read_gauge(s, gid, a)
        elif name == "record_reading":
            res, images = tool_record(s, gid, a)
        elif name == "request_human_review":
            res, images = tool_review(s, gid, a, actor)
        elif name == "create_work_order":
            res, images = tool_work_order(s, gid, a, actor)
        elif name == "finish_gauge":
            res, images = tool_finish(s, gid, a)
        else:  # pragma: no cover
            raise ToolError(name)
        dt = (time.perf_counter() - t0) * 1000
        s.emit(actor, "tool_result", tool=name, ok=True, result=res, images=images, latency_ms=round(dt, 1))
        return {"ok": True, "result": res}
    except (ToolError, guard.GuardViolation) as e:
        kind = "guard" if isinstance(e, guard.GuardViolation) else "tool_error"
        dt = (time.perf_counter() - t0) * 1000
        s.emit("guard" if kind == "guard" else actor, kind, tool=name, ok=False, error=str(e), latency_ms=round(dt, 1))
        return {"ok": False, "error": str(e)}
    except Exception as e:  # an unexpected failure is reported to the agent, never kills the round
        dt = (time.perf_counter() - t0) * 1000
        msg = f"internal error in {name}: {type(e).__name__}: {str(e)[:200]}"
        s.emit(actor, "tool_error", tool=name, ok=False, error=msg, latency_ms=round(dt, 1))
        return {"ok": False, "error": msg}
