"""Safety guard: deterministic rules every agent (LLM or policy) must pass.

The guard is what makes autonomy *appropriate*: the agent decides how to look,
but it cannot invent measurements, record untrustworthy values, hide an alarm,
lower a work-order priority, or dispatch anything without a human.
"""
from __future__ import annotations

from .session import AUTO_ACCEPT_CONFIDENCE, MAX_CAPTURES_PER_GAUGE, MAX_TOOL_CALLS_PER_GAUGE

MAX_CAPTURES = MAX_CAPTURES_PER_GAUGE


class GuardViolation(Exception):
    pass


def _num(v, lo, hi, default):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, x))


def _bool(v, default=False):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    if v is None:
        return default
    return bool(v)


def normalise(name: str, a: dict) -> dict:
    if name == "capture":
        view = a.get("view", "front")
        if view not in ("front", "left", "right"):
            raise GuardViolation(f"view must be front|left|right, got {view!r}")
        return {
            "view": view,
            "zoom": _num(a.get("zoom", 1.0), 0.6, 1.6, 1.0),
            "ev_bias": _num(a.get("ev_bias", 0.0), -3.0, 2.0, 0.0),
            "burst": int(_num(a.get("burst", 1), 1, 9, 1)),
            "hdr": _bool(a.get("hdr", False)),
            "light": _bool(a.get("light"), None) if a.get("light") is not None else None,
            "fast_shutter": _bool(a.get("fast_shutter", False)),
        }
    if name == "read_gauge":
        if not a.get("capture_id"):
            raise GuardViolation("capture_id is required")
        ops = a.get("enhance") or []
        if isinstance(ops, str):
            ops = [ops]
        ops = [o for o in ops if o in ("clahe", "deglare", "denoise")][:3]
        return {"capture_id": str(a["capture_id"]), "enhance": ops}
    if name == "record_reading":
        if not a.get("reading_id"):
            raise GuardViolation("reading_id is required: only values measured by read_gauge can be recorded")
        return {"reading_id": str(a["reading_id"]), "note": str(a.get("note", ""))[:300]}
    if name == "request_human_review":
        return {"reason": str(a.get("reason") or "agent requested review"), "reading_id": a.get("reading_id")}
    if name == "create_work_order":
        p = str(a.get("priority", "P3")).upper()
        if p not in ("P1", "P2", "P3", "P4"):
            p = "P3"
        return {"priority": p, "title": str(a.get("title") or "Gauge inspection finding"),
                "description": str(a.get("description") or ""), "reading_id": a.get("reading_id")}
    if name == "finish_gauge":
        return {"summary": str(a.get("summary", ""))}
    return a


def check_budget(s, gid: str, name: str) -> None:
    oc = s.outcomes[gid]
    if name == "capture" and oc.captures >= MAX_CAPTURES_PER_GAUGE:
        raise GuardViolation(f"capture budget exhausted ({MAX_CAPTURES_PER_GAUGE}); record a trustworthy reading or "
                             "request_human_review")
    if oc.tool_calls > MAX_TOOL_CALLS_PER_GAUGE and name not in ("request_human_review", "finish_gauge",
                                                                  "create_work_order"):
        raise GuardViolation("tool-call budget exhausted; request_human_review and finish_gauge")


def check_record(s, gid: str, rec) -> None:
    if rec.confidence < AUTO_ACCEPT_CONFIDENCE:
        raise GuardViolation(
            f"confidence {rec.confidence:.2f} < {AUTO_ACCEPT_CONFIDENCE}: improve the evidence (address: "
            f"{', '.join(rec.issues) or 'needle evidence'}) or request_human_review")
    if rec.check.get("status") == "uncertain":
        lim = rec.check.get("nearest_limit", {})
        raise GuardViolation(
            f"2-sigma interval {rec.check.get('interval_2sigma')} straddles the {lim.get('name')} limit "
            f"({lim.get('value')}); tighten the reading (more frames / better view) or request_human_review")


PRIORITY_FLOOR = {"alarm": "P2", "instrument_fault": "P2", "warning": "P3"}
_RANK = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}


def check_work_order(s, gid: str, a: dict):
    for wo in s.work_orders.values():
        if wo.gauge_id == gid and wo.status != "rejected":
            raise GuardViolation(f"work order {wo.id} already exists for {gid}")
    oc = s.outcomes[gid]
    status = oc.status
    spec = s.registry.gauges[gid]
    floor = PRIORITY_FLOOR.get(status)
    if status == "alarm" and spec.criticality == "high":
        floor = "P1"
    prio = a["priority"]
    note = ""
    if floor and _RANK[prio] > _RANK[floor]:
        note = f"priority raised from {prio} to {floor} by policy ({status}, criticality {spec.criticality})"
        prio = floor
    return prio, note


def check_finish(s, gid: str) -> None:
    oc = s.outcomes[gid]
    if oc.reading_id is None and oc.review_id is None:
        raise GuardViolation("nothing recorded: record_reading or request_human_review before finishing")
    if oc.status in ("alarm", "instrument_fault") and oc.work_order_id is None:
        raise GuardViolation(f"recorded status is {oc.status}: create_work_order before finishing")
    # an alarm seen in any trustworthy-ish reading must not be silently dropped
    if oc.review_id is None and oc.work_order_id is None:
        for r in s.readings.values():
            if r.gauge_id == gid and r.confidence >= 0.4 and r.check.get("status") in ("alarm", "instrument_fault") \
                    and oc.status not in ("alarm", "instrument_fault"):
                raise GuardViolation(f"reading {r.id} indicated {r.check['status']} (confidence {r.confidence:.2f}); "
                                     "it must be reviewed by a human or covered by a work order")
