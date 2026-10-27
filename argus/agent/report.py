"""Shift hand-over report (template, optionally polished by the LLM)."""
from __future__ import annotations

from .session import Session


def facts(s: Session) -> dict:
    rows = []
    for gid, oc in s.outcomes.items():
        spec = s.registry.gauges[gid]
        rd = s.readings.get(oc.reading_id) if oc.reading_id else None
        rows.append({
            "gauge": gid, "name": spec.name, "status": oc.status,
            "value": None if rd is None else round(rd.value, 4), "pm2sigma": None if rd is None else round(2 * rd.sigma, 4),
            "units": spec.units, "work_order": oc.work_order_id, "review": oc.review_id, "captures": oc.captures,
            "flutter": None if rd is None or not rd.flutter else round(rd.flutter, 4),
            "trend_note": None,
        })
    return {"round": s.id, "robot_time_s": round(s.camera.time_s, 1), "gauges": rows, "notes": s.notes,
            "ambient_light": s.ambient_sensor()}


def template_report(s: Session) -> str:
    f = facts(s)
    n = len(f["gauges"])
    auto = [g for g in f["gauges"] if g["value"] is not None]
    esc = [g for g in f["gauges"] if g["review"]]
    wos = [g for g in f["gauges"] if g["work_order"]]
    lines = [f"Round #{s.number}: {n} gauges in {f['robot_time_s']:.0f} s of robot time; "
             f"{len(auto)} read automatically, {len(esc)} escalated to the operator, "
             f"{len(wos)} work order{'' if len(wos) == 1 else 's'} drafted."]
    for g in f["gauges"]:
        if g["status"] in ("alarm", "instrument_fault"):
            lines.append(f"- ACTION {g['gauge']} ({g['name']}): {g['status'].replace('_', ' ')} at {g['value']} "
                         f"{g['units']}; {g['work_order']} awaits approval.")
    for g in f["gauges"]:
        if g["status"] == "warning":
            lines.append(f"- WATCH {g['gauge']}: {g['value']} {g['units']} is in the warning band.")
    for g in esc:
        lines.append(f"- REVIEW {g['gauge']}: automated reading not trustworthy; please confirm on site or in "
                     "the review queue.")
    for g in f["gauges"]:
        spec = s.registry.gauges[g["gauge"]]
        if g["flutter"] and g["flutter"] > 0.006 * spec.span:
            lines.append(f"- NOTE {g['gauge']}: needle flutter ±{g['flutter']} {g['units']} suggests pulsation.")
    return "\n".join(lines)


def shift_report(s: Session, llm_agent=None) -> str:
    base = template_report(s)
    if llm_agent is None:
        return base
    try:
        return llm_agent.write_report(s, facts(s), base) or base
    except Exception:
        return base
