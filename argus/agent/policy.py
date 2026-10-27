"""Deterministic active-perception policy.

Encodes the same camera physics the LLM is told about, as explicit rules.  It is
(1) the fallback when Bedrock is unavailable, (2) a strong, reproducible
baseline, and (3) the "no LLM" mode for offline demos.
"""
from __future__ import annotations

from typing import Optional

from .session import AUTO_ACCEPT_CONFIDENCE, MAX_CAPTURES_PER_GAUGE, Session
from .tools import call_tool

ACTOR = "policy"


def _issues(res: dict) -> list:
    q = res.get("quality", {})
    return [i["code"] for i in q.get("issues", [])]


def next_plan(plan: dict, issues: list, reading: Optional[dict], tried: set, ambient: float) -> tuple[Optional[dict], str]:
    """Pick the next capture settings from what OpenCV reported.  Returns (plan, rationale)."""
    p = dict(plan)
    views_left = [v for v in ("left", "right", "front") if v not in {t[0] for t in tried}]

    def other_view(reason):
        if views_left:
            q = dict(p, view=views_left[0], hdr=False, ev_bias=0.0)
            return q, f"{reason} -> move to the {views_left[0]} view"
        return None, f"{reason} and all views tried"

    iss = set(issues)
    if "no_dial" in iss or ("fiducial_missing" in iss and ("defocus_blur" in iss or "motion_blur" in iss)):
        if "defocus_blur" in iss and ("af_retry", p["view"]) not in tried:
            tried.add(("af_retry", p["view"]))
            return dict(p), "image out of focus and tag not found -> retry autofocus"
        if "motion_blur" in iss and not p.get("fast_shutter"):
            return dict(p, fast_shutter=True, burst=5), "motion blur hides the tag -> fast shutter, 5-frame burst"
        return other_view("dial/tag not found")
    if "glare_on_needle" in iss or "glare_on_scale" in iss:
        if p.get("light") and "glare_light_off" not in {t[0] for t in tried}:
            tried.add(("glare_light_off", p["view"]))
            return dict(p, light=False, burst=max(p.get("burst", 1), 5)), \
                "glare may be the robot's own spotlight -> light off, 5-frame burst for noise"
        if not p.get("hdr") and ("hdr", p["view"]) not in tried and "glare_on_needle" not in iss:
            tried.add(("hdr", p["view"]))
            return dict(p, hdr=True, burst=3), "partial glare on the scale -> HDR exposure bracket"
        return other_view("specular glare on the dial (mirror reflection moves with the viewpoint)")
    if "occluded" in iss or ("needle_not_clear" in iss and not iss & {"noisy", "underexposed"}):
        return other_view("something in front of the dial / needle unclear (parallax changes with viewpoint)")
    if "very_oblique_view" in iss:
        return other_view("view too oblique")
    if "needle_flutter" in iss or "frames_inconsistent" in iss:
        return dict(p, fast_shutter=True, burst=9, hdr=False), \
            "needle flutter between frames -> 9 fast-shutter frames, centre by midrange"
    if "motion_blur" in iss:
        if not p.get("fast_shutter"):
            return dict(p, fast_shutter=True, burst=7), "directional blur from vibration -> 4 ms shutter, 7-frame burst"
        return dict(p, burst=9), "still blurred -> 9-frame burst median"
    if "defocus_blur" in iss:
        if ("af_retry", p["view"]) not in tried:
            tried.add(("af_retry", p["view"]))
            return dict(p), "isotropic blur -> autofocus retry"
        return other_view("focus failed twice")
    if "underexposed" in iss or "noisy" in iss:
        if not p.get("light") and ambient < 0.3:
            return dict(p, light=True), "dark scene -> robot spotlight on"
        if p.get("burst", 1) < 5:
            return dict(p, burst=5), "sensor noise -> 5-frame burst"
        if p.get("ev_bias", 0) < 1.0:
            return dict(p, ev_bias=p.get("ev_bias", 0) + 1.0, burst=7), "still noisy -> +1 EV, 7 frames"
        return other_view("noise persists")
    if "overexposed" in iss:
        return dict(p, ev_bias=p.get("ev_bias", 0) - 1.5), "face clipped -> -1.5 EV"
    if "low_resolution" in iss:
        return dict(p, zoom=min(1.5, p.get("zoom", 1.0) * 1.3)), "dial too small -> zoom in"
    if "oblique_view" in iss:
        return other_view("oblique view")
    if reading is not None and reading.get("limit_check", {}).get("status") == "uncertain":
        if p.get("burst", 1) < 7:
            return dict(p, burst=7), "interval straddles a limit -> 7-frame burst to tighten sigma"
        return other_view("still straddling a limit")
    if "scale_not_verified" in iss:
        if p.get("burst", 1) < 5:
            return dict(p, burst=5), "scale ticks not matched -> burst for SNR"
        return other_view("scale still unverified")
    return other_view("low confidence without a specific cause")


def acceptable(rd: dict) -> bool:
    return (rd.get("confidence", 0) >= AUTO_ACCEPT_CONFIDENCE
            and rd.get("limit_check", {}).get("status") != "uncertain")


def inspect_gauge(s: Session, gid: str, actor: str = ACTOR) -> None:
    ambient = s.ambient_sensor()
    plan = dict(view="front", zoom=1.0, ev_bias=0.0, burst=3, hdr=False, light=ambient < 0.15, fast_shutter=False)
    tried: set = set()
    sticky: dict = {}
    best = None
    rationale = ("night-level ambient light -> start with the spotlight on" if plan["light"]
                 else "start with a frontal auto-exposed capture")
    for _ in range(MAX_CAPTURES_PER_GAUGE):
        tried.add((plan["view"], plan.get("hdr"), plan.get("burst"), plan.get("light"), plan.get("ev_bias"),
                   plan.get("fast_shutter")))
        r = call_tool(s, "capture", plan, actor, rationale)
        if not r["ok"]:
            break
        cap = r["result"]
        issues = _issues(cap)
        rr = call_tool(s, "read_gauge", {"capture_id": cap["capture_id"]}, actor,
                       "measure the needle on this capture" if not issues else
                       "read it anyway; confidence will reflect: " + ", ".join(i.replace("_", " ") for i in issues))
        rd = rr["result"] if rr["ok"] else None
        if rd and "low_contrast" in issues and rd.get("confidence", 0) < AUTO_ACCEPT_CONFIDENCE:
            r2 = call_tool(s, "read_gauge", {"capture_id": cap["capture_id"], "enhance": ["clahe"]}, actor,
                           "low contrast (haze) -> re-read with CLAHE")
            if r2["ok"] and r2["result"].get("confidence", 0) > rd.get("confidence", 0):
                rd = r2["result"]
        if rd and rd.get("reading_id"):
            if best is None or rd["confidence"] > best["confidence"]:
                best = rd
            if acceptable(rd):
                best = rd
                break
            issues = sorted(set(issues) | set(rd.get("issues", [])))
        if {"needle_flutter", "frames_inconsistent"} & set(issues):
            sticky["flutter"] = True
        nxt, why = next_plan(plan, issues, rd, tried, ambient)
        if nxt is not None and sticky.get("flutter") and not nxt.get("hdr"):
            if nxt.get("burst", 1) < 9 or not nxt.get("fast_shutter"):
                why += "; keep 9 fast-shutter frames (flutter seen earlier on this gauge)"
            nxt = dict(nxt, burst=9, fast_shutter=True)
        if nxt is None:
            rationale = why
            break
        key = (nxt["view"], nxt.get("hdr"), nxt.get("burst"), nxt.get("light"), nxt.get("ev_bias"),
               nxt.get("fast_shutter"))
        if key in tried and why.find("autofocus") < 0:
            nxt2, why2 = next_plan(nxt, ["occluded"], rd, tried | {(nxt["view"],)}, ambient)
            if nxt2 is None:
                rationale = why2
                break
            nxt, why = nxt2, why + "; " + why2
        plan, rationale = nxt, why
    _close(s, gid, best, actor)


def _close(s: Session, gid: str, best: Optional[dict], actor: str) -> None:
    spec = s.registry.gauges[gid]
    if best is not None and acceptable(best):
        rec = call_tool(s, "record_reading", {"reading_id": best["reading_id"]}, actor,
                        f"confidence {best['confidence']:.2f} >= {AUTO_ACCEPT_CONFIDENCE} and the interval is clear of "
                        "limits -> record")
        if rec["ok"]:
            st = rec["result"]["status"]
            if st in ("alarm", "instrument_fault"):
                lc = best["limit_check"]
                prio = "P1" if (st == "alarm" and spec.criticality == "high") else "P2"
                tr = best.get("trend", {})
                desc = (f"{spec.name} ({gid}) read {best['value']:.4g} ± {2 * best['sigma']:.2g} {spec.units} (2σ) "
                        f"on the robot round. Status: {st.replace('_', ' ')}. {lc.get('explanation', '')} "
                        f"Limits: {spec.limits_text()}. Previous rounds averaged {tr.get('previous_mean')} {spec.units} "
                        f"({tr.get('change_pct_span', 0):+.0f}% of span). Evidence images attached.")
                call_tool(s, "create_work_order", {"priority": prio, "title": f"{gid} {st.replace('_', ' ')}: "
                                                   f"{best['value']:.4g} {spec.units}", "description": desc,
                                                   "reading_id": best["reading_id"]}, actor,
                          f"{st} -> draft a {prio} work order for human approval")
            elif st == "warning":
                s.notes.append(f"{gid} in warning band: {best['value']:.4g} {spec.units}")
            call_tool(s, "finish_gauge", {"summary": f"{st}: {best['value']:.4g} ± {2 * best['sigma']:.2g} "
                                                     f"{spec.units} ({best['method']})"}, actor)
            return
    reason = "could not obtain a trustworthy reading within budget"
    if best is not None:
        issues = ", ".join(i.replace("_", " ") for i in best.get("issues", [])) or "none"
        reason += (f"; best attempt {best['value']:.4g} {spec.units} at confidence {best['confidence']:.2f} "
                   f"(issues: {issues}; limit check: {best.get('limit_check', {}).get('status')})")
    call_tool(s, "request_human_review", {"reason": reason, "reading_id": best["reading_id"] if best else None}, actor,
              "escalate rather than record an untrustworthy value")
    call_tool(s, "finish_gauge", {"summary": "escalated to operator: " + reason}, actor)
