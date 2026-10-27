"""Round supervisor: walks the route, hands each gauge to an inspector agent, enforces closure."""
from __future__ import annotations

import time
import traceback
from typing import Optional

from . import policy
from .session import Session
from .tools import call_tool


def _force_close(s: Session, gid: str, why: str) -> None:
    oc = s.outcomes[gid]
    s.emit("guard", "forced_handoff", reason=why)
    if oc.reading_id is None and oc.review_id is None:
        call_tool(s, "request_human_review", {"reason": f"agent did not complete the gauge: {why}"}, "guard")
    if oc.status in ("alarm", "instrument_fault") and oc.work_order_id is None:
        call_tool(s, "create_work_order", {"priority": "P2", "title": f"{gid} {oc.status} (auto-raised by guard)",
                                           "description": f"Guard raised this work order because the agent finished "
                                                          f"without one. {why}"}, "guard")
    if not oc.finished:
        r = call_tool(s, "finish_gauge", {"summary": f"closed by guard: {why}"}, "guard")
        if not r["ok"]:
            oc.finished = True
    oc.forced = True


def run_round(s: Session, llm_agent=None, on_gauge_done=None) -> Session:
    s.status = "running"
    s.emit("system", "round_start", profile=s.profile, seed=s.seed, agent=s.agent_mode,
           ambient_light=s.ambient_sensor(), route=[g.id for g in s.route()])
    try:
        for spec in s.route():
            gid = spec.id
            s.current_gauge = gid
            t_robot0 = s.camera.time_s
            ctx = s.gauge_context(gid)
            s.emit("system", "gauge_start", context=ctx)
            used = "policy"
            if s.agent_mode == "llm" and llm_agent is not None:
                used = llm_agent.inspect_gauge(s, gid)  # returns "llm" or "policy-fallback"
            else:
                policy.inspect_gauge(s, gid)
            oc = s.outcomes[gid]
            oc.agent = used
            if not oc.finished:
                _force_close(s, gid, "inspector returned without finish_gauge")
            oc.robot_time_s = round(s.camera.time_s - t_robot0, 1)
            s.emit("system", "gauge_done", outcome={"status": oc.status, "reading_id": oc.reading_id,
                                                     "review_id": oc.review_id, "work_order_id": oc.work_order_id,
                                                     "captures": oc.captures, "agent": used,
                                                     "robot_time_s": oc.robot_time_s})
            s.release_frames(gid)
            if on_gauge_done:
                on_gauge_done(s, gid)
        s.current_gauge = None
        from .report import shift_report
        s.report = shift_report(s, llm_agent if s.agent_mode == "llm" else None)
        s.status = "done"
        s.emit("system", "round_done", robot_time_s=round(s.camera.time_s, 1), usage=s.usage)
    except Exception as e:  # never leave a round half-open
        s.status = "error"
        s.error = f"{type(e).__name__}: {e}"
        s.emit("system", "round_error", error=s.error, tb=traceback.format_exc()[-2000:])
    return s
