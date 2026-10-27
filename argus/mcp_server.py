"""MCP server: the same OpenCV 5 tools and safety guard, for any MCP client (Claude Code, Kiro, ...).

    python -m argus.mcp_server                 # stdio
    claude mcp add argus -- python -m argus.mcp_server

An MCP-connected agent drives the simulated robot exactly like the built-in Bedrock inspector: it cannot
record a value that OpenCV did not measure, cannot record low-confidence readings, and cannot dispatch work.
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Optional

import cv2
from mcp.server.mcpserver import MCPServer

from .agent.session import Session
from .agent.tools import call_tool
from .registry import GaugeSpec
from .vision.pipeline import analyze_capture, overlay

mcp = MCPServer(
    name="argus-opencv",
    instructions=(
        "ARGUS gauge-reading tools backed by OpenCV 5. Call start_round, then for each gauge: begin_gauge, "
        "capture, read_gauge, and either record_reading (confidence >= 0.6, interval clear of limits) or "
        "request_human_review; alarms/faults need create_work_order; then finish_gauge. Glare moves when the "
        "view changes; vibration needs fast_shutter + burst; low light needs light or more frames."),
)
STATE: dict = {"session": None}


def _s() -> Session:
    s = STATE["session"]
    if s is None:
        raise ValueError("no round: call start_round first")
    return s


def _call(name: str, args: dict, why: str = "") -> dict:
    r = call_tool(_s(), name, args, "mcp-agent", why)
    return r["result"] if r["ok"] else {"error": r["error"]}


@mcp.tool(description="Start a simulated inspection round. profile: random|glare|night|vibration|clean|hard.")
def start_round(profile: str = "hard", seed: int = 7, n_gauges: int = 5) -> dict:
    s = Session(profile=profile, seed=seed, agent_mode="mcp", n_gauges=n_gauges)
    s.status = "running"
    STATE["session"] = s
    return {"round_id": s.id, "ambient_light": s.ambient_sensor(),
            "route": [{"gauge_id": g.id, "name": g.name, "criticality": g.criticality} for g in s.route()]}


@mcp.tool(description="Status of every gauge on the route.")
def list_gauges() -> dict:
    s = _s()
    return {"gauges": [{"gauge_id": gid, "status": oc.status, "finished": oc.finished, "captures": oc.captures}
                       for gid, oc in s.outcomes.items()]}


@mcp.tool(description="Make a gauge the current one and get its context (range, limits, history, light level).")
def begin_gauge(gauge_id: str) -> dict:
    s = _s()
    if gauge_id not in s.outcomes:
        return {"error": f"unknown gauge {gauge_id}"}
    s.current_gauge = gauge_id
    ctx = s.gauge_context(gauge_id)
    s.emit("system", "gauge_start", context=ctx)
    return ctx


@mcp.tool(description="Capture the current gauge. view front|left|right; zoom 0.6-1.6; ev_bias -3..2; burst 1-9; "
                      "hdr; light; fast_shutter. Returns capture_id and OpenCV quality issues.")
def capture(view: str = "front", zoom: float = 1.0, ev_bias: float = 0.0, burst: int = 3, hdr: bool = False,
            light: Optional[bool] = None, fast_shutter: bool = False, rationale: str = "") -> dict:
    return _call("capture", dict(view=view, zoom=zoom, ev_bias=ev_bias, burst=burst, hdr=hdr, light=light,
                                 fast_shutter=fast_shutter), rationale)


@mcp.tool(description="Measure the needle on a capture with OpenCV 5; enhance may include clahe, deglare, denoise.")
def read_gauge(capture_id: str, enhance: Optional[list[str]] = None, rationale: str = "") -> dict:
    return _call("read_gauge", {"capture_id": capture_id, "enhance": enhance or []}, rationale)


@mcp.tool(description="Record a reading by reading_id (guarded: confidence >= 0.6 and interval clear of limits).")
def record_reading(reading_id: str, note: str = "") -> dict:
    return _call("record_reading", {"reading_id": reading_id, "note": note})


@mcp.tool(description="Escalate the current gauge to the control-room operator.")
def request_human_review(reason: str, reading_id: Optional[str] = None) -> dict:
    return _call("request_human_review", {"reason": reason, "reading_id": reading_id})


@mcp.tool(description="Draft a work order (P1-P4) for human approval; required for alarm and instrument_fault.")
def create_work_order(priority: str, title: str, description: str, reading_id: Optional[str] = None) -> dict:
    return _call("create_work_order", {"priority": priority, "title": title, "description": description,
                                       "reading_id": reading_id})


@mcp.tool(description="Close the current gauge (the guard checks the outcome is complete).")
def finish_gauge(summary: str) -> dict:
    return _call("finish_gauge", {"summary": summary})


@mcp.tool(description="Save the evidence image for a capture or reading to a temporary file and return its path.")
def save_image(image_id: str) -> dict:
    data = _s().store.get(image_id)
    if not data:
        return {"error": "unknown image_id"}
    path = os.path.join(tempfile.gettempdir(), f"argus-{image_id}.jpg")
    with open(path, "wb") as f:
        f.write(data)
    return {"path": path}


@mcp.tool(description="Read a gauge from a local photo (no fiducial). Provide the scale; major_ticks enables "
                      "scale/rotation verification.")
def read_photo(path: str, min_value: float, max_value: float, units: str = "", major_ticks: int = 0,
               theta_min: float = -135.0, theta_max: float = 135.0, needle_color: str = "black") -> dict:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": f"cannot read image {path}"}
    spec = GaugeSpec(id="PHOTO", name="photo", units=units, min_value=min_value, max_value=max_value,
                     theta_min=theta_min, theta_max=theta_max, major_ticks=major_ticks if major_ticks >= 2 else 9,
                     needle_color=needle_color, marker_id=-1)
    an = analyze_capture([img], spec, None, upload=True, ticks=major_ticks >= 2)
    out_path = os.path.join(tempfile.gettempdir(), "argus-photo-overlay.jpg")
    cv2.imwrite(out_path, overlay(an, spec, "photo"))
    rd = an.reading
    return json.loads(json.dumps({"value": rd.get("value"), "sigma": rd.get("sigma"),
                                  "confidence": rd.get("confidence"),
                                  "issues": [i for i in an.quality.get("issues", []) if i != "fiducial_missing"],
                                  "overlay_path": out_path}, default=float))


def main():
    mcp.run("stdio")


if __name__ == "__main__":
    main()
