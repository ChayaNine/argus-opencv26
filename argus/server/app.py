"""ARGUS web service: runs inspection rounds, serves the live UI, takes human decisions."""
from __future__ import annotations

import json
import logging
import math
import os
import random
import threading
import time
from collections import OrderedDict, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import __version__
from ..agent.session import Session
from ..agent.supervisor import run_round
from ..config import CONFIG
from ..registry import GaugeSpec
from ..runtime import runtime_info
from ..vision import privacy
from ..vision.pipeline import analyze_capture, encode_jpeg, overlay
from ..vision.sim import PROFILES
from . import cloud

log = logging.getLogger("argus")
logging.basicConfig(level=os.environ.get("ARGUS_LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")

WEB = Path(__file__).resolve().parents[2] / "web"
app = FastAPI(title="ARGUS", version=__version__, docs_url="/api/docs", redoc_url=None)

_rt = runtime_info(light=True)
_stream = f"{_rt.get('ec2', {}).get('instance-id', 'local')}-{time.strftime('%Y%m%d')}"
archive = cloud.Archive()
tracelogs = cloud.TraceLogs(_stream)
metrics = cloud.Metrics()
pool = ThreadPoolExecutor(max_workers=CONFIG.max_concurrent_rounds, thread_name_prefix="round")
SESSIONS: "OrderedDict[str, Session]" = OrderedDict()
LOCK = threading.Lock()
UPLOADS: "OrderedDict[str, bytes]" = OrderedDict()
COUNTERS = defaultdict(float)
_rate: dict = defaultdict(deque)
_llm_day = {"day": time.strftime("%Y-%m-%d"), "n": 0}
_llm = None


def llm_agent():
    global _llm
    if not CONFIG.llm_enabled:
        return None
    if _llm is None:
        from ..agent.llm import BedrockInspector
        _llm = BedrockInspector.from_env()
    else:
        _llm.circuit_open = False
        _llm.consecutive_failures = 0
    return _llm


# ---------------------------------------------------------------------------
# middleware: origin verification (CloudFront secret header), security headers


@app.middleware("http")
async def guard_and_headers(request: Request, call_next):
    if CONFIG.origin_secret and request.url.path != "/api/health":
        client = request.client.host if request.client else ""
        if client not in ("127.0.0.1", "::1") and request.headers.get("x-origin-verify") != CONFIG.origin_secret:
            return PlainTextResponse("forbidden", status_code=403)
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Content-Security-Policy"] = (
        "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
    return resp


def client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_ok(key: str, limit: int, window_s: float = 3600) -> bool:
    now = time.time()
    dq = _rate[key]
    while dq and now - dq[0] > window_s:
        dq.popleft()
    if len(dq) >= limit:
        return False
    dq.append(now)
    return True


# ---------------------------------------------------------------------------
# rounds


class RoundRequest(BaseModel):
    profile: str = Field("random")
    seed: Optional[int] = None
    agent: str = Field("policy")
    n_gauges: int = Field(5, ge=2, le=8)


def _archive_round(s: Session):
    try:
        data = json.dumps(s.export_trace(), default=_json_default).encode()
        CONFIG.data_dir.mkdir(parents=True, exist_ok=True)
        (CONFIG.data_dir / "rounds").mkdir(exist_ok=True)
        (CONFIG.data_dir / "rounds" / f"{s.id}.json").write_bytes(data)
        archive.put(f"rounds/{s.id}/trace.json", data, "application/json")
        metrics.round_done(s.snapshot())
    except Exception as e:  # pragma: no cover
        log.warning("archive failed: %s", e)


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _run(s: Session, agent):
    try:
        run_round(s, llm_agent=agent)
    finally:
        COUNTERS["rounds_done"] += 1
        COUNTERS["gauges"] += len(s.outcomes)
        COUNTERS["escalations"] += sum(1 for o in s.outcomes.values() if o.review_id)
        COUNTERS["fallbacks"] += s.usage.get("fallbacks", 0)
        COUNTERS["llm_cost_usd"] += s.usage.get("cost_usd", 0.0)
        COUNTERS["compute_ms"] += sum(o.compute_ms for o in s.outcomes.values())
        _archive_round(s)


@app.post("/api/rounds")
def start_round(req: RoundRequest, request: Request):
    ip = client_ip(request)
    if req.profile not in PROFILES:
        raise HTTPException(400, f"unknown profile; choose one of {list(PROFILES)}")
    if req.agent not in ("policy", "llm"):
        raise HTTPException(400, "agent must be 'policy' or 'llm'")
    running = sum(1 for s in SESSIONS.values() if s.status in ("created", "running"))
    if running >= CONFIG.max_concurrent_rounds * 3:
        raise HTTPException(429, "the robot is busy with other rounds; try again in a minute")
    if not rate_ok(f"round:{ip}", CONFIG.rounds_per_ip_hour):
        raise HTTPException(429, "rate limit: too many rounds from this address in the last hour")
    agent = None
    if req.agent == "llm":
        if not CONFIG.llm_enabled:
            raise HTTPException(409, "the Bedrock LLM agent is not enabled on this deployment; use the policy agent")
        today = time.strftime("%Y-%m-%d")
        if _llm_day["day"] != today:
            _llm_day.update(day=today, n=0)
        if _llm_day["n"] >= CONFIG.llm_daily_cap:
            raise HTTPException(429, "daily LLM budget reached (cost control); the policy agent is still available")
        if not rate_ok(f"llm:{ip}", CONFIG.llm_rounds_per_ip_hour):
            raise HTTPException(429, "rate limit for LLM rounds from this address; try the policy agent")
        _llm_day["n"] += 1
        agent = llm_agent()
    seed = req.seed if req.seed is not None else random.randint(1, 10_000_000)
    s = Session(profile=req.profile, seed=seed, agent_mode=req.agent, n_gauges=req.n_gauges)
    s.listeners.append(lambda ev, rid=s.id: tracelogs.event(rid, ev))
    s.store.on_put = lambda iid, data, rid=s.id: archive.put(f"rounds/{rid}/img/{iid}.jpg", data, "image/jpeg")
    with LOCK:
        SESSIONS[s.id] = s
        while len(SESSIONS) > 40:
            old_id, old = next(iter(SESSIONS.items()))
            if old.status in ("created", "running"):
                break
            SESSIONS.pop(old_id)
    pool.submit(_run, s, agent)
    return {"id": s.id, "number": s.number, "seed": seed, "profile": req.profile, "agent": req.agent}


@app.get("/api/rounds")
def list_rounds():
    with LOCK:
        items = list(SESSIONS.values())[-20:]
    return [s.brief() for s in reversed(items)]


def _get(rid: str) -> Session:
    s = SESSIONS.get(rid)
    if s is None:
        raise HTTPException(404, "round not found (rounds are kept in memory for the life of the server)")
    return s


@app.get("/api/rounds/{rid}")
def get_round(rid: str, since: int = 0):
    s = _get(rid)
    return JSONResponse(json.loads(json.dumps(s.snapshot(since), default=_json_default)))


@app.get("/api/rounds/{rid}/trace.json")
def get_trace(rid: str):
    s = _get(rid)
    data = json.dumps(s.export_trace(), default=_json_default, indent=1)
    return Response(data, media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="{rid}-trace.json"'})


@app.get("/api/images/{iid}")
def get_image(iid: str):
    for s in reversed(list(SESSIONS.values())):
        b = s.store.get(iid)
        if b:
            return Response(b, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})
    b = UPLOADS.get(iid)
    if b:
        return Response(b, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
    raise HTTPException(404, "image not found")


# ---------------------------------------------------------------------------
# human-in-the-loop decisions


class ReviewDecision(BaseModel):
    action: str  # confirm | correct | reject
    value: Optional[float] = None
    reviewer: str = Field("operator", max_length=60)
    note: str = Field("", max_length=400)


class WODecision(BaseModel):
    action: str  # approve | reject
    reviewer: str = Field("supervisor", max_length=60)
    note: str = Field("", max_length=400)


def _find(kind: str, oid: str):
    for s in reversed(list(SESSIONS.values())):
        d = s.reviews if kind == "review" else s.work_orders
        if oid in d:
            return s, d[oid]
    raise HTTPException(404, f"{kind} {oid} not found")


@app.post("/api/rounds/{rid}/reviews/{oid}")
def decide_review(rid: str, oid: str, dec: ReviewDecision):
    s = _get(rid)
    rv = s.reviews.get(oid)
    if rv is None:
        raise HTTPException(404, "review not found")
    if rv.status != "pending":
        raise HTTPException(409, f"already {rv.status}")
    spec = s.registry.gauges[rv.gauge_id]
    with s.lock:
        if dec.action == "confirm":
            if rv.reading_id is None:
                raise HTTPException(400, "nothing to confirm: enter the value you read (correct)")
            rv.final_value = s.readings[rv.reading_id].value
            rv.status = "confirmed"
        elif dec.action == "correct":
            if dec.value is None or not math.isfinite(dec.value):
                raise HTTPException(400, "value required")
            rv.final_value = float(dec.value)
            rv.status = "corrected"
        elif dec.action == "reject":
            rv.status = "rejected"
        else:
            raise HTTPException(400, "action must be confirm|correct|reject")
        rv.decided_by, rv.note, rv.decided_at = dec.reviewer, dec.note, time.time()
        oc = s.outcomes[rv.gauge_id]
        if rv.final_value is not None:
            st = spec.status_of(rv.final_value)
            oc.status = f"{st} (operator)"
            s.emit("human", "human_review", gauge_id=rv.gauge_id, review_id=rv.id, decision=rv.status,
                   value=rv.final_value, by=dec.reviewer, status=st)
            if st == "alarm" and oc.work_order_id is None:
                from ..agent.session import WorkOrder
                wo = WorkOrder(s.new_id("WO-"), rv.gauge_id, "P1" if spec.criticality == "high" else "P2",
                               f"{rv.gauge_id} alarm confirmed by operator: {rv.final_value:g} {spec.units}",
                               f"Operator {dec.reviewer} reported {rv.final_value:g} {spec.units} for {spec.name}, "
                               f"outside alarm limits ({spec.limits_text()}). Raised automatically for approval.",
                               rv.reading_id, rv.evidence[-2:], "system")
                s.work_orders[wo.id] = wo
                oc.work_order_id = wo.id
                s.emit("system", "work_order_drafted", gauge_id=rv.gauge_id, work_order_id=wo.id)
        else:
            s.emit("human", "human_review", gauge_id=rv.gauge_id, review_id=rv.id, decision=rv.status,
                   by=dec.reviewer, note=dec.note)
    return asdict(rv)


@app.post("/api/rounds/{rid}/workorders/{oid}")
def decide_wo(rid: str, oid: str, dec: WODecision):
    s = _get(rid)
    wo = s.work_orders.get(oid)
    if wo is None:
        raise HTTPException(404, "work order not found")
    if wo.status != "pending_approval":
        raise HTTPException(409, f"already {wo.status}")
    with s.lock:
        if dec.action == "approve":
            spec = s.registry.gauges[wo.gauge_id]
            res = cloud.dispatch_work_order(asdict(wo), {"id": spec.id, "name": spec.name}, dec.reviewer)
            wo.status = "dispatched"
            wo.dispatch_ref = f"{res.get('channel')}:{res.get('ref')}"
        elif dec.action == "reject":
            wo.status = "rejected"
        else:
            raise HTTPException(400, "action must be approve|reject")
        wo.approved_by, wo.decided_at = dec.reviewer, time.time()
        s.emit("human", "human_work_order", gauge_id=wo.gauge_id, work_order_id=wo.id, decision=wo.status,
               by=dec.reviewer, dispatch_ref=wo.dispatch_ref, note=dec.note)
    return asdict(wo)


# ---------------------------------------------------------------------------
# single-photo analysis (bring your own gauge)

GUIDANCE = {
    "glare_on_needle": "Step to one side: the reflection on the glass moves when you move, the needle does not.",
    "glare_on_scale": "There is a reflection on the glass; a small step sideways usually clears it.",
    "overexposed": "The dial is washed out; tap the dial on your screen to set exposure, or shade it.",
    "underexposed": "Too dark: turn on a light or move closer to a light source.",
    "noisy": "Low light: hold the phone still and use more light.",
    "defocus_blur": "Out of focus: tap the dial to focus and hold still.",
    "motion_blur": "Motion blur: brace your hands, or the gauge is vibrating (take several photos).",
    "oblique_view": "Face the dial more squarely.",
    "very_oblique_view": "Face the dial squarely; the angle is too steep.",
    "low_resolution": "Move closer so the dial fills about half the frame.",
    "low_contrast": "The glass may be dirty or fogged; wipe it if it is safe to do so.",
    "occluded": "Something covers part of the dial; change position.",
    "needle_not_clear": "The needle is hard to see; change angle or lighting.",
    "no_dial": "No round dial found; fill more of the frame with the gauge.",
    "scale_not_verified": "Enter how many numbered marks the dial has, so ARGUS can check the scale and the photo's rotation.",
}


@app.post("/api/analyze")
async def analyze_photo(request: Request, file: UploadFile = File(...), min_value: float = Form(...),
                        max_value: float = Form(...), units: str = Form("bar"), major_ticks: int = Form(0),
                        theta_min: float = Form(-135.0), theta_max: float = Form(135.0),
                        needle_color: str = Form("black")):
    ip = client_ip(request)
    if not rate_ok(f"upload:{ip}", CONFIG.uploads_per_ip_hour):
        raise HTTPException(429, "rate limit for photo analysis")
    data = await file.read(int(CONFIG.max_upload_mb * 1024 * 1024) + 1)
    if len(data) > CONFIG.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"image larger than {CONFIG.max_upload_mb} MB")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "could not decode the image (use JPEG or PNG)")
    if not (max_value > min_value) or not (theta_max > theta_min) or not (0 <= major_ticks <= 25):
        raise HTTPException(400, "check the scale parameters")
    h, w = img.shape[:2]
    if max(h, w) > 2000:
        s = 2000 / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    img, n_faces = privacy.blur_faces(img)  # before anything is kept
    spec = GaugeSpec(id="PHOTO", name="Uploaded gauge", units=units[:12], min_value=min_value, max_value=max_value,
                     theta_min=theta_min, theta_max=theta_max, major_ticks=major_ticks if major_ticks >= 2 else 9,
                     needle_color="red" if needle_color == "red" else "black", marker_id=-1)
    an = analyze_capture([img], spec, None, upload=True, ticks=major_ticks >= 2)
    ov = overlay(an, spec, title="photo")
    iid = f"up-{os.urandom(5).hex()}"
    UPLOADS[iid] = encode_jpeg(ov, 85)
    while len(UPLOADS) > 60:
        UPLOADS.popitem(last=False)
    issues = [i for i in an.quality.get("issues", []) if i != "fiducial_missing"]  # photos have no tag
    rd = an.reading
    accept = rd.get("ok") and rd.get("confidence", 0) >= 0.6
    out = {
        "ok": bool(rd.get("ok")), "value": rd.get("value"), "sigma": rd.get("sigma"), "units": spec.units,
        "confidence": rd.get("confidence"), "accepted": bool(accept), "issues": issues,
        "guidance": [GUIDANCE[i] for i in issues if i in GUIDANCE][:3], "overlay": f"/api/images/{iid}",
        "faces_blurred": n_faces, "localization": an.localization.get("method"),
        "scale_verified": rd.get("scale_verified"), "timings_ms": an.timings_ms,
        "decision": ("reading accepted" if accept else
                     "not trustworthy yet: follow the guidance and take another photo, or read it yourself"),
        "privacy": "processed in memory; faces blurred; the photo itself is not stored",
    }
    return JSONResponse(json.loads(json.dumps(out, default=_json_default)))


# ---------------------------------------------------------------------------
# evidence, health, runtime


def _load_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


@app.get("/api/evidence")
def evidence():
    res = CONFIG.results_dir
    ev = (_load_json(res / "eval" / "latest" / "summary.json") or archive.get_json("evidence/eval_summary.json")
          or _load_json(res / "eval" / "final" / "summary.json"))
    bench = _load_json(res / "bench" / "compare.json") or archive.get_json("evidence/bench_compare.json")
    return {"eval": ev, "bench": bench}


@app.get("/api/evidence/{name}")
def evidence_file(name: str):
    latest = CONFIG.results_dir / "eval" / "latest" / "strategies.png"
    allowed = {"strategies.png": latest if latest.exists() else CONFIG.results_dir / "eval" / "final" / "strategies.png",
               "bench.png": CONFIG.results_dir / "bench" / "bench.png"}
    p = allowed.get(name)
    if p is None or not p.exists():
        raise HTTPException(404, "not found")
    return FileResponse(p)


@app.get("/api/runtime")
def runtime():
    info = runtime_info()
    info["privacy_face_detector"] = privacy.available()
    return info


@app.get("/api/config")
def config():
    return {"version": __version__, "profiles": PROFILES, "llm_enabled": CONFIG.llm_enabled,
            "llm_model": CONFIG.llm_model if CONFIG.llm_enabled else None,
            "auto_accept_confidence": 0.6, "archive": archive.enabled, "cloudwatch": tracelogs.enabled,
            "sns": bool(CONFIG.sns_topic_arn)}


@app.get("/api/health")
def health():
    return {"ok": True, "version": __version__, "runtime": _rt.get("label"),
            "rounds_running": sum(1 for s in SESSIONS.values() if s.status == "running"),
            "archive_errors": archive.errors, "uploaded": archive.uploaded}


@app.get("/metrics")
def prom():
    lines = [f"argus_{k}_total {v}" for k, v in COUNTERS.items()]
    lines.append(f"argus_rounds_running {sum(1 for s in SESSIONS.values() if s.status == 'running')}")
    return PlainTextResponse("\n".join(lines) + "\n")


if WEB.exists():
    app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/favicon.ico")
def favicon():
    return FileResponse(WEB / "favicon.svg", media_type="image/svg+xml")
