"""Inspection-round state: robot, captures, readings, reviews, work orders and the trace."""
from __future__ import annotations

import itertools
import math
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

import cv2
import numpy as np

from ..registry import GaugeSpec
from ..vision import sim
from ..vision.pipeline import Analysis, encode_jpeg

AUTO_ACCEPT_CONFIDENCE = 0.6
MAX_CAPTURES_PER_GAUGE = 6
MAX_TOOL_CALLS_PER_GAUGE = 22


class ImageStore:
    """In-memory JPEG store (bounded); the server mirrors it to S3 when configured."""

    def __init__(self, max_items: int = 800):
        self._d: OrderedDict[str, bytes] = OrderedDict()
        self._lock = threading.Lock()
        self.max_items = max_items
        self.on_put: Optional[Callable[[str, bytes], None]] = None

    def put(self, img: np.ndarray, prefix: str = "img", max_w: int = 1024, quality: int = 84) -> str:
        if img.shape[1] > max_w:
            s = max_w / img.shape[1]
            img = cv2.resize(img, (max_w, int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
        data = encode_jpeg(img, quality)
        iid = f"{prefix}-{uuid.uuid4().hex[:10]}"
        with self._lock:
            self._d[iid] = data
            while len(self._d) > self.max_items:
                self._d.popitem(last=False)
        if self.on_put:
            try:
                self.on_put(iid, data)
            except Exception:
                pass
        return iid

    def get(self, iid: str) -> Optional[bytes]:
        with self._lock:
            return self._d.get(iid)


@dataclass
class Capture:
    id: str
    gauge_id: str
    view: str
    params: dict
    settings: list
    frames: list
    predicted_H: Any
    hdr: bool
    robot_time_s: float
    image_id: str
    truth: list = field(default_factory=list, repr=False)
    analyses: dict = field(default_factory=dict, repr=False)


@dataclass
class ReadingRec:
    id: str
    gauge_id: str
    capture_id: str
    value: float
    sigma: float
    confidence: float
    units: str
    method: str
    flags: list
    issues: list
    overlay_id: str
    check: dict
    flutter: float = 0.0
    theta: float = float("nan")
    enhance: list = field(default_factory=list)


@dataclass
class Review:
    id: str
    gauge_id: str
    reason: str
    reading_id: Optional[str]
    evidence: list
    created_by: str
    status: str = "pending"  # pending | confirmed | corrected | rejected
    decided_by: Optional[str] = None
    final_value: Optional[float] = None
    note: str = ""
    created_at: float = field(default_factory=time.time)
    decided_at: Optional[float] = None


@dataclass
class WorkOrder:
    id: str
    gauge_id: str
    priority: str
    title: str
    description: str
    reading_id: Optional[str]
    evidence: list
    created_by: str
    status: str = "pending_approval"  # pending_approval | dispatched | rejected
    approved_by: Optional[str] = None
    dispatch_ref: Optional[str] = None
    priority_note: str = ""
    created_at: float = field(default_factory=time.time)
    decided_at: Optional[float] = None


@dataclass
class GaugeOutcome:
    gauge_id: str
    status: str = "pending"  # normal | warning | alarm | instrument_fault | escalated | pending
    reading_id: Optional[str] = None
    review_id: Optional[str] = None
    work_order_id: Optional[str] = None
    captures: int = 0
    tool_calls: int = 0
    robot_time_s: float = 0.0
    compute_ms: float = 0.0
    agent: str = ""
    finished: bool = False
    summary: str = ""
    forced: bool = False


class Session:
    _ids = itertools.count(1)

    def __init__(self, profile: str = "random", seed: int = 0, agent_mode: str = "policy", n_gauges: int = 5,
                 scene: Optional[sim.Scene] = None, keep_truth: bool = True):
        self.id = f"round-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.number = next(Session._ids)
        self.created = time.time()
        self.profile = profile
        self.seed = seed
        self.agent_mode = agent_mode
        self.scene = scene or sim.make_scene(seed=seed, profile=profile, n_gauges=n_gauges)
        self.camera = sim.RobotCamera(self.scene, seed=seed + 7)
        self.registry = self.scene.registry
        self.store = ImageStore()
        self.captures: dict[str, Capture] = {}
        self.readings: dict[str, ReadingRec] = {}
        self.reviews: dict[str, Review] = {}
        self.work_orders: dict[str, WorkOrder] = {}
        self.outcomes: dict[str, GaugeOutcome] = OrderedDict((s.id, GaugeOutcome(s.id)) for s in self.registry)
        self.trace: list[dict] = []
        self.usage = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "fallbacks": 0,
                      "llm_latency_ms": 0.0}
        self.status = "created"
        self.error: Optional[str] = None
        self.current_gauge: Optional[str] = None
        self.report: Optional[str] = None
        self.keep_truth = keep_truth
        self.lock = threading.RLock()
        self.listeners: list[Callable[[dict], None]] = []
        self._seq = 0
        self._cid = itertools.count(1)
        self.notes: list[str] = []

    # ---- trace
    def emit(self, actor: str, kind: str, **kw) -> dict:
        with self.lock:
            self._seq += 1
            ev = {"seq": self._seq, "t": round(time.time() - self.created, 3), "robot_t": round(self.camera.time_s, 1),
                  "gauge_id": kw.pop("gauge_id", self.current_gauge), "actor": actor, "kind": kind}
            ev.update(kw)
            self.trace.append(ev)
        for fn in list(self.listeners):
            try:
                fn(ev)
            except Exception:
                pass
        return ev

    # ---- context the agents see
    def ambient_sensor(self) -> float:
        """Robot light-sensor reading, normalised: ~1 daylight, <0.15 night."""
        return round(float(self.scene.ambient), 3)

    def gauge_context(self, gid: str) -> dict:
        spec = self.registry.gauges[gid]
        sg = self.scene.gauge(gid)
        return {
            "gauge_id": gid, "name": spec.name, "units": spec.units, "range": [spec.min_value, spec.max_value],
            "limits": {k: v for k, v in spec.limits().items() if v is not None}, "criticality": spec.criticality,
            "location": spec.location, "previous_readings": sg.history,
            "ambient_light": self.ambient_sensor(),
            "budget": {"max_captures": MAX_CAPTURES_PER_GAUGE, "max_tool_calls": MAX_TOOL_CALLS_PER_GAUGE},
        }

    def route(self) -> list[GaugeSpec]:
        return list(self.registry)

    def new_id(self, prefix: str) -> str:
        return f"{prefix}{next(self._cid)}"

    # ---- serialisation for the UI / S3
    def snapshot(self, since: int = 0) -> dict:
        with self.lock:
            gauges = []
            for gid, oc in self.outcomes.items():
                spec = self.registry.gauges[gid]
                rd = self.readings.get(oc.reading_id) if oc.reading_id else None
                last = None
                for r in reversed(list(self.readings.values())):
                    if r.gauge_id == gid:
                        last = r
                        break
                gauges.append({
                    "spec": spec.to_dict(), "outcome": asdict(oc),
                    "recorded": _reading_public(rd), "latest": _reading_public(last),
                    "captures": [c.image_id for c in self.captures.values() if c.gauge_id == gid],
                })
            return {
                "id": self.id, "number": self.number, "created": self.created, "status": self.status,
                "error": self.error, "profile": self.profile, "seed": self.seed,
                "agent_mode": self.agent_mode, "current_gauge": self.current_gauge,
                "robot_time_s": round(self.camera.time_s, 1), "ambient_light": self.ambient_sensor(),
                "gauges": gauges,
                "reviews": [asdict(r) for r in self.reviews.values()],
                "work_orders": [asdict(w) for w in self.work_orders.values()],
                "usage": {k: (round(v, 5) if isinstance(v, float) else v) for k, v in self.usage.items()},
                "report": self.report, "notes": self.notes,
                "trace": [e for e in self.trace if e["seq"] > since],
                "trace_len": len(self.trace),
            }

    def brief(self) -> dict:
        """One row for the rounds list."""
        with self.lock:
            ocs = list(self.outcomes.values())
            return {
                "id": self.id, "number": self.number, "status": self.status, "profile": self.profile,
                "agent": self.agent_mode, "seed": self.seed, "created": self.created, "gauges": len(ocs),
                "done": sum(1 for o in ocs if o.finished), "robot_time_s": round(self.camera.time_s, 1),
                "alarms": sum(1 for o in ocs if o.status in ("alarm", "instrument_fault")),
                "warnings": sum(1 for o in ocs if o.status == "warning"),
                "escalated": sum(1 for o in ocs if o.review_id),
                "pending_human": sum(1 for r in self.reviews.values() if r.status == "pending")
                + sum(1 for w in self.work_orders.values() if w.status == "pending_approval"),
                "fallbacks": self.usage.get("fallbacks", 0), "cost_usd": round(self.usage.get("cost_usd", 0.0), 5),
            }

    def export_trace(self) -> dict:
        snap = self.snapshot()
        snap["trace"] = list(self.trace)
        snap["scenario"] = {"profile": self.profile, "seed": self.seed}
        return snap

    def release_frames(self, gid: str):
        for c in self.captures.values():
            if c.gauge_id == gid:
                c.frames = c.frames[:1] if not c.hdr else c.frames
                c.analyses = {k: _strip(a) for k, a in c.analyses.items()}


def _strip(a: Analysis) -> Analysis:
    a.canon = None
    a.work_image = None
    a.frame_readings = []
    return a


def _reading_public(r: Optional[ReadingRec]) -> Optional[dict]:
    if r is None:
        return None
    d = asdict(r)
    for k, v in list(d.items()):
        if isinstance(v, float) and not math.isfinite(v):
            d[k] = None
    return d
