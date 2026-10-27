"""HTTP API: rounds, human decisions, photo analysis, security."""
import time

import cv2
import pytest
from fastapi.testclient import TestClient

from argus.server import app as appmod
from argus.vision import sim


@pytest.fixture(scope="module")
def client():
    return TestClient(appmod.app)


def _wait(client, rid, timeout=120):
    t = time.time()
    while time.time() - t < timeout:
        snap = client.get(f"/api/rounds/{rid}").json()
        if snap["status"] in ("done", "error"):
            return snap
        time.sleep(0.5)
    raise AssertionError("round did not finish")


def test_health_runtime_config(client):
    assert client.get("/api/health").json()["ok"]
    rt = client.get("/api/runtime").json()
    assert rt["opencv_major"] >= 5
    cfg = client.get("/api/config").json()
    assert "hard" in cfg["profiles"]
    r = client.get("/")
    assert r.status_code == 200 and "Content-Security-Policy" in r.headers


def test_round_and_human_decisions(client):
    r = client.post("/api/rounds", json={"profile": "hard", "seed": 3, "agent": "policy"})
    assert r.status_code == 200
    rid = r.json()["id"]
    snap = _wait(client, rid)
    assert snap["status"] == "done"
    assert len(snap["gauges"]) == 5
    if snap["work_orders"]:
        wo = snap["work_orders"][0]
        d = client.post(f"/api/rounds/{rid}/workorders/{wo['id']}", json={"action": "approve", "reviewer": "test"})
        assert d.status_code == 200 and d.json()["status"] == "dispatched"
        again = client.post(f"/api/rounds/{rid}/workorders/{wo['id']}", json={"action": "approve"})
        assert again.status_code == 409
    if snap["reviews"]:
        rv = snap["reviews"][0]
        d = client.post(f"/api/rounds/{rid}/reviews/{rv['id']}", json={"action": "correct", "value": 1.0})
        assert d.status_code == 200 and d.json()["status"] == "corrected"
    tr = client.get(f"/api/rounds/{rid}/trace.json")
    assert tr.status_code == 200 and b"tool_call" in tr.content
    img = snap["gauges"][0]["captures"][0]
    assert client.get(f"/api/images/{img}").headers["content-type"] == "image/jpeg"


def test_llm_disabled_is_explicit(client):
    r = client.post("/api/rounds", json={"profile": "clean", "agent": "llm"})
    assert r.status_code == 409


def test_photo_analysis(client, tmp_path):
    sc = sim.make_scene(seed=12, profile="clean")
    cam = sim.RobotCamera(sc, seed=1)
    sg = sc.gauges[0]
    img = cam.capture(sg.spec.id, "front", zoom=1.35)["frames"][0][:600]
    p = tmp_path / "g.jpg"
    cv2.imwrite(str(p), img)
    with open(p, "rb") as f:
        r = client.post("/api/analyze", files={"file": ("g.jpg", f, "image/jpeg")},
                        data={"min_value": sg.spec.min_value, "max_value": sg.spec.max_value, "units": sg.spec.units,
                              "major_ticks": sg.spec.major_ticks})
    d = r.json()
    assert r.status_code == 200 and d["ok"]
    assert abs(d["value"] - sg.true_value) / sg.spec.span < 0.01
    assert "not stored" in d["privacy"]
    bad = client.post("/api/analyze", files={"file": ("x.jpg", b"not an image", "image/jpeg")},
                      data={"min_value": 0, "max_value": 10})
    assert bad.status_code == 400


def test_origin_secret_enforced(monkeypatch):
    monkeypatch.setattr(appmod.CONFIG, "origin_secret", "s3cret-value")
    c = TestClient(appmod.app, client=("203.0.113.9", 5000))
    assert c.get("/api/config").status_code == 403
    assert c.get("/api/config", headers={"X-Origin-Verify": "s3cret-value"}).status_code == 200
    assert c.get("/api/health").status_code == 200
