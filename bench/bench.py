"""COOL vs stock OpenCV 5: time the real ARGUS vision workload on identical inputs.

    python -m bench.bench --label cool-c8g          # run with the COOL interpreter
    python -m bench.bench --label stock-c8g         # run with a pip opencv-python-headless==5.0.0.93 venv

Measures
  pipeline      analyze_capture() on 24 single frames (localize, rectify, assess, read): latency
  burst         analyze_capture() on a 9-frame burst (per-frame needle reads + fusion): latency
  throughput    reads/second with one process per vCPU (cv2.setNumThreads(1) each): saturation
  ops           micro-benchmarks of the OpenCV calls the pipeline is made of
The JSON records exactly which cv2 build ran (path, version, build flags) and on which CPU/instance.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from argus.registry import GaugeSpec  # noqa: E402
from argus.runtime import runtime_info  # noqa: E402
from argus.vision import perception as P  # noqa: E402
from argus.vision.pipeline import analyze_capture  # noqa: E402

INP = Path(__file__).resolve().parent / "inputs"


def _spec(d: dict) -> GaugeSpec:
    keys = GaugeSpec.__dataclass_fields__.keys()
    return GaugeSpec(**{k: v for k, v in d.items() if k in keys})


def load():
    man = json.loads((INP / "manifest.json").read_text())
    singles = [(cv2.imread(str(INP / m["file"])), _spec(m["spec"]), np.array(m["predicted_H"])) for m in man["single"]]
    b = man["burst"]
    burst = ([cv2.imread(str(INP / f)) for f in b["files"]], _spec(b["spec"]), np.array(b["predicted_H"]))
    return singles, burst


def stats(xs):
    xs = sorted(xs)
    return {"n": len(xs), "median_ms": round(statistics.median(xs), 3), "mean_ms": round(statistics.mean(xs), 3),
            "p95_ms": round(xs[int(0.95 * (len(xs) - 1))], 3), "min_ms": round(xs[0], 3)}


def timeit(fn, reps=30, warm=3):
    for _ in range(warm):
        fn()
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t) * 1000)
    return stats(out)


def bench_ops(img: np.ndarray, reps: int):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canon = cv2.resize(img[:, 320:960], (512, 512))
    clahe = cv2.createCLAHE(2.5, (8, 8))
    H = np.array([[0.9, 0.05, 20], [-0.03, 0.95, 10], [1e-5, 2e-5, 1]])
    mask = np.zeros(canon.shape[:2], np.uint8)
    cv2.circle(mask, (200, 200), 30, 255, -1)
    det = P.detector() if hasattr(cv2, "aruco") else None
    kern = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    mertens = cv2.createMergeMertens() if hasattr(cv2, "createMergeMertens") else None
    frames3 = [canon, cv2.convertScaleAbs(canon, alpha=0.5), cv2.convertScaleAbs(canon, alpha=1.5)]
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ops = {
        "cvtColor BGR2GRAY 720p": lambda: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
        "resize INTER_AREA 0.5x": lambda: cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
        "GaussianBlur 5x5 720p": lambda: cv2.GaussianBlur(img, (5, 5), 0),
        "Canny 720p": lambda: cv2.Canny(gray, 80, 160),
        "adaptiveThreshold 720p": lambda: cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                                cv2.THRESH_BINARY, 31, 5),
        "findContours 720p": lambda: cv2.findContours(thr, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE),
        "CLAHE 720p gray": lambda: clahe.apply(gray),
        "warpPerspective to 512": lambda: cv2.warpPerspective(img, H, (512, 512)),
        "warpPolar 512->720x240": lambda: P.to_polar(canon),
        "HoughCircles ALT 640": lambda: cv2.HoughCircles(cv2.resize(gray, (640, 360)), cv2.HOUGH_GRADIENT_ALT, 1.5,
                                                         20, param1=150, param2=0.8, minRadius=20, maxRadius=170),
        "ArUco detectMarkers 720p": lambda: det.detectMarkers(gray),
        "Laplacian 720p": lambda: cv2.Laplacian(gray, cv2.CV_32F),
        "filter2D noise kernel 720p": lambda: cv2.filter2D(gray.astype(np.float32), -1, kern),
        "MergeMertens 3x512": lambda: mertens.process(frames3),
        "inpaint Telea 512": lambda: cv2.inpaint(canon, mask, 5, cv2.INPAINT_TELEA),
        "fastNlMeansDenoisingColored 512": lambda: cv2.fastNlMeansDenoisingColored(canon, None, 7, 7, 5, 15),
    }
    try:
        from argus.vision import privacy
        if privacy.available():
            d = privacy._detector()
            small = cv2.resize(img, (640, 360))
            d.setInputSize((640, 360))
            ops["DNN YuNet face detect 640x360"] = lambda: d.detect(small)
    except Exception:
        pass
    out = {}
    for k, f in ops.items():
        try:
            out[k] = timeit(f, reps=reps)
        except (cv2.error, AttributeError) as e:
            print(f"  skip {k}: {e}", flush=True)
    return out


_W = None


def _winit():
    global _W
    cv2.setNumThreads(1)
    _W = load()[0]


def _wrun(n):
    out = 0
    for i in range(n):
        img, spec, Hp = _W[i % len(_W)]
        analyze_capture([img], spec, Hp)
        out += 1
    return out


def throughput(n_proc: int, seconds: float) -> dict:
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_proc, initializer=_winit) as pool:
        pool.map(_wrun, [2] * n_proc)  # warm
        t0 = time.perf_counter()
        done = 0
        while time.perf_counter() - t0 < seconds:
            done += sum(pool.map(_wrun, [6] * n_proc))
        dt = time.perf_counter() - t0
    return {"processes": n_proc, "reads": done, "seconds": round(dt, 2), "reads_per_s": round(done / dt, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--reps", type=int, default=5, help="passes over the 24 inputs for latency")
    ap.add_argument("--op-reps", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=20.0, help="throughput measurement window")
    ap.add_argument("--out", default=str(ROOT / "results" / "bench"))
    ap.add_argument("--threads", type=int, default=0, help="cv2.setNumThreads for latency runs (0 = library default)")
    a = ap.parse_args()
    if a.threads:
        cv2.setNumThreads(a.threads)
    singles, burst = load()
    rt = runtime_info()
    print(f"[{a.label}] {rt['label']} threads={cv2.getNumThreads()}", flush=True)
    # correctness check: the build must produce the same readings
    readings = []
    for img, spec, Hp in singles:
        an = analyze_capture([img], spec, Hp)
        readings.append(round(an.reading.get("value", float("nan")), 4) if an.reading.get("ok") else None)
    lat = []
    for _ in range(a.reps):
        for img, spec, Hp in singles:
            t = time.perf_counter()
            analyze_capture([img], spec, Hp)
            lat.append((time.perf_counter() - t) * 1000)
    blat = []
    for _ in range(max(3, a.reps)):
        t = time.perf_counter()
        analyze_capture(burst[0], burst[1], burst[2])
        blat.append((time.perf_counter() - t) * 1000)
    print("  pipeline", stats(lat), flush=True)
    ops = bench_ops(singles[0][0], a.op_reps)
    tp1 = throughput(1, a.seconds / 2)
    tpn = throughput(os.cpu_count() or 1, a.seconds)
    print("  throughput", tp1, tpn, flush=True)
    res = {"label": a.label, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "runtime": rt,
           "cv2_threads": cv2.getNumThreads(), "pipeline": stats(lat), "burst9": stats(blat),
           "throughput_1proc": tp1, "throughput_all": tpn, "ops": ops, "readings": readings}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"bench_{a.label}.json"
    p.write_text(json.dumps(res, indent=1))
    print("wrote", p)


if __name__ == "__main__":
    main()
