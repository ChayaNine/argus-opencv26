"""Reader precision on clean frontal captures (single frame), against the true needle angle.

    python -m eval.precision --seeds 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from argus.vision import geometry as g  # noqa: E402
from argus.vision import sim  # noqa: E402
from argus.vision.pipeline import analyze_capture  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="results/eval/final/precision.json")
    a = ap.parse_args()
    errs, skipped = [], 0
    for sd in range(2001, 2001 + a.seeds):
        sc = sim.make_scene(seed=sd, profile="clean")
        cam = sim.RobotCamera(sc, seed=sd)
        for sgg in sc.gauges:
            cap = cam.capture(sgg.spec.id, "front")
            if cap["defocus"]:
                skipped += 1
                continue
            an = analyze_capture(cap["frames"], sgg.spec, cap["predicted_H"])
            if not an.reading.get("ok"):
                continue
            d = g.wrap180(an.reading["theta"] - cap["truth"][0]["theta"])
            errs.append(abs(d) / (sgg.spec.theta_max - sgg.spec.theta_min) * 100)
    e = np.array(errs)
    res = {"n": int(len(e)), "skipped_autofocus_failures": skipped, "median_pct_span": round(float(np.median(e)), 3),
           "p90_pct_span": round(float(np.percentile(e, 90)), 3), "max_pct_span": round(float(e.max()), 3),
           "median_deg": round(float(np.median(e)) * 2.7, 3)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(res)


if __name__ == "__main__":
    main()
