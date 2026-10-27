"""Render the fixed benchmark inputs once, so every machine times identical pixels.

    python -m bench.make_inputs            # writes bench/inputs/*.jpg + manifest.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from argus.vision import sim  # noqa: E402

OUT = Path(__file__).resolve().parent / "inputs"


def main():
    OUT.mkdir(exist_ok=True)
    manifest = []
    profiles = [("clean", 501), ("glare", 502), ("night", 503), ("vibration", 504), ("hard", 505), ("random", 506)]
    for prof, seed in profiles:
        sc = sim.make_scene(seed=seed, profile=prof, n_gauges=4)
        cam = sim.RobotCamera(sc, seed=seed)
        for sg in sc.gauges:
            cap = cam.capture(sg.spec.id, "front", burst=1)
            name = f"{prof}_{sg.spec.id}.jpg"
            cv2.imwrite(str(OUT / name), cap["frames"][0], [cv2.IMWRITE_JPEG_QUALITY, 92])
            manifest.append({"file": name, "spec": sg.spec.to_dict(), "predicted_H": cap["predicted_H"].tolist(),
                             "profile": prof})
    # one 9-frame burst for the multi-frame workload
    sc = sim.make_scene(seed=507, profile="vibration", n_gauges=2)
    cam = sim.RobotCamera(sc, seed=507)
    sg = sc.gauges[0]
    cap = cam.capture(sg.spec.id, "front", burst=9, fast_shutter=True)
    burst = []
    for i, f in enumerate(cap["frames"]):
        name = f"burst_{i}.jpg"
        cv2.imwrite(str(OUT / name), f, [cv2.IMWRITE_JPEG_QUALITY, 92])
        burst.append(name)
    (OUT / "manifest.json").write_text(json.dumps({"single": manifest, "burst": {
        "files": burst, "spec": sg.spec.to_dict(), "predicted_H": cap["predicted_H"].tolist()}}, indent=1))
    print(f"wrote {len(manifest)} single frames and a {len(burst)}-frame burst to {OUT}")


if __name__ == "__main__":
    main()
