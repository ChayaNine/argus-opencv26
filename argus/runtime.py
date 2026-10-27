"""Runtime verification: which OpenCV build is executing, on what hardware.

This is the evidence for "COOL executes the core workload on Graviton": the
module path (COOL installs under /opt/cool), the version, the build flags
(KleidiCV / SVE / NEON / custom HAL), the CPU architecture and part number, and
the EC2 instance type from IMDSv2.
"""
from __future__ import annotations

import functools
import json
import os
import platform
import re
import urllib.request

import cv2

ARM_PARTS = {"0xd0c": "Neoverse-N1 (Graviton2)", "0xd40": "Neoverse-V1 (Graviton3)",
             "0xd4f": "Neoverse-V2 (Graviton4)", "0xd84": "Neoverse-V3 (Graviton5)"}


def _cpu() -> dict:
    out = {"machine": platform.machine(), "cpus": os.cpu_count()}
    try:
        txt = open("/proc/cpuinfo").read()
        m = re.search(r"CPU part\s*:\s*(0x[0-9a-f]+)", txt)
        if m:
            out["arm_part"] = m.group(1)
            out["arm_core"] = ARM_PARTS.get(m.group(1), "unknown")
        f = re.search(r"Features\s*:\s*(.*)", txt)
        if f:
            feats = f.group(1).split()
            out["sve"] = "sve" in feats
            out["sve2"] = "sve2" in feats
            out["neon_asimd"] = "asimd" in feats
        mn = re.search(r"model name\s*:\s*(.*)", txt)
        if mn:
            out["model_name"] = mn.group(1).strip()
    except OSError:
        pass
    return out


@functools.lru_cache(maxsize=1)
def _imds() -> dict:
    if os.environ.get("ARGUS_NO_IMDS"):
        return {}
    try:
        req = urllib.request.Request("http://169.254.169.254/latest/api/token", method="PUT",
                                     headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"})
        tok = urllib.request.urlopen(req, timeout=0.4).read().decode()
        out = {}
        for k in ("instance-type", "placement/region", "placement/availability-zone", "ami-id", "instance-id"):
            r = urllib.request.Request(f"http://169.254.169.254/latest/meta-data/{k}",
                                       headers={"X-aws-ec2-metadata-token": tok})
            out[k.split("/")[-1]] = urllib.request.urlopen(r, timeout=0.4).read().decode()
        try:
            r = urllib.request.Request("http://169.254.169.254/latest/meta-data/product-codes",
                                       headers={"X-aws-ec2-metadata-token": tok})
            out["marketplace_product_codes"] = urllib.request.urlopen(r, timeout=0.4).read().decode().split()
        except Exception:
            out["marketplace_product_codes"] = []
        return out
    except Exception:
        return {}


@functools.lru_cache(maxsize=1)
def _build_highlights() -> dict:
    bi = cv2.getBuildInformation()
    keep = []
    for line in bi.splitlines():
        l = line.lower()
        if any(k in l for k in ("kleidi", "custom hal", "baseline", "dispatched", "parallel framework", "sve",
                                "neon", "version control", "lapack", "dnn")):
            keep.append(line.strip())
    return {"lines": keep[:24], "kleidicv": "kleidi" in bi.lower(), "sve_in_build": "sve" in bi.lower()}


def runtime_info(light: bool = False) -> dict:
    path = os.path.dirname(cv2.__file__)
    cool_env = os.environ.get("ARGUS_COOL_PATH", "")
    is_cool = path.startswith("/opt/cool") or (bool(cool_env) and path.startswith(cool_env))
    info = {
        "opencv_version": cv2.__version__,
        "opencv_major": int(cv2.__version__.split(".")[0]),
        "opencv_path": path,
        "cool": bool(is_cool),
        "opencv_threads": cv2.getNumThreads(),
        "python": platform.python_version(),
        "cpu": _cpu(),
        "ec2": _imds(),
    }
    hl = _build_highlights()
    info["build"] = {"kleidicv": hl["kleidicv"], "sve_in_build": hl["sve_in_build"]}
    if not light:
        info["build"]["highlights"] = hl["lines"]
    ec2 = info["ec2"]
    arch = info["cpu"].get("machine", "")
    label = f"OpenCV {cv2.__version__}"
    label += " · COOL" if is_cool else " · stock build"
    label += f" · {arch}"
    if info["cpu"].get("arm_core"):
        label += f" · {info['cpu']['arm_core']}"
    if ec2.get("instance-type"):
        label += f" · {ec2['instance-type']}"
    info["label"] = label
    return info


if __name__ == "__main__":
    print(json.dumps(runtime_info(), indent=2))
