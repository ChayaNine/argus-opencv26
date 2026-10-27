"""Merge benchmark JSONs into a comparison table, speed-ups and cost per 1,000 reads.

    python -m bench.compare results/bench/bench_*.json

Cost per 1,000 reads = (EC2 on-demand $/h + COOL software $/h) / (reads per second on all cores * 3600) * 1000.
Prices are read from bench/prices.json (on-demand, us-east-1 by default); edit them for your region.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRICES = json.loads((Path(__file__).resolve().parent / "prices.json").read_text())


def main(paths):
    runs = [json.loads(Path(p).read_text()) for p in paths]
    configs = []
    for r in runs:
        rt = r["runtime"]
        it = rt.get("ec2", {}).get("instance-type") or r["label"]
        cool = bool(rt.get("cool"))
        ec2 = PRICES["ec2_on_demand_usd_per_hour"].get(it)
        fee = PRICES["cool_software_usd_per_hour"].get(it, 0.0) if cool else 0.0
        rps = r["throughput_all"]["reads_per_s"]
        cost = None if ec2 is None or not rps else (ec2 + fee) / (rps * 3600) * 1000
        configs.append({
            "label": r["label"], "instance_type": it, "cool": cool, "opencv_version": rt["opencv_version"],
            "opencv_path": rt["opencv_path"], "cpu": rt.get("cpu", {}).get("arm_core") or rt.get("cpu", {}).get(
                "model_name", rt.get("cpu", {}).get("machine")), "kleidicv": rt.get("build", {}).get("kleidicv"),
            "pipeline": {"median_ms": r["pipeline"]["median_ms"], "p95_ms": r["pipeline"]["p95_ms"],
                         "burst9_median_ms": r["burst9"]["median_ms"],
                         "throughput_rps_1": r["throughput_1proc"]["reads_per_s"], "throughput_rps_all": rps},
            "price_usd_per_hour": None if ec2 is None else round(ec2 + fee, 4), "cost_per_1k_reads_usd": cost,
            "ops": {k: v["median_ms"] for k, v in r["ops"].items()}, "readings": r.get("readings"),
            "time": r.get("time"),
        })
    # speed-ups vs the stock build on the same instance type
    for c in configs:
        base = next((b for b in configs if b["instance_type"] == c["instance_type"] and not b["cool"]), None)
        if c["cool"] and base:
            c["speedup_vs_stock_same_instance"] = round(base["pipeline"]["median_ms"] / c["pipeline"]["median_ms"], 3)
            c["throughput_gain_vs_stock_same_instance"] = round(
                c["pipeline"]["throughput_rps_all"] / base["pipeline"]["throughput_rps_all"], 3)
            c["op_speedups"] = {k: round(base["ops"][k] / v, 3) for k, v in c["ops"].items() if k in base["ops"] and v}
            same = [a == b for a, b in zip(c["readings"] or [], base["readings"] or [])]
            c["identical_readings"] = f"{sum(same)}/{len(same)}"
    configs.sort(key=lambda c: (not c["cool"], c["instance_type"]))
    note = ("Same 24 rendered frames on every machine (bench/inputs). Latency: single process, library-default "
            "threads. Throughput: one process per vCPU with cv2.setNumThreads(1). Prices: " + PRICES["source"])
    out = {"configs": configs, "note": note, "prices": PRICES}
    od = ROOT / "results" / "bench"
    od.mkdir(parents=True, exist_ok=True)
    (od / "compare.json").write_text(json.dumps(out, indent=1))
    md = ["| Configuration | OpenCV | Pipeline median ms | p95 ms | reads/s (all cores) | $/1k reads | speed-up vs stock |",
          "|---|---|---|---|---|---|---|"]
    for c in configs:
        cost = "" if c["cost_per_1k_reads_usd"] is None else "$%.5f" % c["cost_per_1k_reads_usd"]
        md.append(f"| {c['label']} ({c['instance_type']}) | {c['opencv_version']}{' COOL' if c['cool'] else ''} | "
                  f"{c['pipeline']['median_ms']:.2f} | {c['pipeline']['p95_ms']:.2f} | "
                  f"{c['pipeline']['throughput_rps_all']:.1f} | {cost} | "
                  f"{c.get('speedup_vs_stock_same_instance', '')} |")
    (od / "compare.md").write_text("\n".join(md) + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cool = next((c for c in configs if c["cool"] and c.get("op_speedups")), None)
        if cool:
            items = sorted(cool["op_speedups"].items(), key=lambda kv: kv[1])
            fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(items) + 1.2), dpi=150)
            ax.barh([k for k, _ in items], [v for _, v in items], color=["#2f7d6d" if v >= 1 else "#c26b5a"
                                                                          for _, v in items])
            ax.axvline(1.0, color="#555", lw=0.8)
            ax.set_xlabel("speed-up of COOL over stock OpenCV 5 on the same instance (x)")
            ax.tick_params(axis="y", labelsize=7.5)
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            fig.savefig(od / "bench.png")
    except Exception as e:  # pragma: no cover
        print("chart skipped:", e)
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1:] or sorted(str(p) for p in (ROOT / "results" / "bench").glob("bench_*.json")))
