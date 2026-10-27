"""Ground-truth evaluation: naive automation vs uncertainty gating vs active-perception agents.

    python -m eval.run_eval --scenes 40 --strategies overview,fixed,fixed_gated,agent_policy
    python -m eval.run_eval --scenes 10 --strategies agent_llm          # needs Bedrock access

Strategies
  overview      one wide shot of the whole panel, every gauge read from it, every value accepted
  fixed         one frontal auto-exposed frame per gauge, value accepted (typical scripted robot round)
  fixed_gated   same frame, but escalate when the reader's gated confidence < 0.6 (our uncertainty alone)
  agent_policy  ARGUS active perception, deterministic policy + safety guard
  agent_llm     ARGUS active perception, Bedrock LLM inspector + safety guard (policy fallback)

Key safety metrics
  silent_failure   a value was recorded automatically and it is wrong by > 2 % of span
                   (or a faulted gauge was recorded as healthy) -- the error nobody notices
  missed_critical  the gauge was truly in alarm / instrument fault and the system neither raised a
                   work order nor escalated it
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from argus.agent.session import AUTO_ACCEPT_CONFIDENCE, Session  # noqa: E402
from argus.agent.supervisor import run_round  # noqa: E402
from argus.agent.tools import _limit_check  # noqa: E402
from argus.runtime import runtime_info  # noqa: E402
from argus.vision import sim  # noqa: E402
from argus.vision.pipeline import analyze_capture, encode_jpeg, overlay  # noqa: E402

CRITICAL = ("alarm", "instrument_fault")


def _row(seed, sg, **kw):
    spec = sg.spec
    d = dict(seed=seed, gauge_id=spec.id, name=spec.name, conditions="+".join(sg.conditions),
             true_value=round(sg.true_value, 5), true_status=sg.true_status(), fault=sg.fault or "",
             span=spec.span, criticality=spec.criticality)
    d.update(kw)
    return d


def _score(d):
    """Derive error/safety columns for one gauge outcome."""
    span = d["span"]
    rec = d.get("recorded", False)
    err = None
    if rec and d.get("value") is not None and not d["fault"]:
        err = (d["value"] - d["true_value"]) / span * 100.0
    d["err_pct_span"] = None if err is None else round(err, 4)
    pred = d.get("status_pred", "")
    if rec:
        if d["fault"]:
            silent = pred not in CRITICAL and not d.get("work_order")
        else:
            silent = abs(err) > 2.0
    else:
        silent = False
    d["silent_failure"] = bool(silent)
    flagged = bool(d.get("work_order")) or bool(d.get("escalated")) or pred in CRITICAL
    d["missed_critical"] = bool(d["true_status"] in CRITICAL and not flagged)
    d["false_alarm"] = bool(d["true_status"] == "normal" and d.get("work_order"))
    d["status_correct"] = bool(rec and (pred == d["true_status"] or (d["fault"] and pred in CRITICAL)))
    if rec and err is not None and d.get("sigma"):
        d["within_2sigma"] = bool(abs(d["value"] - d["true_value"]) <= 2 * d["sigma"])
    return d


def strat_overview(seed, profile):
    sc = sim.make_scene(seed=seed, profile=profile)
    cam = sim.RobotCamera(sc, seed=seed + 7)
    t0 = time.perf_counter()
    cap = cam.overview()
    rows = []
    for sg in sc.gauges:
        an = analyze_capture(cap["frames"], sg.spec, None)
        rd = an.reading
        ok = bool(an.ok and rd.get("ok"))
        val = rd.get("value") if ok else None
        st = _limit_check(sg.spec, val, rd["sigma"], rd.get("flags", []))["status"] if ok else ""
        if st == "uncertain":
            st = sg.spec.status_of(val)
        rows.append(_row(seed, sg, strategy="overview", recorded=ok, value=val, sigma=rd.get("sigma"),
                         confidence=rd.get("confidence"), status_pred=st, escalated=not ok,
                         work_order=ok and st in CRITICAL, captures=1 / len(sc.gauges),
                         robot_time_s=cam.time_s / len(sc.gauges),
                         compute_ms=an.timings_ms.get("total", 0)))
    return rows, []


def strat_fixed(seed, profile, gated=False):
    sc = sim.make_scene(seed=seed, profile=profile)
    cam = sim.RobotCamera(sc, seed=seed + 7)
    rows, gallery = [], []
    for sg in sc.gauges:
        t_r = cam.time_s
        cap = cam.capture(sg.spec.id, "front")
        an = analyze_capture(cap["frames"], sg.spec, cap["predicted_H"])
        rd = an.reading
        ok = bool(an.ok and rd.get("ok"))
        conf = rd.get("confidence", 0.0) if ok else 0.0
        accept = ok and (not gated or conf >= AUTO_ACCEPT_CONFIDENCE)
        val = rd.get("value") if ok else None
        st = ""
        if accept:
            st = _limit_check(sg.spec, val, rd["sigma"], rd.get("flags", []))["status"]
            if st == "uncertain":
                st = "uncertain" if gated else sg.spec.status_of(val)
            if st == "uncertain":
                accept = False
        rows.append(_row(seed, sg, strategy="fixed_gated" if gated else "fixed", recorded=accept,
                         value=val if accept else None, sigma=rd.get("sigma") if accept else None, confidence=conf,
                         status_pred=st if accept else "", escalated=not accept,
                         work_order=accept and st in CRITICAL, captures=1, robot_time_s=cam.time_s - t_r,
                         compute_ms=an.timings_ms.get("total", 0)))
    return rows, gallery


def strat_agent(seed, profile, mode="policy", llm=None, gallery_dir=None):
    s = Session(profile=profile, seed=seed, agent_mode=mode)
    run_round(s, llm_agent=llm)
    rows, gallery = [], []
    for gid, oc in s.outcomes.items():
        sg = s.scene.gauge(gid)
        rd = s.readings.get(oc.reading_id) if oc.reading_id else None
        rec = rd is not None
        d = _row(seed, sg, strategy=f"agent_{mode}", recorded=rec, value=rd.value if rec else None,
                 sigma=rd.sigma if rec else None, confidence=rd.confidence if rec else None,
                 status_pred=oc.status if rec else "", escalated=oc.review_id is not None,
                 work_order=oc.work_order_id is not None, captures=oc.captures, robot_time_s=oc.robot_time_s,
                 compute_ms=round(oc.compute_ms, 1), agent_used=oc.agent, forced=oc.forced,
                 tool_calls=oc.tool_calls)
        rows.append(d)
        if gallery_dir is not None:
            ov = rd.overlay_id if rd else None
            if ov is None:
                for r in reversed(list(s.readings.values())):
                    if r.gauge_id == gid:
                        ov = r.overlay_id
                        break
            if ov:
                gallery.append((f"s{seed}_{gid}_{'rec' if rec else 'esc'}", s.store.get(ov), d))
    usage = dict(s.usage)
    for d in rows:
        d["llm_cost_usd_round"] = usage.get("cost_usd", 0.0)
        d["llm_tokens_round"] = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
    return rows, gallery


def _worker(args):
    strategy, seed, profile, gallery = args
    import cv2
    cv2.setNumThreads(1)
    if strategy == "overview":
        r, g = strat_overview(seed, profile)
    elif strategy == "fixed":
        r, g = strat_fixed(seed, profile, gated=False)
    elif strategy == "fixed_gated":
        r, g = strat_fixed(seed, profile, gated=True)
    elif strategy == "agent_policy":
        r, g = strat_agent(seed, profile, "policy", gallery_dir=True if gallery else None)
    elif strategy == "agent_llm":
        from argus.agent.llm import BedrockInspector
        r, g = strat_agent(seed, profile, "llm", llm=BedrockInspector.from_env(), gallery_dir=True if gallery else None)
    else:
        raise ValueError(strategy)
    return [_score(d) for d in r], g


def summarise(rows):
    by = defaultdict(list)
    for d in rows:
        by[d["strategy"]].append(d)
    out = {}
    for k, rs in by.items():
        n = len(rs)
        rec = [d for d in rs if d["recorded"]]
        errs = [abs(d["err_pct_span"]) for d in rec if d["err_pct_span"] is not None]
        crit = [d for d in rs if d["true_status"] in CRITICAL]
        cov = [d["within_2sigma"] for d in rec if "within_2sigma" in d]
        out[k] = {
            "gauges": n,
            "automation_rate": round(len(rec) / n, 4),
            "escalation_rate": round(sum(1 for d in rs if d["escalated"] and not d["recorded"]) / n, 4),
            "silent_failures": sum(d["silent_failure"] for d in rs),
            "silent_failure_rate": round(sum(d["silent_failure"] for d in rs) / n, 4),
            "critical_gauges": len(crit),
            "missed_critical": sum(d["missed_critical"] for d in rs),
            "false_alarms": sum(d["false_alarm"] for d in rs),
            "mae_pct_span": round(float(np.mean(errs)), 4) if errs else None,
            "median_err_pct_span": round(float(np.median(errs)), 4) if errs else None,
            "p95_err_pct_span": round(float(np.percentile(errs, 95)), 4) if errs else None,
            "within_1pct": round(float(np.mean([e <= 1.0 for e in errs])), 4) if errs else None,
            "status_accuracy_recorded": round(float(np.mean([d["status_correct"] for d in rec])), 4) if rec else None,
            "sigma_coverage_2sigma": round(float(np.mean(cov)), 4) if cov else None,
            "captures_per_gauge": round(float(np.mean([d["captures"] for d in rs])), 3),
            "robot_s_per_gauge": round(float(np.mean([d["robot_time_s"] for d in rs])), 2),
            "compute_ms_per_gauge": round(float(np.mean([d["compute_ms"] for d in rs])), 1),
        }
        if k == "agent_llm":
            seeds = {}
            for d in rs:
                seeds[d["seed"]] = d.get("llm_cost_usd_round", 0)
            out[k]["llm_cost_usd_per_gauge"] = round(sum(seeds.values()) / n, 5)
            out[k]["policy_fallbacks"] = sum(1 for d in rs if d.get("agent_used") == "policy-fallback")
    return out


def by_condition(rows, strategies):
    conds = ["clean", "glare", "night", "vibration", "oblique", "occlusion", "haze", "fault"]
    table = {}
    for c in conds:
        table[c] = {}
        for st in strategies:
            rs = [d for d in rows if d["strategy"] == st and c in d["conditions"].split("+")]
            if not rs:
                continue
            table[c][st] = {
                "n": len(rs),
                "automated": round(sum(d["recorded"] for d in rs) / len(rs), 3),
                "silent": sum(d["silent_failure"] for d in rs),
                "missed_critical": sum(d["missed_critical"] for d in rs),
            }
    return table


def charts(summary, cond, outdir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    files = []
    strategies = list(summary.keys())
    labels = {"overview": "Wide shot", "fixed": "Fixed PTZ\n(naive)", "fixed_gated": "Fixed PTZ\n+ gating",
              "agent_policy": "ARGUS\npolicy agent", "agent_llm": "ARGUS\nLLM agent",
              "agent_policy_same_seeds": "ARGUS policy\n(LLM seeds)"}
    colors = {"overview": "#9aa3ab", "fixed": "#c26b5a", "fixed_gated": "#d9a441", "agent_policy": "#2f7d6d",
              "agent_llm": "#3a6fb0", "agent_policy_same_seeds": "#6fa89b"}
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), dpi=150)
    for ax, (key, title, pct) in zip(axes, [("silent_failure_rate", "Silent failures (lower is better)", True),
                                            ("automation_rate", "Gauges read automatically", True),
                                            ("missed_critical", "Missed alarms / faults", False)]):
        vals = [summary[s][key] * (100 if pct else 1) for s in strategies]
        ax.bar([labels.get(s, s) for s in strategies], vals, color=[colors.get(s, "#666") for s in strategies])
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.1f}%" if pct else f"{int(v)}", ha="center", va="bottom", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=7.5)
    fig.tight_layout()
    p = outdir / "strategies.png"
    fig.savefig(p)
    plt.close(fig)
    files.append(p.name)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--profile", default="random")
    ap.add_argument("--strategies", default="overview,fixed,fixed_gated,agent_policy")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--out", default="results/eval")
    ap.add_argument("--gallery", action="store_true")
    ap.add_argument("--llm-scenes", type=int, default=None,
                    help="run agent_llm on only the first N scenes (Bedrock cost); the policy agent is also "
                         "summarised on the same seeds for a like-for-like comparison")
    a = ap.parse_args()
    strategies = a.strategies.split(",")
    outdir = Path(a.out) / time.strftime("%Y%m%d-%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)
    n_llm = a.scenes if a.llm_scenes is None else min(a.scenes, a.llm_scenes)
    jobs = [(st, a.seed0 + i, a.profile, a.gallery) for st in strategies if st != "agent_llm" for i in range(a.scenes)]
    llm_jobs = [("agent_llm", a.seed0 + i, a.profile, a.gallery) for i in range(n_llm)] if "agent_llm" in strategies else []
    t0 = time.time()
    rows, gallery = [], []
    if a.workers > 1:
        with mp.get_context("spawn").Pool(a.workers) as pool:
            for r, g in pool.imap_unordered(_worker, jobs):
                rows += r
                gallery += g
                print(f"\r{len(rows)} gauge outcomes", end="", flush=True)
    else:
        for j in jobs:
            r, g = _worker(j)
            rows += r
            gallery += g
            print(f"\r{len(rows)} gauge outcomes", end="", flush=True)
    for j in llm_jobs:  # sequential: Bedrock rate limits and cost
        r, g = _worker(j)
        rows += r
        gallery += g
        print(f"\r{len(rows)} gauge outcomes (LLM)", end="", flush=True)
    print()
    if llm_jobs and n_llm < a.scenes:
        seeds = {j[1] for j in llm_jobs}
        for d in [d for d in rows if d["strategy"] == "agent_policy" and d["seed"] in seeds]:
            rows.append(dict(d, strategy="agent_policy_same_seeds"))
    summ = summarise(rows)
    cond = by_condition(rows, strategies)
    meta = {"scenes": a.scenes, "llm_scenes": n_llm if llm_jobs else 0, "seed0": a.seed0, "profile": a.profile,
            "wall_s": round(time.time() - t0, 1),
            "runtime": runtime_info(light=True), "auto_accept_confidence": AUTO_ACCEPT_CONFIDENCE}
    (outdir / "summary.json").write_text(json.dumps({"meta": meta, "summary": summ, "by_condition": cond}, indent=2))
    keys = sorted({k for d in rows for k in d.keys()})
    with open(outdir / "per_gauge.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for d in rows:
            w.writerow(d)
    files = charts(summ, cond, outdir)
    if gallery:
        gd = outdir / "gallery"
        gd.mkdir(exist_ok=True)
        # keep the interesting ones: escalations and the largest recorded errors
        gallery.sort(key=lambda x: (-(abs(x[2].get("err_pct_span") or 0)), x[0]))
        for name, data, d in gallery[:40]:
            if data:
                (gd / f"{name}.jpg").write_bytes(data)
    print(json.dumps(summ, indent=1))
    print("wrote", outdir)
    latest = Path(a.out) / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(outdir.name)
    except OSError:
        pass


if __name__ == "__main__":
    main()
