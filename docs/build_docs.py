"""Fill README.md and docs/REPORT.md from the measured results.

    python docs/build_docs.py

Uses results/eval/aws/summary.json (includes the Bedrock agent) if present, else results/eval/final/summary.json,
plus results/eval/final/precision.json and results/bench/compare.json when available.  Re-run after the AWS
benchmark and evaluation; every number in the docs then comes from a results file.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABEL = {"overview": "Wide shot", "fixed": "Fixed PTZ (naive)", "fixed_gated": "Fixed PTZ + gating",
         "agent_policy": "**ARGUS policy agent**", "agent_llm": "**ARGUS LLM agent (Bedrock)**",
         "agent_policy_same_seeds": "ARGUS policy (LLM seeds)"}
ROWS = [("silent_failure_rate", "Silent failures (wrong value recorded)", "pct"),
        ("missed_critical", "Missed alarms / instrument faults", "crit"),
        ("false_alarms", "False work orders", "int"),
        ("automation_rate", "Read automatically", "pct"),
        ("within_1pct", "Recorded within ±1% of span", "pct"),
        ("median_err_pct_span", "Median error (% of span)", "f3"),
        ("sigma_coverage_2sigma", "Truth inside reported ±2σ", "pct"),
        ("status_accuracy_recorded", "Status correct (recorded)", "pct"),
        ("captures_per_gauge", "Captures per gauge", "f2"),
        ("robot_s_per_gauge", "Robot seconds per gauge", "f1")]


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def fmt(v, kind, crit=None):
    if v is None:
        return "n/a"
    if kind == "pct":
        return f"{v * 100:.1f}%"
    if kind == "crit":
        return f"{v} of {crit}"
    if kind == "int":
        return str(v)
    if kind == "f3":
        return f"{v:.3f}"
    if kind == "f2":
        return f"{v:.2f}"
    return f"{v:.1f}"


def main_table(S):
    keys = [k for k in ["overview", "fixed", "fixed_gated", "agent_policy", "agent_llm", "agent_policy_same_seeds"]
            if k in S]
    lines = ["| | " + " | ".join(LABEL[k] for k in keys) + " |", "|---|" + "---|" * len(keys)]
    for key, label, kind in ROWS:
        cells = [fmt(S[k].get(key), kind, S[k].get("critical_gauges")) for k in keys]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    if "agent_llm" in S:
        lines.append("| LLM cost per gauge | " + " | ".join(
            (f"${S[k]['llm_cost_usd_per_gauge']:.4f}" if k == "agent_llm" else "") for k in keys) + " |")
    return "\n".join(lines)


def cond_table(C, S):
    keys = [k for k in ["fixed", "fixed_gated", "agent_policy", "agent_llm"] if k in S]
    lines = ["| Condition (gauges) | " + " | ".join(LABEL[k].replace("**", "") for k in keys) + " |",
             "|---|" + "---|" * len(keys)]
    for c, row in C.items():
        if not row:
            continue
        n = next(iter(row.values()))["n"]
        cells = []
        for k in keys:
            r = row.get(k)
            cells.append("n/a" if not r else f"{r['automated'] * 100:.0f}% auto, {r['silent']} silent, "
                                              f"{r['missed_critical']} missed")
        lines.append(f"| {c} ({n}) | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def bench_md(B):
    lines = ["| Configuration | OpenCV | Pipeline median ms | p95 ms | Burst-9 ms | Reads/s (all vCPUs) | $ per 1k reads "
             "| vs stock, same instance |", "|---|---|---|---|---|---|---|---|"]
    for c in B["configs"]:
        cost = "" if c.get("cost_per_1k_reads_usd") is None else f"${c['cost_per_1k_reads_usd']:.5f}"
        sp = c.get("speedup_vs_stock_same_instance")
        lines.append(f"| {c['label']} ({c['instance_type']}) | {c['opencv_version']}{' COOL' if c['cool'] else ''} | "
                     f"{c['pipeline']['median_ms']:.2f} | {c['pipeline']['p95_ms']:.2f} | "
                     f"{c['pipeline']['burst9_median_ms']:.1f} | {c['pipeline']['throughput_rps_all']:.1f} | {cost} | "
                     f"{'' if sp is None else f'{sp:.2f}x latency, ' + format(c['throughput_gain_vs_stock_same_instance'], '.2f') + 'x throughput'} |")
    cool = next((c for c in B["configs"] if c.get("op_speedups")), None)
    if cool:
        top = sorted(cool["op_speedups"].items(), key=lambda kv: -kv[1])[:5]
        lines.append("")
        lines.append("Largest per-operation speed-ups of COOL over stock OpenCV 5 on the same instance: " +
                     ", ".join(f"{k} {v:.2f}x" for k, v in top) + f". Identical readings: "
                     f"{cool.get('identical_readings', 'n/a')}.")
    lines.append("")
    lines.append(B.get("note", ""))
    return "\n".join(lines)


def main():
    aws = load(ROOT / "results/eval/aws/summary.json")
    final = load(ROOT / "results/eval/final/summary.json")
    ev = aws or final
    S, C, M = ev["summary"], ev["by_condition"], ev["meta"]
    P = load(ROOT / "results/eval/final/precision.json") or {}
    B = load(ROOT / "results/bench/compare.json")
    A, N, G = S["agent_policy"], S["fixed"], S["fixed_gated"]
    vals = {
        "main_table": main_table(S), "cond_table": cond_table(C, S),
        "n_gauges": A["gauges"], "scenes": M["scenes"], "seed0": M["seed0"], "seed1": M["seed0"] + M["scenes"] - 1,
        "llm_scenes": M.get("llm_scenes") or "N",
        "clean_med": P.get("median_pct_span", "n/a"), "clean_p90": P.get("p90_pct_span", "n/a"),
        "clean_n": P.get("n", "n/a"), "clean_deg": P.get("median_deg", "n/a"),
        "naive_silent": f"{N['silent_failure_rate'] * 100:.0f}", "naive_missed": N["missed_critical"],
        "critical": N["critical_gauges"], "naive_false": N["false_alarms"],
        "gated_esc": f"{G['escalation_rate'] * 100:.0f}",
        "agent_auto": f"{A['automation_rate'] * 100:.0f}", "agent_silent_n": A["silent_failures"],
        "agent_missed": A["missed_critical"], "agent_false": A["false_alarms"],
        "agent_within1": f"{A['within_1pct'] * 100:.1f}", "agent_cov": f"{A['sigma_coverage_2sigma'] * 100:.0f}",
        "agent_caps": f"{A['captures_per_gauge']:.1f}", "agent_robot": f"{A['robot_s_per_gauge']:.0f}",
    }
    for src, dst in [("docs/REPORT.template.md", "docs/REPORT.md"), ("docs/README.template.md", "README.md")]:
        txt = (ROOT / src).read_text()
        for k, v in vals.items():
            txt = txt.replace("{{" + k + "}}", str(v))
        if B:
            start = txt.find("**Results.** TODO-AWS")
            if start >= 0:
                end = txt.find("\n\n", start)
                txt = txt[:start] + "**Results** (measured with `deploy/run_bench.sh` and `deploy/bench_x86.sh`):\n\n" + \
                    bench_md(B) + txt[end:]
            txt = txt.replace("produced by `deploy/run_bench.sh` (TODO-AWS).", "produced by `deploy/run_bench.sh`.")
        if aws and "agent_llm" in S:
            start = txt.find("TODO-AWS: paste the `agent_llm`")
            if start >= 0:
                end = txt.find("\n\n", start)
                L, PS = S["agent_llm"], S.get("agent_policy_same_seeds", A)
                txt = txt[:start] + (
                    f"On the same {M.get('llm_scenes')} rounds, the Bedrock agent "
                    f"read {L['automation_rate'] * 100:.0f}% automatically (policy {PS['automation_rate'] * 100:.0f}%) "
                    f"with {L['silent_failures']} silent failures (policy {PS['silent_failures']}), "
                    f"{L['missed_critical']} missed alarms/faults (policy {PS['missed_critical']}), "
                    f"{L['captures_per_gauge']:.2f} captures per gauge (policy {PS['captures_per_gauge']:.2f}), "
                    f"{L.get('policy_fallbacks', 0)} fallbacks to the policy, and about "
                    f"${L.get('llm_cost_usd_per_gauge', 0):.4f} of Bedrock per gauge. Beyond these numbers, the LLM "
                    "contributes the operator-facing explanations before every action, specific work-order text, "
                    "lessons carried between gauges, and the shift report.") + txt[end:]
        (ROOT / dst).write_text(txt)
        left = txt.count("TODO-AWS")
        print(f"wrote {dst} ({left} TODO-AWS left)")


if __name__ == "__main__":
    main()
