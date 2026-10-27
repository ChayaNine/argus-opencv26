"""Bedrock LLM inspector: an agent that plans captures from OpenCV evidence.

Uses the Amazon Bedrock Converse API with tool use.  Every tool call goes through the
same safety guard as the deterministic policy.  On any Bedrock failure (throttling,
access, timeout, malformed output) the gauge is handed to the policy agent and the
event is recorded in the trace, so a round never stalls.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

from . import policy
from .session import MAX_CAPTURES_PER_GAUGE, Session
from .tools import TOOL_SPECS, call_tool

DEFAULT_MODEL = "global.anthropic.claude-haiku-4-5-20251001-v1:0"

# USD per 1M tokens (input, output).  Estimates for cost reporting only; verify on the Bedrock pricing page.
PRICES = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-6": (5.0, 25.0),
    "nova-lite": (0.06, 0.24),
    "nova-pro": (0.8, 3.2),
    "nova-micro": (0.035, 0.14),
}

SYSTEM_PROMPT = """You are ARGUS, the perception-and-decision agent of an autonomous inspection robot doing an operator round in an industrial plant. For the gauge you are given, obtain a trustworthy reading with the tools, decide what it means, and close it out safely. Every action costs robot time: be efficient, but never trade safety for speed.

How the tools work
- capture: takes 1-9 frames from a walkway position (front/left/right) and returns an OpenCV quality assessment with issue codes and metrics.
- read_gauge: OpenCV 5 measures the needle on a capture and returns value, sigma (1 sigma), confidence 0-1, issues, and a limit_check that already accounts for uncertainty.
- record_reading only accepts confidence >= 0.6 and a limit_check status other than "uncertain".

Camera physics you can exploit
- Specular glare (glare_on_needle / glare_on_scale) is a mirror reflection: moving left/right moves it across the dial. HDR (hdr=true) recovers detail only when the highlight is not fully saturated. The robot's own spotlight can create glare too.
- Vibration (needle_flutter, motion_blur): use fast_shutter=true with burst 7-9; the reader then finds the centre of the oscillation. Flutter itself is a maintenance signal (pulsation).
- Low light (noisy, underexposed): light=true if it is off, more burst frames (noise averages as sqrt(N)), or ev_bias +1.
- Defocus (isotropic blur, tag not found) is usually a transient autofocus miss: capture again.
- oblique_view / very_oblique_view lose resolution: use the side view that faces the dial more squarely. Objects hanging in front of a dial (occluded, needle_not_clear) shift by parallax when you change view.
- low_contrast (haze, dirty glass): read_gauge with enhance=["clahe"] before taking new pictures.
- limit_check "uncertain" means the 2-sigma interval straddles a limit: tighten sigma with more frames or a better view, else escalate.
- Start with burst=3 so flutter can be detected; at night start with light=true.

Rules (a safety guard enforces them; violations are rejected with an explanation)
1. Never state, guess or record a value that read_gauge did not return. Record by reading_id.
2. If the recorded status is alarm or instrument_fault: create_work_order with a specific title and a description citing the value +/- 2 sigma, the limits, the trend, and the evidence; then finish.
3. If you cannot get an acceptable reading within the capture budget (6 captures), request_human_review with a specific reason (what you saw and tried), then finish.
4. You cannot approve or dispatch work orders; humans do.

Style: before each tool call write ONE short sentence (max 25 words) explaining the decision with reference to what the last capture showed. Operators read these. End with finish_gauge(summary) where summary is one line: the outcome plus any lesson useful for the next gauges (e.g. "front view glare from overhead light; left view clean")."""

REPORT_PROMPT = """Write the shift hand-over note for the control room from these round facts. Use only the facts given (no invented numbers). Max 120 words. Start with one summary sentence, then bullets: ACTION (alarms/faults with work-order ids), REVIEW (escalations), WATCH (warnings, notable trends, flutter). Plain text."""


def _price(model_id: str) -> tuple[float, float]:
    env_in, env_out = os.environ.get("ARGUS_LLM_PRICE_IN"), os.environ.get("ARGUS_LLM_PRICE_OUT")
    if env_in and env_out:
        return float(env_in), float(env_out)
    for k, v in PRICES.items():
        if k in model_id:
            return v
    return (3.0, 15.0)


class BedrockError(Exception):
    pass


class BedrockInspector:
    def __init__(self, client: Any, model_id: str = DEFAULT_MODEL, max_turns: int = 16, prompt_cache: bool = True,
                 temperature: float = 0.0):
        self.client = client
        self.model_id = model_id
        self.max_turns = max_turns
        self.prompt_cache = prompt_cache and "anthropic" in model_id
        self.temperature = temperature
        self.consecutive_failures = 0
        self.circuit_open = False

    @classmethod
    def from_env(cls) -> "BedrockInspector":
        import boto3
        from botocore.config import Config
        region = os.environ.get("ARGUS_BEDROCK_REGION") or os.environ.get("AWS_REGION") or "us-east-1"
        client = boto3.client("bedrock-runtime", region_name=region,
                              config=Config(retries={"max_attempts": 4, "mode": "adaptive"}, read_timeout=45,
                                            connect_timeout=5))
        return cls(client, os.environ.get("ARGUS_BEDROCK_MODEL", DEFAULT_MODEL),
                   prompt_cache=os.environ.get("ARGUS_PROMPT_CACHE", "1") == "1")

    # ---- Bedrock plumbing
    def _tool_config(self) -> dict:
        tools = [{"toolSpec": {"name": t["name"], "description": t["description"],
                               "inputSchema": {"json": t["input_schema"]}}} for t in TOOL_SPECS]
        if self.prompt_cache:
            tools.append({"cachePoint": {"type": "default"}})
        return {"tools": tools, "toolChoice": {"auto": {}}}

    def _system(self, text: str) -> list:
        out = [{"text": text}]
        if self.prompt_cache:
            out.append({"cachePoint": {"type": "default"}})
        return out

    def converse(self, s: Session, messages: list, system: str, tools: bool = True, max_tokens: int = 700) -> dict:
        kw = dict(modelId=self.model_id, system=self._system(system), messages=messages,
                  inferenceConfig={"maxTokens": max_tokens, "temperature": self.temperature})
        if tools:
            kw["toolConfig"] = self._tool_config()
        t0 = time.perf_counter()
        try:
            resp = self.client.converse(**kw)
        except Exception as e:  # botocore ClientError, EndpointConnectionError, ReadTimeoutError...
            msg = str(e)
            if self.prompt_cache and ("cache" in msg.lower() or "cachePoint" in msg):
                self.prompt_cache = False  # model without prompt caching: retry once without it
                return self.converse(s, messages, system, tools, max_tokens)
            raise BedrockError(f"{type(e).__name__}: {msg[:300]}") from e
        dt = (time.perf_counter() - t0) * 1000
        u = resp.get("usage", {})
        pin, pout = _price(self.model_id)
        cached_read = u.get("cacheReadInputTokens", 0) or 0
        cached_write = u.get("cacheWriteInputTokens", 0) or 0
        tin = u.get("inputTokens", 0) or 0
        tout = u.get("outputTokens", 0) or 0
        cost = (tin * pin + cached_write * pin * 1.25 + cached_read * pin * 0.1 + tout * pout) / 1e6
        with s.lock:
            s.usage["llm_calls"] += 1
            s.usage["input_tokens"] += tin + cached_read + cached_write
            s.usage["output_tokens"] += tout
            s.usage["cost_usd"] += cost
            s.usage["llm_latency_ms"] += dt
            s.usage.setdefault("cache_read_tokens", 0)
            s.usage["cache_read_tokens"] += cached_read
        s.emit("llm", "llm_call", model=self.model_id, input_tokens=tin, output_tokens=tout,
               cache_read=cached_read, latency_ms=round(dt, 1), stop=resp.get("stopReason"))
        return resp

    # ---- the agent loop for one gauge
    def inspect_gauge(self, s: Session, gid: str) -> str:
        if self.circuit_open:
            s.emit("system", "fallback", reason="Bedrock circuit open after repeated failures; using policy agent")
            s.usage["fallbacks"] += 1
            policy.inspect_gauge(s, gid, actor="policy")
            return "policy-fallback"
        ctx = s.gauge_context(gid)
        lessons = [oc.summary for oc in s.outcomes.values() if oc.finished and oc.summary][-3:]
        intro = (f"Gauge to inspect now:\n{json.dumps(ctx)}\n"
                 + (f"Notes from earlier gauges this round:\n- " + "\n- ".join(lessons) + "\n" if lessons else "")
                 + "Start.")
        messages: list = [{"role": "user", "content": [{"text": intro}]}]
        nudges = 0
        try:
            for _turn in range(self.max_turns):
                resp = self.converse(s, messages, SYSTEM_PROMPT)
                msg = resp["output"]["message"]
                messages.append(msg)
                texts = [b["text"] for b in msg.get("content", []) if "text" in b and b["text"].strip()]
                uses = [b["toolUse"] for b in msg.get("content", []) if "toolUse" in b]
                rationale = " ".join(texts).strip()
                if not uses:
                    if rationale:
                        s.emit("llm", "thought", text=rationale[:600])
                    if s.outcomes[gid].finished:
                        break
                    nudges += 1
                    if nudges > 2:
                        raise BedrockError("model stopped without finishing the gauge")
                    messages.append({"role": "user", "content": [{"text": (
                        "The gauge is not finished. Continue using the tools and end with finish_gauge.")}]})
                    continue
                results = []
                for i, tu in enumerate(uses):
                    r = call_tool(s, tu["name"], tu.get("input") or {}, "llm", rationale if i == 0 else "")
                    payload = r["result"] if r["ok"] else {"error": r["error"]}
                    results.append({"toolResult": {"toolUseId": tu["toolUseId"], "content": [{"json": payload}],
                                                   "status": "success" if r["ok"] else "error"}})
                messages.append({"role": "user", "content": results})
                if s.outcomes[gid].finished:
                    break
            self.consecutive_failures = 0
        except BedrockError as e:
            self.consecutive_failures += 1
            if self.consecutive_failures >= 2:
                self.circuit_open = True
            s.usage["fallbacks"] += 1
            s.emit("system", "fallback", reason=f"LLM agent failed: {e}; handing {gid} to the policy agent")
            if not s.outcomes[gid].finished:
                policy.inspect_gauge(s, gid, actor="policy")
            return "policy-fallback"
        if not s.outcomes[gid].finished:
            s.emit("system", "fallback", reason="LLM turn budget exhausted; policy agent completes the gauge")
            s.usage["fallbacks"] += 1
            policy.inspect_gauge(s, gid, actor="policy")
            return "policy-fallback"
        return "llm"

    def write_report(self, s: Session, facts: dict, base: str) -> Optional[str]:
        if self.circuit_open:
            return None
        try:
            resp = self.converse(s, [{"role": "user", "content": [{"text": REPORT_PROMPT + "\n\nFACTS:\n"
                                                                    + json.dumps(facts)}]}],
                                 "You write concise, factual industrial shift hand-over notes.", tools=False,
                                 max_tokens=400)
            txt = " ".join(b.get("text", "") for b in resp["output"]["message"]["content"]).strip()
            return txt or None
        except BedrockError:
            return None


def check_access(model_id: Optional[str] = None) -> dict:
    """Minimal Converse call to verify model access (used by scripts/check_bedrock.py and /api/health)."""
    insp = BedrockInspector.from_env()
    if model_id:
        insp.model_id = model_id
    t0 = time.perf_counter()
    try:
        r = insp.client.converse(modelId=insp.model_id,
                                 messages=[{"role": "user", "content": [{"text": "Reply with the word ready."}]}],
                                 inferenceConfig={"maxTokens": 10, "temperature": 0})
        txt = " ".join(b.get("text", "") for b in r["output"]["message"]["content"])
        return {"ok": True, "model": insp.model_id, "reply": txt.strip()[:40],
                "latency_ms": round((time.perf_counter() - t0) * 1000)}
    except Exception as e:
        return {"ok": False, "model": insp.model_id, "error": f"{type(e).__name__}: {str(e)[:300]}"}
