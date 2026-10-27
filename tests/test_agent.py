"""Agent loop, safety guard and LLM plumbing."""
import copy

import pytest

from argus.agent import guard
from argus.agent.llm import BedrockInspector
from argus.agent.session import Session
from argus.agent.supervisor import run_round
from argus.agent.tools import call_tool


def test_policy_round_completes_and_is_safe():
    s = Session(profile="hard", seed=3)
    run_round(s)
    assert s.status == "done", s.error
    for gid, oc in s.outcomes.items():
        assert oc.finished
        sg = s.scene.gauge(gid)
        if oc.reading_id:
            rd = s.readings[oc.reading_id]
            assert rd.confidence >= 0.6
            if not sg.fault:
                assert abs(rd.value - sg.true_value) / sg.spec.span < 0.02
        else:
            assert oc.review_id is not None
        if oc.status in ("alarm", "instrument_fault"):
            assert oc.work_order_id is not None
    assert s.report and "Round" in s.report


def _begin(s, i=0):
    gid = list(s.outcomes)[i]
    s.current_gauge = gid
    return gid


def test_guard_blocks_unmeasured_and_low_confidence_records():
    s = Session(profile="glare", seed=21)
    gid = _begin(s)
    r = call_tool(s, "record_reading", {}, "llm")
    assert not r["ok"] and "only values measured" in r["error"]
    r = call_tool(s, "record_reading", {"reading_id": "rd999"}, "llm")
    assert not r["ok"]
    cap = call_tool(s, "capture", {"view": "front"}, "llm")["result"]
    rd = call_tool(s, "read_gauge", {"capture_id": cap["capture_id"]}, "llm")["result"]
    s.readings[rd["reading_id"]].confidence = 0.3
    r = call_tool(s, "record_reading", {"reading_id": rd["reading_id"]}, "llm")
    assert not r["ok"] and "confidence" in r["error"]
    r = call_tool(s, "finish_gauge", {"summary": "done"}, "llm")
    assert not r["ok"] and "nothing recorded" in r["error"]
    assert any(e["kind"] == "guard" for e in s.trace)


def test_guard_requires_work_order_for_alarm_and_raises_priority():
    s = Session(profile="clean", seed=12)
    gid = _begin(s)
    sg = s.scene.gauge(gid)
    sg.true_value = sg.spec.high_alarm + 0.05 * sg.spec.span if sg.spec.high_alarm is not None else sg.true_value
    cap = call_tool(s, "capture", {"view": "front", "burst": 3}, "llm")["result"]
    rd = call_tool(s, "read_gauge", {"capture_id": cap["capture_id"]}, "llm")["result"]
    assert rd["limit_check"]["status"] == "alarm"
    assert call_tool(s, "record_reading", {"reading_id": rd["reading_id"]}, "llm")["ok"]
    r = call_tool(s, "finish_gauge", {"summary": "x"}, "llm")
    assert not r["ok"] and "create_work_order" in r["error"]
    wo = call_tool(s, "create_work_order", {"priority": "P4", "title": "t", "description": "d",
                                            "reading_id": rd["reading_id"]}, "llm")["result"]
    assert wo["priority"] in ("P1", "P2") and "priority_adjusted" in wo
    assert call_tool(s, "finish_gauge", {"summary": "x"}, "llm")["ok"]
    assert s.work_orders[wo["work_order_id"]].status == "pending_approval"


def test_capture_budget_enforced():
    s = Session(profile="clean", seed=12)
    _begin(s)
    for _ in range(guard.MAX_CAPTURES):
        assert call_tool(s, "capture", {"view": "front"}, "llm")["ok"]
    r = call_tool(s, "capture", {"view": "front"}, "llm")
    assert not r["ok"] and "budget" in r["error"]


class ScriptedBedrock:
    """Converse-shaped fake: behaves like a careful model (capture -> read -> record/escalate -> finish)."""

    def __init__(self):
        self.calls = 0

    def converse(self, **kw):
        self.calls += 1
        msgs = kw["messages"]
        if "toolConfig" not in kw:
            return {"output": {"message": {"role": "assistant", "content": [{"text": "Shift note."}]}},
                    "stopReason": "end_turn", "usage": {"inputTokens": 100, "outputTokens": 20}}
        last = msgs[-1]["content"]
        res = None
        for b in last:
            if "toolResult" in b:
                res = b["toolResult"]["content"][0]["json"]
        uid = f"t{self.calls}"
        if res is None:
            tool, inp, txt = "capture", {"view": "front", "burst": 3}, "Start with a frontal 3-frame capture."
        elif "capture_id" in res and "reading_id" not in res:
            tool, inp, txt = "read_gauge", {"capture_id": res["capture_id"]}, "Measure the needle."
        elif "reading_id" in res and "recorded" not in res:
            if res["confidence"] >= 0.6 and res["limit_check"]["status"] != "uncertain":
                tool, inp, txt = "record_reading", {"reading_id": res["reading_id"]}, "Trustworthy: record."
            else:
                tool, inp, txt = "request_human_review", {"reason": "low confidence"}, "Escalate."
        elif res.get("recorded") and res.get("work_order_required"):
            tool, inp, txt = "create_work_order", {"priority": "P2", "title": "alarm", "description": "d"}, "WO."
        elif "error" in res:
            tool, inp, txt = "request_human_review", {"reason": res["error"][:80]}, "Guard said no; escalate."
        else:
            tool, inp, txt = "finish_gauge", {"summary": "done"}, "Close."
        return {"output": {"message": {"role": "assistant", "content": [
            {"text": txt}, {"toolUse": {"toolUseId": uid, "name": tool, "input": inp}}]}},
            "stopReason": "tool_use", "usage": {"inputTokens": 1200, "outputTokens": 60}}


class BrokenBedrock:
    def converse(self, **kw):
        raise RuntimeError("ThrottlingException: Too many requests")


def test_llm_inspector_loop_with_scripted_model():
    s = Session(profile="clean", seed=12, agent_mode="llm")
    insp = BedrockInspector(ScriptedBedrock(), model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    run_round(s, llm_agent=insp)
    assert s.status == "done", s.error
    assert all(oc.finished and oc.agent == "llm" for oc in s.outcomes.values())
    assert s.usage["llm_calls"] > 0 and s.usage["cost_usd"] > 0
    assert any(e["kind"] == "thought" and e["actor"] == "llm" for e in s.trace)
    assert s.report == "Shift note."


def test_llm_failure_falls_back_to_policy_and_opens_circuit():
    s = Session(profile="clean", seed=12, agent_mode="llm")
    insp = BedrockInspector(BrokenBedrock(), model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    run_round(s, llm_agent=insp)
    assert s.status == "done"
    assert all(oc.finished for oc in s.outcomes.values())
    assert all(oc.agent == "policy-fallback" for oc in s.outcomes.values())
    assert insp.circuit_open
    assert any(e["kind"] == "fallback" for e in s.trace)
