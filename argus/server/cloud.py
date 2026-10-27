"""AWS sinks: S3 evidence archive, CloudWatch Logs (structured trace), CloudWatch metrics, SNS dispatch.

All sinks are optional (enabled by environment) and fail soft: the inspection
round never depends on the archive being reachable.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from typing import Optional

from ..config import CONFIG

log = logging.getLogger("argus.cloud")


def _client(name: str):
    import boto3
    from botocore.config import Config
    return boto3.client(name, region_name=CONFIG.region,
                        config=Config(retries={"max_attempts": 3, "mode": "standard"}, connect_timeout=3,
                                      read_timeout=10))


class Archive:
    """Background uploader to S3 (round traces and evidence images)."""

    def __init__(self):
        self.enabled = bool(CONFIG.s3_bucket)
        self.q: "queue.Queue[tuple]" = queue.Queue(maxsize=5000)
        self.errors = 0
        self.uploaded = 0
        if self.enabled:
            threading.Thread(target=self._run, daemon=True, name="s3-archive").start()

    def put(self, key: str, data: bytes, content_type: str):
        if not self.enabled:
            return
        try:
            self.q.put_nowait((key, data, content_type))
        except queue.Full:
            self.errors += 1

    def _run(self):
        s3 = _client("s3")
        while True:
            key, data, ct = self.q.get()
            try:
                s3.put_object(Bucket=CONFIG.s3_bucket, Key=key, Body=data, ContentType=ct,
                              ServerSideEncryption="AES256")
                self.uploaded += 1
            except Exception as e:  # pragma: no cover - network
                self.errors += 1
                log.warning("s3 upload failed for %s: %s", key, e)

    def get_json(self, key: str) -> Optional[dict]:
        if not self.enabled:
            return None
        try:
            obj = _client("s3").get_object(Bucket=CONFIG.s3_bucket, Key=key)
            return json.loads(obj["Body"].read())
        except Exception:
            return None


class TraceLogs:
    """Ships every trace event to CloudWatch Logs as one JSON line (batched)."""

    def __init__(self, stream: str):
        self.enabled = bool(CONFIG.log_group)
        self.stream = stream
        self.buf: list = []
        self.lock = threading.Lock()
        self.errors = 0
        if self.enabled:
            threading.Thread(target=self._run, daemon=True, name="cw-logs").start()

    def event(self, round_id: str, ev: dict):
        if not self.enabled:
            return
        slim = {k: v for k, v in ev.items() if k not in ("result",)}
        if "result" in ev and isinstance(ev["result"], dict):
            r = ev["result"]
            slim["result"] = {k: r.get(k) for k in ("capture_id", "reading_id", "value", "sigma", "confidence",
                                                     "status", "work_order_id", "review_id") if k in r}
            if "quality" in r:
                slim["issues"] = [i["code"] for i in r["quality"].get("issues", [])]
            if "limit_check" in r:
                slim["limit_status"] = r["limit_check"].get("status")
        slim["round_id"] = round_id
        with self.lock:
            self.buf.append({"timestamp": int(time.time() * 1000), "message": json.dumps(slim, default=str)[:20000]})

    def _run(self):
        logs = _client("logs")
        try:
            logs.create_log_stream(logGroupName=CONFIG.log_group, logStreamName=self.stream)
        except Exception:
            pass
        while True:
            time.sleep(2.0)
            with self.lock:
                batch, self.buf = self.buf[:5000], self.buf[5000:]
            if not batch:
                continue
            batch.sort(key=lambda e: e["timestamp"])
            try:
                logs.put_log_events(logGroupName=CONFIG.log_group, logStreamName=self.stream, logEvents=batch)
            except Exception as e:  # pragma: no cover - network
                self.errors += 1
                log.warning("cloudwatch logs failed: %s", e)


class Metrics:
    def __init__(self):
        self.enabled = bool(CONFIG.log_group or CONFIG.s3_bucket)

    def round_done(self, snap: dict):
        if not self.enabled:
            return
        g = snap["gauges"]
        n = max(1, len(g))
        dims = [{"Name": "Agent", "Value": snap["agent_mode"]}]
        esc = sum(1 for x in g if x["outcome"]["review_id"])
        wo = sum(1 for x in g if x["outcome"]["work_order_id"])
        auto = sum(1 for x in g if x["outcome"]["reading_id"])
        comp = sum(x["outcome"]["compute_ms"] for x in g) / n
        robot = snap["robot_time_s"] / n
        u = snap["usage"]
        data = [
            ("RoundsCompleted", 1, "Count"), ("GaugesInspected", len(g), "Count"),
            ("AutomatedReadings", auto, "Count"), ("Escalations", esc, "Count"), ("WorkOrdersDrafted", wo, "Count"),
            ("ComputeMsPerGauge", comp, "Milliseconds"), ("RobotSecondsPerGauge", robot, "Seconds"),
            ("LLMCalls", u.get("llm_calls", 0), "Count"), ("LLMFallbacks", u.get("fallbacks", 0), "Count"),
            ("LLMCostMicroUSD", u.get("cost_usd", 0.0) * 1e6, "Count"),
        ]
        md = [{"MetricName": k, "Dimensions": dims, "Value": float(v), "Unit": unit} for k, v, unit in data]
        threading.Thread(target=self._put, args=(md,), daemon=True).start()

    def _put(self, md):
        try:
            _client("cloudwatch").put_metric_data(Namespace=CONFIG.metric_namespace, MetricData=md)
        except Exception as e:  # pragma: no cover - network
            log.warning("put_metric_data failed: %s", e)


def dispatch_work_order(wo: dict, gauge: dict, approver: str) -> dict:
    """Send an approved work order to the maintenance channel (SNS topic -> email/CMMS webhook)."""
    msg = (f"ARGUS work order {wo['id']} ({wo['priority']}) approved by {approver}\n\n"
           f"{wo['title']}\n\n{wo['description']}\n\nGauge: {gauge.get('id')} {gauge.get('name')}\n")
    if not CONFIG.sns_topic_arn:
        return {"channel": "simulated-cmms", "ref": f"SIM-{wo['id']}", "note": "SNS not configured"}
    try:
        r = _client("sns").publish(TopicArn=CONFIG.sns_topic_arn, Subject=f"[ARGUS] {wo['priority']} {wo['title']}"[:99],
                                   Message=msg)
        return {"channel": "sns", "ref": r.get("MessageId")}
    except Exception as e:  # pragma: no cover - network
        return {"channel": "sns", "ref": None, "error": str(e)[:200]}
