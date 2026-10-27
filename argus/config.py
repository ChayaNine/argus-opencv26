"""Runtime configuration from environment variables (set by deploy/bootstrap.sh on EC2)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _b(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Config:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("ARGUS_DATA_DIR", "./data")))
    results_dir: Path = field(default_factory=lambda: Path(os.environ.get("ARGUS_RESULTS_DIR", "./results")))
    region: str = field(default_factory=lambda: os.environ.get("AWS_REGION", "us-east-1"))
    s3_bucket: str = field(default_factory=lambda: os.environ.get("ARGUS_S3_BUCKET", ""))
    sns_topic_arn: str = field(default_factory=lambda: os.environ.get("ARGUS_SNS_TOPIC_ARN", ""))
    log_group: str = field(default_factory=lambda: os.environ.get("ARGUS_CW_LOG_GROUP", ""))
    metric_namespace: str = field(default_factory=lambda: os.environ.get("ARGUS_CW_NAMESPACE", "ARGUS"))
    origin_secret: str = field(default_factory=lambda: os.environ.get("ARGUS_ORIGIN_SECRET", ""))
    llm_enabled: bool = field(default_factory=lambda: _b("ARGUS_LLM_ENABLED", False))
    llm_model: str = field(default_factory=lambda: os.environ.get(
        "ARGUS_BEDROCK_MODEL", "global.anthropic.claude-haiku-4-5-20251001-v1:0"))
    llm_daily_cap: int = field(default_factory=lambda: int(os.environ.get("ARGUS_LLM_DAILY_CAP", "80")))
    max_concurrent_rounds: int = field(default_factory=lambda: int(os.environ.get("ARGUS_MAX_ROUNDS", "2")))
    rounds_per_ip_hour: int = field(default_factory=lambda: int(os.environ.get("ARGUS_ROUNDS_PER_IP_HOUR", "20")))
    llm_rounds_per_ip_hour: int = field(default_factory=lambda: int(os.environ.get("ARGUS_LLM_PER_IP_HOUR", "6")))
    uploads_per_ip_hour: int = field(default_factory=lambda: int(os.environ.get("ARGUS_UPLOADS_PER_IP_HOUR", "40")))
    max_upload_mb: float = field(default_factory=lambda: float(os.environ.get("ARGUS_MAX_UPLOAD_MB", "8")))


CONFIG = Config()
