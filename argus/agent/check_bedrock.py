"""Check that this machine can call the Bedrock model used by the LLM inspector.

    python -m argus.agent.check_bedrock [--model <id>]

Tries the configured model, then common alternatives, and prints what to fix.
"""
from __future__ import annotations

import argparse
import json
import os

from .llm import DEFAULT_MODEL, check_access

ALTERNATIVES = [
    "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "global.anthropic.claude-sonnet-4-6",
    "us.anthropic.claude-sonnet-4-6",
    "us.amazon.nova-pro-v1:0",
    "us.amazon.nova-lite-v1:0",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("ARGUS_BEDROCK_MODEL", DEFAULT_MODEL))
    a = ap.parse_args()
    r = check_access(a.model)
    print(json.dumps(r, indent=1))
    if r["ok"]:
        print(f"\nOK: the LLM agent can use {a.model}.")
        return
    print("\nThe configured model is not callable. Trying alternatives:")
    for m in ALTERNATIVES:
        if m == a.model:
            continue
        rr = check_access(m)
        print(f"  {m:55s} {'OK' if rr['ok'] else rr.get('error', '')[:90]}")
        if rr["ok"]:
            print(f"\nUse it with:  ./deploy/set_env.sh ARGUS_BEDROCK_MODEL={m}")
            break
    err = r.get("error", "")
    if "AccessDenied" in err:
        print("\nFix: Bedrock console > Model access: enable the Anthropic models (Anthropic asks for a short use-case "
              "form on first use), or check the instance role policy.")
    elif "ValidationException" in err:
        print("\nFix: the model ID or inference profile is not valid in this region; pick one that printed OK.")
    elif "Throttling" in err:
        print("\nThrottled: retry in a minute; the app retries automatically and falls back to the policy agent.")


if __name__ == "__main__":
    main()
