#!/bin/bash
# Ground-truth evaluation on the deployed instance: baselines + policy agent on N scenes, and the Bedrock LLM
# agent on the first M of those scenes.  Publishes the summary to the Evidence tab and S3.
#   ./deploy/run_eval.sh [--scenes 30] [--llm-scenes 8]
source "$(dirname "$0")/lib.sh"
SCENES=30; LLM_SCENES=8
while [ $# -gt 0 ]; do case "$1" in --scenes) SCENES="$2"; shift 2;; --llm-scenes) LLM_SCENES="$2"; shift 2;; --stack) STACK="$2"; shift 2;; *) die "unknown option $1";; esac; done
check_aws
IID=$(out InstanceId); BUCKET=$(out DataBucket)
TMP=$(mktemp)
cat > "$TMP" <<EOF
set -eo pipefail
cd /opt/argus
set -a; source /etc/argus/argus.env; set +a
source /etc/argus/python.env
source "\$ARGUS_ACTIVATE"
STRATS=overview,fixed,fixed_gated,agent_policy
if [ "$LLM_SCENES" -gt 0 ] && [ "\$ARGUS_LLM_ENABLED" = true ]; then STRATS=\$STRATS,agent_llm; fi
python -m eval.run_eval --scenes $SCENES --llm-scenes $LLM_SCENES --strategies "\$STRATS" --gallery --workers \$(nproc)
python deploy/s3tool.py put results/eval/latest/summary.json "$BUCKET" evidence/eval_summary.json
python deploy/s3tool.py put results/eval/latest/strategies.png "$BUCKET" evidence/eval_strategies.png
python deploy/s3tool.py put results/eval/latest/per_gauge.csv "$BUCKET" evidence/eval_per_gauge.csv
chown -R argus:argus results || true
EOF
ssm_run "$IID" "$TMP" 5400
rm -f "$TMP"
mkdir -p "$ROOT/results/eval/aws"
aws s3 cp --only-show-errors "s3://$BUCKET/evidence/eval_summary.json" "$ROOT/results/eval/aws/summary.json"
aws s3 cp --only-show-errors "s3://$BUCKET/evidence/eval_strategies.png" "$ROOT/results/eval/aws/strategies.png" || true
aws s3 cp --only-show-errors "s3://$BUCKET/evidence/eval_per_gauge.csv" "$ROOT/results/eval/aws/per_gauge.csv" || true
say "evaluation copied to results/eval/aws/ and shown in the app's Evidence tab"
python3 - "$ROOT/results/eval/aws/summary.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))["summary"]
keys = ["gauges", "automation_rate", "silent_failure_rate", "missed_critical", "false_alarms", "within_1pct",
        "sigma_coverage_2sigma", "captures_per_gauge", "llm_cost_usd_per_gauge", "policy_fallbacks"]
print(f"{'':26s}" + "".join(f"{s:>18s}" for s in d))
for k in keys:
    print(f"{k:26s}" + "".join(f"{str(d[s].get(k, '')):>18s}" for s in d))
PY
