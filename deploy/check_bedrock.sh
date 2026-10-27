#!/bin/bash
# Verify that the deployed instance can call the Bedrock model the LLM agent uses (runs on the instance).
#   ./deploy/check_bedrock.sh [--model <id>]
source "$(dirname "$0")/lib.sh"
MODEL=""
while [ $# -gt 0 ]; do case "$1" in --model) MODEL="$2"; shift 2;; --stack) STACK="$2"; shift 2;; *) die "unknown option $1";; esac; done
check_aws
IID=$(out InstanceId)
TMP=$(mktemp)
cat > "$TMP" <<EOF
set -e
cd /opt/argus
set -a; source /etc/argus/argus.env; set +a
source /etc/argus/python.env
source "\$ARGUS_ACTIVATE"
python -m argus.agent.check_bedrock ${MODEL:+--model $MODEL}
EOF
ssm_run "$IID" "$TMP" 300
rm -f "$TMP"
