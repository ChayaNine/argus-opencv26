#!/bin/bash
# Change runtime settings on the instance and restart the service, e.g.
#   ./deploy/set_env.sh ARGUS_BEDROCK_MODEL=us.anthropic.claude-sonnet-4-6
#   ./deploy/set_env.sh ARGUS_LLM_ENABLED=false ARGUS_LLM_DAILY_CAP=40
source "$(dirname "$0")/lib.sh"
[ $# -gt 0 ] || die "usage: $0 KEY=VALUE [KEY=VALUE...]"
check_aws
IID=$(out InstanceId)
TMP=$(mktemp)
{
  echo "set -e"
  for kv in "$@"; do
    k="${kv%%=*}"; v="${kv#*=}"
    [[ "$k" =~ ^ARGUS_[A-Z_]+$ ]] || die "only ARGUS_* settings can be changed ($k)"
    [[ "$v" =~ ^[A-Za-z0-9._:/-]*$ ]] || die "unsupported characters in value for $k"
    echo "grep -q '^$k=' /etc/argus/argus.env && sed -i 's#^$k=.*#$k=$v#' /etc/argus/argus.env || echo '$k=$v' >> /etc/argus/argus.env"
  done
  echo "systemctl restart argus && sleep 3 && curl -fsS http://127.0.0.1:8080/api/config"
} > "$TMP"
ssm_run "$IID" "$TMP" 120
rm -f "$TMP"
