#!/bin/bash
# Ship the current code to the running instance (no downtime beyond a service restart).
#   ./deploy/update.sh [--stack argus]
source "$(dirname "$0")/lib.sh"
while [ $# -gt 0 ]; do case "$1" in --stack) STACK="$2"; shift 2;; --region) REGION="$2"; export AWS_DEFAULT_REGION="$2"; shift 2;; *) die "unknown option $1";; esac; done
check_aws
IID=$(out InstanceId); URL=$(out AppUrl)
ensure_artifact_bucket
package
TMP=$(mktemp)
cat > "$TMP" <<EOF
set -eo pipefail
set -a; source /etc/argus/argus.env; set +a
source /etc/argus/python.env
source "\$ARGUS_ACTIVATE"
python - <<'PY'
import boto3
boto3.client("s3").download_file("$(artifact_bucket)", "$ARTIFACT_KEY", "/tmp/argus-new.tgz")
PY
rm -rf /opt/argus.new && mkdir -p /opt/argus.new && tar xzf /tmp/argus-new.tgz -C /opt/argus.new
# keep results produced on the instance (benchmarks, evaluations)
if [ -d /opt/argus/results ]; then cp -a /opt/argus/results/. /opt/argus.new/results/ ; fi
rm -rf /opt/argus.prev; mv /opt/argus /opt/argus.prev; mv /opt/argus.new /opt/argus
sed -i "s#^ARTIFACT_KEY=.*#ARTIFACT_KEY=$ARTIFACT_KEY#" /etc/argus/artifact.env || true
bash /opt/argus/deploy/bootstrap.sh | tail -n 25
EOF
ssm_run "$IID" "$TMP" 1800
rm -f "$TMP"
DIST=$(out DistributionId)
if [ -n "$DIST" ] && [ "$DIST" != None ]; then
  aws cloudfront create-invalidation --distribution-id "$DIST" --paths "/static/*" "/" >/dev/null && say "CloudFront cache invalidated"
fi
wait_healthy "$URL" 5 && say "updated: $URL"
