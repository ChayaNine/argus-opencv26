#!/bin/bash
# COOL vs stock OpenCV 5 on the deployed Graviton4 instance (identical inputs, same hardware).
# Merges any other benchmark JSONs found in s3://<data-bucket>/evidence/ (e.g. the x86 run) and publishes
# compare.json / compare.md / bench.png to the app's Evidence tab and to S3.
#   ./deploy/run_bench.sh [--compare-only] [--seconds 30]
source "$(dirname "$0")/lib.sh"
MODE=full; SECS=30
while [ $# -gt 0 ]; do case "$1" in --compare-only) MODE=compare; shift;; --seconds) SECS="$2"; shift 2;; --stack) STACK="$2"; shift 2;; *) die "unknown option $1";; esac; done
check_aws
IID=$(out InstanceId); BUCKET=$(out DataBucket)
TMP=$(mktemp)
cat > "$TMP" <<EOF
set -eo pipefail
cd /opt/argus
set -a; source /etc/argus/argus.env; set +a
source /etc/argus/python.env
source "\$ARGUS_ACTIVATE"
TOK=\$(curl -s -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')
IT=\$(curl -s -H "X-aws-ec2-metadata-token: \$TOK" http://169.254.169.254/latest/meta-data/instance-type)
mkdir -p results/bench
if [ "$MODE" = full ]; then
  echo "stopping the web service during the benchmark for clean numbers"
  systemctl stop argus || true
  trap 'systemctl start argus' EXIT
  python -m bench.bench --label "cool-\$IT" --seconds $SECS
  /opt/argus-stock/bin/python -m bench.bench --label "stock-\$IT" --seconds $SECS
  systemctl start argus; trap - EXIT
  for f in results/bench/bench_cool-\$IT.json results/bench/bench_stock-\$IT.json; do
    python deploy/s3tool.py put "\$f" "$BUCKET" "evidence/\$(basename \$f)"
  done
fi
python deploy/s3tool.py pull "$BUCKET" evidence/bench_ results/bench .json
rm -f results/bench/bench_compare.json
python -m bench.compare results/bench/bench_*.json
python deploy/s3tool.py put results/bench/compare.json "$BUCKET" evidence/bench_compare.json
python deploy/s3tool.py put results/bench/compare.md "$BUCKET" evidence/bench_compare.md
python deploy/s3tool.py put results/bench/bench.png "$BUCKET" evidence/bench.png
chown -R argus:argus results || true
EOF
ssm_run "$IID" "$TMP" 3600
rm -f "$TMP"
mkdir -p "$ROOT/results/bench"
for k in bench_compare.json bench_compare.md bench.png; do
  aws s3 cp --only-show-errors "s3://$BUCKET/evidence/$k" "$ROOT/results/bench/${k/bench_compare/compare}" || true
done
aws s3 cp --only-show-errors "s3://$BUCKET/evidence/" "$ROOT/results/bench/" --recursive --exclude "*" --include "bench_cool-*" --include "bench_stock-*"
say "results copied to results/bench/ and shown in the app's Evidence tab"
cat "$ROOT/results/bench/compare.md" 2>/dev/null || true
