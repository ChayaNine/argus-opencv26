#!/bin/bash
# Shared helpers for the deploy scripts.  Designed to run in AWS CloudShell (aws CLI v2, python3, tar).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STACK="${STACK:-argus}"
REGION="${REGION:-${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}}"
export AWS_DEFAULT_REGION="$REGION"

say() { printf '\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "'$1' is required (run these scripts in AWS CloudShell, which has it)"; }

check_aws() {
  need aws; need tar; need python3
  ACCOUNT=$(aws sts get-caller-identity --query Account --output text 2>/dev/null) || die "AWS credentials not configured"
  say "AWS account $ACCOUNT, region $REGION, stack $STACK"
}

out() {  # stack output by key
  aws cloudformation describe-stacks --stack-name "$STACK" --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text
}

artifact_bucket() { echo "argus-artifacts-${ACCOUNT}-${REGION}"; }

ensure_artifact_bucket() {
  local b; b=$(artifact_bucket)
  if ! aws s3api head-bucket --bucket "$b" 2>/dev/null; then
    say "creating artifact bucket $b"
    if [ "$REGION" = us-east-1 ]; then aws s3api create-bucket --bucket "$b" >/dev/null
    else aws s3api create-bucket --bucket "$b" --create-bucket-configuration LocationConstraint="$REGION" >/dev/null; fi
    aws s3api put-public-access-block --bucket "$b" --public-access-block-configuration \
      BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
    aws s3api put-bucket-encryption --bucket "$b" --server-side-encryption-configuration \
      '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
  fi
}

package() {  # -> sets ARTIFACT_KEY, uploads tarball
  local ts tmp; ts=$(date -u +%Y%m%d-%H%M%S); tmp=$(mktemp -d)
  ARTIFACT_KEY="argus/argus-${ts}.tgz"
  say "packaging $ROOT"
  ( cd "$ROOT" && tar --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='./data' \
      --exclude='./results/eval/2*' --exclude='./results/eval/latest' --exclude='./.venv' --exclude='*.tgz' \
      -czf "$tmp/argus.tgz" . )
  # ship the most recent local evaluation (if any) so the Evidence tab is populated from the first boot
  if [ -e "$ROOT/results/eval/latest/summary.json" ]; then
    mkdir -p "$tmp/stage/results/eval/latest"
    cp -L "$ROOT/results/eval/latest/summary.json" "$tmp/stage/results/eval/latest/" 2>/dev/null || true
    cp -L "$ROOT/results/eval/latest/strategies.png" "$tmp/stage/results/eval/latest/" 2>/dev/null || true
    ( cd "$tmp" && gunzip argus.tgz && tar -rf argus.tar -C stage results && gzip -f argus.tar && mv argus.tar.gz argus.tgz )
  fi
  aws s3 cp --only-show-errors "$tmp/argus.tgz" "s3://$(artifact_bucket)/$ARTIFACT_KEY"
  rm -rf "$tmp"
  say "uploaded s3://$(artifact_bucket)/$ARTIFACT_KEY"
}

ssm_run() {  # ssm_run <instance-id> <script-file> [timeout-seconds]
  local iid="$1" file="$2" to="${3:-3600}" b64 cid status
  b64=$(base64 -w0 "$file" 2>/dev/null || base64 "$file" | tr -d '\n')
  cid=$(aws ssm send-command --instance-ids "$iid" --document-name AWS-RunShellScript \
        --timeout-seconds "$to" --comment "argus $(basename "$file")" \
        --parameters "{\"commands\":[\"echo $b64 | base64 -d > /tmp/argus-ssm.sh\",\"bash /tmp/argus-ssm.sh\"],\"executionTimeout\":[\"$to\"]}" \
        --query Command.CommandId --output text) || die "SSM send-command failed (is the instance registered in Systems Manager?)"
  say "SSM command $cid running on $iid (this can take several minutes)"
  while true; do
    sleep 10
    status=$(aws ssm get-command-invocation --command-id "$cid" --instance-id "$iid" --query Status --output text 2>/dev/null || echo Pending)
    case "$status" in Pending|InProgress|Delayed) printf '.';; *) echo; break;; esac
  done
  aws ssm get-command-invocation --command-id "$cid" --instance-id "$iid" --query StandardOutputContent --output text | tail -n 60
  local err; err=$(aws ssm get-command-invocation --command-id "$cid" --instance-id "$iid" --query StandardErrorContent --output text | tail -n 20)
  [ -n "$err" ] && [ "$err" != "None" ] && echo "--- stderr (tail) ---" && echo "$err"
  [ "$status" = Success ] || die "remote command finished with status $status"
}

wait_healthy() {  # wait_healthy <url> [minutes]
  local url="$1" mins="${2:-20}" i
  say "waiting for $url/api/health (CloudFront deploys in ~5-10 min; first boot installs packages)"
  for i in $(seq 1 $((mins * 6))); do
    if curl -fsS --max-time 8 "$url/api/health" >/dev/null 2>&1; then echo; say "healthy"; return 0; fi
    printf '.'; sleep 10
  done
  echo; return 1
}
