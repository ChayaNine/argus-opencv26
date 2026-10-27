#!/bin/bash
# One-command deployment of ARGUS to your AWS account.
#
#   ./deploy/deploy.sh --ami ami-0123456789abcdef0 [--email you@example.com] [--region us-east-1]
#                      [--stack argus] [--instance-type c8g.xlarge] [--model <bedrock-model-id>] [--no-llm]
#
# Run without --ami to list COOL AMIs visible to your account after you subscribe in AWS Marketplace.
source "$(dirname "$0")/lib.sh"
AMI=""; EMAIL=""; ITYPE="c8g.xlarge"; MODEL="global.anthropic.claude-haiku-4-5-20251001-v1:0"; LLM=true
while [ $# -gt 0 ]; do
  case "$1" in
    --ami) AMI="$2"; shift 2;;
    --email) EMAIL="$2"; shift 2;;
    --region) REGION="$2"; export AWS_DEFAULT_REGION="$2"; shift 2;;
    --stack) STACK="$2"; shift 2;;
    --instance-type) ITYPE="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --no-llm) LLM=false; shift;;
    -h|--help) sed -n '2,9p' "$0"; exit 0;;
    *) die "unknown option $1";;
  esac
done
check_aws

EXISTS=0
aws cloudformation describe-stacks --stack-name "$STACK" >/dev/null 2>&1 && EXISTS=1

if [ "$EXISTS" = 0 ]; then
  if [ -z "$AMI" ]; then
    say "COOL AMIs from AWS Marketplace visible in $REGION (subscribe first if this list is empty):"
    aws ec2 describe-images --owners aws-marketplace --filters "Name=architecture,Values=arm64" \
      "Name=name,Values=*COOL*,*Cool*,*cool*,*OpenCV*,*opencv*" \
      --query 'reverse(sort_by(Images,&CreationDate))[:10].[ImageId,Name,CreationDate]' --output table || true
    die "pass the Graviton4 COOL AMI with --ami (from Marketplace > Manage subscriptions > Launch, or the list above)"
  fi
  aws ec2 describe-images --image-ids "$AMI" --query 'Images[0].[Name,Architecture]' --output text \
    || die "AMI $AMI not found in $REGION"
  AZ=$(aws ec2 describe-instance-type-offerings --location-type availability-zone \
        --filters "Name=instance-type,Values=$ITYPE" --query 'InstanceTypeOfferings[].Location' --output text | tr '\t' '\n' | sort | head -1)
  [ -n "$AZ" ] || die "$ITYPE is not offered in $REGION"
  say "using $ITYPE in $AZ"
  PL=$(aws ec2 describe-managed-prefix-lists --filters Name=prefix-list-name,Values=com.amazonaws.global.cloudfront.origin-facing \
        --query 'PrefixLists[0].PrefixListId' --output text 2>/dev/null || echo "")
  [ "$PL" = None ] && PL=""
  [ -n "$PL" ] && say "origin port restricted to CloudFront prefix list $PL"
  SECRET=$(python3 -c "import secrets; print(secrets.token_hex(24))")
  ensure_artifact_bucket
  package
  say "deploying CloudFormation stack $STACK (about 10 minutes)"
  aws cloudformation deploy --stack-name "$STACK" --template-file "$ROOT/deploy/cloudformation.yaml" \
    --capabilities CAPABILITY_IAM --no-fail-on-empty-changeset \
    --parameter-overrides CoolAmiId="$AMI" InstanceType="$ITYPE" AvailabilityZone="$AZ" \
      ArtifactBucket="$(artifact_bucket)" ArtifactKey="$ARTIFACT_KEY" EnableLlm="$LLM" BedrockModelId="$MODEL" \
      AlertEmail="$EMAIL" OriginSecret="$SECRET" CloudFrontPrefixList="$PL" \
    --tags Project=ARGUS
else
  say "stack $STACK already exists."
  say "  ship code:            ./deploy/update.sh"
  say "  change model/LLM:     ./deploy/set_env.sh ARGUS_BEDROCK_MODEL=<id>   (or ARGUS_LLM_ENABLED=false)"
  if [ -n "$EMAIL" ]; then
    say "updating the alert email only"
    aws cloudformation deploy --stack-name "$STACK" --template-file "$ROOT/deploy/cloudformation.yaml" \
      --capabilities CAPABILITY_IAM --no-fail-on-empty-changeset --parameter-overrides AlertEmail="$EMAIL"
  fi
fi

URL=$(out AppUrl)
IID=$(out InstanceId)
if ! wait_healthy "$URL" 25; then
  echo "Not healthy yet. Bootstrap log:  aws ssm start-session --target $IID  then  sudo tail -n 100 /var/log/argus-bootstrap.log"
  exit 1
fi
curl -fsS "$URL/api/runtime" | python3 -c "import json,sys; d=json.load(sys.stdin); print('Runtime:', d['label']); print('COOL verified:', d['cool'])"
cat <<EOF

ARGUS is live:   $URL
Dashboard:       $(out DashboardUrl)
Instance:        $IID   (shell: aws ssm start-session --target $IID)

Next:
  ./deploy/check_bedrock.sh          # confirm the LLM agent can call Bedrock
  ./deploy/run_bench.sh              # COOL vs stock OpenCV 5 on this Graviton4 instance
  ./deploy/bench_x86.sh              # optional x86 baseline (c7i.xlarge, terminates itself)
  ./deploy/run_eval.sh               # ground-truth evaluation incl. the Bedrock agent
EOF
[ -n "$EMAIL" ] && echo "Check $EMAIL and confirm the SNS subscription to receive dispatched work orders."
