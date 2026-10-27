#!/bin/bash
# Optional x86 baseline: launch a c7i.xlarge (same vCPU/RAM class as c8g.xlarge) with stock OpenCV 5, run the
# identical benchmark, publish the JSON to the evidence bucket, and terminate itself.
#   ./deploy/bench_x86.sh [--instance-type c7i.xlarge]
source "$(dirname "$0")/lib.sh"
ITYPE=c7i.xlarge
while [ $# -gt 0 ]; do case "$1" in --instance-type) ITYPE="$2"; shift 2;; --stack) STACK="$2"; shift 2;; *) die "unknown option $1";; esac; done
check_aws
BUCKET=$(out DataBucket); PROFILE=$(out InstanceProfileName); SG=$(out SecurityGroupId); SUBNET=$(out SubnetId)
AMI=$(aws ssm get-parameter --name /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id \
       --query Parameter.Value --output text) || die "could not resolve the Ubuntu 24.04 AMI"
ensure_artifact_bucket
package
UD=$(mktemp)
cat > "$UD" <<EOF
#!/bin/bash
exec > /var/log/argus-bench-x86.log 2>&1
set -x
export DEBIAN_FRONTEND=noninteractive AWS_DEFAULT_REGION=$REGION
apt-get update -y && apt-get install -y python3-venv
python3 -m venv /opt/v && /opt/v/bin/pip install -q boto3==1.43.118
/opt/v/bin/python -c "import boto3; boto3.client('s3').download_file('$(artifact_bucket)', '$ARTIFACT_KEY', '/tmp/a.tgz')"
mkdir -p /opt/argus && tar xzf /tmp/a.tgz -C /opt/argus && cd /opt/argus
/opt/v/bin/pip install -q -r requirements.txt -r requirements-opencv.txt
/opt/v/bin/python -m bench.bench --label stock-$ITYPE --seconds 30
/opt/v/bin/python -c "import boto3; boto3.client('s3').upload_file('results/bench/bench_stock-$ITYPE.json', '$BUCKET', 'evidence/bench_stock-$ITYPE.json')"
shutdown -h now
EOF
say "launching $ITYPE (Ubuntu 24.04, stock OpenCV 5) for the x86 baseline"
IID=$(aws ec2 run-instances --image-id "$AMI" --instance-type "$ITYPE" --subnet-id "$SUBNET" --security-group-ids "$SG" \
      --iam-instance-profile Name="$PROFILE" --instance-initiated-shutdown-behavior terminate \
      --metadata-options HttpTokens=required,HttpPutResponseHopLimit=2 \
      --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=20,VolumeType=gp3,Encrypted=true}' \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$STACK-bench-x86},{Key=Project,Value=ARGUS}]" \
      --user-data "file://$UD" --query 'Instances[0].InstanceId' --output text)
rm -f "$UD"
say "instance $IID running the benchmark; waiting until it terminates itself (about 10 minutes)"
aws ec2 wait instance-terminated --instance-ids "$IID" || true
if aws s3 ls "s3://$BUCKET/evidence/bench_stock-$ITYPE.json" >/dev/null 2>&1; then
  say "x86 result published. Merging:"
  "$(dirname "$0")/run_bench.sh" --compare-only
else
  die "no result found; the instance may have failed (console output: aws ec2 get-console-output --instance-id $IID)"
fi
