#!/bin/bash
# Delete everything this project created (after judging ends on November 9).
#   ./deploy/teardown.sh [--stack argus] [--yes]
source "$(dirname "$0")/lib.sh"
YES=0
while [ $# -gt 0 ]; do case "$1" in --yes) YES=1; shift;; --stack) STACK="$2"; shift 2;; *) die "unknown option $1";; esac; done
check_aws
BUCKET=$(out DataBucket 2>/dev/null || true)
if [ "$YES" != 1 ]; then
  read -r -p "Delete stack $STACK, its evidence bucket ${BUCKET:-?} and the artifact bucket? Type 'delete': " a
  [ "$a" = delete ] || die "aborted"
fi
[ -n "$BUCKET" ] && [ "$BUCKET" != None ] && { say "emptying $BUCKET"; aws s3 rm "s3://$BUCKET" --recursive --only-show-errors || true; }
say "deleting stack $STACK"
aws cloudformation delete-stack --stack-name "$STACK"
aws cloudformation wait stack-delete-complete --stack-name "$STACK" || true
[ -n "$BUCKET" ] && [ "$BUCKET" != None ] && aws s3api delete-bucket --bucket "$BUCKET" 2>/dev/null || true
AB=$(artifact_bucket)
aws s3 rm "s3://$AB" --recursive --only-show-errors 2>/dev/null || true
aws s3api delete-bucket --bucket "$AB" 2>/dev/null || true
say "done. Also cancel the COOL subscription: AWS Marketplace > Manage subscriptions > COOL > Cancel subscription."
