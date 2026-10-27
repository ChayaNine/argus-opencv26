#!/bin/bash
# systemd entry point: enter the exact environment COOL expects, then serve.
set -e
if [ -n "${ARGUS_ACTIVATE:-}" ]; then
  # shellcheck disable=SC1090
  source "$ARGUS_ACTIVATE"
fi
if [ -n "${ARGUS_COOL:-}" ] && [ "$ARGUS_COOL" = 1 ]; then export ARGUS_COOL_PATH=/opt/cool; fi
cd /opt/argus
exec python -m uvicorn argus.server.app:app --host 0.0.0.0 --port 8080 --workers 1 \
  --proxy-headers --forwarded-allow-ips='*' --timeout-keep-alive 30 --log-level info
