#!/bin/bash
# Runs ON the EC2 instance (from UserData, and again on every update).
# - finds the COOL OpenCV 5 interpreter and proves it is the one that loads
# - installs the app's pure-Python dependencies next to it without touching COOL's numpy/cv2
# - prepares a stock OpenCV 5 venv for the like-for-like benchmark
# - installs a hardened systemd service and waits for it to be healthy
set -euo pipefail
APP=/opt/argus
LOG=/var/log/argus-bootstrap.log
exec > >(tee -a "$LOG") 2>&1
echo "=== ARGUS bootstrap $(date -Is) ==="
export DEBIAN_FRONTEND=noninteractive

if [ -z "${ARGUS_BOOTSTRAP_TEST:-}" ]; then
  apt-get update -y || true
  apt-get install -y --no-install-recommends python3-venv python3-pip curl ca-certificates || true
fi

# ---------------------------------------------------------------- find COOL
mkdir -p /etc/argus
cat > /etc/argus/check_cool.py <<'PY'
import cv2, json, os, sys
p = os.path.dirname(cv2.__file__)
ok = p.startswith("/opt/cool") and cv2.__version__.split(".")[0] == "5"
print(json.dumps({"version": cv2.__version__, "path": p}))
sys.exit(0 if ok else 1)
PY
ARGUS_PY=""; ARGUS_ACTIVATE=""; ARGUS_COOL=0
for v in /opt/cool/venvs/python_3.12 /opt/cool/venvs/python_3.11 /opt/cool/venvs/python_3.13 /opt/cool/venvs/python_3.10; do
  # use the vendor's activate script so library paths are exactly what COOL expects
  if [ -x "$v/bin/python" ] && out=$(bash -c "source '$v/bin/activate' && python /etc/argus/check_cool.py"); then
    ARGUS_PY="$v/bin/python"; ARGUS_ACTIVATE="$v/bin/activate"; ARGUS_COOL=1
    echo "COOL interpreter: $ARGUS_PY -> $out"; break
  fi
done
if [ -z "$ARGUS_PY" ] && [ -d /opt/cool ]; then
  # manual environment, as documented on the Marketplace listing, plus our own venv for app packages
  for sp in /opt/cool/python_3.12/site-packages /opt/cool/python_3.12/site-packages/cv2/python-3.12; do
    if out=$(env LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib:/opt/ffmpeg/lib PYTHONPATH="$sp" python3 /etc/argus/check_cool.py); then
      [ -x /opt/argus-venv/bin/python ] || python3 -m venv /opt/argus-venv
      {
        echo "source /opt/argus-venv/bin/activate"
        echo 'export LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib:/opt/ffmpeg/lib:${LD_LIBRARY_PATH:-}'
        echo "export PYTHONPATH=$sp"
      } > /etc/argus/cool-env.sh
      ARGUS_PY=/opt/argus-venv/bin/python; ARGUS_ACTIVATE=/etc/argus/cool-env.sh; ARGUS_COOL=1
      echo "COOL via PYTHONPATH=$sp -> $out"; break
    fi
  done
fi
if [ -z "$ARGUS_PY" ]; then
  echo "WARNING: COOL OpenCV 5 not found on this image; falling back to stock OpenCV 5 (the COOL award needs the COOL AMI)."
  [ -x /opt/argus-venv/bin/python ] || python3 -m venv /opt/argus-venv
  /opt/argus-venv/bin/pip install --quiet -r "$APP/requirements-opencv.txt"
  ARGUS_PY=/opt/argus-venv/bin/python; ARGUS_ACTIVATE=/opt/argus-venv/bin/activate; ARGUS_COOL=0
fi

# ---------------------------------------------------------------- app dependencies (never replace COOL's numpy/cv2)
run_py() { bash -c "source '$ARGUS_ACTIVATE' && $*"; }
run_py "python -m pip --version" || run_py "python -m ensurepip --upgrade" || true
CONSTRAINTS=/etc/argus/constraints.txt
run_py "python -m pip freeze" | grep -iE '^(numpy|opencv[-a-z]*)==' > "$CONSTRAINTS" || true
echo "pinned from the existing environment:"; cat "$CONSTRAINTS" || true
PIPX="python -m pip install --quiet --disable-pip-version-check"
run_py "$PIPX -r $APP/requirements.txt -c $CONSTRAINTS"
# charts are optional: try the pinned matplotlib, then any version compatible with COOL's numpy, else skip
run_py "$PIPX -r $APP/requirements-eval.txt -c $CONSTRAINTS" || run_py "$PIPX matplotlib -c $CONSTRAINTS" \
  || echo "matplotlib not installed (charts in eval/bench reports will be skipped)"
# guard: the import must still resolve to COOL after installing
if [ "$ARGUS_COOL" = 1 ]; then
  run_py "python -c \"import cv2,sys; print('cv2 after install:', cv2.__version__, cv2.__file__); sys.exit(0 if cv2.__file__.startswith('/opt/cool') else 1)\""
fi

if [ -n "${ARGUS_BOOTSTRAP_TEST:-}" ]; then
  echo "TEST MODE: ARGUS_PY=$ARGUS_PY ARGUS_ACTIVATE=$ARGUS_ACTIVATE ARGUS_COOL=$ARGUS_COOL"; exit 0
fi

# ---------------------------------------------------------------- stock OpenCV 5 venv (benchmark baseline, same machine)
if [ ! -x /opt/argus-stock/bin/python ]; then
  python3 -m venv /opt/argus-stock
fi
/opt/argus-stock/bin/pip install --quiet --disable-pip-version-check -r "$APP/requirements.txt" -r "$APP/requirements-opencv.txt" -r "$APP/requirements-eval.txt" || echo "stock venv install failed (benchmark baseline unavailable)"

# ---------------------------------------------------------------- service user, dirs, unit
id argus >/dev/null 2>&1 || useradd --system --home /var/lib/argus --shell /usr/sbin/nologin argus
mkdir -p /var/lib/argus "$APP/results/eval" "$APP/results/bench"
chown -R argus:argus /var/lib/argus "$APP/results"
chmod 755 "$APP"
cat > /etc/argus/python.env <<EOF
ARGUS_PY=$ARGUS_PY
ARGUS_ACTIVATE=$ARGUS_ACTIVATE
ARGUS_COOL=$ARGUS_COOL
EOF
chgrp argus /etc/argus/argus.env && chmod 640 /etc/argus/argus.env
SERVICE_USER=argus
if ! sudo -u argus bash -c "source '$ARGUS_ACTIVATE' && cd $APP && python -c 'import cv2, argus.server.app'" ; then
  echo "WARNING: the service cannot import as user 'argus' (COOL entitlement or permissions); running as root"
  SERVICE_USER=root
fi
cat > /etc/systemd/system/argus.service <<EOF
[Unit]
Description=ARGUS inspection-round service (OpenCV 5 / COOL on Graviton)
After=network-online.target
Wants=network-online.target

[Service]
User=$SERVICE_USER
Group=$SERVICE_USER
WorkingDirectory=$APP
EnvironmentFile=/etc/argus/argus.env
EnvironmentFile=/etc/argus/python.env
ExecStart=/bin/bash $APP/deploy/start.sh
Restart=always
RestartSec=3
NoNewPrivileges=yes
PrivateTmp=yes
ProtectHome=yes
ProtectSystem=full
ReadWritePaths=/var/lib/argus $APP/results
LimitNOFILE=8192

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable argus.service
systemctl restart argus.service

# ---------------------------------------------------------------- health + COOL evidence
for i in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:8080/api/health >/dev/null 2>&1; then break; fi
  sleep 2
done
curl -fsS http://127.0.0.1:8080/api/runtime | tee "$APP/results/runtime.json" || { journalctl -u argus -n 80 --no-pager; exit 1; }
run_py "cd $APP && python -c \"
import json, os, boto3
b = os.environ.get('ARGUS_S3_BUCKET')
if b:
    iid = json.load(open('results/runtime.json')).get('ec2', {}).get('instance-id', 'unknown')
    boto3.client('s3').upload_file('results/runtime.json', b, f'evidence/runtime_{iid}.json')
\"" || true
echo "=== ARGUS ready $(date -Is) (COOL=$ARGUS_COOL, user=$SERVICE_USER) ==="
