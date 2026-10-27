# Deploying ARGUS to AWS

The whole stack deploys with one command from **AWS CloudShell**, the browser terminal inside the AWS console. CloudShell already has the AWS CLI, your credentials, Python and tar, so you install nothing locally.

Total hands-on time is about 20 minutes, plus about 15 minutes of waiting. Region: **us-east-1** (N. Virginia) is recommended because it has Graviton4 (`c8g`) instances, the COOL AMI, and Claude on Amazon Bedrock.

## What it costs

| Item | Rate | Through judging (about 16 days) |
|---|---|---|
| EC2 `c8g.xlarge` (Graviton4, 4 vCPU, 8 GiB) | $0.1595/h on demand, us-east-1 | about $61 |
| COOL software fee (after the 7-day free trial) | $0.01/h on `c8g.xlarge` | about $2 |
| Bedrock, Claude Haiku 4.5 | about $0.03 per gauge, $0.15 per 5-gauge round | capped by `LlmDailyCap` (80 rounds/day by default) |
| x86 benchmark (`c7i.xlarge`, terminates itself) | $0.1785/h for about 15 minutes | about $0.05 |
| CloudFront, S3, CloudWatch, SNS | small | under $5 |

If you pass `--email`, the stack creates a **$60/month AWS Budget** that emails you at 80%. The LLM agent has a daily cap and per-IP rate limits, so a busy demo cannot run up a bill.

## 0. Before you start (5 minutes, do this first)

1. **Check the EC2 vCPU quota.** Open Service Quotas, then EC2, then "Running On-Demand Standard (A, C, D, H, I, M, R, T, Z) instances". You need at least 8 vCPUs: 4 for the app and 4 for the optional x86 benchmark. New accounts sometimes start lower; request an increase now, because approval can take a while.
2. **Subscribe to COOL.** Open the [COOL for AWS Graviton4 listing](https://aws.amazon.com/marketplace/pp/prodview-fdvbfiewzuehs).
   - Choose *View purchase options* (or *Continue to Subscribe*), accept the terms, and wait until it shows as subscribed. That takes a minute or two. It has a 7-day free trial.
   - Choose *Continue to Configuration*. Pick the latest software version (v3.1 is "Built on OpenCV 5.0") and region **US East (N. Virginia)**.
   - **Copy the AMI ID** shown (`ami-...`). You do not need to launch from the console.
3. **Enable Claude on Bedrock.** In the Amazon Bedrock console (us-east-1), open *Model access* (or *Model catalog*). Make sure **Anthropic Claude Haiku 4.5** is available to your account. Anthropic models ask for a short use-case form the first time.

## 1. Get the code into CloudShell

1. Open the AWS console in us-east-1 and click the CloudShell icon (`>_`) in the top bar.
2. Choose *Actions*, then *Upload file*, and upload `argus-rounds.zip`. Then run:

```bash
unzip -q argus-rounds.zip && cd argus-rounds
chmod +x deploy/*.sh
```

If your code is already on GitHub, `git clone https://github.com/<you>/argus-rounds && cd argus-rounds` works too.

## 2. Deploy (about 15 minutes, mostly waiting)

```bash
./deploy/deploy.sh --ami ami-XXXXXXXXXXXXXXXXX --email you@example.com
```

What the script does:

- Picks an Availability Zone that offers `c8g.xlarge`.
- Restricts the origin port to CloudFront's managed prefix list.
- Generates the origin secret.
- Packages the code to a private S3 bucket.
- Deploys `deploy/cloudformation.yaml`: VPC, Graviton4 instance from the COOL AMI, least-privilege IAM role, CloudFront HTTPS, S3, SNS, CloudWatch logs, dashboard and alarms, and the budget.
- Waits until `https://<id>.cloudfront.net/api/health` answers.

On first boot, the instance runs `deploy/bootstrap.sh`. It:

- Finds COOL's Python 3.12 environment and proves `cv2` is OpenCV 5 loaded from `/opt/cool`.
- Installs the app's pure-Python packages without touching COOL's numpy/cv2.
- Builds a stock OpenCV 5 venv for the like-for-like benchmark.
- Starts a hardened systemd service as an unprivileged user.

When it finishes, the script prints your **public URL**. If you used `--email`, **confirm the SNS subscription email** so approved work orders reach you.

Run without `--ami` to list COOL AMIs your account can see.

## 3. Verify (2 minutes)

Open the URL. The runner badge at the top right must read **`c8g.xlarge, Graviton4, OpenCV 5.0.x COOL`** with a green dot. Click it: the Runner dialog shows the build loaded from `/opt/cool`, the Neoverse-V2 CPU, the instance type and the Marketplace product code. In *Run round*, the inspector list must offer **LLM agent on Amazon Bedrock**.

Then:

```bash
./deploy/check_bedrock.sh        # the instance calls Bedrock with its role and prints OK
curl -s https://<your-url>/api/runtime | python3 -m json.tool | head -40   # COOL evidence: path, build flags, CPU part
```

If `check_bedrock.sh` fails, it tries other model IDs and tells you which works. Switch with:

```bash
./deploy/set_env.sh ARGUS_BEDROCK_MODEL=<id that printed OK>
```

## 4. Produce the evidence (about 45 minutes, unattended)

```bash
./deploy/run_bench.sh                          # COOL vs stock OpenCV 5 on the same c8g.xlarge (~6 min)
./deploy/bench_x86.sh                          # optional: stock OpenCV 5 on c7i.xlarge, merged automatically (~12 min)
./deploy/run_eval.sh --scenes 30 --llm-scenes 8   # ground truth incl. the Bedrock agent (~25 min, ~$1-2 of Bedrock)
```

Results appear in the app's **Evidence** tab, in `s3://<data-bucket>/evidence/`, and in `results/` in CloudShell. Put the final numbers into `docs/REPORT.md` (the sections marked `TODO-AWS`).

## 5. Use it

The console is laid out like a GitHub Actions run, so most people know where to click:

- **Run round** (green button on *Rounds*): pick *Hard conditions* and *LLM agent on Amazon Bedrock*. The page follows the robot from gauge to gauge. Each gauge page is a step log; steps labelled *Re-planned from what it saw* are the agent changing its next capture because of what OpenCV reported.
- **Round summary:** status, the route graph with live dials and ±2σ cones, annotations, approvals and the shift hand-over.
- **Review work orders** (yellow banner): *Approve and dispatch* publishes to SNS, so you get an email. **Review readings** (purple banner): confirm the robot's best attempt, enter your own reading, or send someone to check.
- **Read a photo:** drag in a phone photo of a real analog gauge, pick a common scale or enter it, and analyze.
- **Trace (JSON)** in the round sidebar: every perception result, decision, guard check and human action.
- **CloudWatch dashboard:** URL printed by `deploy.sh`. It shows automated vs escalated readings, fallbacks, LLM cost, CPU, and a Logs Insights table of the agent's decisions.

## 6. Ship code changes

```bash
./deploy/update.sh               # packages, copies to the instance via SSM, re-runs bootstrap, keeps results
```

## 7. MCP (optional, for Claude Code or Kiro)

```bash
pip install -r requirements-dev.txt
claude mcp add argus -- python -m argus.mcp_server
```

Then ask Claude Code to "run an ARGUS round on the hard profile". It drives the same OpenCV tools through the same safety guard.

## Run locally (no AWS)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest -q                       # 19 tests
python -m uvicorn argus.server.app:app --port 8080    # http://localhost:8080 (policy agent; LLM off)
python -m eval.run_eval --scenes 30       # ground-truth evaluation (~4 min on 2 cores)
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `You must accept terms ... Marketplace` during deploy | Finish step 0.2, wait 2 minutes, re-run `deploy.sh`. |
| `VcpuLimitExceeded` or `InsufficientInstanceCapacity` | Raise the quota (step 0.1), or try `--instance-type m8g.xlarge`. |
| Health check times out | Open Systems Manager, *Session Manager*, connect to the instance, then run `sudo tail -n 100 /var/log/argus-bootstrap.log` and `sudo journalctl -u argus -n 100`. |
| Header shows **Stock OpenCV build** | The AMI is not the COOL Graviton4 AMI, or COOL could not be loaded. Check the bootstrap log for "COOL interpreter". |
| `403 forbidden` when you open the EC2 IP directly | Expected: only CloudFront, with its secret header, can reach the origin. Use the CloudFront URL. |
| LLM rounds fall back to the policy | `./deploy/check_bedrock.sh`, then fix model access or switch the model with `set_env.sh`. The **LLMFallbacks** alarm emails you. |
| CloudFront 502 or 504 right after deploy | The instance is still bootstrapping; wait 5 minutes. |

## After judging (November 10)

```bash
./deploy/teardown.sh
```

Then cancel the COOL subscription: AWS Marketplace, *Manage subscriptions*, COOL, *Cancel subscription*.
