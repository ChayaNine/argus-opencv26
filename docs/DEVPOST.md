# Devpost submission text

Copy each section into the matching Devpost field. Numbers marked TODO-AWS come from `results/` after the AWS runs (see `docs/SUBMISSION.md`).

## Project name
ARGUS: operator rounds that know when not to trust their own eyes

## Tagline (under 200 characters)
OpenCV 5 on Graviton (COOL) reads analog gauges; a Bedrock agent changes how it looks when the evidence is bad, and escalates instead of guessing. 0 silent failures in 300 trials.

## Inspiration
Every brownfield plant has analog gauges that people walk around and read. Robots can photograph them, but automated readings fail exactly where plants are hard: glare on glass, pump vibration, night shifts, oblique mounts, a tag hanging in front of the dial. The dangerous failure isn't a missing reading; it's a wrong one recorded with confidence. We wanted a system that measures, knows when its own evidence is bad, does something about it, and hands off to a person when it can't fix it.

## What it does
An inspection robot's PTZ camera captures each gauge. **OpenCV 5**, running on **AWS Graviton4 with the Cloud-Optimized OpenCV Library (COOL)**, does the measuring:

- rectifies the dial from its ArUco tag;
- re-centres on the needle's pivot;
- finds the needle in polar coordinates;
- verifies the printed scale against the expected tick pattern;
- reports a calibrated ±2σ, plus a list of what's wrong with the evidence (glare on the needle, flutter, blur, noise, oblique view, occlusion, low contrast).

An agent, **Claude on Amazon Bedrock via the Converse API with tool use**, acts on that list:

- glare is a mirror reflection, so it **moves** and the reflection leaves the needle;
- flutter: nine fast-shutter frames, centred by midrange;
- night: spotlight, or more frames when the spotlight itself causes glare;
- haze: CLAHE;
- a reading whose interval straddles a limit: tighten it, or escalate.

A deterministic **safety guard** checks every call: only measured values can be recorded, no low-confidence records, alarms must get a work order, and priority can be raised but never lowered. People confirm escalations and approve every work order, which is then published to **Amazon SNS**.

The operator console is laid out like a GitHub Actions run, so it feels familiar from the first click. The rounds list has a *Run round* button. The round summary has a live route graph of dials with ±2σ cones. Each gauge page is a step log showing every capture, reading and decision, with the agent's reasoning ("re-planned from what it saw"). *Review work orders* and *Review readings* are the human gates. Photo mode reads your own gauge photo or tells you how to retake it.

## How we built it
- **OpenCV 5 perception:**
  - ArUco + `getPerspectiveTransform`;
  - `warpPolar` rim search + `fitEllipse`;
  - `HoughCircles(HOUGH_GRADIENT_ALT)` for the hub and for tag-less photos;
  - `warpPolar` needle coverage profile + Huber `fitLine` cross-check;
  - tick-template cross-correlation for scale and roll;
  - `AlignMTB` + `MergeMertens` HDR; `CLAHE`, Telea `inpaint`, NL-means;
  - Immerkær noise, structure-tensor motion-blur detection;
  - `FaceDetectorYN` (YuNet on OpenCV 5's DNN engine) to blur faces in uploads.
- **Agent:** six tools shared by the Bedrock inspector, a deterministic policy (fallback and no-LLM mode), an **MCP server** (Claude Code, Kiro) and the HTTP API. One Converse conversation per gauge; lessons passed between gauges; prompt caching; a circuit breaker.
- **Physics-based robot simulator for ground truth:**
  - pinhole PTZ optics, 3D gauge mounting;
  - specular glare from mirror geometry;
  - a shot- and read-noise sensor model;
  - shutter-dependent motion blur, needle flutter;
  - parallax occluders, haze, autofocus failures, stuck and over-range faults.
- **AWS:** one CloudFormation template.
  - CloudFront HTTPS; an origin secret plus the CloudFront prefix list;
  - a c8g.xlarge from the COOL Marketplace AMI with IMDSv2, least-privilege IAM, no SSH (Systems Manager), auto-recovery;
  - S3 evidence archive, CloudWatch JSON trace logs, metrics, dashboard and alarm, SNS, a budget.
  - One-command deploy, update, benchmark and evaluation scripts.
- **COOL benchmark:** identical inputs, the full ARGUS pipeline plus 17 OpenCV operations, timed on COOL vs stock OpenCV 5 on the *same* Graviton4 instance, and on x86. Reported as latency, throughput and $ per 1,000 reads.

## Results
Ground truth, 300 gauge inspections per strategy (simulator; see the report):

| | Naive robot round | + uncertainty gating | ARGUS |
|---|---|---|---|
| Silent failures | 32.0% | 1.7% | **0.0%** |
| Missed alarms or faults (of 69) | 15 | 0 | **0** |
| False work orders | 33 | 0 | **0** |
| Read automatically | 100% | 62% | **92%** |
| Recorded within ±1% of span | 61% | 90% | **99.6%** |
| Truth inside the reported ±2σ | 58% | 86% | **98%** |

- Reader precision on clean frames: median 0.06% of span.
- COOL vs stock OpenCV 5 on the same c8g.xlarge: TODO-AWS (from `results/bench/compare.md`).
- Bedrock agent: TODO-AWS (from `results/eval/aws/summary.json`).

## Challenges we ran into
- **Pivot errors.** A few pixels of pivot error bias a polar needle angle by degrees, and scale calibration can't see a translation. We fixed it by finding the hub cap and re-centring on it.
- **Flutter.** A 3-frame median can't find the centre of a 4–12 Hz needle oscillation. The midrange of nine fast frames can, because the sampled needle follows an arcsine distribution.
- **Evidence vs. confidence.** A uniformly clipped white face isn't glare: the needle survives overexposure but not a localised reflection. We kept a strong-looking needle peak inside a glare blob from counting as evidence.
- **Measurement on AWS.** We made COOL execution provable at runtime, and the benchmark like-for-like: same instance, same pixels, same readings.

## Accomplishments that we're proud of
Zero silent failures and zero missed alarms across 300 hard trials. The agent's actions follow the physics (it moves to clear glare). The uncertainty is calibrated. And every claim is reproducible from a script in the repo.

## What we learned
Knowing when not to trust a measurement is worth more than a slightly better measurement. Confidence has to be built from evidence quality, not just from how peaked a detector's output looks.

## What's next for ARGUS
- A camera adapter for real robot PTZ APIs, and site validation with real gauges.
- More dial types (non-linear and dual scales, digital displays).
- Trend alerts across rounds.
- Learned priors for which view works for each gauge.

## Built with
OpenCV 5, Cloud-Optimized OpenCV Library (COOL), AWS Graviton4 (EC2 c8g), Amazon Bedrock (Claude Haiku 4.5, Converse API), Amazon CloudFront, Amazon S3, Amazon CloudWatch, Amazon SNS, AWS Systems Manager, AWS CloudFormation, AWS Budgets, Model Context Protocol, Python, FastAPI, NumPy

## Special awards
- **Best Use of COOL:** COOL executes the whole perception workload on Graviton4. Runtime proof is at `/api/runtime` and in the console's runner badge; the like-for-like benchmark is in `bench/` and `results/bench/`.
- **Agentic Vision:** a perception→decision→action loop in which OpenCV 5 output changes the next tool call (see the trace rows "Re-planned from what it saw"). Includes a guard, fallback, observability and human approval.
