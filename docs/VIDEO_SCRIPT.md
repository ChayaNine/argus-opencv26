# Video script (4:50, hard limit 5:00)

The rules require the video to show **the team, the application working, its architecture, and its principal results**. Each of those has its own segment below. Record at 1080p.

Before recording:

- Open four browser tabs: the CloudFront URL (Rounds page), the app's Evidence page, the CloudWatch dashboard, and your email inbox.
- Pre-run one *Hard* round with the LLM agent, so a work order and a review are waiting in case the live round is slow.
- Use seed `3` on *Hard*: it reliably shows glare, night spotlight glare, flutter, an alarm and an escalation.

Recording tips: on macOS use QuickTime screen recording plus the camera; on Windows use Xbox Game Bar (Win+G) or OBS. Speak slowly; the script below is about 600 words, which fits.

---

## 0:00–0:20 Team and the one-line pitch (camera on you)

> "Hi, I'm ___ [and this is ___]. This is ARGUS: an inspection robot's eyes and judgement for analog gauges, built on OpenCV 5 running on AWS Graviton with the Cloud-Optimized OpenCV Library, and an agent on Amazon Bedrock that decides what to look at next."

## 0:20–0:50 The problem (screen: a glare capture from the timeline, then the naive result row in Evidence)

> "Plants still have thousands of analog gauges that people walk around and read. Robots can photograph them, but a single photo fails silently: glare on the glass, pump vibration, night-time noise, oblique mounting. In our ground-truth test, a robot that takes one frame and trusts it recorded a wrong value for almost a third of gauges, and missed 15 of 69 alarms and broken gauges. The dangerous part is that nobody notices."

## 0:50–2:40 Live demo (screen: the CloudFront URL, full screen)

1. **Runner badge (5 s).** Point at the badge at the top right.
   > "This is the deployed app on CloudFront. The badge is read from the running process: OpenCV 5, the COOL build, on a Graviton4 c8g.xlarge."
2. **Run a round:** green *Run round* button, *Hard conditions*, *LLM agent on Amazon Bedrock*, seed 3, *Run round* (10 s).
   > "It's laid out like a CI run: the robot walks a pump skid at night, and the page follows it gauge by gauge."
3. **First re-planned step (30 s).** Point at the yellow *Warning:* lines and the evidence thumbnail (magenta = clipped glare), then the step labelled *Re-planned from what it saw*.
   > "OpenCV rectifies the dial from its fiducial tag, finds the needle in polar coordinates and checks the printed scale. It also reports *glare on the needle*. So the agent doesn't trust it. Glare is a mirror reflection, so it moves the robot to the left view. Here's the new capture: clean, confidence 1.0, recorded."
4. **Spotlight row (15 s).**
   > "At night it turned on its spotlight, and its own light caused glare. It noticed, turned it off, and averaged more frames instead."
5. **Flutter row (15 s).**
   > "This pump gauge is fluttering. Three frames disagreed, so it took nine fast-shutter frames and used the midrange, which is the right estimator for an oscillating needle. That flutter is itself flagged as pulsation."
6. **Alarm (15 s).** Show the *Drafted work order* step on the alarm gauge.
   > "This one is in alarm. The safety guard won't let the agent close the gauge without a work order, and it enforces a minimum priority. The agent can draft it, but only a person can dispatch it."
7. **Human in the loop (20 s).** On the round summary, click *Review work orders*, then *Approve and dispatch*, then switch to the email tab and show the SNS email arriving. Then click *Review readings* to show the escalated gauge with its evidence.
   > "When it can't get trustworthy evidence within its budget, it escalates with the images instead of guessing."
8. **Trace (10 s).** Click *Show model calls* in a step log, then *Trace (JSON)* in the sidebar.
   > "Every perception result, decision, guard check and human action is logged here and in CloudWatch."

## 2:40–3:20 Architecture (screen: How it works page, then the Runner dialog, then the CloudWatch dashboard)

> "CloudFront terminates HTTPS and is the only thing that can reach the Graviton4 instance. The instance runs from the COOL Marketplace AMI, and you can see the proof at runtime: cv2 loads from /opt/cool, OpenCV 5, Neoverse-V2. The agent talks to Bedrock through the Converse API with tool use. If Bedrock fails, the gauge is handed to a deterministic policy with the same tools and guard. Evidence goes to S3, every decision to CloudWatch Logs and metrics, and approved work orders to SNS. There's no SSH: deployment, benchmarks and evaluation run through Systems Manager, all from one CloudFormation template."

## 3:20–4:20 Results (screen: Evidence page, then Read a photo)

> "We measured against simulator ground truth across glare, night, vibration, oblique, occlusion, haze and instrument faults."

Point at the table:

> "Naive automation: 32% silent failures and 15 missed alarms or broken gauges. ARGUS: zero silent failures, zero missed, zero false work orders. It reads 92% of gauges without a person, 99.6% of its recorded readings are within one percent of span, and its plus-or-minus two-sigma interval contains the truth 98% of the time." *(These are the 60-round local numbers; use the AWS run's numbers if you re-ran the evaluation there.)*

Then the COOL table:

> "On the same c8g.xlarge, COOL runs our full pipeline ___× faster than stock OpenCV 5, which is ___ dollars per thousand gauge reads." *(Fill in from `results/bench/compare.md`.)*

Then *Read a photo*, uploading one real gauge photo (15 s):

> "And it reads a real photo with no tag, or tells you how to retake it."

## 4:20–4:50 Limitations and close (camera on you)

> "Limits: our numbers come from a physics-based simulator with ground truth; real deployment needs a camera adapter and real-site validation, and the photo mode shows where we are. ARGUS is advisory: alarms still belong to the plant's control and safety systems. Thanks for watching."

---

Upload as **Unlisted** on YouTube, or as a public Vimeo link, and paste the link into Devpost.
