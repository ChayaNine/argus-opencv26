# Submission checklist and timeline

**Deadline:** Monday Oct 26, 11:59 p.m. Pacific = **Tuesday Oct 27, 2:59 p.m. Taipei time.**
Aim to submit by **Tuesday 10:00 Taipei**, leaving a five-hour buffer.

## Timeline (Taipei time)

| When | What | Done when |
|---|---|---|
| Sat evening | Step 0 of `DEPLOY.md`: quota, COOL subscription and AMI ID, Bedrock access | AMI ID copied; Haiku 4.5 accessible |
| Sat evening | `./deploy/deploy.sh --ami ... --email ...`, then confirm the SNS email | URL opens, header shows **COOL build** |
| Sun morning | `check_bedrock.sh`; run one LLM round on *Hard*; approve a work order and receive the email | timeline shows "Agent:" rationales |
| Sun | `run_bench.sh`, `bench_x86.sh`, `run_eval.sh --llm-scenes 8` | Evidence tab shows COOL table and the LLM column |
| Sun | Push the repo to GitHub (private repos must be shared with judges; see Devpost rules) | link works in a private window |
| Sun evening | `python3 docs/build_docs.py` refills `README.md` and `docs/REPORT.md` from `results/` (benchmark + Bedrock numbers); copy the same numbers into `docs/DEVPOST.md`; commit and push | `grep -rn TODO-AWS README.md docs/REPORT.md docs/DEVPOST.md` prints nothing |
| Mon | Record the video with `docs/VIDEO_SCRIPT.md`; upload unlisted | under 5:00, team on camera |
| Mon evening | Export the report to PDF (open `docs/REPORT.md` on GitHub, then the browser's Print, then Save as PDF); fill Devpost; **submit a complete draft** | Devpost shows "Submitted" |
| Tue morning | Final checks below; fix only typos | done by 10:00 |

## Devpost fields

- [ ] Title: **ARGUS: operator rounds that know when not to trust their own eyes**
- [ ] Tagline, description, "built with": from `docs/DEVPOST.md`
- [ ] Opt into **Best Use of COOL** and **Agentic Vision** (both qualify; see the rubric map in `README.md`)
- [ ] Working URL (CloudFront) and the line "no login needed; LLM rounds are rate-limited for cost"
- [ ] Repository link (public, or private with judge access)
- [ ] Video link (YouTube unlisted or Vimeo), under 5 minutes
- [ ] Technical report PDF (`docs/REPORT.md` exported), plus images: `docs/img/architecture.png`, `docs/img/agent_workflow.png`, `results/eval/final/strategies.png`, `results/bench/bench.png`, and a console screenshot

## Final checks (Tuesday morning)

- [ ] Open the URL in a private window **on a phone**. Start a *Hard* round with the LLM agent; it completes.
- [ ] `/api/runtime` shows `"cool": true`, `opencv_version` 5.x, `arm_core` Neoverse-V2, `instance-type` c8g.xlarge.
- [ ] Evidence tab shows both the evaluation and the COOL benchmark.
- [ ] The CloudWatch dashboard has data and the budget email is confirmed.
- [ ] The repo README rubric map links resolve.
- [ ] Leave the stack running until **November 9** (judging ends); tear it down on November 10.

## Honesty rules we follow (the judges can disqualify misrepresentation)

- Accuracy numbers come from the simulator with ground truth; say so every time.
- COOL numbers come only from `run_bench.sh` on the deployed instances. Report them as measured, even if the speed-up is small.
- The LLM agent's numbers come only from `run_eval.sh` with Bedrock. If it is not better than the policy on some metric, say so; its value is also in explanations, work-order text and handling the open cases.
- Human review is real in the UI. Escalations are not counted as automated readings.
