# Published-file audit

**Question.** Which already published files were never put through the decision gate?

**Source.** The pushed branch `experiment/audit-published`, one commit on top of
`experiment/extended-calibration-app` (2e8c973f), kept as
[`audit-published.patch`](audit-published.patch). It adds
`.github/workflows/audit-published.yml`, which ExecuPack's main also carries. One audit it led
to, of the published Qwen3-0.6B, ran on an EC2 box: `P1/runs/audit-qwen3-06b.log` in
[`aws-runs-2026-11`](../aws-runs-2026-11/).
