# The EC2 export boxes, November 2026

Four EC2 boxes ran ExecuPack's exports, the GPTQ rollout and a few checks until 2026-11-09, when
they were shut down. Before that, everything worth keeping was copied to the Mac;
[`ARCHIVE.md`](ARCHIVE.md) is the README written then, and describes the whole archive.

What is here is its text: for each box (`Q1`, `Q3`, `P1`, `P3`), the job and setup scripts, the
`runs/` folders with their logs, export and verify reports, gate results and job status files
(`.rc` exit codes, `.err` stderr, `.ok` and empty `published-*` markers), the byte checks under
`codecheck/results/`, and the one-off checks:

- `P1/expt.py`, `P1/parity.py`, `P1/rtn.sh` and their outputs.
- `P3/think_check.py` and `P3/think/`: the Qwen3-4B-Thinking check.
- `Q1/basegate.py`, `Q1/regate.py`, `Q1/lfm3_check.py` and their logs.

The `gptq-*` run folders are the production rollout's records rather than experiments, kept
because their reports and logs are the evidence behind what was published.

Not here, and still in `~/execupack-aws-archive-2026-11-09` on the Mac: `codes/` (17 GB of GPTQ
solves, the inputs to rebuild any export) and `inventory/` (account, instance, volume, network
and key records). Model files, tokenizers and downloads were left out; the AWS account and
instance ids in `ARCHIVE.md` and the boxes' private addresses in six fetch logs are redacted.
Scripts are kept exactly as they ran: they `cd` into an ExecuPack checkout on the box.
