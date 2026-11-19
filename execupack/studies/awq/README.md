# AWQ beside GPTQ

**Question.** Does an AWQ solve on the same int4 grid, with its scaled norms carried through
the export, keep more of a model than GPTQ, alone or with GPTQ on top? Each is measured against
the fp32 reference, round to nearest, and the file already published.

**Status.** No finding yet. The code exists only as three commits on the branch `awq-compare`,
which was never pushed: [`awq-compare.patch`](awq-compare.patch) holds them
(`git am` onto `executorch-1.6`, b5affc76). They add `scripts/awq_compare.py`, `pipeline/awq.py`,
a resume path for a stopped reference, and a calibration option that renders rows with a
reasoning template's thinking off.

**Runs.** `runs/` is the worktree's gitignored `work/awq/` as it stood on 2026-11-19, text only:
per model (`lfm25-1.2b`, `qwen3-1.7b`, `llama32-1b`, `t135`), one folder per arm (`ref`, `rtn`,
`published`, `gptq`, `awq`, `awq_gptq`, and the Qwen no-thinking variants) with `result-*.json`,
`rows-*.json`, `gate.json` and logs; `analyze.py` makes the comparison from them;
`failed-spendlimit/` holds the runs a cloud spend limit stopped. `chain.sh` and
`qwen_nothink.sh` are kept exactly as they ran: they `cd` into the `execupack-awq` checkout.
