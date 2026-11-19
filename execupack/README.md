# ExecuPack experiments

The measurements behind [ExecuPack](https://github.com/ExperimentalMachines/execupack), the pipeline that exports open-weight models to
ExecuTorch for XNNPACK, Vulkan, Qualcomm's QNN and MediaTek's NeuroPilot, and publishes them.

## Layout

- [`research/`](research/): the research log, `README.md`, one row per finding with its
  evidence, and `evidence/` with the logs, reports and reproducer scripts each row cites. Read
  from the pushed branch `executorch-1.6` (b5affc76), `origin/main` plus two commits (the
  ExecuTorch 1.6.0 export upgrade and finding 48);
  finding 49 and its write-up `evidence/mtk-a16w4-gptq.md` were never committed and come from
  the `mtk-a16w4` worktree.
- [`harness/codecheck/`](harness/codecheck/): byte-level checks that a `.pte` holds the int4
  codes its export report names. Not wired into the pipeline or CI.
- [`studies/`](studies/): the experiments that are not, or not yet, a finding.

| Study | What | Where it came from |
|---|---|---|
| [`awq`](studies/awq/) | AWQ beside GPTQ on the same int4 grid | the local-only branch `awq-compare` (as a patch) and its gitignored `work/awq` |
| [`mtk-a16w4`](studies/mtk-a16w4/) | MediaTek A16W4 against A16W8, with and without GPTQ (finding 49) | the `mtk-a16w4` worktree: uncommitted edits (as a patch), scripts and gitignored results |
| [`calibration-corpus`](studies/calibration-corpus/) | A larger and differently composed GPTQ calibration corpus | the pushed `experiment/extended-calibration*` and `publish/lfm12b-gptq257` branches, as patches |
| [`published-audit`](studies/published-audit/) | A workflow to gate the published files nobody had gated | the pushed `experiment/audit-published` branch, as a patch |
| [`aws-runs-2026-11`](studies/aws-runs-2026-11/) | What the four EC2 export boxes held when they were shut down | `~/execupack-aws-archive-2026-11-09`, text only |

## What stayed in ExecuPack, and why

- `pipeline/`, its tests, `config/`, `requirements/` and CI: the product.
- `docs/PLAN.md`: a roadmap with run notes, not a record.
- `.github/workflows/probe-runner.yml` and `audit-published.yml`: measurement workflows, but a
  workflow only runs from `.github/`.
- The pipeline's code, README and `config/` cite `docs/research` by path and finding number, so
  that folder has to stay there until those citations point here.

Running a script under `research/evidence/` or `harness/codecheck/` that imports `pipeline`
needs an ExecuPack checkout on `PYTHONPATH` or in `EXECUPACK`, as each script's docstring says.
The `.patch` files made by `git format-patch` apply to an ExecuPack checkout with `git am` at
the base their README names; `mtk-a16w4/uncommitted.patch` is a plain diff, for `git apply`.

## What did not come across

Model files, tokenizers, Hugging Face downloads and caches, solve and export working folders,
vendor license copies and the boxes' shell dotfiles (2,397 files, each listed as `# excluded` in
[`PROVENANCE.tsv`](PROVENANCE.tsv) with its reason), the archive's 17 GB of
GPTQ solves (`codes/`) and its `inventory/` (account, instance, network and key records). Both
stay in `~/execupack-aws-archive-2026-11-09` on the Mac. Redacted on the way: the archive
README's AWS account and instance ids, and the boxes' private addresses in six fetch logs.
