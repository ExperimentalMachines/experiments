# On-device evals: parity, public benchmarks, exported windows

**Question.** Do the app's two runtimes, llama.cpp and ExecuTorch, give the same answers to the
same prompts on every chip; how do the shortlisted models score on GSM8K, IFEval and BFCL on each
phone; and does the context window a compiled model was exported with change anything.

**Write-ups** (in OpenWeights): [`backend-parity.md`](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/backend-parity.md), [`parity-five-socs.md`](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/parity-five-socs.md), [`executorch-families.md`](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/executorch-families.md), [`public-benchmarks.md`](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/public-benchmarks.md), [`benchmark-matrix.md`](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/benchmark-matrix.md), [`window-matrix.md`](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/window-matrix.md), [`executorch-window-matrix.md`](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/executorch-window-matrix.md), [`executorch-state-and-recipes.md`](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/executorch-state-and-recipes.md), [`first-turn-latency.md`](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/first-turn-latency.md).

**Runs.** `runs/` is the eval harness's results folder as it stood at OpenWeights 81734dee,
2026-08-31 to 2026-09-24, kept whole: `compare.py`, `bench/report.py` and `bench/window_report.py`
each read it as one pool and pick their own files by content, so splitting it would change what
they regenerate. A file name's prefix names the phone or lab (`elite-`, `exynos-`,
`exynos2500-`, `tensor-`, `qdc-`, `prod-`); `*.bench.json` and `*.bench.graded.json` are public
benchmark reports, `*-8da4w-<n>k*` the window matrix, the rest parity reports and their logs.

- `controls/`: the Gemma 3 loop control.
- `logcats/`: the Test Lab logcats, xz-compressed, cut to the app's own lines (see below).
- `invalid-nobos/`: window cells set aside because the tokenizer added no BOS token.
- `qwen3-32k-poco-rerun/`: the POCO's Qwen3 32k rerun.

A few files read by the same scripts belong to smaller questions: `*-probe*` and `*-speed*`
(core selection, 2026-09-03), `poco-pd-decode-probe.json` (the NPU decode probe, see
[`npu-pd-disaggregation.md`](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/npu-pd-disaggregation.md)), and the warm-prefix rerun's logcat.

**Harness.** [`harness/app-evals/`](../../harness/app-evals/): `run_matrix.sh` (adb) and
`run_matrix_ftl.sh` (Firebase Test Lab) for parity, `bench/` for the public benchmarks and the
window matrix, `cold_start_probe.sh` for the first turn.

Each renders a write-up; give `--out` a path in an OpenWeights checkout to refresh it there.
`compare.py` on this folder reproduces the published `backend-parity.md` line for line below its
header, and `bench/report.py` reproduces the benchmark matrix.

```sh
D=openweights/studies/on-device-evals/runs
python3 openweights/harness/app-evals/compare.py $D --out "$OPENWEIGHTS"/docs/research/backend-parity.md
python3 openweights/harness/app-evals/bench/report.py $D --out "$OPENWEIGHTS"/docs/research/benchmark-matrix.md
python3 openweights/harness/app-evals/bench/window_report.py $D \
  --sizes openweights/harness/app-evals/bench/window_sizes.json --out "$OPENWEIGHTS"/docs/research/window-matrix.md
```

**Changed on import.** The Test Lab logcats (`logcats/` and the top-level `*.logcat.xz`) were
whole devices' logs: every process, with network names and addresses and test accounts. Only
the app's lines came across ([`tools/logcat_filter.py`](../../../tools/logcat_filter.py)); the full
logs are in OpenWeights at [81734dee](https://github.com/ExperimentalMachines/openweights/tree/81734deed5048f5691518d0e739662c50e51e1a3/tools/eval/results). Two tokenizers that had been
committed into the folder (`elite-LFM2.5-2.6B-8da4w-{4k,32k}.tokenizer.json`) stayed out: they
are model files, and `compare.py` crashed reading them as reports.
