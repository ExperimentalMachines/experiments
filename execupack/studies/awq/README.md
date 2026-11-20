# AWQ beside GPTQ

**Whether a different way of choosing 4-bit weights, AWQ, keeps more of a model than the GPTQ
solve ExecuPack publishes.**

`unpublished` `2 machines` `19 Nov 2026`

> **The question.** ExecuPack's phone CPU and GPU files store weights as 4-bit integers
> ([8da4w](../../../docs/glossary.md#8da4w)) chosen by a [GPTQ](../../../docs/glossary.md#gptq)
> solve. Would [AWQ](../../../docs/glossary.md#awq), which rescales weights by how strongly each
> input channel is used, keep more of the original model, alone or with GPTQ applied on top?
> Every method is measured against the full-precision model, against plain
> [rounding to nearest](../../../docs/glossary.md#round-to-nearest-rtn), and against the file
> already published.

> [!IMPORTANT]
> **What we found (unpublished, no finding yet)**
> - This is not a finding: the code sits on a branch that was never pushed and no row of the
>   research log cites it. The numbers below are read from the stored results.
> - **No method wins everywhere.** On LFM2.5 1.2B, AWQ is clearly worse than GPTQ (held-out
>   [KL](../../../docs/glossary.md#kl-divergence) **0.285 against 0.117**) and fails the
>   [publish gate](../../../docs/glossary.md#decision-gate) with 0 of 12 tool-call rows kept, as
>   plain rounding does. On Qwen3 1.7B AWQ is better (0.504 against 1.056). On Llama 3.2 1B AWQ
>   alone is worse (0.411 against 0.142), but AWQ followed by GPTQ is the best arm (**0.072**).
> - On Qwen3 1.7B, calibrating GPTQ on prompts rendered with reasoning switched off, as the app
>   sends a turn, brought it to **0.464**, not distinguishable from AWQ: how the calibration rows
>   were formatted moved quality about as much as the algorithm.
> - The ordinary GPTQ build of Qwen3 1.7B, like the published file, kept 0 of the 12 held-out
>   questions where the original model opens with a [tool call](../../../docs/glossary.md#tool-call)
>   (AWQ kept 7, GPTQ calibrated with reasoning off 5), yet it passed the publish gate 12 of 12.

## Key numbers

| What | Result | Where |
|---|---|---|
| Held-out KL, GPTQ / AWQ / AWQ then GPTQ | **0.117** / **0.285** / 0.119 | LFM2.5 1.2B, CPU file, 2,048 tokens, ExecuTorch 1.6.0 |
| AWQ on the publish gate | fails, 0 of 12 tool-call rows kept | LFM2.5 1.2B |
| Held-out KL, GPTQ / AWQ / GPTQ calibrated with reasoning off | 1.056 / 0.504 / 0.464 | Qwen3 1.7B |
| Held-out questions where the original opens with a tool call, kept | GPTQ 0 of 12 (published file also 0), AWQ 7 of 12, GPTQ with reasoning off 5 of 12 | Qwen3 1.7B |
| Held-out KL, GPTQ / AWQ / AWQ then GPTQ | 0.142 / 0.411 / **0.072** | Llama 3.2 1B |
| AWQ on Llama's gate | 12 of 12 rows agree, yet it fails: mean probability 0.505 against the original's 0.884 | Llama 3.2 1B |
| Plain rounding, held-out top-1 agreement | 0.064 (Llama), 0.383 (Qwen3) | round to nearest, no solve |
| Solve time, GPTQ / AWQ / AWQ then GPTQ | 815.5 / 1,096.5 / 2,438.2 s (LFM2.5); 1,014.7 / 2,979.7 / 2,540.2 s (Qwen3); 806.9 / 1,237.0 / 1,797.9 s (Llama) | cloud sandbox, 8 cores |

## Why it matters

Nothing in the product changed: the code lives only on an unpushed branch, and the published
files stay GPTQ. For a developer choosing a 4-bit method: no single algorithm wins across model
families, and AWQ's own per-layer objective (0.66 to 0.88 of plain rounding's error) did not
predict end-to-end quality. On a reasoning model, how the calibration rows are rendered mattered
as much as swapping GPTQ for AWQ. And a publish gate of twelve rows passed files that kept none
of the 12 held-out tool calls, so a gate is a floor, not proof.

## What was tested

- **Machines.** Modal cloud sandboxes of 40,960 MiB and 8 cores for the three real models; a
  Mac for the SmolLM2 plumbing test. No phone. See [all devices](../../../docs/devices.md).
- **Runtimes.** [ExecuTorch](../../../docs/glossary.md#executorch) 1.6.0 with torch 2.15.0 and
  torchao 0.18.0; [XNNPACK](../../../docs/glossary.md#xnnpack) (CPU) files at a 2,048-token
  [window](../../../docs/glossary.md#context-window).
- **Models.** LFM2.5 1.2B Instruct, Qwen3 1.7B and Llama 3.2 1B Instruct; SmolLM2 135M Instruct
  as a plumbing test with 6 held-out rows.
- **Method.** Every arm uses the same grid (symmetric 4-bit in groups of 32, bf16 scales, 8-bit
  per-token activations), the same 104 calibration rows and the same window, so only the 4-bit
  codes differ. Arms: round to nearest, the published GPTQ file, a fresh GPTQ solve, AWQ, AWQ then
  GPTQ, and for Qwen3 GPTQ calibrated with reasoning off. Each arm is scored by the publish gate,
  by first-token KL and agreement on 141 held-out questions, and by KL along the original's own
  replies, with paired 95% bootstrap intervals (4,000 resamples).
- **When.** 19 November 2026.

## What's in this folder

| Run | What it holds |
|---|---|
| [`runs/lfm25-1.2b/`](runs/lfm25-1.2b/) | LFM2.5 1.2B: the full-precision reference and one folder per arm (round to nearest, published, GPTQ, AWQ, AWQ then GPTQ), each with its result, per-question rows and export report, and a log per arm. 26 files. |
| [`runs/qwen3-1.7b/`](runs/qwen3-1.7b/) | Qwen3 1.7B: the same arms, plus the reasoning-off calibration rows, their reference and GPTQ arm, and the reference resumed after a spend-limit stop. 36 files. |
| [`runs/llama32-1b/`](runs/llama32-1b/) | Llama 3.2 1B: the same arms, with Meta's license and use policy beside each export. 34 files. |
| [`runs/t135/`](runs/t135/) | SmolLM2 135M plumbing test on the Mac: reference, round to nearest, published and AWQ. 15 files. |
| [`runs/failed-spendlimit/`](runs/failed-spendlimit/) | Four cloud runs a spend limit stopped, all redone. 4 files. |

Each arm's `result-*.json` holds the gate verdict and the KL summaries, `rows-*.json` the
per-question values the bootstrap resamples, and `export/` the export report. At the top of
`runs/` sit `analyze.py` (the comparison tables), `chain.sh` (one model on Modal),
`qwen_nothink.sh` (the Qwen3 reasoning-off arm), `simcheck.py` and `heldout.json` (the 141
held-out questions). [`awq-compare.patch`](awq-compare.patch) holds the code.

## Run it again

The code is three commits on the local branch `awq-compare`, never pushed, kept here as
`awq-compare.patch`: apply it with `git am` onto ExecuPack's `executorch-1.6` branch at b5affc76.
It adds `scripts/awq_compare.py`, `pipeline/awq.py`, a resume path for a stopped reference and a
calibration option that renders rows with a reasoning template's thinking off. `runs/` is the
worktree's gitignored `work/awq/` as it stood on 19 November 2026, text only; `chain.sh` and
`qwen_nothink.sh` are kept exactly as they ran. The tables
recompute from this repository alone; a full rerun needs a Modal account, a Hugging Face token
with access to Llama 3.2 and the published `.pte` files. The references' `rows.pt` and
`evalref.pt` were left out, so a new arm needs a new reference (27.3 and 28.8 minutes of sandbox
time each for the two that ran in full).

```sh
# The comparison tables, from the stored results (no models needed)
cd <experiments>/execupack/studies/awq/runs
python3 analyze.py lfm25-1.2b qwen3-1.7b llama32-1b t135

# A full rerun: the branch onto an ExecuPack checkout
cd <execupack checkout>
git switch -c awq-compare b5affc76
git am <experiments>/execupack/studies/awq/awq-compare.patch

# One model, every arm, on Modal. The scripts cd into a checkout named execupack-awq, read
# HF_TOKEN from the Hugging Face token cache and expect work/awq/heldout.json;
# METHODS limits the arms.
chain.sh <name> <model id> <revision> <published file>
chain.sh lfm25-1.2b LiquidAI/LFM2.5-1.2B-Instruct 0f604ada3f766f9f257460c4c9f0b5d6f69d431b \
    xnnpack/LFM2.5-1.2B-Instruct-8da4w-gptq-2k.pte

# The Qwen3 arm calibrated with reasoning off
qwen_nothink.sh
```

## Caveats

> [!NOTE]
> - Unpublished: the branch's third commit states the reasoning-off hypothesis, and no write-up
>   interprets the arm that tests it.
> - `analyze.py` does not print the reasoning-off arm. Its comparison with AWQ was derived for
>   this page with the same bootstrap: the interval, -0.3092 to +0.2032, spans zero.
> - KL here is over the original's top 64 tokens plus one bucket for the rest, a lower bound on
>   the full KL.
> - The SmolLM2 test has no GPTQ arm and only 6 held-out rows. Its AWQ-then-GPTQ arm failed on
>   the Mac with a Cholesky error, the same kind of Mac-only solve failure as research finding 44;
>   that the cause is the same is not shown.
> - Four runs stopped by a cloud spend limit end "Modal Sandbox exited 137"; they were redone.

## Read more

- No write-up yet. The comparison logic is [`analyze.py`](runs/analyze.py).
- Related: [MediaTek 4-bit weights](../mtk-a16w4/) (GPTQ on MediaTek's grid, the same three
  models), [calibration corpus](../calibration-corpus/), and findings 29 (why every 4-bit file is
  GPTQ-solved) and 44 (solves on a Mac) in the [research log](../../research/README.md).
- Words used here: [glossary](../../../docs/glossary.md).
