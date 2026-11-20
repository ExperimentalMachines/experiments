# Published-file audit

**A workflow that re-grades files already on Hugging Face, published before any quality gate
existed, against the full-precision models they came from.**

`written up here, in part` `1 machine` `21 Sep to 9 Nov 2026`

> **The question.** A [smoke test](../../../docs/glossary.md#smoke-test) only proves a file
> generates text, which a file with quietly rounded weights also does. Do the files ExecuPack
> published before it had a [decision gate](../../../docs/glossary.md#decision-gate) still make
> the same [tool-call](../../../docs/glossary.md#tool-call) decisions as their full-precision
> source, checked without rebuilding or republishing them?

> [!TIP]
> **What we found**
> - The one audit kept here, of the published Qwen3 0.6B 4-bit CPU file at 2,048 tokens,
>   **failed**: its mean probability on the original's first token was **0.0857 against 0.6189**,
>   it kept 0 of 4 graded tool-call rows, and only 2 of 4 quiet rows answered like the original.
> - That result is research finding 43's "October GPTQ file". Qwen3 0.6B now ships full-precision
>   linear layers on the CPU and no GPU folder.
> - The commit that added the workflow reports two earlier audits, SmolLM2 135M at 0.190 and
>   SmolLM2 360M at 0.221 against a full-precision mean near 0.58; no log of those two is in this
>   repository.

## Key numbers

| What | Result | Where |
|---|---|---|
| Gate mean, published file against the original | **0.0857** against **0.6189** (shortfall 0.5332) | Qwen3 0.6B, GPTQ 4-bit CPU file, 2,048 tokens |
| Tool-call rows kept | 0 of 4 graded | the same file |
| Quiet rows answered like the original | 2 of 4 | the same file |
| Worst quiet row, "What is 8 times 7?" | original 0.873, file 0.0334 | the same file |
| Earlier audits, from the commit message only | SmolLM2 135M 0.190, SmolLM2 360M 0.221, original near 0.58 | SmolLM2 files built 13 Sep 2026 |

## Why it matters

A gate at publish time protects only the files built after it; this workflow is how the earlier
ones get checked. The Qwen3 0.6B audit is what moved that model to full-precision linear layers.
For a developer: grade a quantized file against its own full-precision model's decisions, not
only on whether it generates. A file that still answers "Paris" can have lost its tool calls.

## What was tested

- **Machines.** The kept audit ran on one Amazon EC2 r7i.4xlarge box. The workflow targets an
  8 vCPU x86 Blacksmith runner, because the gate runs ExecuTorch's C++ runner, which fails on ARM
  (research finding 30). No phone. See [all devices](../../../docs/devices.md).
- **Runtimes.** ExecuTorch's C++ runner, driven through ExecuPack's audit command.
- **Models.** Qwen3 0.6B (the kept audit); SmolLM2 135M and 360M (earlier audits, numbers only in
  the commit message). The workflow was written for files built on 13 September 2026 with
  round-to-nearest 4-bit weights and the host-dependent embedding search of finding 27.
- **Method.** For each source model, work out the full-precision model's choices on the gate's
  prompts, download the published file, and grade it the same way. Nothing is published.
- **When.** The workflow was written on 21 September 2026; the kept audit ran on 9 November 2026.

## What's in this folder

| Run | What it holds |
|---|---|
| [`audit-published.patch`](audit-published.patch) | The pushed branch `experiment/audit-published`, one commit on top of `experiment/extended-calibration-app` (2e8c973f), adding `.github/workflows/audit-published.yml`. ExecuPack's main also carries the workflow. 1 file. |
| [`../aws-runs-2026-11/P1/runs/audit-qwen3-06b.log`](../aws-runs-2026-11/P1/runs/audit-qwen3-06b.log) | The Qwen3 0.6B audit's full output, ending in the gate's JSON, kept with the EC2 box that ran it. 1 file, 132 lines. |

The patch is a `git format-patch` file; the log is console output whose last block is the gate
result, one entry per prompt with the original's and the file's probability.

## Run it again

You need an ExecuPack checkout: the `pipeline` package is not in this repository. The workflow is
on ExecuPack's main, or `git am` the patch onto `experiment/extended-calibration-app` (2e8c973f).
It needs the `HF_TOKEN` repository secret and a Blacksmith x86 runner; locally, x86 Linux and the
full-precision source model from Hugging Face.

```sh
# Locally, one model (finding 43 used the first form)
python -m pipeline audit Qwen/Qwen3-0.6B --context 2048
python -m pipeline audit <model> --work audit-work [--context <window>]

# Or start the "Audit published" workflow by hand, with inputs
#   models   space-separated source ids, e.g. "Qwen/Qwen3-0.6B meta-llama/Llama-3.2-1B-Instruct"
#   context  optional window
```

## Caveats

> [!NOTE]
> - The sources disagree on the batch the workflow was written for: the commit says "Nine XNNPACK
>   repositories", the workflow comment "Twelve models".
> - The SmolLM2 numbers exist only in the commit message.
> - No record here says whether the other published files were audited.
> - Only 4 of the 12 tool-call rows were graded: the gate skips rows where the full-precision
>   model is not confident.

## Read more

- Finding 43 in the [research log](../../research/README.md) cites this audit.
- Related: [the EC2 export boxes](../aws-runs-2026-11/) (where it ran, and the rollout it fed),
  [calibration corpus](../calibration-corpus/) (the branch this one sits on), and the byte-level
  checks of published files in [codecheck](../../harness/codecheck/).
- Words used here: [glossary](../../../docs/glossary.md).
