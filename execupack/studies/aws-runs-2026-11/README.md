# The EC2 export boxes, November 2026

**What four rented Amazon EC2 machines built when ExecuPack re-exported its published models with
GPTQ-solved 4-bit weights, and how each file was checked before it went public.**

`archive` `4 machines` `31 Oct to 9 Nov 2026`

> **The question.** This is the record of a production rollout, not a designed experiment. When
> ExecuPack rebuilt its published phone models with [GPTQ](../../../docs/glossary.md#gptq) 4-bit
> weights for the CPU ([XNNPACK](../../../docs/glossary.md#xnnpack)) and the GPU
> ([Vulkan](../../../docs/glossary.md#vulkan)) at up to five
> [context windows](../../../docs/glossary.md#context-window), did each file keep the
> full-precision model's [tool-call](../../../docs/glossary.md#tool-call) decisions, and does
> every published file hold exactly the 4-bit weights its report claims?

> [!TIP]
> **What we found**
> - Most of the 16 models kept the original's decisions on the
>   [decision gate](../../../docs/glossary.md#decision-gate) (Qwen3 4B 12 of 12, mean 0.9999
>   against 0.9999), but **Qwen3 0.6B at 4-bit kept 0 of 4** tool-call rows at every window (mean
>   0.0156 against 0.6189). It shipped with full-precision linear layers instead, at 5.13 times
>   the file size.
> - Phi-4-mini kept 7 of 8 and was published only under a recorded operator override; plainly
>   [rounded](../../../docs/glossary.md#round-to-nearest-rtn), it kept **0 of 8**.
> - **All 120 pre-publish byte checks** matched the GPTQ solve exactly, and a later scan of 169
>   published files found every checksum equal to the Hub's and the report's.
> - A one-token gate is blind on reasoning models: LFM3's first token with reasoning on was "We"
>   on every search row, and the published Qwen3 4B Thinking file, given 600 tokens to reason,
>   reached a tool call on only 6 of 12 search questions.

## Key numbers

| What | Result | Where |
|---|---|---|
| Gate, GPTQ 4-bit file against the original | mean **0.0156** against 0.6189, 0 of 4 tool-call rows | Qwen3 0.6B, CPU, 2,048 tokens, ExecuTorch 1.5.1 |
| The same gate, full-precision linear layers | 0.6177 against 0.6189, 4 of 4 | Qwen3 0.6B, CPU, 2,048 tokens |
| What that fix costs in size | 2,545,178,112 B against 496,570,368 B (5.13x) | Qwen3 0.6B, 2,048 tokens |
| Plain rounding, no solve | 0 of 8, mean 0.0492 against the original's 0.6629 | Phi-4-mini, CPU, 2,048 tokens |
| GPTQ with an 8-bit output head, as published | 7 of 8, mean 0.5066 (2k) and 0.6956 (8k); gate failed, published under override | Phi-4-mini, CPU |
| A reasoning model allowed 600 tokens | 6 of 12 search rows reached a tool call; 0 of 4 quiet rows called | Qwen3 4B Thinking 2507, published 4k CPU file |
| Pre-publish byte checks | 120 of 120 matched the solve exactly | every GPTQ rollout file |
| Later scan of published files | 169 files in 20 repos, every checksum equal to the Hub's and the report's | CPU and GPU files on Hugging Face |

## Why it matters

A 4-bit file that still writes fluent text can have lost the model's decision to call a tool, so
"it runs" is not a release test. The gate plus the byte check is what kept the broken Qwen3 0.6B
file off the Hub and put a full-precision one there instead. For a developer exporting small
models: measure the quantized file's decisions against the unquantized model, model by model,
because the damage does not follow size (SmolLM2 135M kept 4 of 4 where Qwen3 0.6B kept 0 of 4).
These runs are the evidence behind research findings 42 to 45.

## What was tested

- **Machines.** Four Amazon EC2 boxes: two r7i.4xlarge on demand and two r7i.8xlarge on
  persistent spot, all reporting an Intel Xeon Platinum 8488C, with 16 and 32 CPUs. Every smoke
  test and gate ran on the boxes; a Mac ordered the jobs. No phone. See
  [all devices](../../../docs/devices.md).
- **Runtimes.** [ExecuTorch](../../../docs/glossary.md#executorch) 1.5.1 on 165 of 166 export
  reports, with torch 2.14.0 and torchao 0.18.0; one LFM3 build on ExecuTorch 1.6.0.
- **Models.** Sixteen: SmolLM2 135M and 360M; Qwen2.5 0.5B, 1.5B and 3B, and Qwen2.5 Math 1.5B;
  Qwen3 0.6B, 1.7B and 4B, Qwen3 4B Instruct 2507 and Thinking 2507; Llama 3.2 1B and 3B;
  Phi-4-mini; LFM3 1.7B and its base model.
- **Method.** One GPTQ solve per model, then an export per window (2k to 32k) for each backend. A
  file was published only after a [smoke test](../../../docs/glossary.md#smoke-test), the
  decision gate against a full-precision reference computed on Linux, and, for a solved file, a
  byte check that every 4-bit layer equals the solve (120 checks, all matched). A GPU file cites
  the gated CPU file of the same window. Of 166 export reports, 155 are GPTQ, 5 round to nearest
  and 6 full precision; a round-to-nearest file had no solve to check against, and at least one
  (the Qwen3 4B Thinking GPU file) was published.
- **When.** LFM3 on 31 October 2026; a Qwen2.5 1.5B re-export overnight into 9 November; the
  rollout on 9 November 2026, 00:49 to 09:49 UTC. The boxes were shut down that day.

## What's in this folder

Box P1 (r7i.4xlarge):

| Run | What it holds |
|---|---|
| [`P1/runs/fp32-qwen3-06b/`](P1/runs/fp32-qwen3-06b/) | Qwen3 0.6B with full-precision linear layers, CPU, 2k to 32k: gate 4 of 4 at every window, all five published. 15 files. |
| [`P1/runs/gptq-qwen3-06b/`](P1/runs/gptq-qwen3-06b/) | Qwen3 0.6B GPTQ, CPU, 2k to 16k: every window failed the gate, none published (finding 43). 30 files. |
| [`P1/runs/gptq-llama-32-1b-instruct/`](P1/runs/gptq-llama-32-1b-instruct/) | Llama 3.2 1B, both backends, five windows: gate 12 of 12, 10 published. 121 files. |
| [`P1/runs/gptq-qwen25-math-15b-instruct/`](P1/runs/gptq-qwen25-math-15b-instruct/) | Qwen2.5 Math 1.5B: the solve plus the 8k window, gate 3 of 3, 2 published. 25 files. |
| [`P1/runs/gptq-smollm2-360m-instruct/`](P1/runs/gptq-smollm2-360m-instruct/) | SmolLM2 360M: the CPU files passed 2 of 2 and stayed unpublished as the GPU files' gate proxies; the GPU files were published by hand. 86 files. |
| [`P1/runs/phi4-mini/`](P1/runs/phi4-mini/) | Phi-4-mini: rounded GPU files published then withdrawn, GPTQ 4-bit failing the gate, GPTQ with an 8-bit head published at 4k, 8k and 16k under override, and the one-off checks' outputs. 89 files. |
| [`P1/runs/gptq-setup/`](P1/runs/gptq-setup/) | Toolchain setup. 3 files. |
| [`P1/runs/audit-qwen3-06b.log`](P1/runs/audit-qwen3-06b.log) | The audit of the published Qwen3 0.6B file (see [published-file audit](../published-audit/)). 1 file. |
| [`P1/codecheck/`](P1/codecheck/) | The byte-check scripts as run, and the first and third scans of the Hub's files. 32 files. |

Box P3 (r7i.8xlarge):

| Run | What it holds |
|---|---|
| [`P3/runs/gptq-llama-32-3b-instruct/`](P3/runs/gptq-llama-32-3b-instruct/) | Llama 3.2 3B, both backends, five windows: gate 12 of 12, 10 published. 121 files. |
| [`P3/runs/gptq-qwen25-math-15b-instruct/`](P3/runs/gptq-qwen25-math-15b-instruct/) | Qwen2.5 Math 1.5B at 2k, 4k, 16k and 32k: gate 3 of 3, 8 published. 79 files. |
| [`P3/runs/gptq-qwen3-4b/`](P3/runs/gptq-qwen3-4b/) | Qwen3 4B at 32k on both backends: gate 12 of 12, 2 published. 21 files. |
| [`P3/runs/gptq-qwen3-4b-instruct-2507/`](P3/runs/gptq-qwen3-4b-instruct-2507/) | Qwen3 4B Instruct 2507, both backends, five windows: gate 12 of 12, 10 published. 101 files. |
| [`P3/runs/phi4-mini/`](P3/runs/phi4-mini/) | Phi-4-mini at 2k and 32k, both backends, under override: 4 published. 40 files. |
| [`P3/runs/gptq-setup/`](P3/runs/gptq-setup/) | Toolchain setup. 3 files. |
| [`P3/think/`](P3/think/) | Per-question results of the Qwen3 4B Thinking check. 1 file. |
| [`P3/codecheck/`](P3/codecheck/) | Shards of the third Hub scan. 43 files. |

Box Q1 (r7i.4xlarge):

| Run | What it holds |
|---|---|
| [`Q1/runs/lfm3/`](Q1/runs/lfm3/) | LFM3 1.7B GPTQ, CPU, five windows: stopped by hand, re-gated with reasoning off, gate 12 of 12, 5 published. 38 files. |
| [`Q1/runs/lfm3-base/`](Q1/runs/lfm3-base/) | LFM3 1.7B Base, five windows: gate 12 of 12 without measuring tool calling, 5 published. 36 files. |
| [`Q1/runs/et16/`](Q1/runs/et16/) | LFM3 1.7B at 2k, rounded, on ExecuTorch 1.6.0: the smoke test answered "Paris"; unpublished. 3 files. |
| [`Q1/runs/fp32test/`](Q1/runs/fp32test/) | LFM3 1.7B in full precision at 2k: smoke test passed; unpublished. 3 files. |
| [`Q1/runs/qwen25-15b/`](Q1/runs/qwen25-15b/) | Qwen2.5 1.5B CPU GPTQ re-export, 5 of 5 published. 36 files. |
| [`Q1/runs/qwen3-4b-thinking/`](Q1/runs/qwen3-4b-thinking/) | Qwen3 4B Thinking: GPTQ CPU files at 4k, 8k and 16k and rounded GPU files at 8k and 16k published; 32k refused by the size gate. 47 files. |
| [`Q1/runs/gptq-qwen25-05b-instruct/`](Q1/runs/gptq-qwen25-05b-instruct/) | Qwen2.5 0.5B, both backends, five windows: gate 9 of 9, 10 published. 101 files. |
| [`Q1/runs/gptq-qwen25-15b-instruct/`](Q1/runs/gptq-qwen25-15b-instruct/) | Qwen2.5 1.5B, both backends, five windows: gate 12 of 12, 10 published. 101 files. |
| [`Q1/runs/gptq-qwen25-3b-instruct/`](Q1/runs/gptq-qwen25-3b-instruct/) | Qwen2.5 3B at 2k to 16k, both backends: gate 12 of 12, 8 published. 79 files. |
| [`Q1/runs/gptq-qwen3-17b/`](Q1/runs/gptq-qwen3-17b/) | Qwen3 1.7B's solve fetched from the box that made it; nothing built. 3 files. |
| [`Q1/runs/gptq-qwen3-4b/`](Q1/runs/gptq-qwen3-4b/) | Qwen3 4B at 8k on both backends: gate 12 of 12, 2 published. 22 files. |
| [`Q1/runs/gptq-smollm2-135m-instruct/`](Q1/runs/gptq-smollm2-135m-instruct/) | SmolLM2 135M, both backends, five windows: gate 4 of 4, 10 published. 101 files. |
| [`Q1/runs/gptq-setup/`](Q1/runs/gptq-setup/) | Toolchain setup. 3 files. |
| [`Q1/work/`](Q1/work/) | Parameters for the LFM3 check. 2 files. |
| [`Q1/codecheck/`](Q1/codecheck/) | Shards of the third Hub scan. 16 files. |

Box Q3 (r7i.8xlarge):

| Run | What it holds |
|---|---|
| [`Q3/runs/gptq-qwen3-17b/`](Q3/runs/gptq-qwen3-17b/) | Qwen3 1.7B, both backends, five windows: gate 12 of 12, 10 published. 101 files. |
| [`Q3/runs/gptq-qwen3-4b/`](Q3/runs/gptq-qwen3-4b/) | Qwen3 4B: the solve plus 2k, 4k and 16k, gate 12 of 12, 6 published. 64 files. |
| [`Q3/runs/gptq-qwen25-3b-instruct/`](Q3/runs/gptq-qwen25-3b-instruct/) | Qwen2.5 3B: the solve plus 32k on both backends, 2 published. 25 files. |
| [`Q3/runs/gptq-qwen3-4b-thinking-2507/`](Q3/runs/gptq-qwen3-4b-thinking-2507/) | Qwen3 4B Thinking rebuilt with GPTQ on both backends at five windows (the earlier GPU files were rounded): 10 published. 99 files. |
| [`Q3/runs/qwen3-4b-thinking/`](Q3/runs/qwen3-4b-thinking/) | The 32k window Q1 could not hold: 2 published. 22 files. |
| [`Q3/runs/gptq-setup/`](Q3/runs/gptq-setup/) | Toolchain setup. 3 files. |
| [`Q3/codecheck/`](Q3/codecheck/) | The second Hub scan, two shards. 14 files. |

1,761 files in all. Each run keeps its logs (`.log`), export and verify reports and gate results
(`.json`), exit codes (`.rc`), standard error (`.err`), and empty `.ok` and `published-*`
markers; byte-check results are JSON lines, one per scanned file. Each box's top level holds the
job and setup scripts as they ran and the one-off checks (`expt.py`, `parity.py`, `rtn.sh`,
`think_check.py`, `basegate.py`, `regate.py`, `lfm3_check.py`) with their outputs.
[`ARCHIVE.md`](ARCHIVE.md) is the archive's own README, written the day the boxes were shut down.

## Run it again

Not from this repository alone. Missing: the Mac-side `driver.py` that ordered the jobs; `codes/`,
the 17 GB of GPTQ solves needed to rebuild any export, and `inventory/`, both still in
`~/execupack-aws-archive-2026-11-09` on the Mac; and the model weights, tokenizers and virtualenvs. You need an x86 Ubuntu 24.04
machine with about 128 GB of RAM for most windows and about 256 GB for the 4B models at 32k, an
ExecuPack checkout at `~/execupack-main2` with `codecheck/` inside, and a Hugging Face write
token in `~/.config/execupack/hf.env` to publish. The scripts are kept exactly as they ran and
`cd` into that checkout.

```sh
# On a box: set up once, then one job per step (kinds: setup, solve, xnn, vk, vkc, parity, fetch).
# job.sh is P1/job2.sh here. ACCEPT_GATE, ALLOW_UNUSED and NO_PUBLISH change what a job accepts
# and whether it publishes.
nohup bash ~/job.sh <id> setup <model> <rev> <runs> &
nohup bash ~/job.sh <id> <kind> <model> <rev> <runs> [window] &

# What the jobs run inside the ExecuPack checkout
python -m pipeline solve "$MODEL" --revision "$REV" --out "$RUNS/codes.pt" --work "$RUNS/solve-work"
python -m pipeline export-xnnpack ... --context "$W" --codes "$RUNS/codes.pt" --skip-smoke
python -m pipeline verify "$out" --backend xnnpack --work ... --gate "$RUNS/gate.json"
python codecheck/codesmatch.py <pte> codes.pt
python -m pipeline publish-hf "$out" --backend "$backend"

# One model unattended, with a dead-man shutdown after 480 minutes (NO_SHUTDOWN to keep the box)
MODEL=Qwen/Qwen2.5-1.5B-Instruct REV=<sha> RUNS=~/runs/qwen25-15b nohup bash ~/run-export.sh > ~/runs/qwen25-15b/queue.out 2>&1 &
nohup bash ~/run-lfm3.sh > ~/runs/lfm3/queue.out 2>&1 &

# The one-off checks
python expt.py <execupack checkout> <runs dir>
python think_check.py <execupack checkout> <work dir> [max_new_tokens]
python lfm3_check.py <work dir> [--pte <fp32 .pte>]

# Byte-check every published file in the org, in N shards
run_scan.sh <TAG> <N> [make_tasks filter args]
```

## Caveats

> [!NOTE]
> - An archive, not an experiment: the GPTQ folders are the production rollout's records, kept as
>   the evidence behind what was published.
> - The Qwen3 4B Thinking check counts a row as "called" if a tool call appears anywhere and as
>   "still thinking" if the reasoning never closed, so its 6 called and 7 still thinking overlap.
> - Phi-4-mini's eager-model experiment kept 8 of 8 with an 8-bit head, but the exported file
>   measured 7 of 8: the eager result did not carry over. Its parity check kept no output.
> - Phi-4-mini's rounded GPU files at 2k and 4k sit in a withdrawn folder; no log says why.
> - Gate references must be computed on Linux; a Mac's are wrong (research finding 44).
> - AWS account and instance ids in the archive README, and the boxes' private addresses in six
>   fetch logs, were redacted on import.

## Read more

- The archive's own README: [AWS export boxes, archived 2026-11-09](ARCHIVE.md).
- Findings 42 to 45 in the [research log](../../research/README.md) rest on these runs.
- The byte checks: [codecheck](../../harness/codecheck/).
- Related: [published-file audit](../published-audit/) and [AWQ beside GPTQ](../awq/).
- Words used here: [glossary](../../../docs/glossary.md).
