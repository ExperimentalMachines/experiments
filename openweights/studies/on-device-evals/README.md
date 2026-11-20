# On-device evals

**Whether OpenWeights' two runtimes give the same answers on different phones, how the models
they run score on public benchmarks, and what a compiled model's context window changes.**

`written up in OpenWeights` `6 phones` `31 Aug to 24 Sep 2026`

> **The question.** OpenWeights runs the same small open models through two runtimes:
> [llama.cpp](../../../docs/glossary.md#llamacpp) with [GGUF](../../../docs/glossary.md#gguf)
> files, and [ExecuTorch](../../../docs/glossary.md#executorch) with compiled
> [`.pte`](../../../docs/glossary.md#pte) files. With
> [greedy decoding](../../../docs/glossary.md#greedy-decoding), do the two answer the same
> prompts the same way, and does either answer the same way on every chip? How do five model
> families score on GSM8K, IFEval and BFCL on each phone? And does the
> [context window](../../../docs/glossary.md#context-window) a compiled model was exported with
> change anything besides memory?

> [!TIP]
> **What we found**
> - On the app's own seven-case suite llama.cpp graded identically on all five phones, but on
>   public benchmarks greedy decoding is not portable: the grade differs between five chips on
>   **13.3%** of 390 cases with llama.cpp and **31.5%** with ExecuTorch, while the same phone
>   rerun gave byte-identical replies.
> - The GGUF files scored higher on **12 of 14** comparable cells. That is a claim about the
>   publishers' files as shipped, not the runtimes alone: Gemma 3's compiled export loops (0.6 of
>   30 on GSM8K against 13.2), and SmolLM3's GGUF made 0 of 30 BFCL calls on every phone.
> - The exported window costs memory, not answers or speed, within run-to-run variation (a
>   screen, not a proof): Qwen3 1.7B pays **224 KB a token**, and its 32k export was killed by a
>   6 GB per-app limit on three of four phones.
> - [Prefilling](../../../docs/glossary.md#prefill-and-decode) the fixed prompt head at load cut a
>   fresh chat's [first token](../../../docs/glossary.md#time-to-first-token) from **9,326 ms to
>   235 ms**. A later correction stands over the compiled LFM2.5 numbers in these notes: they
>   were taken with a state leak, so their direction holds and their sizes do not.

## Key numbers

| What | Result | Where |
|---|---|---|
| Benchmark cases whose grade differs across five chips | **52 of 390 (13.3%)** llama.cpp; **123 of 390 (31.5%)** ExecuTorch | five phones, five families, GSM8K, IFEval and BFCL |
| The app's seven-case parity suite on llama.cpp | identical on all five phones, 35 grades each | five families |
| Comparable cells where the GGUF side scores higher | **12 of 14** | mean of five chips, 30 prompts a cell |
| Gemma 3 1B on GSM8K, mean passes of 30 | 0.6 ExecuTorch against 13.2 llama.cpp | five chips |
| Memory per token of exported window | 25 to 33 KB (LFM2.5), 224 KB (Qwen3 1.7B), whose 32k export sat at 6.1 to 8.5 GB | ExecuTorch exports, four phones |
| Prefill across exported windows, paired | within 0.3 to 1.3% of the 2k export (LFM2.5 1.2B); Qwen3 32k 5.9% slower | POCO X8 Pro Max, 929-token prompt |
| Fresh chat, time to first token, cold against warmed prefix | **9,326 ms** against **235 ms** | POCO X8 Pro Max, LFM2.5 1.2B Q4_K_M, llama.cpp |
| llama.cpp prefill by thread count | 345 tok/s at 6 threads against 92 at 8 | Galaxy S25 Ultra, Qwen3 1.7B Q8_0 |

## Why it matters

These runs changed the app: thread counts now follow core types instead of clock speeds, after
the thread sweep found two thread-count bugs; the first turn is pre-warmed
([warm prefix and snapshot](../../../docs/glossary.md#warm-prefix-and-snapshot)); and every
compiled family carries its own chat template, stop list and
[BOS token](../../../docs/glossary.md#bos-token), because a `.pte` carries none. For a developer:
greedy decoding is reproducible on one phone, not across phones, so test on more than one chip;
choosing a runtime means comparing what each publisher's file gives up, not which runtime is
faster; and size a compiled model's window from its
[KV cache](../../../docs/glossary.md#kv-cache) cost per token, because a 32k full-attention
export can be killed by the OS at load.

## What was tested

- **Phones.** The POCO X8 Pro Max (MediaTek MT6991, the maintainer's own phone); a Snapdragon 8
  Gen 3 (SM8650) phone on [Qualcomm Device Cloud](../../../docs/glossary.md#qualcomm-device-cloud-qdc);
  a Pixel 10 Pro XL (Tensor G5), a Galaxy S24+ (Exynos 2400) and a Galaxy S25 Ultra (Snapdragon 8
  Elite) on Firebase Test Lab; a Galaxy Z Flip7
  (Exynos 2500) on [Samsung Remote Test Lab](../../../docs/glossary.md#samsung-remote-test-lab-rtl),
  a partial column; and a Galaxy S25 Ultra through
  [Firebase Device Streaming](../../../docs/glossary.md#firebase-device-streaming) for the recipes
  work. See [all devices](../../../docs/devices.md).
- **Runtimes.** ExecuTorch 1.4.0 on the CPU ([XNNPACK](../../../docs/glossary.md#xnnpack)), with
  one Vulkan row; llama.cpp, whose version these write-ups do not state.
- **Models.** Gemma 3 1B, LFM2.5 1.2B, Llama 3.2 3B, Qwen3 1.7B and SmolLM3 3B as the publishers'
  `.pte` and GGUF pairs (usually [8da4w](../../../docs/glossary.md#8da4w) against
  [Q4_K_M](../../../docs/glossary.md#gguf-quantization-types)); the project's own LFM2.5 1.2B,
  LFM2.5 2.6B and Qwen3 1.7B exports at 2k to 32k windows; Llama 3.2 1B and Qwen3.5 2B in the
  recipes work.
- **Method.** The app's instrumented eval classes on the phone: seven agentic parity cases, and
  seeded subsets of GSM8K, IFEval and BFCL, 30 prompts each; greedy, thinking off, replies capped
  at 640 tokens (384 for BFCL).
- **When.** 31 August to 23 September 2026.

## What's in this folder

`runs/` is the eval harness's results folder as it stood in OpenWeights, kept whole on purpose:
`compare.py`, `bench/report.py` and `bench/window_report.py` each read it as one pool and pick
their files by content, so splitting it would change what they regenerate.

| Run | What it holds |
|---|---|
| [`runs/`](runs/) | The pool, 561 files at the top level: parity reports, public-benchmark reports (`*.bench.json`) and their grades (`*.bench.graded.json`), the window matrix (`*-8da4w-<n>k*`), and the drivers' logs. |
| [`runs/controls/`](runs/controls/) | The Gemma 3 loop control: GGUF GSM8K with the repeat penalty off, POCO, ungraded. 1 file. |
| [`runs/logcats/`](runs/logcats/) | The Test Lab benchmark logcats of the three Test Lab phones, xz-compressed. 64 files. |
| [`runs/invalid-nobos/`](runs/invalid-nobos/) | LFM2.5 2.6B window cells set aside because no BOS token was fed. 24 files. |
| [`runs/qwen3-32k-poco-rerun/`](runs/qwen3-32k-poco-rerun/) | The POCO's Qwen3 32k rerun (7 of 90 prompts completed) and the OS's kill line. 2 files. |

A file name's prefix names the phone or lab:

| Prefix | Phone | Files |
|---|---|---|
| `tensor-` | Pixel 10 Pro XL, Test Lab | 151 |
| `exynos-` | Galaxy S24+, Test Lab | 134 |
| `elite-` | Galaxy S25 Ultra, Test Lab | 128 |
| `exynos2500-` | Galaxy Z Flip7, Samsung Remote Test Lab | 16 |
| `qdc-` | Snapdragon 8 Gen 3, Qualcomm Device Cloud | 30 |
| none | POCO X8 Pro Max | 58 |
| `repeat-` | same-file reruns for the window matrix (POCO and Galaxy S24+) | 14 |
| `prod-` | POCO runs of two LFM2.5 32k exports from the compiled tool-calling work | 4 |
| `bench-*.log` | the drivers' own logs | 26 |

Reports are JSON; `.ftl.log` files are Test Lab chunk logs. A few files belong to smaller
questions: `*-probe*` and `*-speed*` (core selection, 3 Sep 2026), `poco-pd-decode-probe.json`
(the NPU write-up's decode probe), `window-probe.log` (the fixed-prompt speed probe) and the
warm-prefix rerun's logcat. Changed on import: the Test Lab logcats were whole devices' logs,
with network names, addresses and test accounts, so only the app's lines came across
([`tools/logcat_filter.py`](../../../tools/logcat_filter.py)); the full logs are in
[OpenWeights at 81734dee](https://github.com/ExperimentalMachines/openweights/tree/81734deed5048f5691518d0e739662c50e51e1a3/tools/eval/results).
Two tokenizers that had been committed into the folder stayed out: they are model files, and
`compare.py` crashed reading them as reports.

## Run it again

You need an OpenWeights checkout in `OPENWEIGHTS`, a JDK and the Android SDK (the scripts build
the engine's test APK), Python 3 with `bench/requirements.txt` for grading, and the model files,
which are not in this repository. The instrumented eval classes live in OpenWeights. Test Lab
needs a Google Cloud project and a `gs://` bucket holding the models; Samsung Remote Test Lab and
Qualcomm Device Cloud need their own accounts. Rendering needs nothing but Python.

```sh
export OPENWEIGHTS=~/path/to/openweights
cd ~/experiments
D=openweights/studies/on-device-evals/runs

# Parity on a phone you hold over adb: pushes the models, builds, installs, pulls, renders
openweights/harness/app-evals/run_matrix.sh <models-dir> [adb-serial]

# One Test Lab phone; engine is executorch, llamacpp, probe, pd-probe, bench-executorch or
# bench-llamacpp. The bucket is required, although the usage line marks it optional.
openweights/harness/app-evals/run_matrix_ftl.sh <model-id> <os-version> <prefix> <engine> gs://<bucket>

# Public benchmarks over adb (results without a prefix are the POCO's column)
openweights/harness/app-evals/bench/run_local.sh [adb-serial]

# Public benchmarks on every Test Lab phone (fails as imported, see Caveats)
PHONES="mustang:36:tensor- e2s:36:exynos- pa3q:36:elite-" openweights/harness/app-evals/bench/run_cloud.sh [parallel-jobs]

# Grade on the host. grade.py rewrites the *.graded.json beside each report, and runs/ is a
# record, so grade a copy
G=$(mktemp -d) && cp -R $D/. $G/ && python3 openweights/harness/app-evals/bench/grade.py $G

# Render. Give --out a path in an OpenWeights checkout to refresh the write-up there.
# backend-parity.md comes back from the whole pool
python3 openweights/harness/app-evals/compare.py $D --out <backend-parity.md>
# benchmark-matrix.md and window-matrix.md come back only from copies without the later files.
# They read the *.graded.json files: from the graded copy $G if you graded above, else the stored ones
T=$(mktemp -d)
rsync -a --exclude '*8da4w-*k*' --exclude 'repeat-*' --exclude 'prod-*' ${G:-$D}/ $T/bench/
rsync -a --exclude 'prod-*' ${G:-$D}/ $T/window/
python3 openweights/harness/app-evals/bench/report.py $T/bench --out <benchmark-matrix.md>
python3 openweights/harness/app-evals/bench/window_report.py $T/window \
  --sizes openweights/harness/app-evals/bench/window_sizes.json --out <window-matrix.md>
# The window matrix's speed-probe tables
python3 openweights/harness/app-evals/bench/probe_report.py $D/window-probe.log

# Why one turn was slow (swap, page cache or throttling), on a debuggable build
openweights/harness/app-evals/cold_start_probe.sh [seconds]
```

Checked on this folder: `compare.py` reproduces the published parity table line for line except
its header; the two copies above reproduce the benchmark matrix and the window matrix line for
line below their headers; `probe_report.py` reproduces the speed-probe tables. The fixed-prompt
probe's driver and the first-turn eval have no driver script here; the POCO's ROM blocks plain
`adb install`, which OpenWeights' `phone-deploy` skill works around.

## Caveats

> [!WARNING]
> - **The compiled LFM2.5 numbers are directional only.** A later write-up found that every
>   compiled LFM2.5 quality number taken with one reused file carried a convolution-state leak:
>   the LFM cells of the public benchmarks and the benchmark matrix, the window matrix's LFM rows
>   and its "same file twice" reading. Their direction survives, their sizes do not, and "the
>   runtime is not bit-reproducible" is no longer established for this family. The 31.5% figure
>   includes those cells.
> - **Two renderers do not reproduce their tables from the whole pool**, contrary to this
>   folder's earlier README. `bench/report.py` reads the later window-matrix, `repeat-` and
>   `prod-` reports as the POCO's column and changes its ExecuTorch LFM2.5 and Qwen3 cells;
>   `bench/window_report.py` reads the `prod-` reports as the POCO and adds a section for them.
>   Leaving those files out, as in "Run it again", reproduces both. `compare.py` reproduces its
>   table from the whole pool.
> - **The Test Lab drivers fail as imported.** `run_matrix_ftl.sh` stops without a fifth
>   argument, the bucket (the old one was deleted on 30 September 2026), but `run_cloud.sh`,
>   `run_cloud_windows.sh` and `run_cloud_rerun.sh` call it with four and send its errors to
>   `/dev/null`, so every job prints FAILED until a bucket is passed through.
> - Four llama.cpp cells in the public-benchmarks write-up's prose table do not match the
>   generated matrix or a recount from the graded files (for example Qwen3 GSM8K 23.2 in the
>   prose against 22.8); the 12-of-14 result is unaffected.
> - Still open: the Tensor G5 and Exynos 2400 llama.cpp decode gap; the SmolLM3 template
>   diagnosis, a hypothesis until a corrected template restores its calls; whether Gemma 3's
>   ExecuTorch looping was the missing BOS token. The window verdict is a screen with one repeat
>   per control cell.
> - The parity write-up calls the POCO "POCO X8 Pro"; the other write-ups say POCO X8 Pro Max.

## The write-ups

### Backend parity: ExecuTorch vs llama.cpp, same prompts, four SoCs
Do the two runtimes grade the same on seven agentic cases on every chip? Mostly yes, and the
differences follow family and quantisation: Qwen3 1.7B passes all 7 on both runtimes on every
chip but one ExecuTorch cell, while LFM2.5's format case fails on ExecuTorch on all five chips
and passes on llama.cpp on all five.

### The parity matrix on five SoCs: what three cloud phones added
What changed with three [Test Lab](../../../docs/glossary.md#firebase-test-lab) phones added?
llama.cpp graded identically on all five, 35 grades a phone; ExecuTorch showed three cross-chip
divergences, none attributed to a kernel. The thread sweep found two thread-count bugs: fixed,
Qwen3 Q8_0's median prefill on the 8 Elite went from 12 to 236 tok/s.

### The ExecuTorch catalogue: seven families, one runtime, measured
What does supporting seven families on ExecuTorch take? Each needs its own prompt template, stop
markers and capability flags, because a `.pte` carries none. The parity matrix caught five
harness bugs, one a runner that turned a 768-token budget into 1,643.

### Public benchmarks on six phones: what GSM8K, IFEval and BFCL say about the two runtimes
How do five families score on each runtime and phone, 30 prompts a set? The GGUF side leads on 12
of 14 comparable cells, and grades differ across chips on 31.5% of cases on ExecuTorch against
13.3% on llama.cpp.

### Public benchmarks: ExecuTorch vs llama.cpp on five SoCs
The generated table behind the previous note: passes, prefill, decode and seconds per prompt for
every family, runtime and phone. Gemma 3 on GSM8K on the POCO: 0 of 30 on ExecuTorch, 14 of 30 on
llama.cpp.

### Exported window matrix
The generated table per exported window (2k to 32k), per phone. Grades move both ways between
windows by about as much as between two runs of the same file; LFM2.5 1.2B prefills at 332 to 343
tok/s at every window while its memory after load goes from 1,050 to 1,768 MB.

### Does the exported context window matter? ExecuTorch on four chips
Is a full sweep of windows, models and chips worth about 50 device-hours? No: no window effect on
answers or tool calls stood out, speed was flat where measured in pairs, and the window costs
memory (224 KB a token for Qwen3 1.7B). It also found that every compiled model the app ran was
fed no BOS token.

### Compiled LFM2.5 state and quantization experiments
Why does the compiled LFM2.5 1.2B rarely call web search? Two causes: a convolution state the
runner's reset never cleared, and [round-to-nearest](../../../docs/glossary.md#round-to-nearest-rtn)
int4 rounding. On 141 held-out questions it searched when needed on 10% (round to nearest), 49%
([GPTQ](../../../docs/glossary.md#gptq)) and 51% (Q4_K_M GGUF). Marked historical: the later
published GPTQ 32k file is not approved.

### The first turn, and why it no longer costs twenty seconds
Why did a first turn take about twenty seconds? A roughly 2,050-token tool and instruction
prefix was read cold at about 100 tok/s. Prefilling it at load and restoring a snapshot brought a
fresh chat's first token from 9,326 ms to 235 ms.

### NPU prefill, CPU decode: what works and what it costs
Linked only for one file here, the decode-half probe; the split works on the POCO but stays off.

## Read more

- Write-ups in OpenWeights:
  [Backend parity: ExecuTorch vs llama.cpp, same prompts, four SoCs](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/backend-parity.md),
  [The parity matrix on five SoCs: what three cloud phones added](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/parity-five-socs.md),
  [The ExecuTorch catalogue: seven families, one runtime, measured](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/executorch-families.md),
  [Public benchmarks on six phones: what GSM8K, IFEval and BFCL say about the two runtimes](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/public-benchmarks.md),
  [Public benchmarks: ExecuTorch vs llama.cpp on five SoCs](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/benchmark-matrix.md),
  [Exported window matrix](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/window-matrix.md),
  [Does the exported context window matter? ExecuTorch on four chips](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/executorch-window-matrix.md),
  [Compiled LFM2.5 state and quantization experiments](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/executorch-state-and-recipes.md),
  [The first turn, and why it no longer costs twenty seconds](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/first-turn-latency.md),
  [NPU prefill, CPU decode: what works and what it costs](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/npu-pd-disaggregation.md).
- Related: [compiled tool calling](../compiled-tool-calling/) (why the compiled LFM2.5 does not
  search), [retrieve or answer](../retrieve-or-answer/), [prefix reuse](../prefix-reuse/) and the
  [Test Lab archive](../test-lab-archive/).
- Harness: [`harness/app-evals/`](../../harness/app-evals/).
- Words used here: [glossary](../../../docs/glossary.md).
