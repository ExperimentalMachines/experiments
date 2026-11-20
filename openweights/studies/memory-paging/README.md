# Memory paging

**What happens to an on-device model when the phone runs short of memory: when swap slows a
reply, how close the app's memory estimate comes, and whether reading paged-out weights back
early helps.**

`written up in OpenWeights` `1 phone` `12 to 24 Oct 2026`

> **The question.** When a model needs more memory than the phone can spare, Android compresses
> part of it into [swap](../../../docs/glossary.md#swap-and-page-faults) (zram). Can the app see
> that from inside, and when does it actually slow a reply? How close is the app's "will this
> model fit" estimate to the real peak memory of each model and runtime? And does reading the
> paged-out weights back in while the person types the next message pay for itself?

> [!TIP]
> **What we found**
> - Being swapped out costs nothing; being paged back in during a turn is what costs, almost all
>   of it in [prefill](../../../docs/glossary.md#prefill-and-decode), at **15 to 27 µs a major
>   page fault**: 0.4 million faults cost 10 s, 1.25 million cost 23 s. The app's own process
>   counters are enough to see it. Those turns also began with a full cache reset, which pages
>   memory too, so how much of the slowdown the idle gap alone caused is not settled.
> - The app's memory estimate was right to within a few percent for GGUF files (**-6% to +2%**)
>   while compiled `.pte` files peaked 13% to 42% below their estimates.
> - Reading the weights back while the person types did not establish a benefit or a cost:
>   **96** pairs answered sooner with it and **96** later. It stays on provisionally.
> - The danger is the kill, not the slowdown: under a 6 GB memory hog, or with a 9.5 GB model,
>   Android killed the app during the load, and nothing inside a turn warned of it.

## Key numbers

| What | Result | Where |
|---|---|---|
| Peak memory against the app's estimate | GGUF files -6% to +2%; `.pte` files peaked 13% to 42% below the estimate | three GGUF files on llama.cpp, three `.pte` on ExecuTorch |
| Memory held with an empty conversation, 16k window | 3,838 MB: the 1,749 MB file plus its 1,792 MiB cache | Qwen3 1.7B Q8_0, llama.cpp |
| Cost of a major page fault during a turn | **15 to 27 µs** | Qwen3 1.7B Q8_0 under a 4 and a 5 GB memory hog |
| Worst turn under a 5 GB hog against none | prefill 28.0 against 54.0 to 55.0 tok/s; 53.6 against 29.9 to 30.6 s; 1,254,711 major faults | Qwen3 1.7B Q8_0, 8k window, 1k-token prompt |
| A 6 GB hog or more | the app killed during the load, after 62 kills of 39 other processes | Qwen3 1.7B Q8_0 |
| Reading the weights back early, 4 GB hog, medians of four turns | major faults in the turn 56,682 against 325,049; first token 22.7 against 23.3 s | Qwen3 1.7B Q8_0 |
| The same across models and conversation shapes | sooner in **96** pairs, later in **96**; smallest single-turn sign test p = 0.289 | three GGUF models, 192 pairs |
| The app's "slowed by low phone memory" flag | fired on 14 of 418 paired turns: 11 were 10% or more slower than their partner, 2 slower by less, 1 not slower | the same matrix |

All on the POCO X8 Pro Max.

## Why it matters

The app now flags a reply "slowed by low phone memory" when its major faults times 20 µs come to
at least a second and to more than 10% of the pass, shows each reply's peak memory, shows live
prefill and decode rates (measured to cost nothing), allows 512 MB instead of 450 MB of runtime
overhead in its fit estimate, and reads paged-out weights back while the person types, on
provisionally. For a developer: on Android an app can read only its own memory counters, and
they are enough; the size of swap does not show thrashing, the fault count does; a plain
transformer's [KV cache](../../../docs/glossary.md#kv-cache) is held in full from the load, so a
wide [context window](../../../docs/glossary.md#context-window) costs its whole cache up front;
and the real risk is being killed during the load, which only a check of free memory before
loading can foresee.

## What was tested

- **Phones.** The POCO X8 Pro Max (MediaTek MT6991, 11.5 GB of RAM, 12 GB of swap, Android 16 /
  HyperOS), the maintainer's own phone with their apps in the background, over adb. See
  [all devices](../../../docs/devices.md).
- **Runtimes.** The app's [llama.cpp](../../../docs/glossary.md#llamacpp) and
  [ExecuTorch](../../../docs/glossary.md#executorch) on the CPU; the write-up does not state
  versions.
- **Models.** As [GGUF](../../../docs/glossary.md#gguf) files: LFM2.5 1.2B QAD Q4_0, Qwen3 1.7B
  Q8_0, LFM2.5-VL 3B Q4_0, LFM2.5 2.6B Q4_K_M and Qwen3.5 9B Q8_0. As [`.pte`](../../../docs/glossary.md#pte)
  files: Qwen3 1.7B 8da4w, Gemma 3 4B and LFM2.5-VL 450M.
- **Method.** OpenWeights' engine tests sampled each model's memory every 100 ms through the load
  and a turn. Pressure came from a helper process holding 0 to 8 GB of random bytes, which zram
  cannot compress, touching every page every two seconds; turns were 20 s apart, the time a
  person spends reading a reply. The prefetch matrix paired each turn with and without the
  early read-back, in a seeded order, across three models, single and four-turn conversations,
  with and without the search tool, under a 4 GB hog and none; the analysis was written down
  before the extra levels ran.
- **When.** 12 October 2026 (peak memory, pressure, the flag and the live readout), 23 October
  (prefetch trials) and the night of 23 to 24 October (the matrix). Battery or charger is noted
  per run in the write-up.

## What's in this folder

| Run | What it holds |
|---|---|
| [`runs/2026-10-12-peak/`](runs/2026-10-12-peak/) | Peak memory of three GGUF models at a 16k window with a 4k-token prompt, plus load-only reports of the three `.pte`. 6 files. |
| [`runs/2026-10-12-peak-pte/`](runs/2026-10-12-peak-pte/) | Peak memory of the three `.pte` with a 1.2k-token turn. 3 files. |
| [`runs/2026-10-12-pressure/`](runs/2026-10-12-pressure/) | Qwen3 1.7B Q8_0 under 0, 4 and 5 GB hogs, and its `.pte` under 0 and 4 GB. 5 files. |
| [`runs/2026-10-12-flag/`](runs/2026-10-12-flag/) | The paging flag measured after it was built: Qwen3 1.7B Q8_0, 0, 4 and 5 GB, three turns each. 3 files. |
| [`runs/2026-10-12-readout/`](runs/2026-10-12-readout/) | What the live readout costs: LFM2.5 1.2B QAD Q4_0, six pairs with and without it. 1 file. |
| [`runs/2026-10-23-prefetch/`](runs/2026-10-23-prefetch/) | The first prefetch trials, with the cache reset before each turn, 0 to 3 GB. 4 files. |
| [`runs/2026-10-23-prefetch2/`](runs/2026-10-23-prefetch2/), [`-prefetch3/`](runs/2026-10-23-prefetch3/) | Two more 4 GB trials of the earlier designs. 1 file each. |
| [`runs/2026-10-23-prefetch4/`](runs/2026-10-23-prefetch4/) | The design the app ships, 3 and 4 GB, nine turns each. 2 files. |
| [`runs/matrix-20261023-2356/`](runs/matrix-20261023-2356/) | The main prefetch matrix: three models, four conversation shapes, 4 GB and none, with the status of each cell. 25 files. |
| [`runs/matrix-20261024-0736-big/`](runs/matrix-20261024-0736-big/) | Qwen3.5 9B Q8_0 with no hog: killed during the load in all four cells, so no reports; the kill lines. 2 files. |
| [`runs/matrix-20261024-0746-b5/`](runs/matrix-20261024-0746-b5/) | The 5 GB level, one cell, stopped when the owner needed the phone. 4 files. |
| [`runs/matrix-predeclared.md`](runs/matrix-predeclared.md) | The two levels added during the matrix, each written down before its data. |

Reports are JSON, one per model and level (`*.mem.json`, `*.readout.json`, `*.prefetch.json`);
each matrix folder has a `cells.tsv` with every cell's status and times. The raw logcats are kept
out by the folder's `.gitignore`, because they name the owner's apps.

## Run it again

You need an OpenWeights checkout in `OPENWEIGHTS` with `local.properties` naming the Android SDK
and an NDK (the scripts compile the memory hog, `balloon.c`, with it), the engine's test APK
built first, the model files pushed to `/data/local/tmp/openweights` on the phone, and an
Android phone over adb with room to swap. The benchmark classes, `MemoryPressureBenchmark` and
`PrefetchMatrixBenchmark`, live in OpenWeights. The tables need Python; `analyze.py` also needs
`uv`, the `gguf` package and the GGUF files, to read cache sizes.

```sh
export OPENWEIGHTS=~/path/to/openweights
(cd "$OPENWEIGHTS" && ./gradlew :core:engine:assembleDebugAndroidTest)
cd ~/experiments

# Peak memory and pressure: one run per hog level, in MB (0 alone measures peak only)
MODELS=Qwen3-1.7B-Q8_0.gguf LEVELS="0 4096 5120" openweights/harness/memory/run_pressure.sh [adb-serial]
# The prefetch trials: the last 10 s of each gap read the weights back on odd turns
MODELS=Qwen3-1.7B-Q8_0.gguf LEVELS=4096 LEAD=10 openweights/harness/memory/run_pressure.sh [adb-serial]
# The live readout's cost
CLASS=LiveReadoutOnDeviceTest MODELS=<model.gguf> openweights/harness/memory/run_pressure.sh [adb-serial]

# The prefetch matrix (defaults: LEVELS="4096 0", all four cells, SEED=20261024)
MODELS=LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf,Qwen3-1.7B-Q8_0.gguf,LFM2.5-2.6B-Q4_K_M.gguf \
  openweights/harness/memory/run_prefetch_matrix.sh <adb-serial>

# Tables
uv run openweights/harness/memory/analyze.py <results-dir> [--models-dir ~/ow-models]
python3 openweights/harness/memory/analyze_matrix.py openweights/studies/memory-paging/runs/matrix-20261023-2356 \
  --expect-models LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf,Qwen3-1.7B-Q8_0.gguf,LFM2.5-2.6B-Q4_K_M.gguf \
  --expect-levels 4096,0 --expect-cells single:false,single:true,multi:false,multi:true
```

Results land in a new folder under this study's `runs/`. `CONTEXT`, `FILL`, `TURNS`, `GAP` and
`ANSWER` pass through to the tests. The three `--expect-*` flags are required. Checked:
`analyze_matrix.py` on the main matrix prints "the prefetch turn answered sooner in 96 pairs and
later in 96".

## Caveats

> [!NOTE]
> - Every 12 October pressure turn began with a reset that zeroes the whole KV cache, which
>   under pressure pages the unused cache back in and can push weights out. The timings are what
>   was observed; how much of the slowdown came from the idle gap rather than the reset is not
>   known, and the later runs with the reset moved do not settle it.
> - The flag's 20 µs a fault is a middle figure, not a constant: the cost of a fault moved with
>   conditions, and the write-up says the flag needs recalibrating on matched runs.
> - The prefetch result is open: the 5 GB level stopped after one cell, the multi-turn cells need
>   more blocks, and one LFM2.5 2.6B pair took 97 s to first token with the prefetch against 34 s
>   without, unexplained. Its effect on other apps is also unsettled.
> - Found and not fixed here: starting a new chat zeroes the whole KV cache the same way, so
>   under pressure it pages the unused cache back in.
> - One phone, one model under pressure on 12 October, the owner's apps present; the hog is an
>   active app, which presses harder than an idle one.
> - The kill notes for the 9.5 GB model list 68 to 193 "other kills" per cell while the write-up
>   says 67 to 192; the two may count the benchmark's own kill differently.

## Read more

- Write-up in OpenWeights:
  [Swap and peak memory on the phone](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/swap-and-peak-memory.md)
- Related: [on-device evals](../on-device-evals/) (where a 32k export was killed by a per-app
  memory limit) and [prefix reuse](../prefix-reuse/).
- Harness: [`harness/memory/`](../../harness/memory/).
- Words used here: [glossary](../../../docs/glossary.md).
