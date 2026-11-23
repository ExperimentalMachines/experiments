# KV cache

**What shrinking the conversation's cache to 8 bits on the CPU, or running in 16-bit mode on the
GPU, does to speed, memory and answers, from 2k to 32k-token windows.**

`published on experimentalmachines.org` `3 reference phones` `20 and 21 Nov 2026`

> **The question.** Every token a model has read stays in its
> [KV cache](../../../docs/glossary.md#kv-cache), which ExecuPack's files keep in 32 bits and size
> for the whole [window](../../../docs/glossary.md#context-window). On a phone, would an 8-bit cache
> on the CPU ([XNNPACK](../../../docs/glossary.md#xnnpack)), or a 16-bit mode on the GPU
> ([Vulkan](../../../docs/glossary.md#vulkan)), make long chats faster and smaller without changing
> the answers? Four models, three windows, each file measured on the phone against the original
> model.

> [!TIP]
> **What we found**
> - **On the CPU, an 8-bit cache makes long chats much faster:** 1.7 to 3.8 times faster writing
>   7.9k tokens into a chat, and up to 5.3 GB less memory.
> - **It nudges the answers, not consistently for the worse.** About 1 in 10 next-token picks
>   change; the distance from the original model ([KL](../../../docs/glossary.md#kl-divergence))
>   and the recall of hidden facts mostly did not move. The one clear worsening: Llama 3.2 1B
>   32.5k tokens in, 9.6% further from the original (one book).
> - **On the GPU, 16-bit mode is 1.05 to 1.29 times faster and uses 0.4 to 2.6 GB less memory,**
>   much of it likely attention's working space rather than the cache. ExecuTorch 1.5.1 cannot
>   build an 8-bit cache for the GPU.
> - **But 16-bit mode broke Llama deep into a long text:** at 32.5k tokens it recalled 0 of 9
>   hidden facts, against 9 of 9 at 32-bit.
> - **Llama 3.2 1B's published GPU files above 8k do not load** on this phone. Reading the prompt
>   in 256-token pieces instead of 2,048 let a 32k file load.

## Key numbers

| What | Result | Where |
|---|---|---|
| Writing speed, 8-bit cache against 32-bit, 7.9k tokens in | 1.7 to 3.8 times faster | CPU files, 8k and 32k windows, four models |
| Memory saved by the 8-bit cache | up to 5.3 GB; 0.04 to 0.35 GB at a 2k window | resident memory on the phone |
| Next-token picks that change with the 8-bit cache | about 1 in 10 (direct KL between the two files 0.013 to 0.037 at 1.9k) | three PG-19 books |
| The one clear worsening, KL to the original | 0.2931 against 0.3212 (+9.6%) | Llama 3.2 1B, 32k file, 32.5k tokens in, one book |
| GPU 16-bit mode against 32-bit | 1.05 to 1.29 times faster writing, 0.4 to 2.6 GB less GPU memory | Llama 3.2 1B and Qwen3 1.7B, 2k to 32k |
| Hidden facts recalled at 32.5k tokens, GPU | 0 of 9 in 16-bit mode, 9 of 9 at 32-bit | Llama 3.2 1B, 32k file built with 256-token prompt pieces |
| Published Llama 3.2 1B GPU files | 16k and 32k abort at load; 8k loads | ExecuTorch 1.5.1, 2,048-token prompt pieces |

## Why it matters

The cache is the part of a file's memory that grows with the window, so it decides how long a
chat a phone can hold and how fast the reply streams late in it. The CPU result is the case for
shipping the 8-bit cache in ExecuPack's 8k and 32k CPU files, after a check through the app's own
runner. The GPU result is a warning: 16-bit mode changes the arithmetic as well as the cache, it
held up at 2k and 8k, and it failed outright at 32.5k tokens on Llama, so the GPU files stay
32-bit. The published files that do not load are a product bug this study found.

## What was tested

- **Phones.** Three Snapdragon SM8975 reference phones on
  [Qualcomm Device Cloud](../../../docs/glossary.md#qualcomm-device-cloud-qdc), one after another as
  sessions ended. The second and third appear as `unit-10b9a2` and `unit-a4448d`. See
  [all devices](../../../docs/devices.md).
- **Runtime.** [ExecuTorch](../../../docs/glossary.md#executorch) 1.5.1, the version inside
  OpenWeights' app, through a small Java harness on the phone (`harness/phone/`).
- **Models.** Llama 3.2 1B, LFM2.5 1.2B, Qwen3 1.7B and Gemma 3 1B, each with the same
  [GPTQ](../../../docs/glossary.md#gptq)-solved [8da4w](../../../docs/glossary.md#8da4w) weights in
  every file, so only the cache differs. CPU files at 2k, 8k and 32k with a 32-bit and an 8-bit
  cache; GPU files for Llama and Qwen3 only (Gemma 3 does not lower to Vulkan and LFM2.5 has no
  Vulkan kernel for its convolutions).
- **Method.** Three PG-19 books, each with nine hidden facts, read to 1.9k, 7.9k and 32.5k tokens
  as each window allows (Llama and Qwen3 read one book at 32k); at each depth 128 next-token
  predictions scored against the original model in 32-bit on a GPU, then the recall questions;
  the app's 141 held-out questions; writing speed and memory throughout. GPU runs were gated on
  the GPU's temperature and thermal level, one book per process, with the throttle level logged
  every 2 seconds.
- **When.** 20 and 21 November 2026.

## What's in this folder

| Folder | What it holds |
|---|---|
| [`runs/phone-out/`](runs/phone-out/) | The first phone's CPU runs: every model, window and cache, streams and app questions. 52 files. |
| [`runs/phone-out2/`](runs/phone-out2/) | The second phone's CPU rerun, which reproduced the first phone's KL exactly, and the direct KL between the two caches. 17 files. |
| [`runs/gpu-out/`](runs/gpu-out/), [`runs/gpu-out-phone1/`](runs/gpu-out-phone1/) | The GPU pass on the second phone (Llama 2k and 8k, Qwen3 2k), with thermal logs (throttled runs set aside in `gpu-out/throttled/`); the first phone's GPU data. 65 files. |
| [`runs/gpu-out4/`](runs/gpu-out4/) | The third phone: Llama's 32k pair and its repeat, Qwen3 8k, and the load test of the published GPU files. 52 files. |
| [`runs/res/`](runs/res/), [`runs/logs/`](runs/logs/), `runs/status.log` | Each build's export report and log, and the build queue's progress. 62 files. |
| [`runs/report/`](runs/report/), `runs/report.md` | The study's report: the page `harness/report.py` computes from the runs, and its tables in Markdown. |
| `runs/codex-review*.md`, `runs/codex-prompt*.txt` | Four outside reviews of the report and the briefs they were given. |
| [`harness/`](harness/) | The scripts, never committed: the reference builder, the solve and export, the Modal and Colab build drivers, the phone scripts, the merge and the report. |
| [`uncommitted.patch`](uncommitted.patch) | The pipeline edits the study needed (Gemma 3 and the cache options), against ExecuPack at `34c55110`. |

## Run it again

The study needs an ExecuPack checkout at `34c55110` with `uncommitted.patch` applied and the
scripts in `harness/` copied into `scripts/kvstudy/` (and the `kv_*.py` and Gemma files into
`scripts/` and `pipeline/`), a Hugging Face token, and a phone with a root shell. Each script's
docstring gives its inputs. The PG-19 books, the reference token streams and the reference tensors
are not here: `reference.py` rebuilds them.

```sh
# The 32-bit reference the phone is scored against (one GPU)
modal run scripts/kvstudy/reference.py --models llama,lfm,qwen,gemma

# Solve once per model, then export every window and cache from those codes
scripts/kvstudy/build.sh llama lfm qwen

# On the phone, detached so a dropped connection does not stop it
setsid sh /data/local/tmp/kvq/study.sh "2k 8k 32k" "llama lfm qwen gemma"

# The report, from the stored runs (no phone needed)
python scripts/kvstudy/report.py runs/phone-out <reference data> runs/report/narrative.html > report.html
```

## Caveats

> [!NOTE]
> - Measured with the study's harness, not through the app. The app's runner was not checked.
> - The phones' memory size is not in the results.
> - Llama 3.2 1B and Qwen3 1.7B read one book at 32k, so their 32k numbers rest on one book.
> - The GPU throttles within 20 to 25 seconds of sustained load. Runs that throttled are marked
>   or set aside, and the 32k pair throttled during its long prompt read.
> - Why 16-bit mode fails at 32.5k is not isolated; the attention sum in 16 bits is the leading
>   suspect. Why the published files above 8k do not load is also not established: the abort
>   came about 10 ms after the GPU was set up, and the attention buffers are the leading suspect.
> - A first measure of quality, the change in each file's KL to the original, hid the effect of
>   the 8-bit cache because the two changes cancel. The direct KL between the two files is the
>   one to read.

## Read more

- Write-up: [Shrinking the KV cache on a phone](https://experimentalmachines.org/research/kv-cache/)
  on experimentalmachines.org.
- Related: [AWQ beside GPTQ](../awq/), whose long-text round reuses this study's books and
  scorer, and the [research log](../../research/README.md).
- Words used here: [glossary](../../../docs/glossary.md).
