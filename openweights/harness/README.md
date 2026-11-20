# OpenWeights harness

**The code that drives phones and reads what they wrote, for every OpenWeights study.**

Each folder here runs one kind of measurement against the OpenWeights app (an Android app that
runs open-weight LLMs on the phone through [llama.cpp](../../docs/glossary.md#llamacpp) and
[ExecuTorch](../../docs/glossary.md#executorch)) or against its engine, and turns the raw output
into tables. The results land in the study they belong to, under `../studies/<study>/runs/`.

## Start here

```sh
export OPENWEIGHTS=~/path/to/openweights   # the app checkout the scripts build and drive
cd ~/experiments                           # scripts run from this repository's root or by full path

# The Python tests of the graders
python3 -m unittest openweights/harness/app-evals/bench/test_*.py
```

Scripts that build or install the app take its checkout from `OPENWEIGHTS` and stop if it is
unset. The instrumented benchmarks and probes they run (`BenchmarkSuite`, `ThreadCountBenchmark`,
`DecisionSuiteOnDeviceTest` and the rest) stay in OpenWeights under `app/` and `core/`, because
they compile against the engine. Finding the test phone and installing around its ROM's block on
`adb install` is OpenWeights' `phone-deploy` skill. Nothing here holds model files; each script
names the files it expects on the phone or the host.

## The folders

| Folder | What it is for | Start with | Used by |
|---|---|---|---|
| [`app-evals/`](app-evals/) | Runs the app's instrumented eval classes over adb, on [Firebase Test Lab](../../docs/glossary.md#firebase-test-lab) and on [Samsung Remote Test Lab](../../docs/glossary.md#samsung-remote-test-lab-rtl), then grades and renders the results. From OpenWeights' `tools/eval/`. | `run_matrix.sh <models-dir> [adb-serial]` for the two-runtime parity matrix; `bench/run_decisions.sh [adb-serial]` for the search-or-answer suite; `run_matrix_ftl.sh` for Test Lab | [on-device evals](../studies/on-device-evals/), [retrieve or answer](../studies/retrieve-or-answer/), [compiled tool calling](../studies/compiled-tool-calling/), [shipped build](../studies/shipped-build/), [runtime upgrades](../studies/runtime-upgrades/) |
| [`device-streaming/`](device-streaming/) | Makes a [Firebase Device Streaming](../../docs/glossary.md#firebase-device-streaming) phone a plain adb device, checks whether a retail phone's shell can reach its [NPU](../../docs/glossary.md#npu), and uploads one object from a GPU box. From `tools/eval/streaming/`. | its [README](device-streaming/README.md); reservations are billed by the minute | [prefill and decode](../studies/prefill-decode/), [prefix reuse](../studies/prefix-reuse/) |
| [`prefill-decode/`](prefill-decode/) | The [prefill and decode](../../docs/glossary.md#prefill-and-decode) benchmark for Snapdragon (`snapdragon/`) and MediaTek (`mediatek/`), the session drivers, the phone-side scripts, every analysis, and the llama.cpp A/B used for upgrades. From `tools/snapdragon/` and `tools/mediatek/`. | [`snapdragon/README.md`](prefill-decode/snapdragon/README.md) | [prefill and decode](../studies/prefill-decode/), [runtime upgrades](../studies/runtime-upgrades/) |
| [`memory/`](memory/) | Memory-pressure and prefetch runs on the phone, with a balloon that holds memory away from the app, and their tables. From `tools/memory/`. | `run_pressure.sh [adb-serial]`, then `analyze.py <results-dir>` | [memory paging](../studies/memory-paging/) |
| [`npu/`](npu/) | CPU and NPU matrix-multiply benchmarks for MediaTek's NPU, one timed CPU decode step, and the multi-turn replay below. The benchmarks only; the runtime the app links is built in OpenWeights' `tools/npu/`, which its README's build steps refer to. | its [README](npu/README.md); needs an x86_64 Linux host | the NPU question below |
| [`host-prompt/`](host-prompt/) | Evals on a computer against `llama-server` with the app's exact prompt bytes (`prompt_dump.json`): where the date goes, tool routing, the 2026-09-05 behaviours, and what a picture costs. From OpenWeights' `eval/` and `tools/vision-probe-image.py`. | the two questions below | the prompt and picture questions below |

## Questions answered here, with no study folder

Three questions have a harness here but no study folder, because their outputs were working files
outside the repository and the write-ups hold the numbers.

### Does NPU prefill pay on real conversations?

**What it asked.** If MediaTek's NPU reads the prompt and the CPU writes the reply, does a chat get
faster on the conversations people actually have, once the app reuses its
[KV cache](../../docs/glossary.md#kv-cache) of the earlier turns?

**What it found.** No, not enough to build. After cache reuse the median turn prefilled only
**50 tokens**, and writing the reply was 60 to 77% of the time, so NPU prefill with CPU decode is
projected (not measured) at just **1.07x to 1.47x** end to end. Against a CPU using
[KleidiAI](../../docs/glossary.md#kleidiai) as llama.cpp really does, the NPU's matrix multiply is
1.3 to 1.8 times faster, not the 3 to 9 times first reported. The cheaper win needed no NPU: Q4_0
instead of Q4_K_M was 1.26x faster end to end. Measured on 2026-08-30 on the POCO X8 Pro Max
(MediaTek MT6991), LFM2.5 1.2B, 45 replayed conversations.

**Write-up:** [Does NPU prefill pay for itself on real conversations?](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/npu-prefill-multiturn.md)
**Files:** [`npu/multiturn/`](npu/multiturn/): `pull_datasets.py`, `replay.py` (self-play) and
`replay_tf.py` (teacher-forced) against a local `llama-server` on port 8080, then `analyse.py`
and `report.py`; the two sweep records `width_sweep.txt` and `cpu_sweep_kleidiai.txt`. The replay
outputs and the loop that ran the sweeps are not here.

### The prompt itself: where the date goes, how the model routes tools, plan mode and recall

**What it asked.** How should a small on-device model's prompt carry today's date, and how much
do prompt wording and sampler settings change whether it makes a
[tool call](../../docs/glossary.md#tool-call), plans instead of answering, or searches instead of
recalling?

**What it found.** Where the date sits decides whether a greeting gets answered: with the date as
the turn just before "hi", the reply answered the date instead of the user on **21 of 128** tries
on the host, and on the phone the shipped fix (a scoped acknowledgement plus one ordinary exchange)
took LFM2.5 1.2B from 6 of 16 to 0 of 16 and Qwen3 1.7B from 1 of 16 to 0 of 16. With no tool
switched on, the date first in the instructions won instead (4 of 128 on the host). Wording barely
moves routing; mechanisms do: a note attached to "who is X" questions took those lookups to 10 of
10 on both models, one corrective push took plan mode from 2 of 8 and 3 of 8 to 8 of 8, and a
128-token reasoning cap on the tool pass kept 31 of 33 routing cases right with 38% fewer generated
tokens. The host and the phone disagree on some prompt shapes; the phone is the verdict.

**Write-ups:** [Saying hello, and being answered about the date](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/date-in-the-prompt.md),
[Plan mode that answers, a model that remembers wrong, and a search it believes](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/plan-mode-and-recall.md),
the routing matrix in [What makes a 1B model call a tool](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/tool-calling.md),
and the reasoning cap in [The 2026-09-02 sweep](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/qa-sweep-2026-09-02.md).
**Files:** [`host-prompt/`](host-prompt/): `date_structure_eval.py` (both passes; it replaced
`date_placement_eval.py`), `date_notools_eval.py`, `routing_matrix.py`, `behaviour_probe.py`
(plan, entity, credulity) and `prompt_dump.json`. Each takes a `llama-server` port; `routing_matrix.py`
starts its own server.

### What a picture costs

**What it asked.** Why does a turn with a picture take tens of seconds (one user waited 213.8 s),
and which setting actually makes it cheaper without losing what the model can read?

**What it found.** For LFM2.5-VL the cost is set by whether the picture is cut into tiles (up to
ten, a flat 256 tokens each), so asking for fewer image tokens made a turn slower, not faster:
**9.1 s** with the automatic setting against **20.8 s** at a 16-token budget on a Snapdragon 8 Gen 3.
What stands is the fix made the same day: raise the ceiling to 512 tokens so a picture is one
encode, shrink the stored picture by area rather than by its longest edge, and reuse the picture
on follow-ups. On the POCO the first picture turn's prefill then took 43.8 s, 35.4 s of it the vision encoder,
and a follow-up's prefill 0.28 s (reply writing comes on top).

**Write-ups:** [Why a picture takes fifty seconds, and the one number that changes it](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/image-tokens.md),
and the remaining cost in [The vision encoder is the image turn](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/vision-encoder-cost.md).
**Files:** [`host-prompt/`](host-prompt/): `vision-probe-image.py <out.png>` draws the synthetic
probe; `image_speed_probe.py <port> <arm-name> [budgets]` sweeps pictures through a `llama-server`
started with the model and its `--mmproj` projector.

## Known quirks

Carried over as they were, so the scripts still match the runs they produced:

- `npu/multiturn/analyse.py` reads `npu/width_sweep.txt` relative to where it runs, while the file
  sits beside it, so run it from a folder holding a copy at that path. `report.py` imports it, and
  its matrix-multiply table reads the CPU side from `width_sweep.txt`, which is the uncorrected CPU
  path; the corrected CPU sweep is `cpu_sweep_kleidiai.txt`, and no script reads it.
- The Python tests under `app-evals/bench/` run with `python3 -m unittest` from this repository's
  root.
- `host-prompt/routing_matrix.py` checks that the shipped tool prompt is in the dump, and the kept
  `prompt_dump.json` (2026-11-05, 15 tools) fails that check. The September runs used a 16-tool
  dump, and OpenWeights' dump as committed on 2026-09-05 passes the check; to make a fresh one from
  an OpenWeights checkout, run
  `PROMPT_DUMP=$PWD/eval/prompt_dump.json ./gradlew :app:testDebugUnitTest --tests '*PromptDumpTest*'`.
- `host-prompt/behaviour_probe.py` reads the dump from a fixed path inside an OpenWeights checkout,
  not the copy here.
- `host-prompt/image_speed_probe.py` reads six of its seven pictures from the maintainer's own
  Downloads folder; only the synthetic probe can be regenerated.
- In `date_structure_eval.py` the arm named `shipped` is the pre-fix plain pair; what ships is
  `spaced_scoped`.
- The compiled tool-call probe under `app-evals/bench/toolcall_probe/` is kept as it ran; see
  [compiled tool calling](../studies/compiled-tool-calling/README.md).

Words used here: [glossary](../../docs/glossary.md).
