# OpenWeights harness

The code that drives devices and reads what they wrote. Scripts that build or install the app
take its checkout from `OPENWEIGHTS` and stop if it is unset.

| Folder | From OpenWeights | What it does |
|---|---|---|
| [`app-evals/`](app-evals/) | `tools/eval/` | Runs the app's instrumented eval classes over adb (`run_matrix.sh`), on Firebase Test Lab (`run_matrix_ftl.sh`, `bench/run_cloud*.sh`, `bench/run_*ftl*.sh`) and on Samsung Remote Test Lab (`bench/run_rtl.sh`); grades and renders the results (`compare.py`, `bench/`). |
| [`device-streaming/`](device-streaming/) | `tools/eval/streaming/` | A Firebase Device Streaming phone as a plain adb device (`ds_bridge.py`), the NPU access check, and a one-object upload from a GPU box. |
| [`prefill-decode/`](prefill-decode/) | `tools/snapdragon/`, `tools/mediatek/` | The prefill/decode benchmark for Snapdragon and MediaTek, its session drivers, the phone-side scripts, and every analysis. |
| [`memory/`](memory/) | `tools/memory/` | Memory pressure and prefetch runs, and their tables. |
| [`npu/`](npu/) | `tools/npu/` (the benchmarks only) | CPU and NPU matmul benchmarks, one CPU decode step, and the multi-turn replay that asked whether NPU prefill pays ([`npu-prefill-multiturn.md`](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/npu-prefill-multiturn.md)). Its README's build steps for the runtime refer to OpenWeights' `tools/npu/`. |
| [`host-prompt/`](host-prompt/) | `eval/`, `tools/vision-probe-image.py` | Host evals against llama-server with the app's own prompt (`prompt_dump.json`): where the date goes ([`date-in-the-prompt.md`](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/date-in-the-prompt.md)), tool routing (`routing_matrix.py`), the 2026-09-05 behaviours ([`plan-mode-and-recall.md`](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/plan-mode-and-recall.md)), and what a picture costs ([`image-tokens.md`](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/image-tokens.md)). |

Known quirks carried over as they were: `npu/multiturn/analyse.py` reads `npu/width_sweep.txt`
relative to where it runs, while the file sits beside it; the Python tests under
`app-evals/bench/` run with `python3 -m unittest` from this repository's root.
