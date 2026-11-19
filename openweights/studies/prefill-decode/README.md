# Prefill and decode on CPU, GPU and NPU

**Question.** Prefill on one processor and decode on another, over real multi-turn conversations
replayed turn by turn: which pairing is fastest on Snapdragon and on MediaTek, and what heat,
throttling and the thread count do to decode.

**Write-up** (in OpenWeights): [`snapdragon-pd-matrix.md`](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/snapdragon-pd-matrix.md). The cross-vendor runs from
2026-11-16 on (Galaxy S24 Ultra, Galaxy Tab S10+, the Dimensity 9500 pilot) are not written up
yet; their `NOTES.md` files are the record.

**Runs.**

- `2026-10-24-sm8850`: the Snapdragon 8 Elite Gen 5 matrix and its follow-up rounds, on QDC.
- `2026-11-09-poco-mt6991`: the stall guard and ordinary chat against the throttle, on the POCO.
- `2026-11-10-sm8550`: the stall guard's retry schedule on a Snapdragon 8 Gen 2.
- `2026-11-11-sm8850-r52`: every thread count on a 1,751-token prompt.
- `2026-11-16-s24`: the matrix on a Galaxy S24 Ultra (Snapdragon 8 Gen 3), Firebase Device
  Streaming, unrooted.
- `2026-11-16-mt6989`: the matrix on a Galaxy Tab S10+ (Dimensity 9300+), Device Streaming,
  with the NPU through MediaTek's runtime.
- `2026-11-18-mt6993-pilot`: a Galaxy Tab S12+ (Dimensity 9500) on Samsung Remote Test Lab.
  It was never committed to OpenWeights; it came from the working tree.
- `npu-access`: whether a retail Exynos or Tensor unit's shell can reach its NPU.

The `unit` in every row and log names the physical phone by a pseudonym (`unit-` and six hex
digits) instead of its serial; one unit has one name across all runs.

**Harness.** [`harness/prefill-decode/`](../../harness/prefill-decode/): `snapdragon/`
(`pd_bench.cpp`, `pd_bench_sd.cpp`, the session drivers and every `analyze_*.py`) and
`mediatek/` (`pd_bench_mtk.cpp`, `npu_engine.h`, the session driver, the queue files). Device
Streaming units are reached through [`harness/device-streaming/`](../../harness/device-streaming/).
