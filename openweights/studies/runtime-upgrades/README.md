# Runtime upgrades

**Question.** Did each upgrade of the app's runtimes keep its speed and its answers?

**Write-ups** (in OpenWeights): [`llama-cpp-v080.md`](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/llama-cpp-v080.md), [`llama-cpp-v100-executorch-160.md`](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/llama-cpp-v100-executorch-160.md).

**Runs.**

- `2026-10-31-llama-v080`: llama.cpp b10549 against v0.8.0 on an SM8850.
- `2026-11-04-poco-v080`, `2026-11-04-qdc-v080`: v0.8.0 on the POCO and on QDC, with LFM3 and
  the splice crash.
- `2026-11-14-runtime-upgrade`: llama.cpp v1.0.0 and ExecuTorch 1.6.0 on the emulator, the Mac
  and the phones; v1.0.0 was held back on what the Galaxy S24 measured.

**Harness.** `llama_bench_ab.sh` in
[`harness/prefill-decode/snapdragon/`](../../harness/prefill-decode/snapdragon/) builds two
llama.cpp commits with the app's CPU flags and alternates them on one phone; the answer checks
use [`harness/app-evals/`](../../harness/app-evals/).
