# Runtime upgrades

**Whether each new version of llama.cpp and ExecuTorch kept the app's speed, memory and answers
on real phones.**

`written up in OpenWeights` `6 phones, an emulator, a Mac` `31 Oct to 14 Nov 2026`

> **The question.** OpenWeights runs models through two libraries:
> [llama.cpp](../../../docs/glossary.md#llamacpp) for [GGUF](../../../docs/glossary.md#gguf)
> files and [ExecuTorch](../../../docs/glossary.md#executorch) for compiled
> [`.pte`](../../../docs/glossary.md#pte) files. When the app moved to a newer version of each, did
> it stay as fast, use the same memory and give the same answers on the phones people use?

> [!TIP]
> **What we found**
> - **ExecuTorch 1.6.0 shipped; llama.cpp v1.0.0 was put back the morning it landed.** On a Galaxy
>   S24, v1.0.0 wrote replies **20% to 23% slower** at the 1,500 to 6,000 tokens of context a
>   tool-using chat sits in. The Mac (11% to 13%) and the emulator (about 4%) did not predict that
>   phone.
> - ExecuTorch 1.6.0 showed no accuracy difference: 67 against 65 of 100 fresh GSM8K questions
>   (p = 0.82). One line, a faster exponential in its attention, makes all the difference in its
>   replies; putting the old one back reproduces 1.5.1's answers on 100 of 100.
> - **llama.cpp v0.8.0 shipped, and was not free.** In the app the same chat read the same tokens
>   with the same prefix reuse at speeds within noise, but its new [KleidiAI](../../../docs/glossary.md#kleidiai)
>   kernels raised memory after loading by **up to 41%** on a phone whose CPU has SME, the app's
>   default LFM2.5 file is about 10% slower to read and 13% slower to write in llama-bench on the
>   POCO, and Gemma 3's replies changed. It brought LFM3 support and 40% to 47% faster NPU decode
>   in Qualcomm's backend benchmarks, not measured in the app.
> - The upgrade work found an older crash in the app's own engine, not in llama.cpp: Qwen3 1.7B
>   failed 5 of 20 questions on a bad read, now fixed.

## Key numbers

| What | Result | Where |
|---|---|---|
| Decode at 5,832 tokens of context, v0.8.0 against v1.0.0 | 34.4 and 35.1 against **24.8 and 28.9** tok/s (ratio 0.77) | Galaxy S24 on Firebase Test Lab, two units per version, the app's engine |
| Decode at 2,048 and 4,096 tokens, v1.0.0 over v0.8.0 | 0.747 and 0.752 (Qwen3 1.7B Q8_0); 0.833 and 0.789 (LFM2.5 1.2B) | POCO X8 Pro Max, llama-bench with the app's CPU flags, median of 4 rounds |
| The same with an unshipped patch that keeps one attention step in 16-bit | 0.978 and 1.038 (Qwen3); 0.920 and 0.913 (LFM2.5) | POCO X8 Pro Max |
| Memory after loading, b10549 to v0.8.0 | 1948 to 2740 MB (up to 41%) | Snapdragon 8 Elite Gen 5 on QDC, Qwen3.5 2B, the app's engine |
| v0.8.0 with KleidiAI switched off | 16.2 against 25.6 tok/s decode with it on, about 37% slower (the write-up's 34.3% is against b10549's 24.6) | POCO X8 Pro Max, Qwen3 1.7B Q8_0, llama-bench |
| NPU decode, b10549 to v0.8.0 | +46.9% (LFM2.5 1.2B), +40.2% (Qwen3 1.7B) | Snapdragon 8 Elite Gen 5 Hexagon NPU, Qualcomm's preset benchmark tools |
| GSM8K, 100 fresh questions, ExecuTorch 1.5.1 against 1.6.0 | 67 against 65, 8 gained and 10 lost, p = 0.82 | 16 KB emulator, LFM2.5 1.2B `.pte` on the CPU |
| The older engine crash | 5 of 20 questions failed before the fix; AddressSanitizer found 0 bad reads in 35 turns after it | Snapdragon 8 Elite Gen 5; the engine's code replayed on the Mac |

## Why it matters

What ships is ExecuTorch 1.6.0 with llama.cpp v0.8.0. The work also changed the app: its memory
fit estimate now adds a per-format allowance on CPUs that report SME, because there the file size
no longer says what the weights cost, and the reasoning budget now stops at every end tag a model's
template lists. For anyone building on llama.cpp: an upgrade that is level or faster on an empty
context can lose a fifth of its decode speed at the 2,000 to 6,000 tokens a real tool-using chat
carries, and neither a Mac nor an emulator predicts the phone. Measure decode at depth, on a phone.

## What was tested

- **Phones.** The POCO X8 Pro Max (MediaTek MT6991), the maintainer's own phone; Qualcomm reference
  handsets with a Snapdragon 8 Elite Gen 5 (SM8850) and a Snapdragon 8 Gen 2 (SM8550) on
  [Qualcomm Device Cloud](../../../docs/glossary.md#qualcomm-device-cloud-qdc); on
  [Firebase Test Lab](../../../docs/glossary.md#firebase-test-lab) a Galaxy S24 (Snapdragon 8 Gen 3),
  a Pixel 9 Pro (Tensor G4) and an OPPO A79 5G (MediaTek Dimensity), the last crawled by Test Lab's
  automatic UI robot. See [all devices](../../../docs/devices.md).
- **Machines.** A 16 KB-page arm64 Android emulator and a Mac (Apple M5), both shared with other
  work at the time.
- **Runtimes.** llama.cpp b10549, v0.8.0 and v1.0.0; ExecuTorch 1.5.1 and 1.6.0, in the app's own
  builds.
- **Models.** LFM2.5 1.2B (several GGUF files and the default `.pte`), LFM2.5 2.6B, LFM2.5-VL 3B,
  LFM3 1.7B, Qwen3 1.7B, Qwen3.5 2B, Gemma 3 1B, and small Vulkan `.pte` files of SmolLM2 135M and
  Qwen2.5 0.5B.
- **Method.** A/B runs that alternate the old and new build on one phone, llama-bench built with
  the app's CPU flags at several context depths, the app's instrumented engine tests on both builds,
  GSM8K, IFEval and BFCL samples on the emulator, a context-depth decode test on Test Lab, and
  AddressSanitizer replays of the engine on the Mac. Not run: Qualcomm's and MediaTek's NPUs through
  ExecuTorch 1.6.0 on hardware, and ExecuTorch's CPU speed on a phone.
- **When.** 31 October, 4 November, and 13 and 14 November 2026.

## What's in this folder

| Run | What it holds |
|---|---|
| [`runs/2026-10-31-llama-v080/`](runs/2026-10-31-llama-v080/) | llama.cpp b10549 against v0.8.0 on a Snapdragon 8 Elite Gen 5: engine tests on both builds, perplexity and llama-bench on the CPU, GPU and NPU, KleidiAI and memory probes, the phone-side A/B scripts. 81 files. |
| [`runs/2026-11-04-poco-v080/`](runs/2026-11-04-poco-v080/) | v0.8.0 on the POCO: KleidiAI on and off in both orders, the reasoning-budget test before and after its fix, a three-turn chat on both builds, release smoke screenshots. 11 files. |
| [`runs/2026-11-04-qdc-v080/`](runs/2026-11-04-qdc-v080/) | v0.8.0 on QDC with LFM3: engine and tool-turn tests, peak memory against the fit estimate, the Mac budget sweep, and the AddressSanitizer replay of the splice crash. 24 files. |
| [`runs/2026-11-14-runtime-upgrade/`](runs/2026-11-14-runtime-upgrade/) | llama.cpp v1.0.0 and ExecuTorch 1.6.0 on the emulator, the Mac and the phones: benchmarks, parity, the cache contract, the POCO and SM8550 three-way run, the Galaxy S24 context-depth test, the shipping build on three Test Lab phones, and the unshipped 16-bit patch. 112 files. |

The A/B logs are plain text, one line per llama-bench result. The 2026-11-14 folder holds mostly
JSON reports from the app's eval classes; its phone sub-folders (`phones-3way/`, `s24-testlab/`,
`shipping-build-devices/`) and the patch lead (`fa-f16-kq-lead/`) have READMEs of their own.

## Run it again

You need an OpenWeights checkout, a Mac with the Android NDK, and one phone over adb; wake, unlock
and cool the phone first. The QDC and Test Lab runs need a reservation or a Google Cloud project
with a bucket. Models are not in this repository.

```sh
export OPENWEIGHTS=~/path/to/openweights

# Decode at depth, two llama.cpp commits, one phone: build both, then alternate them
openweights/harness/prefill-decode/snapdragon/llama_bench_ab.sh build <old commit> <new commit>
openweights/harness/prefill-decode/snapdragon/llama_bench_ab.sh run <adb serial> <model.gguf on the Mac> 6

# The app's device tests (SpeedProbe, ContextLengthBenchmark, ExecuTorchParityEval and the rest)
# on a Test Lab phone
openweights/harness/app-evals/run_matrix_ftl.sh <model-id> <os-version> <prefix> <engine> <gs-bucket>
```

`llama_bench_ab.sh` builds from the llama.cpp submodule in the OpenWeights checkout with the app's
CPU flags and prints per-depth medians. The answer checks use [`harness/app-evals/`](../../harness/app-evals/).
Several pieces cannot be rerun as kept: the three-way run needs a per-device adb wrapper that was
not kept, the GSM8K grader imports from a path that does not exist here (and needs `langdetect`),
the 2026-10-31 phone scripts need two llama.cpp source trees and Qualcomm's preset tool packages,
and the Mac drivers have no build script.

## Caveats

> [!NOTE]
> - The Galaxy S24 depth record does not name its model. The other Galaxy S24 records use LFM2.5
>   1.2B QAD-Q4_0, so that is likely but not stated.
> - LFM2.5's slowdown in llama-bench on the POCO is not isolated, and the in-app chat does not show it.
> - An unwritten result sits in the 2026-10-31 A/B logs: on the Snapdragon 8 Elite Gen 5, forcing
>   llama.cpp's older armv8.6 CPU variant prefilled Qwen3 1.7B at 291.0 and 293.7 tok/s against 117.0
>   and 119.7 for the variant the app picks. Neither write-up states it.
> - On the Mac, the cache-contract check holds for all seven models, but resident memory rose on
>   three (Qwen3 1.7B 3212 to 4848 MB); the write-up does not mention it, and the Mac was shared.
> - The write-up calls the splice crash an out-of-bounds read; AddressSanitizer classes it as a
>   heap-use-after-free read.
> - Two run READMEs are out of date: the 2026-10-31 one says "not yet written up", and the
>   2026-11-14 one says no phone ran its work, which is true only of the folders it lists.

## Read more

- Write-ups in OpenWeights:
  [llama.cpp b10549 to v0.8.0](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/llama-cpp-v080.md)
  (its release sections belong to [the shipped build](../shipped-build/)) and
  [llama.cpp v0.8.0 to v1.0.0, and ExecuTorch 1.5.1 to 1.6.0](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/llama-cpp-v100-executorch-160.md)
- Related: [the shipped build](../shipped-build/), [prefill and decode](../prefill-decode/) (the
  same QDC phones), [prefix reuse](../prefix-reuse/).
- Harness: `llama_bench_ab.sh` in [`harness/prefill-decode/snapdragon/`](../../harness/prefill-decode/snapdragon/),
  and [`harness/app-evals/`](../../harness/app-evals/).
- Words used here: [glossary](../../../docs/glossary.md).
