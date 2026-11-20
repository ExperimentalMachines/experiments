# Vulkan GPU exports on the phone

**Whether a model compiled for the phone's GPU runs, and whether it beats the same model on the
CPU.**

`written up in OpenWeights` `3 phones` `3 to 4 Oct 2026`

> **The question.** [ExecuTorch](../../../docs/glossary.md#executorch) can compile a model for the
> phone's GPU through [Vulkan](../../../docs/glossary.md#vulkan), as a separate
> [`.pte`](../../../docs/glossary.md#pte) file from the CPU one. Do those GPU files run correctly
> through OpenWeights and through its sister app ExecuServe, which serves models to other apps over
> OpenAI- and Anthropic-style endpoints? And are they faster than the CPU file of the same model?

> [!TIP]
> **What we found**
> - The GPU file runs correctly on all three GPUs tried: nine of nine engine tests on each, and
>   ExecuServe's compatibility suites pass on the POCO's GPU build.
> - **Whether it is faster depends on the GPU.** On the POCO's Mali GPU the CPU file writes replies
>   **2.2 to 3.9 times** as fast; the GPU's only win is reading a long prompt, 1.1 to 1.6 times as
>   fast.
> - On the Snapdragon 8 Elite Gen 5's Adreno GPU, the GPU reads long prompts 1.2 to 2.0 times as fast
>   and writes 1.2 to 1.4 times as fast with a long prompt in the cache, but writes short replies at
>   0.6 to 0.75 times the CPU's speed.
> - One model (Qwen3 0.6B), one phone per GPU, cool phones over a few minutes.

## Key numbers

| What | Result | Where |
|---|---|---|
| Reply writing after a 27-token prompt, CPU file against GPU file | 58.37 against **15.79** tok/s (means of three runs) | POCO X8 Pro Max (Mali GPU), Qwen3 0.6B, through ExecuServe |
| The same after a 702-token prompt | 29.99 against 13.43 tok/s | the same |
| Prompt reading, 702 tokens | 329.16 against 411.35 tok/s | the same |
| Reply writing after a 27-token prompt | 125.14 against 88.06 tok/s | Snapdragon 8 Elite Gen 5 (Adreno GPU) on QDC, through ExecuServe |
| Reply writing after a 702-token prompt | 50.47 against **62.28** tok/s | the same |
| OpenWeights engine test, 937-token prompt, three turns | GPU: prompt 968.0 to 1116.8 tok/s, reply 51.4 to 54.8; CPU: 548.3 to 556.1 and 38.0 to 38.9 | Snapdragon 8 Elite Gen 5, the app's engine |
| Memory the process holds for the same model | Mali GPU 2208 to 2211 MiB against CPU 1186 to 1189; Adreno GPU 264 to 268 against CPU 1164 to 1166 | POCO and Snapdragon 8 Elite Gen 5, engine test |
| Snapdragon 8 Gen 3, GPU file only | prompt 585.6 to 639.2 tok/s, reply 40.4 to 40.7; no CPU run was made | Snapdragon 8 Gen 3 on QDC, engine test |

## Why it matters

The app ships one ExecuTorch library that carries both the CPU and the GPU paths. When a phone's
driver cannot run the GPU shaders, the app learns it once, from the runtime's own error, and stops
offering GPU files on that phone; that build went to production on 4 October 2026. For a developer:
"the GPU is faster" is a property of a GPU's driver and memory system, not of GPUs, so measure per
GPU family, and measure reply writing, which is what a reader waits on.

## What was tested

- **Phones.** The POCO X8 Pro Max (MediaTek MT6991, Mali-G925-Immortalis GPU), the maintainer's
  own phone over wireless debugging; on [Qualcomm Device Cloud](../../../docs/glossary.md#qualcomm-device-cloud-qdc)
  a Snapdragon 8 Gen 3 handset (SM8650, Adreno 750) and a Snapdragon 8 Elite Gen 5 handset (SM8850,
  Adreno). See [all devices](../../../docs/devices.md).
- **Runtimes.** ExecuTorch 1.5.1 in both apps (its Android package with the Vulkan backend);
  ExecuTorch's own command-line runner from v1.4.0, to measure the GPU path without the app.
- **Models.** Qwen3 0.6B at a 2,048-token window: the GPU file (616 MB) against the CPU files, an
  early [round-to-nearest](../../../docs/glossary.md#round-to-nearest-rtn) one and the later
  [GPTQ](../../../docs/glossary.md#gptq) one (497 MB), both [8da4w](../../../docs/glossary.md#8da4w).
  A 32k GPU file was tried on the POCO and did not run. The 1.5.1 runtime bump was also checked
  with Qwen3 1.7B and SmolLM2 135M.
- **Method.** The command-line runner with [greedy decoding](../../../docs/glossary.md#greedy-decoding);
  OpenWeights' instrumented engine test (nine tests, including a 937-token, three-turn throughput
  matrix); ExecuServe's own run history over three 27-token and three 702-token prompts at
  temperature 0.
- **When.** 3 and 4 October 2026.

## What's in this folder

| Run | What it holds |
|---|---|
| [`runs/vulkan-2026-10-04/`](runs/vulkan-2026-10-04/) | The whole session on the three phones: the command-line runner on CPU and GPU, the engine's nine tests on the GPU and CPU files, ExecuServe's run histories and compatibility suites, greedy CPU and GPU answers, the 1.5.1 runtime bump, and in `codex/` every Codex review of the bump, the Vulkan code and the write-up. 43 files. |

The run has its own README, which says for each file whether it was captured when it ran or
transcribed from terminal output afterwards. Timings are in the two ExecuServe CSVs (one row per
request) and in the engine logs; the six `*.instrument.txt` files are the raw test outputs.
ExecuServe's side of the same session is
[`execuserve/studies/2026-10-04-vulkan/`](../../../execuserve/studies/2026-10-04-vulkan/): its
fourteen files are byte for byte in both, because each write-up kept its own copy of the evidence
(two Codex reviews carry different names here). Redacted on import: Qualcomm device-cloud addresses
and cluster names, the phone's home-network address and its wireless-debugging name.

## Run it again

You need an OpenWeights checkout with its engine test APK built, the model files from the
`experimentalmachines/Qwen3-0.6B-ExecuTorch` repository on Hugging Face pushed to the phone, and a
phone whose GPU reports Vulkan 1.1 or later. On QDC, `adb forward` binds on Qualcomm's side, so
call servers from `adb shell` and fetch models with the phone's own `curl`. There is no
Vulkan-specific script in this repository; ExecuTorch's runner and ExecuServe's tools are built from
their own repositories.

```sh
export OPENWEIGHTS=~/path/to/openweights
cd "$OPENWEIGHTS" && ./gradlew :core:engine:assembleDebugAndroidTest

# The engine's nine tests on the GPU file (the numbers are in logcat, not in the test output)
adb install -r -t core/engine/build/outputs/apk/androidTest/debug/engine-debug-androidTest.apk
adb shell am instrument -w -r \
  -e class io.github.alpharomercoma.openweights.core.engine.ExecuTorchOnDeviceTest \
  -e pte /data/local/tmp/owvk/Qwen3-0.6B-vulkan-8da4w-2k.pte \
  -e tokenizer /data/local/tmp/owvk/Qwen3-0.6B-vulkan-8da4w-2k.tokenizer.json \
  io.github.alpharomercoma.openweights.core.engine.test/androidx.test.runner.AndroidJUnitRunner
adb logcat -d -s ExecuTorchOnDevice:I
```

Where HyperOS refuses `adb install`, push the APK and install it with `pm install -r -t --user 0`.
The command-line runner's exact download, push and run lines are at the top of
`poco-llama_main-1.4.0-2026-10-03.log`, the four-question greedy script in
`poco-greedy-cpu-vs-gpu-2026-10-04.log`, and ExecuServe's request recipe in the header of
`sm8850-execuserve-runs-2026-10-04.csv`.

## Caveats

> [!NOTE]
> - Process memory for a GPU file depends on where the driver places its buffers, so it is not the
>   model's size. On the Adreno phone the GPU file showed far less than the CPU file, on the Mali
>   phone far more.
> - The CPU's prompt reading on a 27-token prompt varies a lot run to run (197.08 to 391.30 tok/s on
>   the POCO), so a single short-prompt reading says little.
> - The 32k file's log calls it "killed while allocating its KV cache", but the saved output shows
>   neither a finished load nor the kill, and the write-up says the cause was not captured.
> - The POCO's ExecuServe suites ran on one debug build and its timing CSV names another; both are
>   recorded as they appear.
> - Still open: recommending the CPU file where it is faster (a rule for older Adrenos needs a CPU run
>   on the Snapdragon 8 Gen 3 first), larger models and the hybrid families on the Vulkan backend,
>   and sustained runs on a hot phone.

## Read more

- Write-up in OpenWeights: [Vulkan GPU exports on the phone](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/vulkan-on-device.md)
- ExecuServe's side: [`execuserve/studies/2026-10-04-vulkan/`](../../../execuserve/studies/2026-10-04-vulkan/).
- Related: [runtime upgrades](../runtime-upgrades/) (later ExecuTorch versions) and
  [prefill and decode](../prefill-decode/) (CPU, GPU and NPU pairings through llama.cpp).
- Words used here: [glossary](../../../docs/glossary.md).
