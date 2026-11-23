# Devices

Every phone, tablet and emulator that ran something in these studies, and where it was. Chip
names are the ones the run records and write-ups give. Run records name each physical unit by a
[pseudonym](glossary.md#unit-pseudonym), never by its serial.

The POCO X8 Pro Max is the maintainer's own phone and ran more than anything else. The sources
agree it is a MediaTek MT6991 but disagree on the marketing name of the chip (Dimensity 9400 in
some, Dimensity 9500s in others), so this repository calls it by the part number.

## Every device

| Device | Chip | Where it ran | Studies |
|---|---|---|---|
| POCO X8 Pro Max | MediaTek MT6991, Mali-G925 GPU | Own phone, over USB or wireless debugging | Nearly every OpenWeights study; ExecuPack findings 36 and 38; four of ExecuServe's five studies |
| Snapdragon 8 Elite Gen 5 reference phone | Qualcomm SM8850, Adreno GPU, Hexagon NPU | [Qualcomm Device Cloud](glossary.md#qualcomm-device-cloud-qdc), root shell | [prefill-decode](../openweights/studies/prefill-decode/), [runtime-upgrades](../openweights/studies/runtime-upgrades/), [vulkan](../openweights/studies/vulkan/); ExecuPack findings 38 and 42; ExecuServe Vulkan, NPU and memory limits |
| Snapdragon SM8975 reference phones (three) | Qualcomm SM8975, Adreno GPU | [Qualcomm Device Cloud](glossary.md#qualcomm-device-cloud-qdc), root shell | ExecuPack [KV cache](../execupack/studies/kv-cache/) |
| Snapdragon 8 Gen 3 reference phone | Qualcomm SM8650, Adreno 750 | Qualcomm Device Cloud | [on-device-evals](../openweights/studies/on-device-evals/), [retrieve-or-answer](../openweights/studies/retrieve-or-answer/), [shipped-build](../openweights/studies/shipped-build/), [vulkan](../openweights/studies/vulkan/); ExecuServe Vulkan |
| Snapdragon 8 Gen 2 reference phone | Qualcomm SM8550 | Qualcomm Device Cloud | [prefill-decode](../openweights/studies/prefill-decode/), [runtime-upgrades](../openweights/studies/runtime-upgrades/) |
| Galaxy S24 | Snapdragon 8 Gen 3 | [Firebase Test Lab](glossary.md#firebase-test-lab) | [prefix-reuse](../openweights/studies/prefix-reuse/), [runtime-upgrades](../openweights/studies/runtime-upgrades/) |
| Galaxy S24 Ultra | Snapdragon 8 Gen 3 (SM8650) | [Firebase Device Streaming](glossary.md#firebase-device-streaming) | [prefill-decode](../openweights/studies/prefill-decode/) (cross-vendor runs) |
| Galaxy S24+ | Samsung Exynos 2400 | Firebase Test Lab; Device Streaming | [on-device-evals](../openweights/studies/on-device-evals/), [retrieve-or-answer](../openweights/studies/retrieve-or-answer/), [test-lab-archive](../openweights/studies/test-lab-archive/); NPU access check |
| Galaxy S25 Ultra | Snapdragon 8 Elite (SM8750) | Firebase Test Lab; Device Streaming | [on-device-evals](../openweights/studies/on-device-evals/), [retrieve-or-answer](../openweights/studies/retrieve-or-answer/), [prefix-reuse](../openweights/studies/prefix-reuse/), [test-lab-archive](../openweights/studies/test-lab-archive/) |
| Pixel 9 | Google Tensor G4 | Firebase Test Lab | [prefix-reuse](../openweights/studies/prefix-reuse/) |
| Pixel 9 Pro | Google Tensor G4, Mali-G715 | Firebase Test Lab | [runtime-upgrades](../openweights/studies/runtime-upgrades/) |
| Pixel 10 Pro XL | Google Tensor G5 | Firebase Test Lab; Device Streaming | [on-device-evals](../openweights/studies/on-device-evals/), [retrieve-or-answer](../openweights/studies/retrieve-or-answer/), [test-lab-archive](../openweights/studies/test-lab-archive/); NPU access check |
| Pixel 11 | Google Tensor G6 | Firebase Device Streaming | NPU access check, in [prefill-decode](../openweights/studies/prefill-decode/) |
| OPPO A79 5G | MediaTek Dimensity, Mali-G57 | Firebase Test Lab (automated crawl) | [runtime-upgrades](../openweights/studies/runtime-upgrades/) |
| Galaxy Tab S10+ | MediaTek Dimensity 9300+ (MT6989) | Firebase Test Lab; Device Streaming | [prefix-reuse](../openweights/studies/prefix-reuse/), [prefill-decode](../openweights/studies/prefill-decode/) (cross-vendor runs) |
| Galaxy Tab S12+ | MediaTek Dimensity 9500 (MT6993) | [Samsung Remote Test Lab](glossary.md#samsung-remote-test-lab-rtl), one short booking | [prefill-decode](../openweights/studies/prefill-decode/) (pilot) |
| Galaxy Z Flip7 | Samsung Exynos 2500 | Samsung Remote Test Lab | [on-device-evals](../openweights/studies/on-device-evals/) (partial column) |
| Android emulator, 16 KB pages | arm64 image on the Mac | The maintainer's Mac | [runtime-upgrades](../openweights/studies/runtime-upgrades/), [shipped-build](../openweights/studies/shipped-build/); ExecuServe ExecuTorch 1.6.0 |

## Machines that are not phones

- **An Apple M5 Mac** ran the host prompt evals (llama-server), offline replays and the Qwen3
  divergence reconstruction.
- **Four Amazon EC2 machines** exported and solved ExecuPack's published models in November 2026;
  what they held when they were shut down is in
  [aws-runs-2026-11](../execupack/studies/aws-runs-2026-11/).
- **Build hosts** (a Linux x86_64 box, CI runners, a Modal host for the Qualcomm export) built
  files but measured nothing on a device.

## Where the labs fit

| Lab | What it is good for | What it cannot do |
|---|---|---|
| Own phone | Anything, for as long as needed | Only one chip |
| Firebase Test Lab | Many phone models, unattended, repeatable | No interactive shell; results come back as files |
| Firebase Device Streaming | An interactive adb session on a retail phone | Short reservations, no root |
| Qualcomm Device Cloud | Qualcomm reference phones with a root shell, for NPU work | Reference hardware, not retail phones |
| Samsung Remote Test Lab | Galaxy models not offered elsewhere | Short bookings that can end early |
