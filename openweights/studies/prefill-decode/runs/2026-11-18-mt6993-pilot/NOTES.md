# Dimensity 9500 pilot, 2026-11-18 (Samsung Remote Test Lab, Galaxy Tab S12+ SM-X846B, Moscow)

Unit: ro.soc.model MT6993, Android 17 (SDK 37, user build), 11.5 GB visible, cores 1x 4.21 GHz +
3x 3.5 GHz + 4x 2.7 GHz, Mali-G1-Ultra MC12 (Vulkan), USB powered. Unrooted shell over the lab's
Remote Debug Bridge; one 2 h booking ended after about 20 minutes.

Bench: openweights/harness/prefill-decode/mediatek bench3 (pd_bench_mtk.cpp 926ec7b07614116a, llama.cpp b2e5e9b), Qwen3-0.6B-Q4_0,
6 threads, conversation WildChat-1M-18, 2 turns, --gen 64. LD_LIBRARY_PATH must put /system/lib64
before /system_ext/lib64 and /vendor/lib64 on Android 17, or the binary fails to link.

| cell | turn 2 prompt (457 tokens) | turn 2 decode (63 steps) |
|---|---|---|
| cpu:cpu | 685 ms | 1,257 ms |
| cpu:gpu | 698 ms | 3,135 ms |
| gpu:gpu | 5,274 ms | 3,192 ms |

NPU: the published Dimensity 9400 export (experimentalmachines/Qwen3-0.6B-ExecuTorch mtk/mt6991, 4k)
loads its shared weights, then the Neuron runtime refuses the compiled network ("Found an unsupported
target: EDPA_1_2", "NeuronModel_restoreFromCompiledNetwork - Failed to load compiled network"), and the
bench crashes on the null delegate (exit 139). See npu-neuron-log.txt (Neuron tags only). The public
NeuroPilot Express 8.0.8 compiler knows platform configs up to mt6991; an mt6993 network needs a
newer compiler.
