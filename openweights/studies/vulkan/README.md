# Vulkan on the phone

**Question.** Do ExecuTorch's Vulkan GPU exports run on the test phones through the app and
through ExecuServe, and do they beat the CPU file?

**Write-up** (in OpenWeights): [`vulkan-on-device.md`](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/vulkan-on-device.md).

**Runs.** `runs/vulkan-2026-10-04/`, with its own README and the Codex reviews of the write-up
in `codex/`. ExecuServe's side of the same session is
[`execuserve/studies/2026-10-04-vulkan/`](../../../execuserve/studies/2026-10-04-vulkan/);
fourteen files are byte for byte in both (ExecuServe's logs, CSVs and reviews), because each
write-up kept its own copy of the evidence.

Redacted on import: Qualcomm device-cloud addresses and cluster names, the phone's home-network
address and its wireless-debugging name.
