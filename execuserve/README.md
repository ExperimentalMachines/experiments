# ExecuServe experiments

The measurements behind [ExecuServe](https://github.com/ExperimentalMachines/execuserve), the
Android app that serves ExecuTorch models to other apps on the phone through OpenAI- and
Anthropic-compatible endpoints. Imported from its `docs/results/` at commit bfbfa853, which keeps
ExecuServe's own layout: one write-up per study and an evidence folder of the same name beside it.

| Study | Write-up | Evidence |
|---|---|---|
| First run on the POCO X8 Pro Max, 2026-09-29 | [`2026-09-29-poco-x8-pro-max.md`](studies/2026-09-29-poco-x8-pro-max.md) | none; its figures came from `tools/compat`'s checks |
| Vulkan GPU exports on three phones, 2026-10-04 | [`2026-10-04-vulkan.md`](studies/2026-10-04-vulkan.md) | [`2026-10-04-vulkan/`](studies/2026-10-04-vulkan/) |
| NPU exports on two phones, 2026-10-06 | [`2026-10-06-npu.md`](studies/2026-10-06-npu.md) | [`2026-10-06-npu/`](studies/2026-10-06-npu/) |
| Android 17's memory limits, tried by hand, 2026-11-04 | [`2026-11-04-memory-limits.md`](studies/2026-11-04-memory-limits.md) | [`2026-11-04-memory-limits/`](studies/2026-11-04-memory-limits/) |
| ExecuTorch 1.6.0, 2026-11-13 | [`2026-11-13-executorch-1.6.0.md`](studies/2026-11-13-executorch-1.6.0.md) | [`2026-11-13-executorch-1.6.0/`](studies/2026-11-13-executorch-1.6.0/) |

Notes on where things came from:

- The memory-limits study was never committed to ExecuServe; it came from the working tree.
- `2026-10-06-npu/exports/` holds the logs, reports and configs of the two QNN exports the NPU
  study ran (Qwen3-1.7B at 4k, SmolLM2-135M at 512), from `~/Desktop/execuserve-play/qnn`. The
  `.pte` files and tokenizers were left out.
- The Vulkan session's ExecuServe files are also in
  [`openweights/studies/vulkan/`](../openweights/studies/vulkan/), which kept its own copy.
- Each study's `codex/` folder holds the adversarial reviews of its write-up, kept verbatim.

## What stayed in ExecuServe, and why

The app; `tools/execuserve`, the command-line driver the studies call; `tools/compat`, the API
conformance checks that ExecuServe's README and CONTRIBUTING.md send people to (its `bench.py`
and `soak.py` only measure, but the phone runs call them from there); `tools/executorch`, which
builds the runtime the app ships; and `docs/ARCHITECTURE.md`, whose review log the Vulkan study
links to at the pinned commit.

## Running a study again

The phone scripts take ExecuServe's checkout from `REPO_ROOT`, and the phone's adb serial from
`SER`, which used to be written into them; `memtest.py` reads `SER` too. The phone runs
write a `start.txt` that holds the session's API key, which `.gitignore` refuses.

Carried over as it was: `2026-11-13-executorch-1.6.0/phone/run.sh` and `run2.sh` call
`$HERE/greedy.py`, which is in `../emulator/`.

Redacted on the way: in the Vulkan session's logs, Qualcomm device-cloud addresses, the
cluster name of the device the tunnel reached, and its serial, which becomes the same `unit-`
pseudonym the OpenWeights copy of that log uses. Each change is a `changed` row in
[`PROVENANCE.tsv`](PROVENANCE.tsv).
