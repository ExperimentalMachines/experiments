# OpenWeights experiments

The device harness and every recorded run of [OpenWeights](https://github.com/ExperimentalMachines/openweights),
the Android app that runs open-weight models on the phone, imported from its `tools/` and `eval/`
at commit 81734dee (2026-11-17), plus eight run files that were never committed. The write-ups
stay in the app's `docs/research/`, because the app's KDoc cites them; each study here links to
them at that commit.

## Studies

| Study | Question | Write-ups |
|---|---|---|
| [`on-device-evals`](studies/on-device-evals/) | Do both runtimes answer alike on every chip; public benchmark scores; does the exported window matter | backend-parity, parity-five-socs, public-benchmarks, window-matrix and four more |
| [`retrieve-or-answer`](studies/retrieve-or-answer/) | When should the model search; the doubt gate; TypeSafe's reading | retrieve-or-answer, judged-decisions, typesafe-experiments, recommended-runtime |
| [`compiled-tool-calling`](studies/compiled-tool-calling/) | Why the compiled LFM2.5 narrates a search it never makes | executorch-tool-calling, executorch-tool-calling-qad |
| [`vulkan`](studies/vulkan/) | Do Vulkan GPU exports run and beat the CPU | vulkan-on-device |
| [`test-lab-archive`](studies/test-lab-archive/) | The Test Lab runs kept when the eval bucket was deleted | four, listed there |
| [`memory-paging`](studies/memory-paging/) | Swap, peak memory, and prefetching paged-out weights | swap-and-peak-memory |
| [`prefill-decode`](studies/prefill-decode/) | Prefill on one processor, decode on another; heat and threads | snapdragon-pd-matrix; the cross-vendor runs are not written up yet |
| [`runtime-upgrades`](studies/runtime-upgrades/) | Did each llama.cpp and ExecuTorch upgrade keep speed and answers | llama-cpp-v080, llama-cpp-v100-executorch-160 |
| [`shipped-build`](studies/shipped-build/) | The signed release, measured as it ships | llama-cpp-v080 |
| [`prefix-reuse`](studies/prefix-reuse/) | What a retained prefix saves, and what a second turn re-reads | reply-end-and-reuse |

A few questions have a harness here but no study folder, because their outputs were working files
outside the repository and the write-up holds the numbers: where the date goes in the prompt and
how the model routes tools (`harness/host-prompt/`) and what a picture costs
(`image_speed_probe.py`). Whether NPU prefill pays for itself on real conversations
(`harness/npu/multiturn/`) keeps its two sweep records beside the scripts that read them. See
[`harness/README.md`](harness/README.md).

## What stayed in OpenWeights, and why

- The instrumented benchmarks and probes under `app/` and `core/` (`BenchmarkSuite`,
  `ThreadCountBenchmark`, `DecisionSuiteOnDeviceTest` and the rest): they compile against the
  engine. The harness here builds and runs them from an OpenWeights checkout.
- `docs/research/` and `docs/CONTEXT.md`: the code cites them.
- `tools/npu/` apart from the benchmarks and the multi-turn replay: it builds the runtime the app
  links. Its `build_pd_libs.sh` also compiles `cpu_forward_bench.cpp`, so that file is in both
  places until the script stops needing it.
- `tools/release`, `tools/hooks`, `tools/review`, `tools/executorch`, and the two test-image
  generators the instrumented tests use: product tooling.
- `tools/run_python_tests.sh` still runs `tools/eval/bench`'s tests in OpenWeights' CI.

## What did not come across

- `tools/snapdragon/results/2026-11-04-chat-capture/` (untracked, 38 MB): a dump of the app's
  data from the phone (its databases, preferences and a full logcat) and screen frames for the
  website. It is personal data and site material, not a measurement.
- `tools/eval/results/aar/` (untracked): classes unpacked from an AAR while inspecting it.
- `tools/snapdragon/results/2026-10-24-sm8850/.pull.err` (untracked): the copy loop's adb errors.
- `tools/eval/__pycache__/` and `tools/eval/bench/ifeval/__pycache__/`: compiled Python,
  committed by mistake.
- Two tokenizers committed into `tools/eval/results/`: model files.
- The raw logcats OpenWeights' own `.gitignore` keeps out of git (`*.logcat` under
  `tools/eval/results/`, `*.logcat.txt` under `tools/memory/results/`): they stay on the Mac.
- An uncommitted "Run it yourself" section in `2026-11-16-reuse-boundary/README.md`: every
  tracked file was read from the commit, not the working tree.

Changed on the way: the 100 Test Lab logcats were cut to the app's own lines (10.4 million
lines became 61 thousand), because they were whole devices' logs with network names, addresses
and test accounts; the full logs stay in OpenWeights at 81734dee. Every device serial a run
record named (26 lab units, in `unit` fields, chain logs and adb listings) is replaced by a
stable pseudonym, `unit-` and six hex digits, so results still group by unit (`analyze_pd.py`
on the SM8850 rows gives the same tables, with the pseudonyms in place of the serials). The Google
Cloud projects the Device Streaming runs used (two ids and a project number) become
`gcp-project-` and six hex digits the same way, and the two login addresses in their notes and
logs become `account-` and six hex digits. Redacted: the phone's wireless-debugging name, its home-network address, and
the Qualcomm device-cloud addresses and cluster names in the Vulkan session's logs. Each change is a
`changed` row in [`PROVENANCE.tsv`](PROVENANCE.tsv).

## Running the harness

```sh
export OPENWEIGHTS=~/mobile-inference   # the app checkout the scripts build and drive
```

Scripts run from this repository's root or by their full path; results land in the study they
belong to. The exception is the tool-call probe, kept as it ran (see
[`compiled-tool-calling`](studies/compiled-tool-calling/README.md)). The phone-side procedure (finding the phone, installing around the ROM's block on
`adb install`) is OpenWeights' `phone-deploy` skill.
