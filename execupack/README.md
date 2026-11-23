# ExecuPack experiments

**The measurements behind [ExecuPack](https://github.com/ExperimentalMachines/execupack), the
pipeline that compiles open-weight chat models for phones and publishes them.**

`49 findings` `6 studies` `Sep to Nov 2026`

ExecuPack takes models from Hugging Face and exports them to ExecuTorch [`.pte`](../docs/glossary.md#pte)
files for a phone's CPU ([XNNPACK](../docs/glossary.md#xnnpack)), its GPU
([Vulkan](../docs/glossary.md#vulkan)), Qualcomm's NPU ([QNN](../docs/glossary.md#qnn)) and
MediaTek's NPU ([NeuroPilot](../docs/glossary.md#neuropilot)). Each of those paths broke, ran out of
memory or quietly lost accuracy in ways the vendors did not document. The project keeps every such
finding, with its evidence, so that each pipeline default can point at a measurement.

## What we found

> [!TIP]
> - **Vendor sample code can run cleanly and still compute the wrong model.** MediaTek's stock
>   Qwen path builds its rotary embedding at base 10,000 for a model trained at 1,000,000 (KL 0.082
>   from the original before any quantization); its Gemma 3 path uses one rotary table where the
>   model needs two (KL 0.177). Both were patched to an exact match. Check vendor code against the
>   original model at full precision before quantizing anything.
> - **Plainly rounded 4-bit weights keep a model fluent and can take away its tool use.** In a
>   one-tool test on a Galaxy S25 Ultra, the round-to-nearest LFM2.5 1.2B file made 10% of the searches it needed, against 49% after a
>   [GPTQ](../docs/glossary.md#gptq) solve, so every 4-bit file is now solved. Even solved, Qwen3
>   0.6B kept 0 of 4 tool decisions and ships with full-precision layers.
> - **"It runs" is not a release test.** A file is published only after a smoke test, a
>   [decision gate](../docs/glossary.md#decision-gate) against the full-precision model, and a byte
>   check that its 4-bit codes are the ones the solve produced.
> - **The build machine is an input.** One quantizer's search depended on the build CPU, so the
>   same model built on two runners wrote different text; a GPTQ solve on an Apple M5 Mac produced
>   NaN. Solves now run on Linux only.
> - **With MediaTek's NPU quantization, 4-bit weights are 3 to 170 times further from the
>   original** than 8-bit ones, simulated on a GPU host. GPTQ helps two of three models and makes
>   LFM2.5 worse, so 8-bit stays the published recipe.
>   → [mtk-a16w4](studies/mtk-a16w4/)

## The research log at a glance

The [research log](research/README.md) has 49 findings, dated 12 September to 19 November 2026,
each with its evidence. Grouped by what they are about, in plain words. A finding is never
edited to fit a later result: a correction is a new row, and the tables below say which rows
were corrected or superseded.

### Toolchain breakage: wheels, SDKs and the dynamic loader

Qualcomm's and MediaTek's export paths failed on packaging before any model work began.

| Finding | What it found |
|---|---|
| [3](research/README.md) | Qualcomm's LLM package imports every model family's converter when it loads, so it needs torchtune and torchvision even for Qwen. |
| [4](research/README.md) | Qualcomm's export script fails without a build folder even when it only compiles. |
| [5](research/README.md) | The ExecuTorch wheel leaves out the model parameter files Qualcomm's scripts point at; ExecuPack carries its own copies. |
| [6](research/README.md) | Qualcomm's backend sets up its C++ library only in the process that imports it, so a child process cannot load the NPU library. |
| [7](research/README.md) | The same setup changes the library search path too late for the loader, so the export fails at the very end, after a full calibration. |
| [14](research/README.md) | MediaTek's LLM export scripts are not in the wheel, its converter runs only on Python 3.10, and the scripts need transformers 4.x: a separate tool environment. |
| [16](research/README.md) | MediaTek's NeuroPilot SDK comes with only a confidential license PDF, so the SDK is never committed, cached or published. |
| [19](research/README.md) | ExecuTorch 1.4.0 calls a MediaTek function missing from the SDK version ExecuTorch's own CI installs; ExecuTorch's CI never reaches that path. |
| [21](research/README.md) | The newer SDK that has the function cannot find the C++ library it ships beside it; fixed by pointing the loader at that folder. |

### Export memory: sizing the build machine

How much memory an export needs, and the estimate ExecuPack uses to refuse builds that cannot fit.

| Finding | What it found |
|---|---|
| [2](research/README.md) | A CPU export's memory grows with the square of the context window: Qwen3 0.6B at 32,768 tokens needs 30,064,771,072 bytes just for attention masks. |
| [15](research/README.md) | MediaTek's calibration keeps every layer's full-precision cache for every step; at a 2,048-token cache the build machine ran out. Corrected by 18. |
| [18](research/README.md) | Correction to 15: the calibration file holds 9 prompts, not 8, so the memory estimates were too low. |
| [23](research/README.md) | The first MediaTek export used up to 1.37 times the estimated memory; the true peak could only be bounded. Superseded by 25. |
| [24](research/README.md) | The automatic exporter never builds MediaTek's working 512-token window, because it is not one of the configured sizes. Open. |
| [25](research/README.md) | MediaTek's peak memory, measured: 29,592,731,648 bytes against an estimate of 21,139,292,160, while preparing calibration inputs. The estimate was refitted and 2,048 tokens is now skipped for Qwen3 0.6B. |
| [31](research/README.md) | Counting only the attention layers of hybrid LFM2 models takes their 32k estimate from about 28.4 GB to 16.3 GB (1.2B) and 50.0 GB to 23.4 GB (2.6B), so windows once refused now fit. |
| [33](research/README.md) | MediaTek's calibration can stream in one pass with bit-identical results, in 78 s instead of 214 s and with no calibration disk at all. Replaces the fix of 17 and 20. |
| [39](research/README.md) | MediaTek LFM2.5 builds at 16k succeeded only when each chunk held at most one attention layer; why the other build failed is not known. |
| [40](research/README.md) | A smaller cloud runner with no swap allowed caps which windows fit: every 32k and most 16k builds fall outside it. |
| [47](research/README.md) | MediaTek's converter holds weights in one protobuf capped at 2 GB, so Gemma 2 2B's output head cannot convert and Gemma 2 has no MediaTek build. |

### Export time and where exports run

What takes the time in an export, and which machines ran them.

| Finding | What it found |
|---|---|
| [1](research/README.md) | GitHub's free runner has 4 CPUs and ships with 3 GiB of swap, so ExecuPack frees disk and adds 24 GiB of swap. |
| [8](research/README.md) | The first Qualcomm NPU exports of Qwen3 0.6B (2,048 tokens) took 3,637 s and 2,434 s and were published. |
| [10](research/README.md) | A 4,096-token Qualcomm export also fits the free runner (5,987 s); a candidate default, not adopted. |
| [11](research/README.md) | Calibration and Qualcomm's compile are 87 to 93 percent of a Qualcomm export; one scale search is about 99 percent of the calibration (an estimate). |
| [12](research/README.md) | The same calibration took 1,761 s on one hosted runner and 936 s on another. |
| [13](research/README.md) | Qualcomm calibration grew 3.92 times from 2k to 4k tokens where 2 times was predicted; unexplained. Open. |
| [17](research/README.md) | MediaTek's calibration read each stored row back as Python lists, 2,045 to 2,538 s per prompt. Fixed by 20, replaced by 33. |
| [20](research/README.md) | The fix for 17: all 9 prompts calibrated in 138 s. |
| [22](research/README.md) | The first MediaTek export (Qwen3 0.6B, MT6991, 512 tokens) built; its full-precision embedding table is 59.5 percent of the download. |
| [26](research/README.md) | The first published MediaTek export: Qwen3 0.6B for MT6989, 512 tokens, 1,343.2 s. |
| [28](research/README.md) | The export toolchain works on ARM Linux, and exports moved to an 8-CPU ARM runner chosen for its 24 GB of memory; how it scales with CPUs is not established. |
| [41](research/README.md) | Modal's cloud sandboxes fit every window ever built and gave a byte-identical file, but exported more slowly; a version difference confounds the comparison. |

### Vendor model code that runs but computes the wrong thing

Sample code that compiles and generates text, yet does not match the original model.

| Finding | What it found |
|---|---|
| [9](research/README.md) | Qualcomm's compiled files report the same start and stop tokens for every model; apps must not trust them. |
| [32](research/README.md) | LFM2.5 2.6B exports through the 1.2B model's code with its own parameters, even though ExecuTorch lists no 2.6B. |
| [36](research/README.md) | MediaTek's LFM2 export let padding leak into the convolution state (KL 0.165 against 0.0024 once fixed) and used so much memory that 16k builds crashed an 11.5 GB phone. |
| [37](research/README.md) | MediaTek's Qwen and Llama code ignores each model's RoPE base (Qwen3 0.6B at KL 0.082 before any quantization) and its mask leaks on Qwen2.5; patched to exact, Qwen2.5 refused. |
| [42](research/README.md) | LFM3 1.7B cannot be described by ExecuTorch 1.5.1's model code; a hand-built version matches the original (top-1 agreement 100 percent) and runs on a Snapdragon 8 Elite Gen 5 phone. |
| [46](research/README.md) | MediaTek's Gemma 3 code uses one RoPE table where the model needs two (KL 0.177); one table per layer type is exact. |

### Quantization quality: rounding, GPTQ, calibration and precision

How much each way of shrinking the weights costs, measured against the full-precision model.

| Finding | What it found |
|---|---|
| [29](research/README.md) | Plain 4-bit rounding cost LFM2.5 1.2B its tool use (10 percent of needed searches against 49 percent with GPTQ), so every 4-bit file is now GPTQ-solved. |
| [34](research/README.md) | MediaTek's short calibration prompts under-calibrate windows past 512 tokens; a long-context corpus replaced them. |
| [35](research/README.md) | The better corpus barely helps MediaTek's 4-bit weights; 8-bit weights are 110 times closer to the original, so precision is the lever. |
| [43](research/README.md) | Qwen3 0.6B loses its tool-call decision at 4-bit while 1.7B and 4B keep it; 0.6B ships with full-precision layers. |
| [45](research/README.md) | The 4-bit weights inside CPU and GPU files can be read back exactly, which tells a rounded file from a GPTQ-solved one. |
| [49](research/README.md) | MediaTek's 4-bit weights are 3 to 170 times further from the original than its 8-bit ones, even with GPTQ; 8-bit stays the published recipe. Not yet committed to ExecuPack. |

### Same file, different host, runtime or phone

Whether the machine that builds or runs a file changes what it does.

| Finding | What it found |
|---|---|
| [27](research/README.md) | One embedding quantizer's search depends on the build CPU, so the same model built on two runners wrote different text; it was switched off. |
| [30](research/README.md) | ExecuTorch 1.4.0's runner crashes on ARM though the export works there, so the smoke test runs on x86; the cause is not found. |
| [38](research/README.md) | GPU (Vulkan) files run correctly on Mali and Adreno, but on the POCO's Mali GPU the CPU file decodes 2.2 to 3.9 times faster. |
| [44](research/README.md) | A GPTQ solve on an Apple M5 Mac produces NaN and wrong reference answers, so solves run on Linux only. |
| [48](research/README.md) | ExecuTorch 1.6.0 reads every published file and runs every published CPU file; the 4-bit files' outputs shift slightly (KL 0.0025 to 0.064); CPU and GPU exports moved to it, held back from publishing until the apps ship it. |

## Studies

Experiments that are not, or not yet, a finding in the research log.

| Study | The question | What we found | Source |
|---|---|---|---|
| [MediaTek A16W4](studies/mtk-a16w4/) | Can MediaTek's NPU take 4-bit weights, with GPTQ? | 3 to 170 times further from the original than 8-bit; GPTQ restores the decision gate on two of three models. Not yet run on a phone. | The `mtk-a16w4` worktree (finding 49) |
| [AWQ](studies/awq/) | Does AWQ keep more of the model than GPTQ at 4 bits? | No method won everywhere: AWQ was 25% to 62% closer to the original on long text, but further on the app's short questions and failed the publish gate on 2 of 3 models. Published files stay GPTQ. | The local-only branch `awq-compare` and its uncommitted round-2 scripts |
| [KV cache](studies/kv-cache/) | Does an 8-bit cache on the CPU, or 16-bit mode on the GPU, keep speed and answers? | On the CPU, 1.7 to 3.8 times faster 7.9k tokens in and up to 5.3 GB less memory, answers barely moved. On the GPU, 16-bit mode is faster but broke Llama at 32.5k tokens; published GPU files above 8k do not load. | The `kv-study` worktree, uncommitted |
| [Calibration corpus](studies/calibration-corpus/) | Does GPTQ keep more of a model with a larger calibration set? | Not settled: size and content changed at once, and the run that would separate them left no result here. | The pushed `experiment/extended-calibration*` and `publish/lfm12b-gptq257` branches |
| [Published audit](studies/published-audit/) | Do files published before any gate existed still decide like the original model? | A workflow that checks them without rebuilding; the one audit log kept failed (Qwen3 0.6B, 0 of 4). | The pushed `experiment/audit-published` branch |
| [AWS runs, November 2026](studies/aws-runs-2026-11/) | What did the four EC2 export machines produce and check before they were shut down? | The rollout of GPTQ-solved files: all 120 byte checks matched; Qwen3 0.6B failed its gate at every window and Phi-4-mini shipped under a recorded override. | The archive of the four machines, text only |

## The byte-level code check

[`harness/codecheck/`](harness/codecheck/) reads the 4-bit weights back out of a published `.pte`
and says whether they were solved or plainly rounded, from the file's bytes alone (finding 45). Run
over every published file during the November rollout, it found every file's checksum matching its
report and the Hub, and every file's rounding matching its claim. It is not yet wired into the
pipeline or CI.

## Running the scripts

A script under `research/evidence/` or `harness/codecheck/` that imports `pipeline` needs an
ExecuPack checkout on `PYTHONPATH` or in `EXECUPACK`, as its docstring says:

```sh
export EXECUPACK=~/path/to/execupack
export PYTHONPATH=$EXECUPACK
```

The `.patch` files are experiment branches. Those made by `git format-patch` apply to an ExecuPack
checkout with `git am` at the base commit their study names; `mtk-a16w4/uncommitted.patch` is a
plain diff, for `git apply`. Model files, solves and export folders are not in this repository.

## Where these files came from

`tools/import_execupack.py` read the research log and the code checks from the pushed branch
`executorch-1.6` (`b5affc76`: `origin/main` plus the ExecuTorch 1.6.0 export upgrade and finding
48). Finding 49 and its write-up were never committed and come from the `mtk-a16w4` worktree; the
other studies came from the branches, worktrees and archive named in the table above. The pipeline
itself, its CI and its roadmap stayed in ExecuPack, and so does a copy of `docs/research`, because
the pipeline's code cites it by path and finding number. Every file has a row in
[`PROVENANCE.tsv`](PROVENANCE.tsv), including the 2,397 files left out on purpose and why; see
[how this repository was built](../docs/provenance.md).
