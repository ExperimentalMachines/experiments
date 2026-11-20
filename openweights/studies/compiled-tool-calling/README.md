# Compiled tool calling

**Why the compiled LFM2.5 writes "based on my search" without ever calling the search tool,
while the GGUF of the same weights calls it.**

`written up in OpenWeights` `1 phone and a Mac` `13 Sep 2026`

> **The question.** OpenWeights shipped LFM2.5 1.2B, a small model from Liquid AI, in two forms:
> a [GGUF](../../../docs/glossary.md#gguf) file for [llama.cpp](../../../docs/glossary.md#llamacpp)
> and a compiled [`.pte`](../../../docs/glossary.md#pte) file for
> [ExecuTorch](../../../docs/glossary.md#executorch). Asked something it should look up, the
> compiled one wrote "Based on my search, ..." and called nothing, while the GGUF made the
> [tool call](../../../docs/glossary.md#tool-call). This study fed the same prompt bytes to each
> file and read the model's probability of opening a tool call, to find what in the export
> carries the loss.

> [!TIP]
> **What we found**
> - Two things had been measured as one. Most of the GGUF's calls were the model obeying a note
>   the app attaches to "who is X" questions, telling it to search: the GGUF called on **42 of
>   42** such rows of the question suite, the compiled export on **0 of 42**.
> - Inside the export, the loss is the 4-bit weight
>   [quantization](../../../docs/glossary.md#quantization) of the shipped recipe: on the nine
>   noted probe rows the tool-call token falls to 0.00 to 0.09, where the GGUF gives 0.66 to
>   0.87. The same recipe kept Qwen3 1.7B at 0.96 to 0.99 but collapsed Qwen3.5 2B to 0.03 to
>   0.42: the two [hybrid](../../../docs/glossary.md#hybrid-model) families lost the call and the
>   pure-attention one did not, "a pattern, not a proof".
> - A mixed export (attention at 4 bits, the rest at 8) was the first compiled LFM2.5 to call
>   tools on the phone, **32%** of needed searches against 3%, but it was not shipped: it is 1.7
>   times the size and 2.3 times the memory of the QAD GGUF, which searched on 60%.
> - A correction four days later stands: the probe reused one runner, so a convolution state
>   leaked from one prompt into the next. That leak was the whole of the unquantised export's
>   own gap, but the 4-bit collapse survives a clean state (0.004 to 0.08).

## Key numbers

| What | Result | Where |
|---|---|---|
| Probability of the tool-call token, "What is Hanover the capital of?" | GGUF Q4_K_M 0.84, fp32 reference 0.85, shipped 8da4w export **0.004** | LFM2.5 1.2B on a Mac: llama.cpp, transformers, ExecuTorch 1.4.0 |
| Calls on the 42 suite rows carrying the app's search note | GGUF **42 of 42**, 8da4w export **0 of 42** | LFM2.5 1.2B, 160-question suite, POCO X8 Pro Max |
| The same 8da4w recipe on two other families | Qwen3 1.7B 0.96 to 0.99 on nine of ten noted rows (one at 0.52); Qwen3.5 2B 0.03 to 0.42, its reference 0.87 to 0.95 | Mac probe |
| Tool schema indented instead of one line, same GGUF | Hanover 0.84 becomes 0.48 | LFM2.5 1.2B Q4_K_M, Mac |
| Searched when needed, 75 rows | 8da4w 3% (2), attention-int4 export 32% (24), QAD Q4_0 GGUF 60% (45), Q4_K_M GGUF 44% (33) | POCO X8 Pro Max, 160 rows, greedy, one session, on battery |
| Replies that narrated a search with no call | 8da4w 22%, attention-int4 3%, QAD GGUF 1%, Q4_K_M 0% | same run |
| File size and resident memory | attention-int4 export 1253 MB and 2.33 GB; Q4_K_M GGUF 697 MB and 1.06 GB | same run |
| A later recipe in the same format | GPTQ-solved int4 codes recover most calls at 795 MB, not every call or metric | LFM2.5 1.2B, ExecuTorch |

## Why it matters

OpenWeights kept recommending the GGUF for LFM2.5, with the QAD Q4_0 file ahead of Q4_K_M on
this run, and did not ship any of these compiled exports. Two bugs in the app, found on the way,
were fixed regardless: the compiled path spelled each tool's JSON schema indented where the
model's own template writes it on one line, which alone cost 0.36 of probability on the deciding
token in one example, and long prompts were cut into pieces after a space, which changed the
tokens. For a developer: a plain [round-to-nearest](../../../docs/glossary.md#round-to-nearest-rtn)
int4 export can erase a small model's tool calling, and the loss can depend on the model family.
Before shipping an export, read the probability of the tool-call token on your own prompts and
compare it with the unquantised model.

## What was tested

- **Phones.** The POCO X8 Pro Max (MediaTek MT6991, the maintainer's own phone, over adb): two
  40-row reruns on the charger and a 160-row run of four model files on battery. See
  [all devices](../../../docs/devices.md).
- **Machines.** A Mac running the app's own llama.cpp build (b10549, Metal), the ExecuTorch 1.4.0
  Python runner, and transformers in fp32 as the reference.
- **Runtimes.** llama.cpp and ExecuTorch with [XNNPACK](../../../docs/glossary.md#xnnpack), whose
  fast int4 path requires 8-bit activations.
- **Models.** LFM2.5 1.2B Instruct as GGUF Q4_K_M, Liquid's QAD Q4_0 GGUF, fp32, the shipped
  [8da4w](../../../docs/glossary.md#8da4w) 32k export, and about two dozen 2k variants (head kept
  in fp32, 4-bit on one module family at a time, QAD-derived, [AWQ](../../../docs/glossary.md#awq)-folded);
  Qwen3 1.7B (GGUF Q8_0, 8da4w); Qwen3.5 2B (GGUF Q4_K_M, an fp32 export of 9.5 GB and an 8da4w
  export of 576 MB).
- **Method.** Sixteen rows of the question suite, chosen on purpose (ten where the GGUF called
  and the export did not, six where neither did), then 24 rows drawn by a seed that nobody chose.
  Identical prompt bytes into every file; [greedy](../../../docs/glossary.md#greedy-decoding);
  the measure is the probability that the reply's first token opens a tool call. The phone runs
  used the app's own loop on all 160 questions.
- **When.** 13 September 2026. Reviewed the same day by Codex, Gemini and Mistral Vibe.

## What's in this folder

| Run | What it holds |
|---|---|
| [`runs/toolcall-probe/`](runs/toolcall-probe/) | Every probe output from the Mac (tool-call probability, top tokens and 28 greedy tokens per row) for each file tried, the two prompt sets, four review briefs and six reviews. 56 files. |

Outputs are JSON, named by runner: `gguf_*` (llama.cpp server), `ref_*` (transformers fp32),
`pte_*` (one per export; `_trailer_hf` is the sixteen chosen rows, `_fresh` the 24 drawn rows),
`qwen_probe_out.json` and `qwen35_probe_out.json`. The prompts are in
`sample_prompts_trailer_hf.json` and `sample_prompts_fresh.json`; the reviews are the Markdown
files. The phone rows live in [retrieve or answer](../retrieve-or-answer/) (`prod-`, `rerun2-` and
`rerun3-` in its `runs/decisions/`), and the phone's public-benchmark reports in
[on-device evals](../on-device-evals/) (`prod-*`).

## Run it again

Only in part from this repository. The probe needs the export environment
(`~/ow-models/etexport/.venv` with ExecuTorch 1.4.0, transformers and tokenizers), the Hugging
Face checkpoints under `~/ow-models/etexport/hf/`, the ExecuTorch source tree the export scripts
run from, the GGUF files, and a llama.cpp server built from the app's llama.cpp. None of these is
here. The bisection exports also need a patched `quantize.py` in that environment
(`quantize_skip.patch` and `patch_quantize*.py`, switched by `OW_SKIP_OUTPUT`, `OW_SKIP_REGEX`,
`OW_INT8_REGEX` and `OW_Q40_GGUF`).

```sh
cd ~/experiments
P=openweights/harness/app-evals/bench/toolcall_probe
R=openweights/studies/compiled-tool-calling/runs/toolcall-probe
PY=~/ow-models/etexport/.venv/bin/python

# The GGUF side: start llama-server from the app's llama.cpp build on port 8098 with
# --jinja --temp 0, then render the sixteen rows with the app's note and probe them
$PY $P/trailer_run.py

# A .pte, and the fp32 reference, on the same prompt bytes
$PY $P/pte_run.py $R/sample_prompts_trailer_hf.json <out.json> <export.pte>
$PY $P/ref_run.py $R/sample_prompts_trailer_hf.json <out.json>

# The bisection exports (each cds into ~/ow-models/etexport)
$P/export_qad.sh

# The phone runs: the decision-suite drivers of retrieve-or-answer, with OPENWEIGHTS set
```

`trailer_run.py` writes its outputs beside itself. The scripts are kept as they ran on 13
September 2026: several (`fresh2.sh`, `awq.py`, `phone_stats.py`, `save_and_time.py`, `tab_*.py`,
`calib_check.py`, `export_q40.sh`) read or write a session scratch folder that no longer exists,
and `save_and_time.py` is what copied the outputs that mattered into this folder. `prompts.py`
was changed on import to read the question rows from
[retrieve or answer](../retrieve-or-answer/runs/decisions/).

## Caveats

> [!WARNING]
> - **Every `.pte` number after a run's first prompt carries the state leak**, in the probe and
>   in the phone's compiled rows (the phone harness also reused one engine). The int4 collapse
>   stands; the unquantised export's gap and the sizes of the other compiled numbers do not.
> - The write-up says the shipped export leaves "the ten named rows at 0.00 to 0.09, seven of
>   them under 0.01". Its own table has nine noted rows in that range, five of them under 0.01;
>   the tenth row the GGUF called is at 0.110.
> - The sixteen probe rows were chosen on purpose, which bounds what they can claim; the 24 rows
>   drawn by a seed nobody chose show the same ordering of the files.
> - The phone comparison is one phone, one session, on battery; a second phone and a held-out
>   draw were owed before any of it became a recommendation.
> - These results do not approve the later published GPTQ 32k export, whose review is on hold.
> - The Mac's per-row times are not reported: the runs overlapped.

## Read more

- Write-ups in OpenWeights:
  [Why the compiled LFM2.5 narrates a search it never makes](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/executorch-tool-calling.md)
  and
  [An ExecuTorch export of LFM2.5 from the quantisation-aware weights](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/executorch-tool-calling-qad.md).
- The correction:
  [Compiled LFM2.5 state and quantization experiments](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/executorch-state-and-recipes.md).
- Related: [retrieve or answer](../retrieve-or-answer/) (the question suite and its phone rows)
  and [on-device evals](../on-device-evals/).
- Harness: [`harness/app-evals/bench/toolcall_probe/`](../../harness/app-evals/bench/toolcall_probe/).
- Words used here: [glossary](../../../docs/glossary.md).
