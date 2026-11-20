# Retrieve or answer

**When a small model on a phone should search the web instead of answering, and what the app
around it can do to help it decide.**

`written up in OpenWeights` `4 phones` `9 to 18 Sep 2026`

> **The question.** OpenWeights gives its model a web-search
> [tool call](../../../docs/glossary.md#tool-call). A model of about a billion parameters does
> not know what it does not know: does it search when its own memory cannot answer, and only
> then? This study asked 160 seeded questions from three public retrieve-or-answer sets through
> the app's own loop, per model, runtime, rule and phone. Two follow-ups asked whether the
> model's own token probabilities predict a wrong answer, and whether a typed judgement model
> can show where the loop's text heuristics fail.

> [!TIP]
> **What we found**
> - The compiled LFM2.5 1.2B almost never searched on its own: **1 to 4** of 75 must-search
>   questions on four phones, and about one reply in five claimed a search it never made. The
>   GGUF of the same weights searched on 33 to 34 and claimed none. A later correction stands:
>   those compiled rows carried a state leak, so only their direction holds; the clean held-out
>   comparison is **10% against 51%** searched when needed.
> - Two rules shipped: run the search the model says it will make (the intent rule), and search
>   when the least likely of the reply's first 20 tokens had a probability under 0.2 (the doubt
>   gate). The gate took the GGUF from **12% to 24%** correct on must-search questions (paired
>   15/6, p = 0.08) and kept its direction on a held-out seed (5/1); the authors call it
>   provisional.
> - The model's own probabilities flag a wrong answer at an
>   [AUROC](../../../docs/glossary.md#auroc) of **0.78** (LFM2.5) and 0.70 (Qwen3), but only
>   llama.cpp exposes them: the ExecuTorch runner gives no logits, so the gate never fires there.
> - Search is capped by findability: the right answer was in the snippets for only 26 to 35% of
>   searches. Asking the model "are you sure you know?" ranked wrong answers worse than its token
>   probabilities did, so those judgements were removed.

## Key numbers

| What | Result | Where |
|---|---|---|
| Searched when needed, the model decides | **1 to 4 of 75** compiled against 33 to 34 of 75 GGUF | four phones, LFM2.5 1.2B, ExecuTorch 8da4w and llama.cpp Q4_K_M |
| Replies that claim a search never made | 32 to 35 of 160 compiled against 0 of 160 GGUF | same runs |
| Held out, state cleared, one tool | searched when needed 10% (round to nearest) against 51% (Q4_K_M); claimed a search 25% against 1% | Galaxy S25 Ultra, Test Lab, LFM2.5 1.2B |
| Least likely of the first 20 tokens as a wrong-answer detector | AUROC 0.78 (LFM2.5 Q4_K_M), 0.70 (Qwen3 1.7B Q8_0) | offline, against a local llama.cpp server |
| Doubt gate against the intent rule alone | correct on must-search rows **24% against 12%**; paired 15/6, McNemar p = 0.08; median turn 9.1 s against 5.3 s | POCO X8 Pro Max, LFM2.5 1.2B Q4_K_M |
| The same on a held-out seed | correct 48% against 45%, must-search 33% against 29%, paired 5/1 | POCO X8 Pro Max, LFM2.5 1.2B Q4_K_M |
| Sixteen tool definitions in the prompt instead of one | median turn 14.9 s against 3.7 s, first token at 9.9 s, correct 29% against 30% | POCO X8 Pro Max, compiled LFM2.5 |
| Right answer present in the search snippets | 26 to 35% of searches | phones, live DuckDuckGo |

## Why it matters

The study moved the app's recommended LFM2.5 back from the compiled export to the GGUF, and it
shipped the intent rule and the doubt gate, a [decision gate](../../../docs/glossary.md#decision-gate)
read from the model's own token probabilities; both are on. For anyone building an on-device
agent: a tool-use failure can belong to the exported model file rather than to the model; a
small model's token probabilities carry a usable "I might be wrong" signal, but only on a
runtime that exposes logits; and search itself is limited by whether the answer can be found at
all.

## What was tested

- **Phones.** The POCO X8 Pro Max (MediaTek MT6991, the maintainer's own phone, over adb); on
  [Firebase Test Lab](../../../docs/glossary.md#firebase-test-lab) a Pixel 10 Pro XL (Tensor G5),
  a Galaxy S24+ (Exynos 2400) and a Galaxy S25 Ultra (Snapdragon 8 Elite). The judgement write-up's
  offline sweeps ran on a MacBook with an Apple M5 and 24 GB. See [all devices](../../../docs/devices.md).
- **Runtimes.** [ExecuTorch](../../../docs/glossary.md#executorch) for the compiled export and
  [llama.cpp](../../../docs/glossary.md#llamacpp) for the GGUF files; these write-ups do not
  state versions.
- **Models.** LFM2.5 1.2B Instruct as a compiled [8da4w](../../../docs/glossary.md#8da4w) 32k
  export (a [round-to-nearest](../../../docs/glossary.md#round-to-nearest-rtn) file, not the later
  published GPTQ one) and as Q4_K_M and QAD Q4_0 [GGUF](../../../docs/glossary.md#gguf) files;
  Qwen3 1.7B Q8_0; offline also Qwen3.5 2B and LFM2.5 2.6B (dropped).
- **Method.** 160 questions drawn with one seed from RetrievalQA (80), PopQA (40) and FreshQA
  (40), each labelled by its set for whether the answer lives in the weights, and a held-out
  draw for confirmation. Every row ran through the app's real loop with real DuckDuckGo searches,
  [greedy](../../../docs/glossary.md#greedy-decoding), thinking off, a 4k window. Each "arm" is
  one rule the app can vary (answer bare, let the model decide, honour its stated intent, the
  doubt gate, all sixteen tools in the prompt). The offline gates ran the same rows against a
  local llama.cpp server.
- **When.** 9 to 18 September 2026.

## What's in this folder

| Run | What it holds |
|---|---|
| [`runs/decisions/`](runs/decisions/) | One JSON line per question per run (the first line a header: model, arm, phone, chip, thermal state, start time), the Test Lab chunk logs, and the grader's `summary.json`. 176 files. |
| [`runs/gate/`](runs/gate/) | The offline one-token gate: bare answers and gate readings per model, two seeds, and 48 questions that need no search. 23 files. |
| [`runs/confidence/`](runs/confidence/) | Offline token log-probabilities for LFM2.5 Q4_K_M and Qwen3 1.7B Q8_0, the 160 rows and the 48 that need no search. 4 files. |
| [`runs/typesafe/`](runs/typesafe/) | TypeSafe's readings of the loop's text matchers and of the results, with the two request caches, so the tables reproduce without a key. 6 files. |

In `decisions/`, a file name's prefix names the phone: none, `cut-`, `five-`, `fix-`, `held-`,
`min3-` and `qad-` are the POCO; `tensor-`, `exynos-` and `elite-` the Test Lab phones; `gate-`,
`pixel-gate-`, `pte-`, `pixel-pte-`, `read-` and `readv005-` the 17 September judgement runs.
Later studies added rows here too: `prod-`, `rerun2-` and `rerun3-` belong to
[compiled tool calling](../compiled-tool-calling/), `poco-s9-` to the ExecuTorch recipes work,
and `qdc-*` (Qualcomm Device Cloud, November) to the [shipped build](../shipped-build/) and
later runs. `VOID-window-too-small-*` files are kept as evidence and skipped by the grader. More
decision rows that never reached `decisions/` are in the
[Test Lab archive](../test-lab-archive/runs/ftl/README.md).

## Run it again

You need an OpenWeights checkout in `OPENWEIGHTS` with its debug app and test APKs, an Android
phone over adb (or a Google Cloud project with Test Lab and a `gs://` bucket holding the
models, since the original bucket was deleted on 30 September 2026), and the model files. All
are public on Hugging Face except the compiled 8da4w 32k export, which is only on the
maintainer's machine. The offline gates need a local `llama-server`; TypeSafe labelling needs
`TYPESAFE_API_KEY`, though its reports run from the committed caches.

```sh
export OPENWEIGHTS=~/path/to/openweights
cd ~/experiments
B=openweights/harness/app-evals/bench

# Draw the questions (already committed as decisions.json, decisions-seed8.json, decisions-seed9-heldout.json)
python3 $B/pull_decisions.py [--seed 7] [--cache <dir>]

# On a phone over adb; keep it on the charger or awake
MODELS=<a.pte,b.gguf> ARMS=driven-search,intent-search $B/run_decisions.sh [adb-serial]

# On every Test Lab phone, in chunks that fit Test Lab's 45 minutes
BUCKET=gs://<bucket> PHONES="mustang:36:tensor- e2s:36:exynos- pa3q:36:elite-" $B/run_decisions_cloud.sh [parallel-jobs]
# Or one Test Lab phone with every file sent from this machine (no bucket)
PROJECT=<gcp project> MODELS_DIR=<dir> MODELS=<a.gguf> ARMS=driven-search DECISIONS=<question file> \
  $B/run_decisions_ftl_local.sh <prefix> [device] [version]

# Grade and tabulate. The grader rewrites summary.json and runs/ is a record, so work on a copy
G=$(mktemp -d) && cp -R openweights/studies/retrieve-or-answer/runs/decisions/. $G/
python3 $B/grade_decisions.py $G [--against driven-search]
python3 $B/report_decisions.py $G [--min-rows 160]
python3 -m unittest openweights/harness/app-evals/bench/test_grade_decisions.py

# Offline gates against a local server
llama-server -m <model.gguf> --port 8092 -c 4096 -ngl 99            # add --jinja for gate.py
python3 $B/confidence.py collect 8092 <tag>
python3 $B/confidence.py report LFM2.5-1.2B-Instruct-Q4_K_M Qwen3-1.7B-Q8_0   # reads the stored rows only
python3 $B/gate.py bare 8092 <tag> [seed8]

# TypeSafe readings, from the cached answers
python3 $B/typesafe_results.py report <cache.json>
```

Checked: `confidence.py report` on the stored rows gives back the AUROCs 0.78 and 0.70. Live
DuckDuckGo results change between runs, so a rerun will not match row for row. The gate, lament
and page-read arms need an OpenWeights checkout from before their code was removed on 5 November
2026 (commit 3d150219).

## Caveats

> [!WARNING]
> - **The compiled LFM2.5 rows carry a state leak.** A later write-up found that every compiled
>   LFM2.5 row here was taken with a convolution state left over from the previous question, so
>   the direction of those results survives and their sizes do not. The clean held-out run is the
>   10% against 51% above. The GGUF rows and the doubt gate, which runs on llama.cpp only, are
>   outside the leak (an inference; the write-ups do not say so).
> - **The "correct" columns written before 17 September 2026 are 0 to 8 rows of 160 too high.**
>   The grader matched short answer aliases inside words ("CA" in "located"). Regraded, 71 of 76
>   paired comparisons keep their direction and the gate result holds (14/6, held out 6/1); the
>   published tables were left as written.
> - Three numbers in the write-ups disagree with their own data or with today's grader: the
>   prose says the intent arm searches "27 to 32 of 75" while its table rows show 22 to 28; the
>   intent-against-driven paired counts on the Tensor, Exynos and 8 Elite regrade to 13/8, 12/7
>   and 13/5 against the 10/3, 10/5 and 13/5 in the prose; and the doubt gate's cutoff row at 0.2
>   regrades to 58 sent, 56 wrong, 2 right against the 58, 54, 4 written before the grader fix.
> - `runs/decisions/summary.json` is stale: it holds 73 runs, and grading the folder now finds 94.
> - The doubt gate's size still needs a third held-out draw and a second phone before it is
>   quoted as a number.

## Read more

- Write-ups in OpenWeights:
  [Retrieve or answer: the decision measured on public rows, and the rule that came out of it](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/retrieve-or-answer.md),
  [Decisions read from the model's distribution instead of its prose (2026-09-17)](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/judged-decisions.md),
  [The loop's text heuristics, read by a typed judgement model (2026-09-17)](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/typesafe-experiments.md),
  [Which runtime the shortlist recommends, read from the whole table](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/recommended-runtime.md).
- Related: [compiled tool calling](../compiled-tool-calling/) (why the compiled model does not
  search), [on-device evals](../on-device-evals/) (the state-leak correction and the public
  benchmarks) and the [Test Lab archive](../test-lab-archive/).
- Harness: [`harness/app-evals/bench/`](../../harness/app-evals/bench/).
- Words used here: [glossary](../../../docs/glossary.md).
