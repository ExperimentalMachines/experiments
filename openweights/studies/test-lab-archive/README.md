# Test Lab archive

**The Firebase Test Lab runs kept when their storage bucket was deleted, and the findings they
back.**

`archive` `4 devices` `10 to 17 Sep 2026`

> **The question.** OpenWeights' runs on [Firebase Test Lab](../../../docs/glossary.md#firebase-test-lab)
> wrote everything to a cloud storage bucket, which was deleted on 30 September 2026. Before it
> went, every file was compared with the repository and the local archives, and what existed
> nowhere else and was text came here: result rows, test outcomes and device logs. This is an
> archive, not an experiment: what do these files show, and which findings rest on them?

> [!TIP]
> **What we found**
> - On a [hybrid model](../../../docs/glossary.md#hybrid-model), a search the app made on the
>   model's behalf forced [llama.cpp](../../../docs/glossary.md#llamacpp) to re-read the whole
>   conversation. Keeping checkpoints of the model's recurrent state cut the tokens re-read from
>   **866 to 323** and the row from 11,364 to 8,152 ms on a Pixel 10 Pro XL, with no measurable
>   cost on rows answered in one pass.
> - Every release build from version code **604 to 612** aborted on the first token of any
>   [GGUF](../../../docs/glossary.md#gguf) reply, while debug builds were fine. A probe that
>   installed the app through the Play Store confirmed Play was serving the fixed 613, which
>   answered "hi" and stayed alive.
> - The app's judgement call (the engine asking the model a closed question and reading the
>   answer's probabilities) replayed exactly on a Pixel 10 Pro XL: each judged turn read exactly
>   5 tokens.
> - Three speed probes of the Play build kept here are in no write-up: one prompt each, read from
>   the app's on-screen stats.

## Key numbers

| What | Result | Where |
|---|---|---|
| Tokens read for the search pass, before and after the checkpoint fix | **866 against 323**; a second row 949 against 446 | Pixel 10 Pro XL (Tensor G5), LFM2.5 1.2B QAD Q4_0, llama.cpp |
| That pass's prefill, and the whole row | 5,489 against 2,544 ms; 11,364 against 8,152 ms | same |
| Rows answered in one pass on both builds, median | 3,564 against 3,422 ms (12 rows) | same |
| Release builds that aborted on every GGUF reply | version codes 604 to 612 | every phone, release builds only |
| The Play-installed build answering "hi" | version code 613, 172 tok/s prefill, 25 decode, process alive | Pixel 10 Pro XL, Test Lab |
| A judged turn on the phone | read exactly 5 tokens; a judgement took 336 to 726 ms over 6 rounds | Pixel 10 Pro XL, LFM2.5 1.2B QAD Q4_0 |
| Play build speed, one prompt each (not written up) | 180 and 27 tok/s (Q4_0), 137 and 24 (Q4_K_M), 181 and 28 (QAD Q4_0), prefill and decode | Pixel 10 Pro XL, LFM2.5 1.2B |
| Logs kept | When the bucket was archived, 368.8 MB of logcat became 18 MB compressed. The copies here are cut to the app's own lines: 33 files, 0.4 MB | the archive |

## Why it matters

Two lessons travel beyond this app. On hybrid models such as LFM2.5, any rewrite of prompt bytes
the cache has already read costs a full re-read unless the engine keeps checkpoints of the
recurrent state; OpenWeights now keeps them. And a release-only failure needs a release-build
test: debug builds cannot show a fault introduced by the release shrinker, and the build's guard
checked a method's name while the native code looks it up by its full signature; the fix checks
the signature. The archive also keeps four half-runs of the question suite that were never merged
into [retrieve or answer](../retrieve-or-answer/), so that every number already published from
there still regenerates unchanged.

## What was tested

- **Phones.** On Firebase Test Lab: a Pixel 10 Pro XL (Tensor G5) for the checkpoint runs, the
  judgement run, the Play probe and some question-suite runs; a Galaxy S24+ (Exynos 2400) and a
  Galaxy S25 Ultra (Snapdragon 8 Elite) for question-suite runs; a Galaxy Tab S10+ for the Play
  probe, where the harness could not reach the Install button. The release crash itself was seen
  on the POCO X8 Pro Max (MediaTek MT6991) and an emulator, whose records are not here. See
  [all devices](../../../docs/devices.md).
- **Runtimes.** llama.cpp and [ExecuTorch](../../../docs/glossary.md#executorch).
- **Models.** LFM2.5 1.2B as QAD Q4_0, Q4_0 and Q4_K_M GGUF files and as a compiled
  [8da4w](../../../docs/glossary.md#8da4w) 32k export; Qwen3 1.7B Q8_0.
- **Method.** Instrumented Android tests on Test Lab phones, 45 minutes a run. The checkpoint
  comparison ran 20 RetrievalQA questions through the app's loop with the doubt gate on, once on
  a build before the fix and once after. The Play probe installed the production app through the
  Play Store on a signed-in lab phone and sent it one message.
- **When.** Question-suite runs on 10 September 2026, the Play probe and the checkpoint runs on
  13 and 14 September, the judgement run on 17 September.

## What's in this folder

| Run | What it holds |
|---|---|
| [`runs/ftl/checkpoint-ab/`](runs/ftl/checkpoint-ab/) | Two before-and-after pairs of the checkpoint fix on a Pixel 10 Pro XL; the second pair is the one the write-up reports, and its logs hold the engine's "rollback refused" lines. 16 files. |
| [`runs/ftl/play-probe/`](runs/ftl/play-probe/) | Eight Play Store probes: three "hi" runs on a Pixel 10 Pro XL and a Galaxy Tab S10+ (only the Pixel in the third passed), three passing speed probes and two where Play did not install. 33 files. |
| [`runs/ftl/judge/`](runs/ftl/judge/) | The 17 September judgement run on a Pixel 10 Pro XL, both tests passing. 5 files. |
| [`runs/ftl/decisions/`](runs/ftl/decisions/) | Fifteen question-suite runs from 10 September on the three Test Lab phones whose rows or logs were not already in retrieve or answer. 54 files. |

Each run folder holds one `<device>-36-en-portrait/` folder with Test Lab's JUnit record
(`test_result_1.xml`), its raw record (`instrumentation.results`), the device log (`logcat.xz`)
and, where rows were pulled, a `decisions-*.jsonl`. [`runs/ftl/README.md`](runs/ftl/README.md)
says which folder backs which write-up and lists the four half-runs kept here. The logs were
cut to the app's own lines on import ([`tools/logcat_filter.py`](../../../tools/logcat_filter.py));
the full ones are in
[OpenWeights at 81734dee](https://github.com/ExperimentalMachines/openweights/tree/81734deed5048f5691518d0e739662c50e51e1a3/tools/eval/results/ftl).
Not here: screenshots, screen recordings and the APKs Play delivered (in the maintainer's local
archive), the debug APKs (rebuildable from their commits) and the models.

## Run it again

Only in part. You need an OpenWeights checkout in `OPENWEIGHTS`, a Google Cloud project with
Test Lab, and a `gs://` bucket holding the models in `BUCKET`, since the original was deleted.
The Play probe and the judgement run cannot be repeated as they were.

```sh
export OPENWEIGHTS=~/path/to/openweights
cd ~/experiments

# The checkpoint comparison: run once on a build with the fix and once on a build before it
# (the write-up does not name that build's commit)
BUCKET=gs://<bucket> MODELS=LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf ARMS=doubt-search SETS=retrievalqa FROM=0 ROWS=20 \
  openweights/harness/app-evals/bench/run_decisions_ftl.sh mustang 36 <prefix>

# The Play probe lives in OpenWeights, not here, and gets whatever Play serves today
BUCKET=gs://<bucket> $OPENWEIGHTS/tools/release/probe_play_ftl.sh LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf mustang:36 gts10pwifi:36
```

The Play probe script is
[`tools/release/probe_play_ftl.sh` in OpenWeights](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/tools/release/probe_play_ftl.sh);
`PROMPT=` sends another message instead of "hi". The judgement test was deleted on 5 November
2026, so it needs an OpenWeights checkout from before commit 3d150219, and no script here
launches it. The question-suite runs rerun as in
[retrieve or answer](../retrieve-or-answer/).

## Caveats

> [!NOTE]
> - The first checkpoint pair was discarded: the phone given the fixed build reported thermal
>   throttling from its first row and its prefill decayed. Only the second pair is reported.
> - The Play probe's result is what Play served on 13 September 2026; a rerun cannot reproduce
>   it. The Galaxy Tab S10+ never reached Install, a harness limit and not a result.
> - Four question-suite runs here are halves of a bare-answer arm whose rows never reached
>   retrieve or answer; two more repeat rows that are there, and one Galaxy S24+ run of the
>   intent rule answers 50 of its 80 questions differently from the run that was kept.
> - The three speed probes are one prompt each, read from the on-screen stats in the device log,
>   and are not in any write-up.

## Read more

- Write-ups in OpenWeights:
  [A search the loop made for the model re-read the conversation (2026-09-14)](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/dropped-pass-reread.md),
  [The release build aborted on every GGUF reply (2026-09-14)](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/release-token-callback-abort.md),
  [The loop's text heuristics, read by a typed judgement model (2026-09-17)](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/typesafe-experiments.md),
  [Decisions read from the model's distribution instead of its prose (2026-09-17)](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/judged-decisions.md),
  [Retrieve or answer: the decision measured on public rows, and the rule that came out of it](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/retrieve-or-answer.md).
- Related: [retrieve or answer](../retrieve-or-answer/) and [prefix reuse](../prefix-reuse/).
- Harness: [`harness/app-evals/bench/`](../../harness/app-evals/bench/).
- Words used here: [glossary](../../../docs/glossary.md).
