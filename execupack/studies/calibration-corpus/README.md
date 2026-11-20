# A larger GPTQ calibration corpus

**Whether calibrating GPTQ on more example prompts keeps more of a model, and whether any change
comes from the number of prompts or their kind.**

`not written up` `1 CI runner` `20 to 21 Sep 2026`

> **The question.** A [GPTQ](../../../docs/glossary.md#gptq) solve picks each 4-bit weight using
> calibration prompts, and ExecuPack ships 104 of them. Does a larger corpus of 377 rows keep
> more of a model, and is a difference due to the size or to what was added? A second arm of 257
> rows, with no arithmetic word problems, was set up to tell the two apart.

> [!IMPORTANT]
> **What we found (not settled, never written up)**
> - No finding cites this experiment and there is no run data: the commit messages in the two
>   patches are its only record.
> - The one recorded result: on Qwen3 1.7B the 377-row corpus made the model call tools far more
>   (**35% to 60%** on the sixteen-tool arm) while answering worse (**53% to 45%** correct). Size
>   and composition changed at once, so the run cannot say which caused it.
> - The 257-row arm that would separate them has no recorded result. A branch was prepared to
>   publish LFM2.5 1.2B builds from it under their own names, but nothing here shows that a
>   publish happened.
> - Adding step-by-step word problems grew the tokens far faster than the rows: 3.6 times the rows
>   became 6 times the tokens on Qwen3 1.7B.

## Key numbers

| What | Result | Where |
|---|---|---|
| Calibration rows: shipped, full extended corpus, arm B | 104, 377, 257 | GPTQ calibration for the CPU files |
| Rows added | 273: 120 GSM8K-style word problems and 153 app-shaped questions | the extended corpus |
| Advice the commit cites from vLLM's LLM Compressor | 128 to 512 calibration sequences, starting at 128 | |
| Hand-built exports on a 277-row lab corpus, recall on the sixteen-tool arm | 100% against 85% (heretic finetune), 75% against 70% (base); its author: "Three rows, so not a result" | LFM2.5 1.2B |
| Rows against tokens, 377-row arm | 3.6x the rows, 6x the tokens | Qwen3 1.7B |
| Tool calls and correct answers, 377-row arm | 35% to 60%, 53% to 45% | Qwen3 1.7B, sixteen-tool arm |

## Why it matters

The published recipe did not move: the extended corpus stays off unless an environment variable
asks for it, and a test pins that. For a developer: vary the size and the content of a
calibration set separately. Adding reasoning-heavy rows changes the token mix far more than the
row count suggests, and here it shifted behaviour (more tool calls, fewer right answers) rather
than simply improving it.

## What was tested

- **Machines.** ExecuPack's CPU export workflow on GitHub Actions; the runner type is not
  recorded. The scores come from an OpenWeights phone evaluation, the "sixteen-tool arm", and the
  patches do not name the phone. See [all devices](../../../docs/devices.md).
- **Runtimes.** ExecuPack's [XNNPACK](../../../docs/glossary.md#xnnpack) export with a GPTQ solve;
  the patches do not state the ExecuTorch version.
- **Models.** Qwen3 1.7B (the 377-row arm); LFM2.5 1.2B and its heretic finetune (the earlier
  hand-built exports).
- **Method.** Three corpus modes behind one switch: off (the shipped 104 rows), full (all 273
  extra rows) and app-only (the 153 app-shaped rows, 257 in all). Three lab-corpus questions that
  were also in the graded decision set, and one duplicate, were removed first.
- **When.** 20 and 21 September 2026.

## What's in this folder

| Run | What it holds |
|---|---|
| [`calibration-corpus.patch`](calibration-corpus.patch) | The pushed branches `experiment/extended-calibration` (two commits) and `experiment/extended-calibration-app` (one more), from 0660a3c9: the 273-row extended corpus, the corpus modes in the calibration code, tests and the workflow variable. 1 file, 556 lines. |
| [`publish-lfm12b-gptq257.patch`](publish-lfm12b-gptq257.patch) | The pushed branch `publish/lfm12b-gptq257` on top: LFM2.5 1.2B GPTQ-257 exports go to their own repositories (suffix `-GPTQ-257-ExecuTorch`) and never replace the 104-row exports; plus two link renames. 1 file, 90 lines. |

There are no run folders. Both files were made with `git format-patch`, so each commit message,
which is the experiment's record, is kept in full.

## Run it again

Apply both patches with `git am` onto an ExecuPack checkout at 0660a3c9, the publish patch second.
A run needs GitHub Actions on an ExecuPack fork, a Hugging Face token to publish, and the
OpenWeights phone evaluation (the sixteen-tool arm) to score the result; that evaluation is not in
this repository, and the patches give no dispatch command. The branch's `export-xnnpack.yml` sets
the switch for the job and takes `model_id` and `revision` inputs.

```sh
cd <execupack checkout>
git switch -c extended-calibration 0660a3c9
git am <experiments>/execupack/studies/calibration-corpus/calibration-corpus.patch
git am <experiments>/execupack/studies/calibration-corpus/publish-lfm12b-gptq257.patch   # optional

# The switch: 1 or full adds all 273 rows, app adds the 153 app-shaped rows, anything else is off
export OW_CALIB_EXTENDED=app

# The branch's tests (no torch needed); the graded-set test skips without the OpenWeights decision set
pytest tests/test_calibration_extended.py
```

## Caveats

> [!NOTE]
> - Arm B's outcome is not recorded anywhere here, and whether a GPTQ-257 repository exists on
>   Hugging Face was not checked.
> - The 377-row result is one model, with size and composition confounded.
> - The earlier 277-row signal rests on three rows; its author calls it "not a result".
> - The publish patch's two later commits only rename links in a research note.

## Read more

- No write-up: the commit messages inside [`calibration-corpus.patch`](calibration-corpus.patch)
  are the record.
- Related: [AWQ beside GPTQ](../awq/) (which also varied how Qwen3's calibration rows are
  rendered), [published-file audit](../published-audit/) (built on this branch), and finding 29 in
  the [research log](../../research/README.md) (why the files are GPTQ-solved at all).
- Words used here: [glossary](../../../docs/glossary.md).
