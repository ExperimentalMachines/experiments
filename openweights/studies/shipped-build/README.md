# The shipped build

**The app tested exactly as people get it: the signed release, its own prompt and tools, its real
screen, and the network deliberately broken.**

`written up in OpenWeights` `1 phone, an emulator, a Mac` `4 to 6 Nov 2026`

> **The question.** A test build is not what users run. Does OpenWeights behave well as it ships:
> the signed release, its own system prompt, its three default tools and its real screen? And what
> goes wrong when a thinking model runs out of [tool-call](../../../docs/glossary.md#tool-call)
> rounds, or when every web search engine fails?

> [!TIP]
> **What we found**
> - **With every search engine failing, the signed release made up answers.** Its recommended
>   compiled LFM2.5 1.2B said "approximately ¥250,000" for 500 USD and "approximately 22°C" for the
>   weather in Tokyo. A new rule replaces such replies with a fixed sentence; on signed release 731,
>   on a real phone with the app's network blocked, **0 of 12** search questions got an invented
>   answer (3 refusals and 9 of the sentence).
> - When LFM3 1.7B ran out of tool rounds, closing its reasoning first and telling it "No more tool
>   calls" raised clean finishes from **18 to 30 of 34** and cut the 90th-percentile time from
>   97.1 s to 65.2 s. Correct answers went 8, 10, 7, which the write-up calls "not a gain".
> - On the app's own configuration over 160 public questions the new build looked better (LFM3 75
>   to 85, p = 0.041; the LFM2.5 GGUF 55 to 64, p = 0.012), but a rerun at the final code brought the
>   GGUF to 59, so that gain "was smaller than it looked". These runs cannot show the builds are equal.

## Key numbers

| What | Result | Where |
|---|---|---|
| LFM3 turns out of rounds that finished with no reasoning in the reply: as it was, reasoning closed at 128 tokens, the final rule | 18, 20, **30** of 34 | Snapdragon 8 Gen 3 on QDC, LFM3 1.7B Q4_K_M, llama.cpp v0.8.0 |
| The same 34 turns still thinking at the 600-token cap | 11, 1, 1 | the same |
| The same 34 turns, 90th-percentile time | 97.1 s, 93.9 s, 65.2 s | the same |
| The app's own configuration, correct of 160, old build then new | compiled LFM2.5 53, 55 (p = 0.82); LFM2.5 GGUF 55, 64 (p = 0.012); LFM3 75, 85 (p = 0.041) | Snapdragon 8 Gen 3 on QDC, one unit |
| Median seconds per question on the new build | compiled LFM2.5 8.3, GGUF 9.8, LFM3 28.3 | the same |
| The same configuration at the final code | compiled 55 then 53; GGUF 64 then 59; neither difference significant | one Snapdragon 8 Gen 3, not necessarily the same unit |
| Every search engine failing, release 731 | **0 of 12** invented: 3 refusals, 9 "No search engine could be reached" | Snapdragon 8 Gen 3 on QDC, three models, the release's own screen |

## Why it matters

The findings changed the turn loop: a turn out of rounds now closes its reasoning first and
answers. They added the failed-search rule and a live-data rule (a reply like "I don't have
real-time weather data" is treated as a request to search), and they shipped as signed release 731.
For anyone building on-device agents: a small model handed a failed tool result will often write a
plausible number instead of saying the tool failed, and only a test of the release build, through
its real screen, with the network deliberately broken, found it.

## What was tested

- **Phones.** One Qualcomm handset with a Snapdragon 8 Gen 3 (SM8650, Android 14) on
  [Qualcomm Device Cloud](../../../docs/glossary.md#qualcomm-device-cloud-qdc); the final run may
  have been on a different SM8650 unit. See [all devices](../../../docs/devices.md).
- **Machines.** A 16 KB-page Android 16 emulator on the Mac, standing in for a phone on behaviour,
  not on speed, after a phone reservation never connected; and the Mac itself, running the app's
  own engine code to choose the wording of the last pass before spending phone time.
- **Runtimes.** [llama.cpp](../../../docs/glossary.md#llamacpp) v0.8.0 for the GGUF files;
  [ExecuTorch](../../../docs/glossary.md#executorch) for the compiled model (its version is not
  recorded in these runs).
- **Models.** LFM2.5 1.2B compiled ([8da4w](../../../docs/glossary.md#8da4w), a 4,096-token window),
  LFM2.5 1.2B Q4_K_M and LFM3 1.7B Q4_K_M as GGUF files (a 32,768-token window); Qwen3 1.7B and
  LFM2.5 2.6B on the 34 hardest questions as a crash check.
- **Method.** The 160-question decision suite (80 RetrievalQA, 40 PopQA, 40 FreshQA; 75 need a
  search, 60 do not, 25 are borderline) run through the app's own prompt, default tools, search
  rules and default sampler; everyday questions (weather in Tokyo and Manila, 500 USD in yen, the
  price of Bitcoin, the capital of France) typed into the release's real screen with search working
  and with every engine failing: the emulator's DNS pointed nowhere, the phone's app network blocked.
- **When.** 4 November 2026 by the phone's clock, and 6 November 2026.

## What's in this folder

| Run | What it holds |
|---|---|
| [`runs/2026-11-05-qdc-last-round/`](runs/2026-11-05-qdc-last-round/) | LFM3 out of tool rounds on the Snapdragon 8 Gen 3: the 160-question runs as it was and closed at 128, the 34-question final-rule run, the reasoning-budget test, a screenshot of reasoning on screen, and `mac-wordings/` with the seven last-pass wordings tried on the Mac. 22 files. |
| [`runs/2026-11-05-qdc-shipped/`](runs/2026-11-05-qdc-shipped/) | The app's own configuration, old and new build, three models; Qwen3 and LFM2.5 2.6B on the 34 hardest; engine and app device tests; logcats and search screenshots; `mac/` with the language-line and long-answer runs. 38 files. |
| [`runs/2026-11-06-emulator-release/`](runs/2026-11-06-emulator-release/) | The signed release on the 16 KB emulator through its screen: search working, search dead before and after the fix, the accidental DNS loss that first showed the invented weather, and the scripts. 78 files. |
| [`runs/2026-11-06-qdc-final/`](runs/2026-11-06-qdc-final/) | Signed release 731 on the Snapdragon 8 Gen 3 through its screen, with search working and with the app's network blocked, the final engine and app tests, and the comparison with the earlier build. 48 files. |

The two 2026-11-05 folders have no README of their own; the write-up is their record. The
machine-readable decision rows these runs produced are in
[`retrieve-or-answer/runs/decisions/`](../retrieve-or-answer/runs/decisions/) (`qdc-sm8650-shipped-*`,
and for the last round `qdc-sm8650-base-decisions-*`, `qdc-sm8650-budget-decisions-*` and
`qdc-sm8650-answer-decisions-*`), one JSON object per question. The scripts kept here read them from there and run from this repository's root. The
screen runs are plain text, one file per question and model.

## Run it again

Regrading needs only this repository. New rows need an OpenWeights checkout with the debug and
test APKs built and the model files already on the phone under `/data/local/tmp/openweights/eval`.
The screen runs need the signed release APK, which only the upload key can produce, and the phone
run a QDC reservation with a root shell for the network block.

```sh
# From the repository root: regrade the stored rows
python3 -B openweights/harness/app-evals/bench/compare_shipped.py \
  LFM2.5-1.2B-Instruct-8da4w-gptq-4k LFM2.5-1.2B-Instruct-Q4_K_M LFM3-1.7B-Q4_K_M
python3 -B openweights/studies/shipped-build/runs/2026-11-06-qdc-final/scripts/compare-final.py
python3 -B openweights/harness/app-evals/bench/compare_limit_turns.py <run.jsonl> [<other.jsonl> ...]

# New decision rows on a phone over adb, in the app's own configuration
export OPENWEIGHTS=~/path/to/openweights
MODELS=<a.pte,b.gguf> ARMS=shipped PREFIX=<prefix> INSTALL=1 \
  openweights/harness/app-evals/bench/run_decisions.sh [adb serial]

# The release on the emulator with every search engine failing (needs QDC_SCRATCH, an AVD named es-16k)
openweights/studies/shipped-build/runs/2026-11-06-emulator-release/scripts/dead-emu.sh <apk or -> <out dir>
```

`-B` keeps Python from writing a cache folder into the repository. The QDC chain as it ran cannot be
rerun from here: it calls two phone-side scripts (`run-shipped.sh`, `dl-reg.sh`) that were not kept,
and the Mac wording scorer imports a `lastround` module that is not here either.

## Caveats

> [!NOTE]
> - LFM3 was not rerun through all 160 questions at the final code: it got 13 rows before the
>   reservation ended, so its 85 stands from the earlier build.
> - LFM2.5 can still invent a figure when the search snippet holds none: the GGUF on 5 of 5 rows and
>   the compiled model on 1 of 6. Few rows, and open.
> - LFM3's private reasoning is read by the loop and shows on screen for about six seconds before it
>   collapses. This is what stands between LFM3 and being the default, and it is not fixed.
> - How often the live-data rule fired is unknown: the rows do not keep the text of a pass the loop
>   dropped.
> - Correct counts depend on the definition. The write-up drops answers cut off mid-thought, so the
>   regrading script can report a higher count for LFM3 on questions that need a search (22 against
>   the write-up's 19). The write-up's p = 0.82 at the final code is 0.815 in the stored comparison.
> - The folders are dated 5 November, but the phone's clock put those runs on 4 November.

## Read more

- Write-up in OpenWeights: [llama.cpp b10549 to v0.8.0](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/llama-cpp-v080.md),
  its sections "The whole decision suite, and the last round", "What the app really does", "When
  search fails, and the last check of the release" and "On a real phone".
- Related: [retrieve or answer](../retrieve-or-answer/) (the decision suite and its rows) and
  [runtime upgrades](../runtime-upgrades/) (the same write-up's upgrade half).
- Harness: [`harness/app-evals/`](../../harness/app-evals/).
- Words used here: [glossary](../../../docs/glossary.md).
