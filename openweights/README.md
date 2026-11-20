# OpenWeights experiments

**The measurements behind [OpenWeights](https://github.com/ExperimentalMachines/openweights), an
Android app that runs open-weight chat models entirely on the phone.**

`10 studies` `16 phones and tablets, 1 emulator` `Aug to Nov 2026`

OpenWeights runs a model two ways: as a [GGUF](../docs/glossary.md#gguf) file through [llama.cpp](../docs/glossary.md#llamacpp),
and as a compiled [`.pte`](../docs/glossary.md#pte) file through [ExecuTorch](../docs/glossary.md#executorch). Nearly every
default in the app (thread counts, cache handling, which file to recommend, how the prompt is laid
out) was set by a measurement on a real phone. These are those measurements: the harness that
drove the phones and every run it recorded.

## What we found

> [!TIP]
> - **Keep the cache, and know how each reply ended.** Reuse made second turns 30 to 41 times
>   faster on every phone, but a reply cut short, cancelled or ended on an unreported stop token
>   silently turned reuse into a full re-read. → [prefix reuse](studies/prefix-reuse/)
> - **Threads are a trap on a throttled phone.** With six of eight cores paused, as the Snapdragon
>   8 Elite Gen 5's hard throttle does, llama.cpp at 4 threads decoded at 0.35 tokens/s and at 2
>   threads at 10.7. The engine now drops
>   to 2 threads after two slow steps (not yet in a Play release). → [prefill and decode](studies/prefill-decode/)
> - **A compiled 4-bit file can lose tool calling while its benchmark scores barely move.** In a
>   one-tool test on a Galaxy S25 Ultra, the plainly rounded compiled LFM2.5 1.2B made 10% of the
>   searches it needed, the GGUF of the same weights 51%.
>   → [compiled tool calling](studies/compiled-tool-calling/), [retrieve or answer](studies/retrieve-or-answer/)
> - **Measure upgrades where chats live.** llama.cpp v1.0.0 was level on an empty context and 20 to
>   23% slower at the 1,500 to 6,000 tokens of a real chat on a Galaxy S24, so it was reverted the
>   morning it landed. → [runtime upgrades](studies/runtime-upgrades/)
> - **Test the build users get.** With every search engine failing, the signed release let the
>   model invent a currency conversion; only a test of the release through its real UI found it.
>   → [shipped build](studies/shipped-build/)

## Studies

| Study | The question | When |
|---|---|---|
| [Prefix reuse](studies/prefix-reuse/) | What does keeping the cache save, and when does reuse silently fail? | Nov 2026 |
| [Prefill and decode](studies/prefill-decode/) | Which processor should read the prompt and which should write the reply; what do heat and threads do? | Oct to Nov 2026 |
| [On-device evals](studies/on-device-evals/) | Do both runtimes answer alike on every chip, how do they score on public benchmarks, and does the compiled window matter? | Aug to Sep 2026 |
| [Retrieve or answer](studies/retrieve-or-answer/) | When should a small on-phone model search the web, and what can the app do about it? | Sep 2026 |
| [Compiled tool calling](studies/compiled-tool-calling/) | Why does the compiled LFM2.5 narrate a search it never makes? | Sep 2026 |
| [Memory and paging](studies/memory-paging/) | When does swap slow a reply, how good is the "will it fit" estimate, and does prefetching help? | Oct 2026 |
| [Runtime upgrades](studies/runtime-upgrades/) | Did each llama.cpp and ExecuTorch upgrade keep speed, memory and answers? | Oct to Nov 2026 |
| [Shipped build](studies/shipped-build/) | Does the signed release, with its real prompt, tools and UI, behave the way the tests did? | Nov 2026 |
| [Vulkan](studies/vulkan/) | Do GPU exports run, and are they faster than the CPU? | Oct 2026 |
| [Test Lab archive](studies/test-lab-archive/) | What did the deleted Test Lab bucket hold that existed nowhere else? | Sep 2026 |

Three more questions have a harness here but no run folder, because their outputs were working
files and the write-up holds the numbers: whether NPU prefill pays on real conversations, where
the date and tools go in the prompt, and what a picture costs. The
[harness README](harness/README.md) covers all three.

## The write-ups

The full write-ups stay in OpenWeights' `docs/research/`, because the app's code comments cite
them by path. Each study page links to its write-ups at the commit these runs were imported from,
[`81734dee`](https://github.com/ExperimentalMachines/openweights/tree/81734deed5048f5691518d0e739662c50e51e1a3/docs/research).
Where a run was never written up, the study page says so and the run folder's README is the
record.

## Running the harness

```sh
export OPENWEIGHTS=~/path/to/openweights   # the app checkout the scripts build and drive
export SER=<adb serial>                    # where a script needs one phone
```

Scripts run from this repository's root or by their full path, and write their results into the
study they belong to. The tool-call probe is the exception: it is kept exactly as it ran (see
[compiled tool calling](studies/compiled-tool-calling/)). The phone-side procedure, from finding
a phone on wireless debugging to installing around a ROM that blocks `adb install`, is
OpenWeights' `phone-deploy` skill. [The harness README](harness/README.md) says what each folder
does.

## Where these files came from

Imported from OpenWeights' `tools/` and `eval/` at commit `81734dee` (17 Nov 2026), plus eight
run files that were never committed, by `tools/import_openweights.py`. Every file has a row in
[`PROVENANCE.tsv`](PROVENANCE.tsv). Device logs were cut to the app's own lines and device serials
and cloud ids replaced by pseudonyms on the way; what stayed in OpenWeights and what was left out
is in [how this repository was built](../docs/provenance.md).
