No files were changed. I found no teacher-forced off-by-one: HF selects logits at `t−1`, and the phone feeds through `t−1` before scoring target `t`. The prefill boundary and continuation position agree. ([reference.py:123](/Users/alpha/execupack-kvstudy/scripts/kvstudy/reference.py:123), [StudyBench.java:160](/Users/alpha/execupack-kvstudy/scripts/kvstudy/phone/StudyBench.java:160))

1. **Critical — The headline claims a distribution-distance bound that was never measured.**

   The narrative says int8 changes the distribution by “at most 0.03 KL against the fp32-cache file.” The calculation is actually
   `KL(HF ‖ int8) − KL(HF ‖ fp32-cache)`. That signed difference does **not** measure or bound `KL(fp32-cache ‖ int8)`; substantial changes can cancel. Furthermore, these are means over positions and a 65-category, top-64-plus-tail distribution—not full-vocabulary distances. Even the reported Llama mean’s nominal upper confidence limit is **0.0390**, exceeding 0.03. ([narrative.html:3](/Users/alpha/execupack-kvstudy/work/kvstudy/report/narrative.html:3), [analyze.py:85](/Users/alpha/execupack-kvstudy/scripts/kvstudy/analyze.py:85), [StudyBench.java:91](/Users/alpha/execupack-kvstudy/scripts/kvstudy/phone/StudyBench.java:91), [report.md:14](/Users/alpha/execupack-kvstudy/work/kvstudy/report.md:14))

   **Fix:** Describe this as the *observed change in mean divergence from HF*. To claim distribution preservation, capture both files’ distributions and calculate their direct divergence, including tails and position-level extremes. The saved scalar metrics cannot reconstruct that comparison.

2. **Major — Replaying four tokens does not restore LFM’s state for needle questions.**

   The harness reuses one module and repeatedly rewinds to `length−4`. LFM’s convolution state is cleared only at position zero; any positive position retains the current state. Thus the first replay starts with state from later stream positions, and subsequent questions start with state from the previous answer. Replaying four raw tokens does not restore every layer’s historical inputs: contaminated representations also overwrite attention-cache entries. HF instead evaluates each question from its complete prefix. ([StudyBench.java:191](/Users/alpha/execupack-kvstudy/scripts/kvstudy/phone/StudyBench.java:191), [lfm2_state.py:40](/Users/alpha/execupack-kvstudy/pipeline/lfm2_state.py:40), [lfm2_state.py:50](/Users/alpha/execupack-kvstudy/pipeline/lfm2_state.py:50), [reference.py:142](/Users/alpha/execupack-kvstudy/scripts/kvstudy/reference.py:142))

   This makes the LFM needle comparison unreliable; it does not establish that the reported answers necessarily change.

   **Fix:** Use a fresh module and replay the complete prefix for each needle, or restore a validated snapshot of **all** model state. Compare against fresh-prefix execution and verify question-order invariance.

3. **Major — The confidence intervals do not support population-level equivalence or “no cache effect.”**

   `boot()` independently resamples tokens after concatenating books. Adjacent positions share context, and reply positions share a question and generated trajectory. The reply analysis flattens **47 replies**, not 141 independent replies. Llama and Qwen’s deepest results contain just **one book and one contiguous 128-token span**. Pairing removes between-condition context differences; it does not make these observations independent. ([analyze.py:39](/Users/alpha/execupack-kvstudy/scripts/kvstudy/analyze.py:39), [analyze.py:51](/Users/alpha/execupack-kvstudy/scripts/kvstudy/analyze.py:51), [analyze.py:138](/Users/alpha/execupack-kvstudy/scripts/kvstudy/analyze.py:138), [awq_compare.py:32](/Users/alpha/execupack-kvstudy/scripts/awq_compare.py:32), [report.md:14](/Users/alpha/execupack-kvstudy/work/kvstudy/report.md:14))

   There are also 24 PG-19 and eight app KL intervals without multiplicity adjustment; several windows repeat the same observations. Failure to exclude zero is not evidence of equivalence. ([narrative.html:23](/Users/alpha/execupack-kvstudy/work/kvstudy/report/narrative.html:23))

   **Fix:** Collect more independent texts; resample paired books and appropriate within-book blocks, and cluster replies by question. Treat one-book results as descriptive. Prespecify acceptable degradation margins and primary comparisons, then perform noninferiority/equivalence analysis with appropriate multiplicity handling.

4. **Major — “On the app questions int8 is never worse” is directly contradicted by the results.**

   Qwen’s first-token top-1 agreement drops **87.9% → 85.1%**, or **124/141 → 120/141**. Its zero-based held-out row 117 worsens from **4.53012 to 13.416 KL**. LFM’s mean reply KL also increases, albeit slightly. Separately, Llama’s deepest sampled-span PPL rises **45.70 → 48.46**, approximately **6%**, while top-1 agreement falls **63.3% → 60.2%**. ([narrative.html:23](/Users/alpha/execupack-kvstudy/work/kvstudy/report/narrative.html:23), [report.md:56](/Users/alpha/execupack-kvstudy/work/kvstudy/report.md:56), [report.md:57](/Users/alpha/execupack-kvstudy/work/kvstudy/report.md:57), [Qwen fp32 results:1](/Users/alpha/execupack-kvstudy/work/kvstudy/phone-out/app-2k-qwen-f32kv.json:1), [Qwen int8 results:1](/Users/alpha/execupack-kvstudy/work/kvstudy/phone-out/app-2k-qwen-q8kv.json:1), [report.md:14](/Users/alpha/execupack-kvstudy/work/kvstudy/report.md:14))

   **Fix:** Report improvements in aggregate KL alongside regressions in other metrics and individual prompts. Define acceptable task-level losses before recommending shipment.

5. **Major — The weight-only explanation for errors is unsupported, and the needle wording understates Gemma’s failures.**

   Both exports include quantized activations and embeddings, and differ from HF in runtime and implementation. Similar failures under both caches cannot isolate 4-bit weights as their cause. HF itself retrieves only **177/204**, so some failures predate export quantization. The earlier x86 comparison also changes the solve and execution platform; its sign reversal cannot establish that the metric “follows the weights, not the cache.” ([narrative.html:12](/Users/alpha/execupack-kvstudy/work/kvstudy/report/narrative.html:12), [narrative.html:22](/Users/alpha/execupack-kvstudy/work/kvstudy/report/narrative.html:22), [narrative.html:23](/Users/alpha/execupack-kvstudy/work/kvstudy/report/narrative.html:23), [report.md:36](/Users/alpha/execupack-kvstudy/work/kvstudy/report.md:36))

   The **166 versus 163** total hides seven gains and four losses. Gemma’s 32k **4–5/27 are successes**, not misses: it misses **22–23/27**. Its 8k totals are equal, but the raw paired results contain two gains and two losses. ([report.md:49](/Users/alpha/execupack-kvstudy/work/kvstudy/report.md:49), [Gemma fp32 results:1](/Users/alpha/execupack-kvstudy/work/kvstudy/phone-out/stream-8k-gemma-f32kv.json:1), [Gemma int8 results:1](/Users/alpha/execupack-kvstudy/work/kvstudy/phone-out/stream-8k-gemma-q8kv.json:1))

   **Fix:** Correct successes versus misses, publish paired outcome transitions, and remove causal attribution to weights alone. Isolate causes with equivalent full-precision ExecuTorch controls and quantization ablations, after fixing needle-state handling.

6. **Major — Timing comparisons lack controlled repetitions, and the run record is incomplete.**

   Alternating intended order did not produce tightly paired runs: LFM 8k int8 starts at **04:07**, while fp32 starts at **06:54**. LFM 2k starts at skin temperatures **44.4°C versus 36.7°C**. Download completions occur during benchmark runs, and the narrative acknowledges contamination of LFM 32k. There are no repeated, randomized measurements of each identical workload with controlled starting conditions. ([progress.log:19](/Users/alpha/execupack-kvstudy/work/kvstudy/phone-out/progress.log:19), [progress.log:60](/Users/alpha/execupack-kvstudy/work/kvstudy/phone-out/progress.log:60), [progress.log:1](/Users/alpha/execupack-kvstudy/work/kvstudy/phone-out/progress.log:1), [progress.log:32](/Users/alpha/execupack-kvstudy/work/kvstudy/phone-out/progress.log:32), [narrative.html:34](/Users/alpha/execupack-kvstudy/work/kvstudy/report/narrative.html:34))

   The supplied log ends at Qwen 32k fp32’s **start**, leaving later 32k results without matching completion/order/temperature records. ([progress.log:76](/Users/alpha/execupack-kvstudy/work/kvstudy/phone-out/progress.log:76))

   **Fix:** Prefetch everything, stop background transfers, run randomized or ABBA repetitions under comparable thermal conditions, and retain complete timestamps, frequency/throttling telemetry, and run identifiers. Report run-level variability.

7. **Major — Qwen’s 17.51× result cannot establish its claimed mechanism or “usability.”**

   The fp32 run reaches the reported 8.4-scale peak while its resident memory subsequently falls. The harness records only RSS and high-water RSS—not swap, faults, reclaim, pressure stalls, or competing memory use. Memory pressure is plausible, but its contribution cannot be measured from these records. ([report.md:74](/Users/alpha/execupack-kvstudy/work/kvstudy/report.md:74), [StudyBench.java:37](/Users/alpha/execupack-kvstudy/scripts/kvstudy/phone/StudyBench.java:37), [Qwen fp32 results:1](/Users/alpha/execupack-kvstudy/work/kvstudy/phone-out/stream-32k-qwen-f32kv.json:1))

   Moreover, the int8 run records **881 seconds—14.7 minutes—of prefill forward time for 32,118 tokens**, and retrieves only **5/9** needles. Calling it “usable” solely from 3.5 decode tok/s has no stated acceptance criterion. ([Qwen int8 results:1](/Users/alpha/execupack-kvstudy/work/kvstudy/phone-out/stream-32k-qwen-q8kv.json:1), [report.md:46](/Users/alpha/execupack-kvstudy/work/kvstudy/report.md:46), [narrative.html:3](/Users/alpha/execupack-kvstudy/work/kvstudy/report/narrative.html:3))

   **Fix:** Present this as a device-specific observation. Measure memory-pressure telemetry, time to first answer, sustained decode, and task success under declared usability requirements.

8. **Major — The harness-to-app performance extrapolation is unvalidated.**

   The timer includes tensor creation/JNI output extraction but excludes the subsequent Java log-softmax and scoring. Those untimed operations still affect allocations, scheduling, and thermal load between steps. The runner cross-check uses explicit warm-up, a different prompt, and 63 generated tokens; it is not the same workload or timing definition. Its decode rates differ from the harness by about **13% for fp32 and 36% for int8**. ([StudyBench.java:63](/Users/alpha/execupack-kvstudy/scripts/kvstudy/phone/StudyBench.java:63), [StudyBench.java:175](/Users/alpha/execupack-kvstudy/scripts/kvstudy/phone/StudyBench.java:175), [KvBench.java:72](/Users/alpha/execupack-kvstudy/scripts/kvstudy/phone/KvBench.java:72), [runner cross-check:1](/Users/alpha/execupack-kvstudy/work/kvstudy/phone-out/xcheck-runner-qwen-2k.jsonl:1))

   The cross-check’s 1,800-token prompt also does not validate production chunk boundaries. The study deliberately uses `get_max_seq_len−1`; its supporting script documents a difference from what `TextLLMRunner` sends. ([StudyBench.java:56](/Users/alpha/execupack-kvstudy/scripts/kvstudy/phone/StudyBench.java:56), [gemma3_pte_check.py:51](/Users/alpha/execupack-kvstudy/scripts/gemma3_pte_check.py:51))

   **Fix:** Separate forward-only and end-to-end measurements. Compare identical prompts and depths with matched warm-up, and verify the actual app runner across chunk boundaries, long contexts, resets, and multiple turns. Measure cold and warm prefill separately rather than attributing their difference to warm-up retrospectively.

9. **Minor — Several performance summaries contradict the table.**

   The stated 1.9k speed-up range excludes LFM 32k’s **1.00×**. The 32.5k range **2.5–3.3×** silently excludes Qwen’s **17.51×**. “A larger window decodes slower, whatever its cache” fails for Qwen int8: **27.0 → 29.7 → 35.9 tok/s** across 2k, 8k, and 32k files. ([narrative.html:3](/Users/alpha/execupack-kvstudy/work/kvstudy/report/narrative.html:3), [narrative.html:20](/Users/alpha/execupack-kvstudy/work/kvstudy/report/narrative.html:20), [report.md:71](/Users/alpha/execupack-kvstudy/work/kvstudy/report.md:71), [report.md:72](/Users/alpha/execupack-kvstudy/work/kvstudy/report.md:72))

   **Fix:** State the actual ranges and explicit exclusions; qualify the window-size observation by model/cache. Generate headline numbers from the collected data instead of embedding them manually.

10. **Minor — Prefill bands contain substantial work outside their labeled depths.**

    Entire chunks are assigned by starting position. A chunk starting at **1,899** with **2,047 tokens** is counted in “0–2k” even though it ends at 3,946. Consequently that column covers different depth ranges in 2k and larger exports. The caption discloses the rule, but the narrative interprets the bands as depth-local measurements. ([analyze.py:164](/Users/alpha/execupack-kvstudy/scripts/kvstudy/analyze.py:164), [Llama 8k results:1](/Users/alpha/execupack-kvstudy/work/kvstudy/phone-out/stream-8k-llama-f32kv.json:1), [narrative.html:33](/Users/alpha/execupack-kvstudy/work/kvstudy/report/narrative.html:33))

    **Fix:** End timed chunks at band boundaries, or label and interpret these explicitly as *chunk-start bands*. Do not compare them as identical context intervals.

11. **Minor — KL arithmetic matches algebraically, but its precision/normalization contract is imperfect.**

    Phone log-softmax uses doubles; `awq_compare.evaluate()` uses float32 log-softmax. App reference serialization additionally rounds log-probabilities to seven decimal places. Some resulting top-64 masses exceed one by up to approximately **2.9×10⁻⁷**, after which the tail clamp produces a distribution that is not normalized. The raw Qwen results contain tiny negative KL values around **−5×10⁻¹⁷**. These are numerical artifacts, not evidence explaining the substantive differences. ([StudyBench.java:73](/Users/alpha/execupack-kvstudy/scripts/kvstudy/phone/StudyBench.java:73), [StudyBench.java:100](/Users/alpha/execupack-kvstudy/scripts/kvstudy/phone/StudyBench.java:100), [awq_compare.py:225](/Users/alpha/execupack-kvstudy/scripts/awq_compare.py:225), [appref.py:16](/Users/alpha/execupack-kvstudy/scripts/kvstudy/appref.py:16))

    **Fix:** Preserve adequate reference precision, store an accurately calculated tail mass, and validate normalization and cross-language metric agreement. Label top-1 explicitly as **agreement with HF**, rather than token correctness.

12. **Minor — The reference description incorrectly implies one GPU reference procedure for everything.**

    The long-stream reference explicitly uses CUDA with TF32 disabled. The app reference comes through `solve.teacher()`, which loads the model on CPU and receives CPU tensors. The blanket reference statement therefore conflates two procedures. The Gemma validation also exercises an eager, pre-lowering path, so it cannot independently exclude errors in the final Android implementation. ([narrative.html:28](/Users/alpha/execupack-kvstudy/work/kvstudy/report/narrative.html:28), [reference.py:73](/Users/alpha/execupack-kvstudy/scripts/kvstudy/reference.py:73), [solve.py:224](/Users/alpha/execupack-kvstudy/pipeline/solve.py:224), [awq_compare.py:113](/Users/alpha/execupack-kvstudy/scripts/awq_compare.py:113), [gemma3_check.py:87](/Users/alpha/execupack-kvstudy/scripts/gemma3_check.py:87))

    **Fix:** Document the stream and app references separately, including device and precision. State exactly which Gemma implementation and context lengths were validated.

13. **Minor — Memory units and the speed chart misrepresent the measurements.**

    Dividing `/proc` KiB values by 1,024 produces **MiB**, but the table labels them MB. Qwen’s peaks are approximately **8.24 → 3.33 GiB**, or **8.85 → 3.58 decimal GB**, rather than the stated 8.4 → 3.4 GB. Separately, the decode chart fixes its minimum at **1 tok/s**, placing Qwen’s **0.2 tok/s** endpoint outside the SVG’s visible range. ([analyze.py:185](/Users/alpha/execupack-kvstudy/scripts/kvstudy/analyze.py:185), [narrative.html:7](/Users/alpha/execupack-kvstudy/work/kvstudy/report/narrative.html:7), [report.py:122](/Users/alpha/execupack-kvstudy/scripts/kvstudy/report.py:122), [report.md:74](/Users/alpha/execupack-kvstudy/work/kvstudy/report.md:74))

    **Fix:** Use consistent binary or decimal units and derive chart limits from all plotted observations.

The verdict **“ship the int8 KV cache in every CPU build” is not supported**. The central distribution-preservation claim uses the wrong interpretation of KL, LFM needle execution does not restore state correctly, the statistical design cannot establish equivalence, and actual regressions are obscured by aggregate language. One phone, sparse teacher-forced spans, and short retrieval answers also cannot establish production behavior across CPU builds. A shipment recommendation needs corrected needle runs, predefined quality-loss limits, broader independent task coverage, controlled timing repetitions, and validation through the actual app runner.
