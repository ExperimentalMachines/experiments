The measurements largely reproduce, but several headline claims remain too strong or numerically wrong. I recomputed the statistics from the JSON, checked the per-book merges, and independently scored the needle token IDs. No files were changed.

1. **Major — The verdict still overstates what the quality measurements establish.** “It showed no drift” does not follow from a range of signed mean changes, and “fp16 left the 2k files unchanged” mistakes failure to detect a difference for equivalence. Qwen 2k’s GPU change is **+0.003394**, with interval **[−0.000399, +0.007316]**. The later wording “did not detectably change” is appropriate.

   Likewise, “small quality cost” should identify the measured cost as increased divergence from HF. At Llama 8k/7.9k, fp16 increased that divergence while **perplexity improved from 40.738 to 40.512**; needle success fell by one. These are mixed measurements, not demonstrated degradation in chat quality. The provisional recommendation and equivalence caveat help, but do not resolve the categorical wording.  
   References: [narrative.html:3](/Users/alpha/execupack-kvstudy/work/kvstudy/report/narrative.html:3), [narrative.html:4](/Users/alpha/execupack-kvstudy/work/kvstudy/report/narrative.html:4), [narrative.html:46](/Users/alpha/execupack-kvstudy/work/kvstudy/report/narrative.html:46).

2. **Major — The launcher does not guarantee the stated thermal admission conditions.** `cool()` returns success after its timeout even when the phone remains hot. Missing temperature, thermal-status, or cap readings default to zero and can also pass the gate. Consequently, “started only once” the cap cleared and temperature reached ≤50 °C is stronger than the implementation guarantees.

   The retained GPU progress log ends during the earlier app run and contains no admission records for the reported per-book runs. Their `.therm` files support sampled cap states, but do not record temperature or device thermal status. This does **not** establish that those runs started hot; it means the claimed admission conditions cannot be verified from the retained evidence.  
   References: [gpu.sh:30](/Users/alpha/execupack-kvstudy/scripts/kvstudy/phone/gpu.sh:30), [gpu.sh:32](/Users/alpha/execupack-kvstudy/scripts/kvstudy/phone/gpu.sh:32), [progress.log:23](/Users/alpha/execupack-kvstudy/work/kvstudy/gpu-out/progress.log:23), [narrative.html:40](/Users/alpha/execupack-kvstudy/work/kvstudy/report/narrative.html:40).

3. **Moderate — Thermal attribution is an estimate, and the prefill comparison needs the same caveat as decode.** The merger assumes decode wall time equals timed forwards **×1.3**, with a separate assumed **0.5-second** startup offset for reconstruction. Neither overhead is measured. The first sampled “checkpoint done” can occur after completion; padded windows can include prefill or needle work. Also, `cap_mhz` stores the maximum **observed clock**, not the thermal ceiling.

   The reported **19.8086 → 26.8976 tok/s, 1.3579×** at 7.9k reproduces. Calling it a comparison under different caps is honest; presenting the cap ranges as exact decode attribution is not. The **1.7978× later-prefill gain** is also thermally confounded and should not fall under the verdict’s “Uncapped” framing. These short, cooling-separated runs do not establish sustained steady-state throughput.  
   References: [merge_books.py:48](/Users/alpha/execupack-kvstudy/scripts/kvstudy/merge_books.py:48), [merge_books.py:56](/Users/alpha/execupack-kvstudy/scripts/kvstudy/merge_books.py:56), [merge_books.py:62](/Users/alpha/execupack-kvstudy/scripts/kvstudy/merge_books.py:62), [narrative.html:4](/Users/alpha/execupack-kvstudy/work/kvstudy/report/narrative.html:4), [narrative.html:42](/Users/alpha/execupack-kvstudy/work/kvstudy/report/narrative.html:42).

4. **Moderate — The headline top-token disagreement range is wrong.** Across the reported CPU rerun cells it is **6.25–14.58%**, not 8–11%. At 7.9k, LFM differs on **24/384** positions and Gemma on **56/384**. The 8–11% approximation describes the 1.9k cells reasonably, but the headline explicitly includes both depths.

   “Roughly a tenth” of the divergence attributed to four-bit weights is also loose: direct KL divided by the fp32-cache file’s KL to HF ranges from approximately **4.9% to 23.4%**. These differently directed KL quantities do not provide an additive attribution of error, and the HF comparison includes activation, embedding, and runtime differences too.  
   References: [narrative.html:3](/Users/alpha/execupack-kvstudy/work/kvstudy/report/narrative.html:3), [narrative.html:7](/Users/alpha/execupack-kvstudy/work/kvstudy/report/narrative.html:7), [LFM raw JSON:1](/Users/alpha/execupack-kvstudy/work/kvstudy/phone-out2/stream-8k-lfm-q8kv.json:1), [Gemma raw JSON:1](/Users/alpha/execupack-kvstudy/work/kvstudy/phone-out2/stream-8k-gemma-q8kv.json:1).

5. **Moderate — The cross-phone speed claim misses another exception and does not bound confounding.** LFM 2k changed from **65.798 → 71.058 tok/s** with fp32 (**+7.99%**) and **71.083 → 75.746** with int8 (**+6.56%**). Thus Qwen 2k is not the sole exception to “within about 5%.” Qwen’s second-phone ratio is **1.9880×**, rounding to **1.99×**, not 1.98×.

   All saved CPU checkpoint KL values do match exactly across phones, at their stored precision. The GPU Llama 2k speeds reproduce within **0.24%**, supporting that claim. However, agreement between two uncontrolled runs cannot “bound” a thermal or ordering bias shared by both.  
   References: [narrative.html:32](/Users/alpha/execupack-kvstudy/work/kvstudy/report/narrative.html:32), [narrative.html:59](/Users/alpha/execupack-kvstudy/work/kvstudy/report/narrative.html:59).

6. **Moderate — The memory mechanism is plausible, but the 32k extrapolation and failure attribution need correction.** The measured Llama fp32 KGSL peaks are **2.35738 and 5.61903 GiB**. Linear extrapolation against window size gives **18.66565 GiB at 32k**, not approximately 16 GiB. Sixteen GiB corresponds to the proposed two attention buffers alone, before cache and other allocations.

   The two-buffer explanation has support in the [Vulkan SDPA implementation](https://github.com/pytorch/executorch/blob/v1.5.1/backends/vulkan/runtime/graph/ops/impl/SDPA.cpp#L705), which allocates attention and softmax intermediates using head, sequence, and context dimensions. That supports the mechanism, not an exact allocation accounting or an OOM diagnosis. The explicit “not confirmed” caveat is appropriate. Qwen’s build log establishes **status −9**, but does not establish “killed for memory”; no OOM diagnostic is retained.  
   References: [narrative.html:38](/Users/alpha/execupack-kvstudy/work/kvstudy/report/narrative.html:38), [narrative.html:44](/Users/alpha/execupack-kvstudy/work/kvstudy/report/narrative.html:44), [Qwen export log:53](/Users/alpha/execupack-kvstudy/work/kvstudy/logs/gpu/export-vk-qwen-f32kv-32k.log:53).

7. **Moderate — The bootstrap supports a conditional observation, not well-calibrated inference across books.** The implementation pools **24 sixteen-position blocks**, without resampling books as the independent units. I reproduce Llama 8k’s intervals:

   | Depth | Mean increase | Reported block interval | Whole-book percentile interval |
   |---|---:|---:|---:|
   | 1.9k | +0.012631 | [+0.007157, +0.019050] | [+0.001641, +0.031514] |
   | 7.9k | +0.012063 | [+0.007322, +0.016171] | [+0.010748, +0.013946] |

   **All three book means are positive at both depths.** Unlike the earlier CPU finding, whole-book resampling does not reverse this result. Nevertheless, three books are too few for confident population-level coverage, and one book contributes about **83%** of the aggregate 1.9k increase. “Higher observed mean KL in all three books” is well supported; broader statistical claims need stronger sampling.  
   References: [analyze.py:63](/Users/alpha/execupack-kvstudy/scripts/kvstudy/analyze.py:63), [report.py:62](/Users/alpha/execupack-kvstudy/scripts/kvstudy/report.py:62), [narrative.html:46](/Users/alpha/execupack-kvstudy/work/kvstudy/report/narrative.html:46).

8. **Minor — Several smaller statements need correction or supporting artifacts.**

   - The GPU residual bucket averages **23.29% pooled**, with Qwen 2k approximately **10.42%**; 28% is approximately correct for Llama 2k, not the whole GPU pass.
   - “After about 17 seconds of prefill” does not describe both Llama 8k files: total timed prefill is approximately **18.3–19.0 seconds fp32**, versus **10.8–11.2 seconds fp16**.
   - “Worst row” means **largest increase**, not highest divergence. Llama’s quoted 0.22 → 0.63 row is not its maximum-KL question.
   - The setup still says paired files “differ only in the cache,” contradicting the correctly explained GPU floating-point precision change.
   - I could not reproduce the specific **Qwen 20–25-second / 28%-slowdown** story from the retained throttled files, or find retained phone records establishing the **32k exit-134 / within-one-second** claim. Those observations need their corresponding records.

   References: [narrative.html:14](/Users/alpha/execupack-kvstudy/work/kvstudy/report/narrative.html:14), [narrative.html:38](/Users/alpha/execupack-kvstudy/work/kvstudy/report/narrative.html:38), [narrative.html:40](/Users/alpha/execupack-kvstudy/work/kvstudy/report/narrative.html:40), [narrative.html:42](/Users/alpha/execupack-kvstudy/work/kvstudy/report/narrative.html:42), [narrative.html:46](/Users/alpha/execupack-kvstudy/work/kvstudy/report/narrative.html:46), [report.py:131](/Users/alpha/execupack-kvstudy/scripts/kvstudy/report.py:131).

9. **Minor — Direct KL is mathematically correct, but saved rounding slightly breaks normalization.** The calculation uses the correct direction, **KL(fp32-cache file ‖ reduced-precision file)**, on the same 64 token IDs plus their complement. This coarsened KL is a lower bound on full-vocabulary KL, not an upper bound.

   Six-significant-digit log-probability serialization produces a few selected-token masses above one, by up to **1.27×10⁻⁷**. Clamping the residual does not restore normalization. This is numerically negligible for the reported conclusions, but preserving more precision would make the probability calculation cleaner.  
   References: [StudyBench.java:204](/Users/alpha/execupack-kvstudy/scripts/kvstudy/phone/StudyBench.java:204), [report.py:30](/Users/alpha/execupack-kvstudy/scripts/kvstudy/report.py:30).

The principal GPU calculations check out:

| Model/window/depth | Direct KL mean / maximum | Decode fp32 → fp16, tok/s | Ratio |
|---|---:|---:|---:|
| Llama 2k / 1.9k | 0.001347 / 0.036507 | 47.141 → 56.990 | 1.2089× |
| Qwen 2k / 1.9k | 0.002909 / 0.175976 | 28.680 → 30.237 | 1.0543× |
| Llama 8k / 1.9k | 0.003133 / 0.061529 | 57.195 → 68.194 | 1.1923× |
| Llama 8k / 7.9k | 0.006699 / 0.089848 | 19.809 → 26.898 | 1.3579× |

Prefill ratios reproduce as **1.3027×, 1.0939×, 1.5220× and 1.7978×**. KGSL savings reproduce as **0.66771 GiB** for Llama 2k, **2.42727 GiB** for Llama 8k, and **0.44203 GiB** for Qwen 2k. The merged files match their per-book source measurements.

CPU direct-KL means reproduce the stated **0.013–0.037** range; the largest individual value is **1.035899**, for Qwen at 7.9k. The CPU speed ranges and **4.91445 GiB** maximum resident-memory saving also reproduce. Needle totals are correct: **144 valid original CPU pairs, seven gains and four losses**; fresh-prefix LFM gives **12/12 and 21/21** for both caches. GPU needle counts and the app-question aggregates reproduce, including Llama’s **0.234187 → 0.208651**, with **12 worse / 20 better** rows.

Overall, the page supports a provisional performance-and-memory recommendation and the reported distributional differences. It needs narrower quality wording, estimated thermal labels, corrected headline numbers, and more cautious failure attribution before its conclusions are fully supported.


