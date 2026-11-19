# Compiled tool calling

**Question.** Why does the compiled (ExecuTorch) LFM2.5 narrate a web search it never makes,
when the GGUF of the same model calls the tool? Token-level probes of the same prompts across
quantisations, exports and runtimes.

**Write-ups** (in OpenWeights): [`executorch-tool-calling.md`](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/executorch-tool-calling.md), [`executorch-tool-calling-qad.md`](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/executorch-tool-calling-qad.md).

**Runs.** `runs/toolcall-probe/`: the probe rows, the per-token probabilities, the briefs and
both reviews.

**Harness.** [`harness/app-evals/bench/toolcall_probe/`](../../harness/app-evals/bench/toolcall_probe/):
the export variants (`export_*.sh`), the probes, and its README. They are kept as they ran on
2026-09-13: several read and write a session scratch folder that no longer exists, and most write
their outputs beside themselves. `save_and_time.py` is what copied the outputs that mattered into
`runs/toolcall-probe/`, where they are now.
