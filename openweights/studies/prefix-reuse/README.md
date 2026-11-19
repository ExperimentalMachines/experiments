# Prefix reuse

**Question.** How much does a retained prompt prefix save against a reset cache, across chips and
both runtimes, and what does a second turn re-read after each way the first one ended (a clean
end, a budget cut, a cancel, an edited history)?

**Write-up** (in OpenWeights): [`reply-end-and-reuse.md`](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/reply-end-and-reuse.md).

**Runs.**

- `2026-11-15-coldwarm-poco`: the same prompt read cold and warm on the POCO.
- `2026-11-16-coldwarm-matrix`: the same with a retained prefix and a reset cache on more chips.
- `2026-11-16-reuse-boundary`: the second turn after each kind of ending.

**Harness.** The instrumented tests in OpenWeights' `core/engine` (`WarmPrefixEval`,
`ExecuTorchOnDeviceTest`); the Test Lab runners used are kept in the run folders (`ftl.sh`).
