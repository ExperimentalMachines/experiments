# The shipped build

**Question.** The app as it really ships, not a test build: the whole decision suite and the
last round on a QDC phone, the signed release on a 16 KB emulator, and the signed final build on
a real phone.

**Write-up** (in OpenWeights): [`llama-cpp-v080.md`](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/llama-cpp-v080.md), its release sections.

**Runs.** `2026-11-05-qdc-last-round`, `2026-11-05-qdc-shipped`, `2026-11-06-emulator-release`,
`2026-11-06-qdc-final`. The decision rows they produced are in
[`retrieve-or-answer/runs/decisions/`](../retrieve-or-answer/runs/decisions/)
(`qdc-sm8650-shipped-*`); the scripts kept in these folders read them from there and run from
this repository's root.
