# Test Lab archive

The Firebase Test Lab runs wrote everything to a bucket that was deleted on 2026-09-30. Before it
went, every file was compared with the repository and the local archives, and what existed
nowhere else and was text is here: result rows, test outcomes and xz-compressed logcats.

It backs four write-ups (in OpenWeights): [`dropped-pass-reread.md`](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/dropped-pass-reread.md), [`release-token-callback-abort.md`](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/release-token-callback-abort.md), [`typesafe-experiments.md`](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/typesafe-experiments.md), [`retrieve-or-answer.md`](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/retrieve-or-answer.md).
[`runs/ftl/README.md`](runs/ftl/README.md) says which folder backs which, and why four decision
runs are kept here rather than appended to
[`retrieve-or-answer`](../retrieve-or-answer/README.md).

The logcats (`logcat.xz`, `*_logcat.xz`) were cut to the app's own lines on import, as in
[`on-device-evals`](../on-device-evals/README.md); the full ones are in OpenWeights at
[81734dee](https://github.com/ExperimentalMachines/openweights/tree/81734deed5048f5691518d0e739662c50e51e1a3/tools/eval/results/ftl).
