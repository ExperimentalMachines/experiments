# Calibration corpus

**Question.** Does GPTQ calibrated on a larger corpus (377 rows instead of 104) keep more of a
model, and is that the corpus's size or its composition? Arm B calibrated on 257 rows with no
word problems, through an app-only corpus mode.

**Source.** Pushed experiment branches in ExecuPack, kept here as patches:

- [`calibration-corpus.patch`](calibration-corpus.patch): `experiment/extended-calibration`
  (two commits) and `experiment/extended-calibration-app` (one more), from 0660a3c9. They add
  `config/calibration-extended.json`, the corpus modes in `pipeline/calibration.py`, and tests.
- [`publish-lfm12b-gptq257.patch`](publish-lfm12b-gptq257.patch): `publish/lfm12b-gptq257` on
  top, which isolates LFM2.5 1.2B GPTQ-257 exports for publishing, and two reference renames.

No finding row cites this experiment; the commit messages in the patches are its record.
