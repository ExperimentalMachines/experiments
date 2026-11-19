# Retrieve or answer

**Question.** When should the app's small model search the web instead of answering? Measured on
seeded samples of public retrieve-or-answer sets, through the app's own tool loop, per model,
runtime, rule (arm) and phone. Two follow-ups: whether the model's own token probabilities
predict a wrong bare answer (the doubt gate), and how a typed judgement model reads the loop's
text matchers (TypeSafe).

**Write-ups** (in OpenWeights): [`retrieve-or-answer.md`](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/retrieve-or-answer.md), [`judged-decisions.md`](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/judged-decisions.md), [`typesafe-experiments.md`](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/typesafe-experiments.md), [`recommended-runtime.md`](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/recommended-runtime.md).

**Runs.**

- `decisions/`: one JSON line per question per run, named by phone prefix, model and arm, with
  the grader's outputs beside them. Later studies added rows here too: the shipped-build runs
  (`qdc-sm8650-shipped-*`) and the last round.
- `gate/`, `confidence/`: the doubt gate, offline against llama-server.
- `typesafe/`: TypeSafe's readings and their caches.

More decision rows that never reached `decisions/` are in
[`test-lab-archive`](../test-lab-archive/runs/ftl/README.md).

**Harness.** [`harness/app-evals/bench/`](../../harness/app-evals/bench/): `pull_decisions.py`
and `decisions*.json` (the question sets), `run_decisions*.sh` (adb, Test Lab), `grade_decisions.py`
and `report_decisions.py`, `gate.py`, `confidence.py`, `local_*.py`, `typesafe_*.py`.

```sh
python3 openweights/harness/app-evals/bench/grade_decisions.py openweights/studies/retrieve-or-answer/runs/decisions
python3 -m unittest openweights/harness/app-evals/bench/test_grade_decisions.py
```
