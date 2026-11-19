# Experiments

The measurements behind ExperimentalMachines' on-device inference projects, kept apart from the
products they shaped. Each project's app or pipeline stays in its own repository; what is here is
the harness that drove the devices, the runs it recorded, and the write-ups that live with their
data.

| Folder | Project | What is here |
|---|---|---|
| [`openweights/`](openweights/) | [OpenWeights](https://github.com/ExperimentalMachines/openweights), the Android app | The device harness, every run since August 2026, the host prompt evals. The write-ups stay in the app's `docs/research/`, which the code cites. |
| [`execupack/`](execupack/) | [ExecuPack](https://github.com/ExperimentalMachines/execupack), the export pipeline | The research log with its evidence, the byte-level code checks, the experiment branches as patches, and runs that only ever existed on one machine. |
| [`execuserve/`](execuserve/) | [ExecuServe](https://github.com/ExperimentalMachines/execuserve), the on-phone server | Every study from `docs/results/`, one write-up and one evidence folder each. |

Each project folder has a README that indexes its studies. Inside a project, `harness/` holds
the code that runs on devices and reads the data, and `studies/<study>/` holds one question's
runs and a README naming the write-up and the harness that produced them.

## Where every file came from

Nothing here was copied by hand. `tools/import_<project>.py` reads each source at a pinned
commit (or, for work that was never committed, from the named folder on disk), places each file,
and records it in `<project>/PROVENANCE.tsv`: the source path, the commit, the source blob, the
blob written here, and whether the bytes changed on the way. They change for five reasons only,
each listed in the import script: a moved path rewritten in prose or code (never in a data
file), an exact edit so a moved script finds its files again, a redaction, a device logcat cut
to the app's own lines ([`tools/logcat_filter.py`](tools/logcat_filter.py)), and a device serial,
cloud project or login address replaced by a stable pseudonym such as `unit-3fa2c1`
([`tools/serials.py`](tools/serials.py)), so results still group by unit.

```sh
python3 tools/import_openweights.py --check   # every file matches its source; nothing here is unaccounted for
python3 tools/check_links.py                  # every relative link resolves
```

A path written in prose here is relative to this repository's root. A path that does not exist
here (`core/engine/...`, `pipeline/...`) belongs to the source project at the commit its
PROVENANCE row names.

## Running the harness

Scripts that build or drive an app take its checkout from the environment and stop if it is not
set: `OPENWEIGHTS` for OpenWeights, `REPO_ROOT` for ExecuServe, `EXECUPACK` for ExecuPack, and
`SER` for a phone's adb serial where a script used to carry one. Results land under the study
they belong to, except where a study README says its scripts are kept exactly as they ran.

See [`AGENTS.md`](AGENTS.md) for the rules of adding a run.
