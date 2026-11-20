# Adding a run

How to put a new measurement in this repository so that the next person can trust it and rerun
it. The short version: a new folder, a README that says what ran, no identifiers, no model
files.

## 1. Pick the study

Each question has one study folder: `<project>/studies/<study>/`. If your run answers an existing
question (a rerun, a new phone, a new model), it goes there. If it asks a new question, start a
new study folder with a README in the shape the others use (see
[prefix-reuse](../openweights/studies/prefix-reuse/) for the model).

## 2. Make a new run folder

```text
<project>/studies/<study>/runs/<yyyy-mm-dd>-<what-or-where>/
    README.md        what ran, on what, from which commit, and what it found
    *.jsonl / *.csv  one row per measured turn or request
    *.log            the device's own log lines for the run, cut to the app
    run.sh, *.py     the exact scripts that produced and summarised it
```

Then record the folder in `<project>/ADDED.tsv` (create it if it is not there), one row per run
folder, tab-separated: the folder's path from the repository root, the date, and what it holds.

```text
# folder	date	what
openweights/studies/prefix-reuse/runs/2026-12-01-pixel10	2026-12-01	cold and warm second turns on a Pixel 10
```

Every file under a project needs a reason to be there: a `PROVENANCE.tsv` row if an import wrote
it, an `ADDED.tsv` folder if it was recorded here, or being a `README.md`. The import check
reports anything else as `no provenance`, and the importer never writes into or removes an
`ADDED.tsv` folder.

A run is a record. Once it is committed, never edit its files; a correction goes in the study's
README with the date and what was wrong, and a rerun is a new folder.

## 3. Say what ran

The run's README names, at minimum:

- the **commit** of the app or pipeline that was built, and any local change on top of it;
- the **device**: marketing name, chip, and where it was (own phone or which lab), using the
  unit's pseudonym if a unit must be named;
- the **model files**, by name and where they come from (they are not committed);
- the **date**, and anything about the conditions that matters (screen on, temperature, charging);
- the **result** in a sentence or a small table, so nobody has to rerun the analysis to know it.

## 4. Keep identifiers and personal data out

| Do not commit | Instead |
|---|---|
| An adb serial | Its pseudonym (below), the same `unit-xxxxxx` name the imported runs use. Scripts read a serial from `SER`. |
| A whole-device logcat | Cut it to the app's lines: `python3 tools/logcat_filter.py full.logcat > app.logcat`. Then search what is staged, including inside `.xz` files. Keep full logs outside git. |
| Cloud project, account or instance ids, login addresses, host names, IP addresses | Leave them out, or use the `gcp-project-` and `account-` pseudonyms the same way. |
| API keys, session keys, `start.txt` files | `.gitignore` refuses the common ones. |
| Weights, `.pte`, `.gguf`, tokenizers, checkpoints | Name the file and where to get or rebuild it. |

A unit's pseudonym, from the repository root:

```sh
python3 -c 'import sys; sys.path[:0] = ["tools"]; import serials; print(serials.pseudonym(sys.argv[1]))' <serial>
```

Before you commit, stage the run and check it (the import check reads the source repositories, so
it needs their checkouts on your machine; without them, run the other two and say so in the
commit):

```sh
python3 tools/import_<project>.py --check
python3 tools/check_links.py
git diff --cached --stat
git grep --cached -nE "unit [A-Za-z0-9]{8,}|projects/[a-z]|@[a-z0-9-]+\.(com|net|org)" -- <your folder>
```

## 5. Update the study README

Add the run to the study's "What's in this folder" table. If it changes an answer, change the
"What we found" alert and say what changed and when. If a chart draws from this study, rerun
`python3 tools/figures.py` and commit the regenerated figures with the run.

## 6. Bringing runs over from a source repository

Runs that were made inside OpenWeights, ExecuPack or ExecuServe are not copied by hand. Move the
pinned commit in `tools/import_<project>.py`, rerun it, read the diff, and commit only when
`python3 tools/import_<project>.py --check` passes. See [how this repository was built](provenance.md).

## Commit messages

A scope, a colon, and a sentence saying what changed and why, the way the source projects write
them: `openweights: the S24 matrix rerun at 4 threads, because ...`. The body says what was
measured.
