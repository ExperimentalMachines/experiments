# Working in this repository

This repository holds measurements: device runs, their harness, and the write-ups that live with
their data. The apps and pipelines they measured are in their own repositories. Read
[`README.md`](README.md) for the layout.

## Rules that are not visible in the files

- **A run is a record.** Never edit a file under `runs/` after it is committed. A correction
  goes in the study's README, with the date and what was wrong. A rerun is a new run folder.
- **Every run says what it ran.** A new run folder records the commit of the app or pipeline it
  was built from, the device, and the date, in a README or NOTES file beside the data.
- **No raw device logs.** A full logcat holds vendor tokens, MAC and IP addresses, SSIDs and
  other apps' lines. Cut a log to the app's own lines before it is staged
  (`tools/logcat_filter.py` does it the way the imports did), then search what is staged
  (`git grep --cached`, and inside any `.xz`) for tokens, addresses and serials. Keep full logs
  outside git.
- **No identifiers.** No adb serials, wireless-debugging names, home-network addresses, cloud
  project, account or instance ids, login addresses, device-cloud hostnames, or API keys. Scripts read a serial from `SER`.
  A record that must say which unit ran it uses the unit's pseudonym, `serials.pseudonym()` in
  `tools/serials.py`, the same name the imported runs use.
- **No model files.** Weights, `.pte`, `.gguf`, tokenizers and checkpoints are downloadable or
  rebuildable; name them and their source instead. `.gitignore` refuses the common ones.
- **Imported files change only through the import scripts.** To bring in newer runs from a
  source project, move the pinned commit in `tools/import_<project>.py`, rerun it, and read the
  diff. `--check` must pass before a commit that touches imported files.
- **Write-ups stay where they are cited.** OpenWeights' `docs/research/` stays in the app
  repository because its code cites it; a study here links to it at a pinned commit.
- **Prose style.** No em dashes or en dashes anywhere. Recast the sentence with a comma, a colon
  or a full stop. Comments say why, not what.

## Checks

```sh
for p in openweights execupack execuserve; do python3 tools/import_$p.py --check; done
python3 tools/check_links.py
```

The `--check` runs read the source repositories on this machine at the pinned commits, so they
need the checkouts the import scripts name.

## Git

Commit subjects follow the source projects: a scope, a colon, and a sentence that says what
changed and why (`openweights: the S24 matrix rerun at 4 threads, because ...`). The body says
what was measured. Agents append their own attribution trailer.
