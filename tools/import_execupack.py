#!/usr/bin/env python3
"""ExecuPack's experiment material: the research log and its evidence, the byte-level code
checks, the experiment branches as patches, and the results that only ever lived on this Mac.

What stays in ExecuPack on purpose: `pipeline/`, its tests and CI, `docs/PLAN.md` (a roadmap
with run notes, not a record), and the two measurement workflows under `.github/`, which only
run from there. The research log is read from `executorch-1.6`, the pushed branch one commit
past `origin/main` that adds finding 48; finding 49 exists only as an uncommitted row in the
`mtk-a16w4` worktree and is appended to it here.

  python3 tools/import_execupack.py [--check]
"""

import re
from pathlib import Path

import subprocess

import serials
from importer import Edit, Project, Source, main

HOME = Path.home()
E = "execupack"
# Text records of a run, including the job status files (`.rc` exit codes, `.err` stderr, `.ok`
# markers), the empty `published-*` markers, and the exported files' checksums.
TEXT = {".py", ".sh", ".md", ".json", ".jsonl", ".log", ".out", ".txt", ".tsv", ".csv", ".yaml", ".yml",
        ".rc", ".err", ".ok", "started", "SHA256SUMS", "published-*"}
BINARY_JUNK = {
    "tokenizer*.json": "tokenizer, downloadable from the model's repository",
    "vocab.json": "tokenizer vocabulary",
    "merges.txt": "tokenizer merges",
    "*/.cache/*": "Hugging Face download cache",
    "*/.venv/*": "virtual environment",
}

ROW_49 = next(line for line in (HOME / "execupack-mtkw4/docs/research/README.md").read_text().splitlines()
              if line.startswith("| 49 |"))

# The patches' headers name their commits' authors. A personal address becomes the GitHub
# no-reply address of the same account, which `git am` accepts as an author.
NOREPLY = "108984668+alpharomercoma@users.noreply.github.com"
PATCH_RANGES = ["0660a3c952cc..2e8c973fc3f3851f8cf20ef62cfeb127cde57781",
                "2e8c973fc3f3851f8cf20ef62cfeb127cde57781..a463a66055ea504b45391d7a0ac3e3e582319f4b",
                "2e8c973fc3f3851f8cf20ef62cfeb127cde57781..34b98f1d9b8200bdbc25b9ae3c97ef4bde391641",
                "b5affc7617b9b0b8283ce588c6715e1b86a31ea6..9a883e99594219e77918c38cade4d68403db5fc0"]
AUTHORS = {a for r in PATCH_RANGES for a in subprocess.run(
    ["git", "-C", str(HOME / "execupack"), "log", "--format=%ae%n%ce", r], capture_output=True, text=True, check=True
).stdout.split() if "noreply" not in a}
TO_NOREPLY = serials.replacer(AUTHORS, lambda _: NOREPLY, r"A-Za-z0-9._%+@-")

RESEARCH = Source(
    label="execupack@executorch-1.6 (b5affc76)",
    url="https://github.com/ExperimentalMachines/execupack",
    protect=[r"(?i)openweights:?\s*\[?`?docs/research/[^\s`)\]]*"],
    checkout=HOME / "execupack",
    commit="b5affc7617b9b0b8283ce588c6715e1b86a31ea6",
    roots=["docs/research", "tools/codecheck"],
    moves=[("docs/research", f"{E}/research"), ("tools/codecheck", f"{E}/harness/codecheck")],
    edits={
        # Finding 49 comes from the mtk-a16w4 worktree, where it was never committed.
        "docs/research/README.md": [Edit(r"(\n\| 48 \|[^\n]*)", lambda m: m.group(1) + "\n" + ROW_49, regex=True)],
        # evidence/ sits at execupack/research/evidence, so parents[2] is execupack/.
        "docs/research/evidence/et160-constdiff.py": [
            Edit('parents[3] / "tools/codecheck"', 'parents[2] / "harness/codecheck"')],
        "docs/research/evidence/et160-structure.py": [
            Edit('parents[3] / "tools/codecheck"', 'parents[2] / "harness/codecheck"')],
    },
    patches={
        # The calibration-corpus experiment: three commits on experiment/extended-calibration
        # and experiment/extended-calibration-app, then the isolated GPTQ-257 publish on top.
        f"{E}/studies/calibration-corpus/calibration-corpus.patch":
            ["format-patch", "--stdout", "0660a3c952cc..2e8c973fc3f3851f8cf20ef62cfeb127cde57781"],
        f"{E}/studies/calibration-corpus/publish-lfm12b-gptq257.patch":
            ["format-patch", "--stdout", "2e8c973fc3f3851f8cf20ef62cfeb127cde57781..a463a66055ea504b45391d7a0ac3e3e582319f4b"],
        f"{E}/studies/published-audit/audit-published.patch":
            ["format-patch", "--stdout", "2e8c973fc3f3851f8cf20ef62cfeb127cde57781..34b98f1d9b8200bdbc25b9ae3c97ef4bde391641"],
        # awq-compare was never pushed: its three commits exist only in this Mac's checkout.
        f"{E}/studies/awq/awq-compare.patch":
            ["format-patch", "--stdout", "b5affc7617b9b0b8283ce588c6715e1b86a31ea6..9a883e99594219e77918c38cade4d68403db5fc0"],
    },
    hooks=[TO_NOREPLY],
)

A16W4 = Source(
    label="execupack-mtkw4 worktree (mtk-a16w4 at 34c55110, uncommitted)",
    checkout=HOME / "execupack-mtkw4",
    commit=None,
    roots=["docs/research/evidence/mtk-a16w4-gptq.md", "scripts/mtk_gptq.py", "scripts/mtk_quant_eval.py",
           "work/a16w4/results"] + [f"work/a16w4/{d.name}" for d in sorted((HOME / "execupack-mtkw4/work/a16w4").iterdir())
                                    if d.is_dir() and d.name != "results"],
    moves=[
        ("docs/research/evidence/mtk-a16w4-gptq.md", f"{E}/research/evidence/mtk-a16w4-gptq.md"),
        ("scripts/mtk_gptq.py", f"{E}/studies/mtk-a16w4/harness/mtk_gptq.py"),
        ("scripts/mtk_quant_eval.py", f"{E}/studies/mtk-a16w4/harness/mtk_quant_eval.py"),
        ("work/a16w4/results", f"{E}/studies/mtk-a16w4/runs"),
        # Each model's A16W4 export: its report, runner config and checksums, not the chunks.
        ("work/a16w4", f"{E}/studies/mtk-a16w4/exports"),
    ],
    suffixes=TEXT,
    exclude={**BINARY_JUNK, "USE_POLICY.md": "the vendor's license text"},
    max_bytes=6_000_000,
    # The product edits the study made (the MTK_WEIGHTS_FROM hook and its tests, the finding
    # row, a PLAN note), as one diff against the worktree's commit.
    patches={f"{E}/studies/mtk-a16w4/uncommitted.patch": ["diff", "HEAD"]},
    rewrite="prose",
)

AWQ = Source(
    label="execupack-awq work/awq (gitignored, 2026-11-19)",
    checkout=HOME / "execupack-awq",
    commit=None,
    roots=["work/awq"],
    moves=[("work/awq", f"{E}/studies/awq/runs")],
    suffixes=TEXT,
    exclude={**BINARY_JUNK, "work/awq/t135/*work/*": "solve and export working folders: weights and checkpoints"},
    max_bytes=6_000_000,
    rewrite="none",  # run records: the scripts ran from the execupack-awq checkout as written
)

ARCHIVE = Source(
    label="execupack-aws-archive-2026-11-09 (the four EC2 export boxes)",
    checkout=HOME / "execupack-aws-archive-2026-11-09",
    commit=None,
    roots=["README.md", "Q1", "Q3", "P1", "P3"],
    # The archive's own README becomes ARCHIVE.md, beside the study's README.
    moves=[("README.md", f"{E}/studies/aws-runs-2026-11/ARCHIVE.md")]
          + [(p, f"{E}/studies/aws-runs-2026-11/{p}") for p in ("Q1", "Q3", "P1", "P3")],
    suffixes=TEXT,
    exclude={**BINARY_JUNK,
             "*/source/*": "model source downloads",
             "*/codecheck/dl/*": "files downloaded for the byte checks",
             "*.metadata": "download metadata"},
    max_bytes=6_000_000,
    redact=[
        ("README.md", r"\b\d{12}\b", "<aws-account>"),
        ("README.md", r"\bi-[0-9a-f]{8,17}\b", "<instance>"),
        ("*/fetch-*.log", r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "<ip>"),
    ],
    rewrite="none",
)

PROJECT = Project(E, [RESEARCH, A16W4, AWQ, ARCHIVE])

if __name__ == "__main__":
    main(PROJECT)
