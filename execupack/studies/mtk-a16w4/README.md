# MediaTek A16W4

**Question.** How far from fp32 is MediaTek's A16W4 (4-bit weights, one scale per row) against
the published A16W8, with and without a GPTQ solve on that grid, and does it pass the publish
gate?

**Finding.** Row 49 of [`research/README.md`](../../research/README.md), written up in
[`research/evidence/mtk-a16w4-gptq.md`](../../research/evidence/mtk-a16w4-gptq.md).

**Source.** All of it was uncommitted in the `mtk-a16w4` worktree on 2026-11-19:

- [`uncommitted.patch`](uncommitted.patch): its edits to tracked files against 34c55110 (a plain
  diff, for `git apply`): the `MTK_WEIGHTS_FROM` hook in `pipeline/export_mtk.py` and its tests,
  the finding row, a PLAN note.
- `harness/`: `mtk_gptq.py` (the GPTQ solve on A16W4's grid) and `mtk_quant_eval.py` (finding
  35's harness, generalised).
- `runs/`: the gitignored `work/a16w4/results/`, text only: per model the plan, corpus, solve
  log, and evaluations.
- `exports/`: each model's A16W4 export for MT6991 from `work/a16w4/<model>/`: its export
  report, runner config and checksums. The chunks and embeddings themselves (5 GB) stayed out.
