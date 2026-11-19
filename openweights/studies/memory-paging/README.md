# Memory paging

**Question.** When a model overflows into zram, when does it slow the app, what does peak memory
look like per model and runtime, and does bringing paged-out weights back while the next message
is typed pay for itself?

**Write-up** (in OpenWeights): [`swap-and-peak-memory.md`](https://github.com/ExperimentalMachines/openweights/blob/81734deed5048f5691518d0e739662c50e51e1a3/docs/research/swap-and-peak-memory.md).

**Runs.**

- `2026-10-12-peak`, `-peak-pte`, `-pressure`, `-flag`, `-readout`: peak memory, outside
  pressure, and the paging flag the app shows.
- `2026-10-23-prefetch` to `-prefetch4`: the prefetch trials.
- `matrix-*`: `PrefetchMatrixBenchmark`, every model in every conversation shape.
  `matrix-predeclared.md` is the analysis written down before the matrix ran.

**Harness.** [`harness/memory/`](../../harness/memory/): `run_pressure.sh` and
`run_prefetch_matrix.sh` drive `MemoryPressureBenchmark` and `PrefetchMatrixBenchmark` from
OpenWeights' `core/engine` instrumented tests; `balloon.c` applies the pressure; `analyze*.py`
make the tables. The runs folder's `.gitignore` keeps raw logcats out.
