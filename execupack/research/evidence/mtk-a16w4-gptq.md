# A16W4 on MediaTek, with and without GPTQ on MediaTek's own grid

2026-11-19, RunPod A100 SXM 80 GB (32 vCPU, 250 GB). The question: is A16W4 on the NPU
significantly further from fp32 than the published A16W8, and does a better weight quantizer
close the gap? The context is that A16W8 files do not load on a Dimensity 9400 (MT6991) phone
and A16W4 halves the weights. AWQ was ruled out before this ran: it failed the publish gate on
the CPU recipe in the parallel AWQ-vs-GPTQ study. GPTQ is what is measured here.

## What A16W4 is

`backends/mediatek/quantizer/qconfig.py` at `EXECUTORCH_COMMIT`, `get_a16w4_quant_config`,
which `NeuropilotQuantizer` calls with `is_per_channel=True`:

- One scale and zero point per **output row, across the whole input dimension, with no groups**.
- Asymmetric over [-8, 7], from `PerChannelMinMaxObserver` (eps 1e-6), then round to nearest.
- Activations are 16-bit per-tensor symmetric from MinMax observers.

A16W8 differs only in the weights: symmetric int8, per row.

## GPTQ that survives MediaTek's quantizer unchanged

`execupack/studies/mtk-a16w4/harness/mtk_gptq.py` solves GPTQ on that grid. It uses `solve.py`'s settings:

- blocks of 128 and 1% damping;
- the same 104 calibration rows, with the app head counted once;
- layer by layer, on a model whose earlier layers are already quantized;
- no activation fake-quant, because MediaTek's activations are 16-bit.

The output head is solved too and written untied. The token embedding the runner reads stays the
original.

The grid is fixed per row from the row's own min and max, as the observer will measure them. Two
columns are pinned: the one holding the row's minimum to code -8, and the one holding its maximum
to 7. GPTQ carries the pins' error forward like any rounding. So the dequantized row still spans
the same [min, max], the observer re-derives the solve's scale and zero point, and rounding returns
the solve's codes. Measured on every linear of all three models: **0 weights re-quantize to a code
other than the one solved**. The export (`MTK_WEIGHTS_FROM`, `export_mtk.replace_weights`), the
calibration of the activations and the runner are MediaTek's, unchanged.

Per-linear output error tr((W-Q) H (W-Q)ᵀ), GPTQ as a fraction of plain rounding on the same grid:

| model | linears | median | range | head | solve |
|---|---|---|---|---|---|
| Llama-3.2-1B-Instruct | 113 | 0.20 | 0.05-0.65 | 0.23 | 307 s |
| Qwen3-1.7B | 197 | 0.21 | 0.04-0.32 | 0.21 | 550 s |
| LFM2.5-1.2B-Instruct | 93 | 0.27 | 0.04-0.48 | 0.33 | 316 s |

## Harness

`execupack/studies/mtk-a16w4/harness/mtk_quant_eval.py` is finding 35's harness, generalised to `qwen.py` and `llama.py` and
to several variants against one fp32 reference:

- **Chunks and calibration.** It builds the export's own chunks and runs `prepare_pt2e` with
  `NeuropilotQuantizer` on them. Calibration is the export's own:
  - alpaca.txt in the family preformatter for qwen/llama, streamed in lockstep. The observers
    are MinMax, so the ranges are the same as the scripts' Arrow replay.
  - lfm2.py's long-context corpus at the window for LFM2.
- **Conversion.** `convert_pt2e(fold_quantize=False)` turns the result into the graph MediaTek
  lowers, with quantize/dequantize simulated.
- **Scoring.** Everything is teacher-forced in 128-token blocks, each variant on its own cache.
- **Two measures:**
  - KL(fp32 ‖ q) over 2,046 positions of PG19 book 25773, held out, as a chat-template user turn.
  - The publish gate's 16 rows (pipeline/gate.py) under the app prompt and tool schema.
- **Window.** 2,048, 4 chunks.

The runs here used `--device cuda`: the same pins with CUDA torch, and the graphs' recorded
device arguments rewritten. Checked against the CPU on SmolLM2-135M at 512, where the CUDA run
was within about 1-3% of the CPU KL:

| | A16W8 | A16W4 | A16W4+GPTQ |
|---|---|---|---|
| CPU | 0.0457 | 1.2216 | 0.8138 |
| CUDA | 0.0442 | 1.2159 | 0.8048 |

The ordering and the gate verdicts are the same, except one RTN row (1/4 against 0/4).

## Results

Book KL with 95% intervals: a block bootstrap over 128-position blocks, 5,000 resamples.
Intervals on differences are paired.

| model | build | KL to fp32 | vs A16W8 | ppl (fp32) | top-1 agree | gate (kept of 12) | first-token KL |
|---|---|---|---|---|---|---|---|
| Llama-3.2-1B-Instruct | A16W8 | 0.046 [0.034, 0.067] | | 29.36 (29.14) | 88.9% | FAIL 7, mean 0.48 vs 0.88 | 0.400 |
| | A16W4 | 0.937 [0.874, 1.011] | +0.89 [+0.84, +0.95], x20 | 65.83 | 51.1% | FAIL 0 | 6.595 |
| | A16W4 + GPTQ | 0.675 [0.642, 0.707] | +0.63 [+0.60, +0.66], x15 | 53.73 | 57.2% | **PASS 12** | 0.165 |
| Qwen3-1.7B | A16W8 | 0.266 [0.248, 0.285] | | 33.47 (28.71) | 75.1% | FAIL 9, mean 0.73 vs 1.00 | 1.594 |
| | A16W4 | 0.985 [0.928, 1.045] | +0.72 [+0.67, +0.77], x4 | 71.80 | 54.6% | FAIL 1 | 3.956 |
| | A16W4 + GPTQ | 0.826 [0.764, 0.882] | +0.56 [+0.50, +0.61], x3 | 57.25 | 58.8% | **PASS 11** | 1.879 |
| LFM2.5-1.2B-Instruct | A16W8 | 0.0054 [0.0050, 0.0058] | | 134.52 (134.99) | 95.6% | PASS 12 | 0.004 |
| | A16W4 | 0.762 [0.728, 0.797] | +0.76 [+0.72, +0.79], x142 | 194.37 | 56.2% | FAIL 0 | 2.366 |
| | A16W4 + GPTQ | 0.917 [0.860, 0.973] | +0.91 [+0.86, +0.97], x170 | 139.96 | 54.2% | FAIL 8, mean 0.46 vs 0.82 | 0.568 |

GPTQ against stock A16W4, book KL (paired):

| model | difference | |
|---|---|---|
| Llama | -0.262 [-0.324, -0.210] | better |
| Qwen | -0.159 [-0.224, -0.095] | better |
| LFM2 | +0.155 [+0.099, +0.210] | worse on the book, though it keeps 8 of 12 gate rows to stock's 0, and first-token KL is 0.57 against 2.37 |

LFM2's A16W8 and A16W4 reproduce finding 35 at a different window: 0.0068 and 0.751 at 4k.

## Where the error is

**Qwen3's floor is the activations, not the weights.** An A16W16 build of Qwen3-1.7B has book
KL 0.270 (ppl 33.85, top-1 76.1%). That is the same as A16W8's 0.266. It is flat along the
window: 0.285 over the first 64 positions, 0.272 over 1,024-2,046. So it is not long-context
range clipping. It is MediaTek's 16-bit per-tensor activations on this model. Its gate passes
12/12 at A16W16 (mean 0.997), so A16W8's gate failure is the 8-bit weights on top.

**The head is not the problem.** In Hugging Face fp32, weights only, book KL:

| model | original body + 4-bit head | 4-bit body + original head | all 4-bit, RTN | all 4-bit, GPTQ |
|---|---|---|---|---|
| Llama | 0.031 | 0.790 RTN / 0.557 GPTQ | 0.825 | 0.581 |
| Qwen | 0.051 | 0.506 / 0.427 | 0.568 | 0.456 |
| LFM2 | 0.025 | 0.731 / 0.891 | 0.759 | 0.921 |

The per-row int4 decoder linears carry it, so building the head's chunk at A16W8 would not help.
The GPTQ checkpoint scores the same KL through Hugging Face as through MediaTek's path (LFM2
0.921 against 0.917), so the export reproduces the solve.

## What this says

- **Yes, A16W4 is significantly further from fp32 than A16W8, on every model, with or without
  GPTQ:** 3-4x on Qwen3 (whose A16W8 is itself limited by the activations), 15-20x on Llama,
  140-170x on LFM2.
- MediaTek's grid is one scale per 2,048-wide row. The CPU recipe uses groups of 32, where GPTQ
  passes the gate on all three. On this grid GPTQ fixes the decision on Llama (12/12) and Qwen
  (11/12), not on LFM2 (8/12), and does not bring general-text KL near A16W8 anywhere.
- The published A16W8 files fail the gate on Llama-3.2-1B (7/12) and Qwen3-1.7B (9/12) on this
  harness, while A16W4+GPTQ passes on both.

## The exports

`MTK_PRECISION=A16W4 MTK_WEIGHTS_FROM=<solve> python -m pipeline export-mtk <model> --soc MT6991
--context 2048`, run three at a time on the pod (27-core quota, OMP_NUM_THREADS=8 each). All three
passed the structural check: every chunk loads, carries both methods and delegates to
NeuropilotBackend. They have not run on a device.

| model | A16W4 chunks (MB) | total | A16W8 2k, published | embedding (fp32, unchanged) | export | peak RSS |
|---|---|---|---|---|---|---|
| Llama-3.2-1B-Instruct | 156, 156, 156, 289 | 758 MB | 1,375 MB | 1,051 MB | 2,318 s | 40 GB |
| Qwen3-1.7B | 237, 237, 237, 394 | 1,105 MB | 1,963 MB | 1,245 MB | 3,833 s | 92 GB |
| LFM2.5-1.2B-Instruct | 141, 141, 146, 214 | 642 MB | 1,225 MB | 537 MB | 1,438 s | 19 GB |

The chunks are 52-56% of A16W8's. The fp32 embedding table the runner reads from disk is the
same size either way, and for Llama and Qwen it is about as large as all the A16W4 chunks
together.

## Found on the way

`export_mtk.absolute_tool` resolved `--tool-python` with `Path.resolve()`, which follows links. A
virtualenv's `bin/python` links to the base interpreter, and Python finds the venv from the path
it was started by. So the export ran MediaTek's tools on the bare Python 3.10 and refused with
"mtk_converter is None". It now uses `os.path.abspath`, with a test using a symlinked venv
python.
