"""GPTQ onto MediaTek's A16W4 grid, written back as a checkpoint MediaTek's export reads unchanged.

What A16W4 does to a weight (backends/mediatek/quantizer/qconfig.py, get_a16w4_quant_config,
which NeuropilotQuantizer calls with is_per_channel=True): one scale and zero point per output
row, across the whole input dimension (no groups), asymmetric over [-8, 7], from
PerChannelMinMaxObserver (eps 1e-6), then round to nearest (torchao's _calculate_qparams for
per_channel_affine, quantize_per_channel). ``mtk_w4`` is that, in torchao's arithmetic.

The solve fixes each row's grid from the row as it is (its min and max, as the observer will
measure them), runs GPTQ on that grid (pipeline/solve.py's: blocks of 128, 1% damping, Hessian
over the same calibration rows with the app head counted once, layer by layer on the model whose
earlier layers are already quantized), and writes the dequantized weights in fp32. For MediaTek's
observer to measure the same grid from those weights, each row must still reach both ends of
it: the column holding the row's minimum is pinned to code -8 and the one holding its maximum to
7, and GPTQ carries the error of the pin into the columns after it like any other rounding.
The observer then recovers the solve's scale and zero point and rounding returns the solve's
codes exactly; ``check`` measures that on every linear, so the export quantizes to what was solved
without the quantizer, the export or the runner knowing GPTQ ran.

The output head is solved too (A16W4 quantizes it per row of the vocabulary). A tied head is
written untied, and the config says so, so the token embedding the runner reads from disk stays
the original.

Calibration: pipeline/calibration.py's rows (the app's prompt and tool schema, plus general
prompts), each continued by the fp32 model's own greedy reply: the set the XNNPACK GPTQ solve
uses. Activations are 16-bit on MediaTek, so unlike solve.py none are fake-quantized here.

    PYTHONPATH=<repo> /opt/awq/bin/python scripts/mtk_gptq.py --hf <hf download> \\
        --mtk-weights <export's weights dir> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from torch import nn

QMIN, QMAX, EPS = -8, 7, 1e-6
DAMP = 0.01
BLOCK = 128


def mtk_w4(w: torch.Tensor) -> torch.Tensor:
    """``w`` (rows x inputs) through A16W4's weight quantizer and back."""
    mn = w.amin(1, keepdim=True).clamp(max=0)
    mx = w.amax(1, keepdim=True).clamp(min=0)
    scale = torch.clamp((mx - mn) / float(QMAX - QMIN), min=EPS)
    zp = torch.clamp(QMIN - torch.round(mn / scale), QMIN, QMAX)
    q = torch.clamp(torch.round(w * (1.0 / scale)) + zp, QMIN, QMAX)
    return (q - zp) * scale


@torch.no_grad()
def solve_weight(weight: torch.Tensor, hessian: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """GPTQ for one linear on A16W4's per-row grid; returns the dequantized weight."""
    w = weight.clone().float()
    out_f, in_f = w.shape
    h = hessian.double().clone()
    dead = torch.diag(h) == 0
    h[dead, dead] = 1
    w[:, dead] = 0
    h += torch.eye(in_f, dtype=h.dtype, device=h.device) * (DAMP * torch.diag(h).mean())
    chol = torch.linalg.cholesky(h)
    hinv = torch.linalg.cholesky(torch.cholesky_inverse(chol), upper=True).float()

    mn = w.amin(1).clamp(max=0)
    mx = w.amax(1).clamp(min=0)
    scale = torch.clamp((mx - mn) / float(QMAX - QMIN), min=EPS)
    inv = 1.0 / scale
    zp = torch.clamp(QMIN - torch.round(mn / scale), QMIN, QMAX)
    lo, hi = w.argmin(1), w.argmax(1)
    pin_lo, pin_hi = zp > QMIN, zp < QMAX  # a clamped zero point sits at 0, which the observer clamps to anyway
    codes = torch.zeros_like(w)
    for b0 in range(0, in_f, BLOCK):
        b1 = min(b0 + BLOCK, in_f)
        wb = w[:, b0:b1].clone()
        eb = torch.zeros_like(wb)
        hb = hinv[b0:b1, b0:b1]
        for i in range(b1 - b0):
            col = b0 + i
            c = wb[:, i]
            q = torch.clamp(torch.round(c * inv) + zp, QMIN, QMAX)
            q = torch.where(pin_lo & (lo == col), torch.full_like(q, QMIN), q)
            q = torch.where(pin_hi & (hi == col), torch.full_like(q, QMAX), q)
            codes[:, col] = q
            err = (c - (q - zp) * scale) / hb[i, i]
            wb[:, i:] -= err.unsqueeze(1) * hb[i, i:].unsqueeze(0)
            eb[:, i] = err
        w[:, b1:] -= eb @ hinv[b0:b1, b1:]
    deq = (codes - zp.unsqueeze(1)) * scale.unsqueeze(1)
    return deq, check(deq, codes)


def check(deq: torch.Tensor, codes: torch.Tensor) -> dict:
    """Whether MediaTek's observer, measuring ``deq``, quantizes it back to ``codes``."""
    mn = deq.amin(1, keepdim=True).clamp(max=0)
    mx = deq.amax(1, keepdim=True).clamp(min=0)
    scale = torch.clamp((mx - mn) / float(QMAX - QMIN), min=EPS)
    zp = torch.clamp(QMIN - torch.round(mn / scale), QMIN, QMAX)
    back = torch.clamp(torch.round(deq * (1.0 / scale)) + zp, QMIN, QMAX)
    return {"mismatched": int((back != codes).sum()), "elements": codes.numel()}


def hessian_error(weight, deq, hessian) -> float:
    """tr((W - Q) H (W - Q)^T) per row: the output error the solve minimises."""
    d = (weight - deq).double()
    return float(((d @ hessian.double()) * d).sum() / weight.shape[0])


class Stop(Exception):
    pass


def hessians(model, rows, linears: dict, stop_at) -> tuple[dict, dict]:
    """X^T X / n per linear over every row's positions from its ``keep_from``."""
    hess = {n: torch.zeros(m.in_features, m.in_features, device="cuda") for n, m in linears.items()}
    counts = dict.fromkeys(linears, 0)
    keep = {"from": 0}
    hooks = []
    for name, module in linears.items():

        def pre(_, args, name=name):
            x = args[0][0, keep["from"] :].float()
            hess[name] += x.T @ x
            counts[name] += x.shape[0]

        hooks.append(module.register_forward_pre_hook(pre))

    def halt(*_):
        raise Stop

    if stop_at is not None:
        hooks.append(stop_at.register_forward_hook(halt))
    try:
        with torch.no_grad():
            for ids, keep_from in rows:
                keep["from"] = keep_from
                try:
                    model(torch.tensor([ids], device="cuda"), use_cache=False)
                except Stop:
                    pass
    finally:
        for h in hooks:
            h.remove()
    return {n: hess[n] / max(counts[n], 1) for n in hess}, counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hf", required=True, help="the Hugging Face download, as the model ships")
    ap.add_argument("--mtk-weights", required=True, help="the export's weights dir (its config.json, tokenizer)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-head", action="store_true", help="leave the output head to plain rounding")
    ap.add_argument("--rows", help="reuse calibration rows from a rows.json")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from pipeline import calibration

    started = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.hf)
    model = AutoModelForCausalLM.from_pretrained(args.hf, dtype=torch.float32, local_files_only=True).cuda().eval()
    eos = model.config.eos_token_id
    pad = eos[0] if isinstance(eos, list | tuple) else eos

    def generate(ids, max_new_tokens):
        with torch.no_grad():
            o = model.generate(
                torch.tensor([ids], device="cuda"), max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad
            )
        return o[0, len(ids) :].tolist()

    if args.rows:
        rows = [tuple(r) for r in json.loads(Path(args.rows).read_text())]
    else:
        rows = calibration.sequences(generate, tokenizer)
    (out / "rows.json").write_text(json.dumps(rows))
    tokens = sum(len(ids) - k for ids, k in rows)
    print(f"{len(rows)} calibration rows, {tokens} counted positions ({time.time() - started:.0f} s)", flush=True)

    head = not args.no_head
    tied = model.lm_head.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()
    if head and tied:
        model.lm_head.weight = nn.Parameter(model.lm_head.weight.detach().clone())
        model.config.tie_word_embeddings = False

    log = {"rows": len(rows), "tokens": tokens, "linears": {}}
    mismatched = 0
    for i, layer in enumerate(model.model.layers):
        linears = {n: m for n, m in layer.named_modules() if isinstance(m, nn.Linear)}
        hess, _ = hessians(model, rows, linears, layer)
        for n, m in linears.items():
            rtn = hessian_error(m.weight.data, mtk_w4(m.weight.data), hess[n])
            deq, verdict = solve_weight(m.weight.data, hess[n])
            gptq = hessian_error(m.weight.data, deq, hess[n])
            m.weight.data = deq
            mismatched += verdict["mismatched"]
            log["linears"][f"layers.{i}.{n}"] = {"rtn": rtn, "gptq": gptq, **verdict}
        del hess
        torch.cuda.empty_cache()
        ratios = [v["gptq"] / max(v["rtn"], 1e-30) for k, v in log["linears"].items() if k.startswith(f"layers.{i}.")]
        print(
            f"layer {i}: GPTQ/RTN output error {min(ratios):.2f}-{max(ratios):.2f}, "
            f"re-quantization mismatches {mismatched} ({time.time() - started:.0f} s)",
            flush=True,
        )
    if head:
        hess, _ = hessians(model, rows, {"lm_head": model.lm_head}, None)
        h = hess["lm_head"]
        rtn = hessian_error(model.lm_head.weight.data, mtk_w4(model.lm_head.weight.data), h)
        deq, verdict = solve_weight(model.lm_head.weight.data, h)
        gptq = hessian_error(model.lm_head.weight.data, deq, h)
        model.lm_head.weight.data = deq
        mismatched += verdict["mismatched"]
        log["linears"]["lm_head"] = {"rtn": rtn, "gptq": gptq, **verdict}
        print(f"head: GPTQ/RTN output error {gptq / rtn:.2f}, mismatches {verdict['mismatched']}", flush=True)
    log["mismatched"] = mismatched
    if mismatched:
        raise SystemExit(f"{mismatched} weights would not quantize back to their solved codes")

    from safetensors.torch import save_file

    state = {}
    for k, v in model.state_dict().items():
        if k == "lm_head.weight" and tied and not head:
            continue
        state[k] = v.detach().to("cpu", torch.float32).contiguous().clone()
    if head and tied and model.config.model_type == "lfm2":
        # modeling_lfm2.py (mediatek-lfm2.patch) reads an untied head as f"{prefix}lm_head.weight",
        # prefix "model."; modeling_common.py (Qwen, Llama) reads "lm_head.weight".
        state["model.lm_head.weight"] = state.pop("lm_head.weight")
    src = Path(args.mtk_weights)
    for f in src.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and f.name != "model.safetensors.index.json":
            shutil.copy2(f, out / f.name)
    save_file(state, str(out / "model.safetensors"), metadata={"format": "pt"})
    overrides = {"tie_word_embeddings": False, "tie_embedding": False} if head and tied else {}
    if overrides:
        cfg = json.loads((out / "config.json").read_text())
        cfg.update(overrides)
        (out / "config.json").write_text(json.dumps(cfg, indent=2))
    (out / "config_overrides.json").write_text(json.dumps(overrides))
    log["seconds"] = round(time.time() - started)
    log["head"] = head
    (out / "gptq.json").write_text(json.dumps(log, indent=1))
    print(f"wrote {out} in {log['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
