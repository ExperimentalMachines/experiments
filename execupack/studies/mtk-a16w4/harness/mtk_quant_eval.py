"""What MediaTek's quantization does to a model, measured on the host before any compile.

Builds the chunks the export builds (examples/mediatek's own model classes and loader), puts
NeuropilotQuantizer's observers on each at a precision (prepare_pt2e), calibrates them on what
the export calibrates on and the way it does, converts them (convert_pt2e: the graph MediaTek
lowers, its quantize/dequantize simulated on the CPU), then runs held-out text teacher-forced
through the fp32 chunks and every quantized variant, each on its own cache, in prompt-graph
blocks as the runner feeds them. Per position it records KL(fp32 || q), both NLLs and whether
the top token agrees; and it puts the publish gate's sixteen rows (pipeline/gate.py, the app's
prompt and tool schema) to every variant.

Same harness as docs/research finding 35 (mtk-calibration-and-precision-ab.py), generalised to
qwen.py and llama.py and to several variants against one reference:

- qwen.py / llama.py calibrate as the scripts do (prepare_model_inputs: each alpaca prompt in the
  preformatter, as one step, then up to response_cap greedy tokens). The scripts store every
  step and replay it per chunk; the observers are MinMax, so feeding each chunk its inputs as they
  are computed records the same ranges (the argument lfm2.py's streaming calibration rests on).
- lfm2.py calibrates with its own calibrate_streaming on the corpus the export builds.

A variant is NAME=PRECISION, or NAME=PRECISION:WEIGHTS for a transformed checkpoint
(scripts/mtk_gptq.py). The reference is always --weights in fp32, so a variant that changes the
fp32 model (AWQ's clipping) is charged for it.

Run with MediaTek's tool python from a copy of examples/mediatek (the scripts import from cwd):

    cd <examples/mediatek> && PYTHONPATH=<repo> /opt/mtk-venv/bin/python <repo>/scripts/mtk_quant_eval.py \\
        --script qwen.py --weights W --chunks 4 --window 2048 --dataset aot_utils/llm_utils/prompts/alpaca.txt \\
        --preformatter aot_utils/llm_utils/preformatter_templates/qwen3.json --book pg25773.txt \\
        --variant a16w8=A16W8 --variant a16w4=A16W4 --variant a16w4-gptq=A16W4:W_GPTQ --out result.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import re
import sys
import time
import types


def stub_mtk() -> None:
    """MediaTek's compiler is only needed to lower; without it (any host but the export's), stub
    the modules the scripts import so the quantizer and the model code still load."""
    try:
        import mtk_converter  # noqa: F401
    except ImportError:
        for name in ("mtk_converter", "mtk_converter.python", "mtk_converter.python.converters", "mtk_neuron"):
            sys.modules[name] = types.ModuleType(name)
        pytorch_mod = types.ModuleType("mtk_converter.python.converters.pytorch")
        pytorch_mod.importer_v2 = types.ModuleType("importer_v2")
        sys.modules["mtk_converter.python.converters.pytorch"] = pytorch_mod


def parse_variant(text: str) -> tuple[str, str, str | None]:
    name, _, spec = text.partition("=")
    precision, _, weights = spec.partition(":")
    if not name or precision not in {"A16W16", "A16W8", "A16W4", "A8W8", "A8W4"}:
        raise SystemExit(f"--variant {text!r}: expected NAME=PRECISION[:WEIGHTS]")
    return name, precision, weights or None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--script", required=True, choices=["qwen.py", "llama.py", "lfm2.py"])
    ap.add_argument("--weights", required=True, help="reference checkpoint, with export_mtk's config.json")
    ap.add_argument("--variant", action="append", required=True, help="NAME=PRECISION[:WEIGHTS]")
    ap.add_argument("--chunks", type=int, required=True)
    ap.add_argument("--window", type=int, required=True)
    ap.add_argument("--prompt-tokens", type=int, default=128)
    ap.add_argument("--dataset", required=True, help="alpaca.txt for qwen/llama, the corpus .jsonl for lfm2")
    ap.add_argument("--preformatter")
    ap.add_argument("--response-cap", type=int, default=9)
    ap.add_argument("--book", required=True, help="held-out text (PG19 book 25773 in finding 35)")
    ap.add_argument("--book-tokens", type=int, help="positions to feed; default the window less one")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("THREADS", "16")))
    ap.add_argument("--device", default="cpu", help="cpu (as the export calibrates) or cuda")
    args = ap.parse_args()
    variants = [parse_variant(v) for v in args.variant]

    stub_mtk()
    examples = os.getcwd()
    sys.path.insert(0, examples)
    sys.path.insert(0, os.path.join(examples, "model_export_scripts"))
    import torch

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    script = importlib.import_module(args.script[:-3])
    from aot_utils.llm_utils.preformatter import Preformatter
    from aot_utils.llm_utils.utils import (
        generate_mask,
        get_embedding_layer,
        get_master_rot_emb,
        load_checkpoints,
        resolve_model_classes,
    )
    from datasets import load_dataset
    from executorch.backends.mediatek import NeuropilotQuantizer, Precision
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    lfm = args.script == "lfm2.py"
    n = args.chunks
    window, ntok = args.window, args.prompt_tokens

    def build(weights: str):
        """The export's chunks, in its order: embedding first, then each chunk pops its layers."""
        config, weight_dir, tokenizer_class, chunk_class = resolve_model_classes(os.path.join(weights, "config.json"))
        nbpc = [(config.num_hidden_layers // n) + (i < (config.num_hidden_layers % n)) for i in range(n)]
        state_dict = load_checkpoints(weight_dir)
        embedding = get_embedding_layer(config, weight_dir, state_dict)
        models = []
        for i, blocks in enumerate(nbpc):
            chunk = chunk_class(
                config, blocks, chunk_idx=i, dtype=torch.float32, include_tail=(i == n - 1), jit_trace=True
            )
            models.append(chunk.load_weights(state_dict, sum(nbpc[:i])))
        del state_dict
        return config, weight_dir, tokenizer_class, models, embedding, nbpc

    config, weight_dir, tokenizer_class, ref_models, embedding, nbpc = build(args.weights)
    dev = torch.device(args.device)
    master_rot_emb = get_master_rot_emb(config, dtype=torch.float32).to(dev)
    head_dim = int(config.head_dim)
    state_rows = config.conv_L_cache - 1 if lfm else 0

    # The scripts' own tokenizer for calibration (qwen.py/llama.py: from_pretrained; lfm2.py falls
    # back to the fast tokenizer file), and transformers' for the chat template of the eval rows.
    if lfm:
        try:
            cal_tokenizer = tokenizer_class.from_pretrained(weight_dir)
        except Exception:
            from transformers import PreTrainedTokenizerFast

            cal_tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=os.path.join(weight_dir, "tokenizer.json"),
                bos_token_id=config.bos_token_id,
                eos_token_id=config.eos_token_id,
                pad_token_id=config.pad_token_id,
            )
        if getattr(cal_tokenizer, "eos_token_id", None) is None:
            cal_tokenizer.eos_token_id = config.eos_token_id
    else:
        cal_tokenizer = tokenizer_class.from_pretrained(weight_dir)
    from transformers import AutoTokenizer

    chat = AutoTokenizer.from_pretrained(args.weights)
    preformatter = Preformatter(args.preformatter) if args.preformatter else None
    eos = torch.tensor(cal_tokenizer.eos_token_id)

    def fresh_cache(models):
        if lfm:
            return [[s.to(dev) for s in chunk] for chunk in script.reset_cache(models, window)]
        return [c.to(dev) for c in script.reset_cache(n, config.num_key_value_heads, nbpc, head_dim, window)]

    def run(graphs, hidden, cache, pos, observers=None):
        """One block through every chunk at position ``pos``. With ``observers`` (the prepared
        graphs), each chunk's observers see its inputs before the fp32 chunk runs."""
        length = hidden.shape[1]
        mask = generate_mask(window, pos, length, length).to(dev)
        pos_emb = master_rot_emb[:, :, pos : pos + length, :]
        conv = tuple(c.to(dev) for c in script.conv_inputs(length, 0, 0, state_rows)) if lfm else ()
        with torch.no_grad():
            for k, graph in enumerate(graphs):
                if lfm:
                    inputs = (hidden, mask, pos_emb, *conv, *cache[k])
                else:
                    inputs = (hidden, mask, pos_emb, *torch.split(cache[k], 1, dim=0))
                if observers is not None:
                    observers[k](*inputs)
                outs = graph(*inputs)
                hidden = outs[0]
                if lfm:
                    cache[k] = [s.clone() for s in outs[1:]]
                else:
                    cache[k] = torch.cat(outs[1 : 1 + 2 * nbpc[k]], dim=0).clone()
        return hidden, cache

    def replies(models, emb, prepared, cache, logits, seq):
        """MediaTek's reply loop: greedy tokens until EOS or response_cap, each observed."""
        nxt = torch.argmax(logits[:, -1, :], dim=-1)
        count = 0
        while True:
            logits, cache = run(models, emb(nxt[:, None].to(torch.int32)), cache, seq, prepared)
            nxt = torch.argmax(logits[:, -1, :], dim=-1)
            if nxt == eos:
                break
            count += 1
            seq += 1
            if count == args.response_cap:
                break

    def calibrate(models, emb, prepared):
        if lfm:
            # lfm2.py calibrate_streaming, step for step: a sample in prompt-graph blocks at
            # advancing positions, then the replies.
            samples = script.load_samples(args.dataset, cal_tokenizer, preformatter)
            for input_ids in samples:
                if len(input_ids) + args.response_cap >= window:
                    raise SystemExit(f"a {len(input_ids)}-token sample does not fit a {window}-token cache")
                cache = fresh_cache(models)
                seq = 0
                for start in range(0, len(input_ids), ntok):
                    block = torch.tensor([input_ids[start : start + ntok]], dtype=torch.int32, device=dev)
                    logits, cache = run(models, emb(block), cache, seq, prepared)
                    seq += block.shape[1]
                replies(models, emb, prepared, cache, logits, seq)
            return len(samples)
        rows = load_dataset("text", data_files=args.dataset, split="train")
        for row in rows:
            text = preformatter.generate_prompt(row["text"]) if preformatter else row["text"]
            ids = cal_tokenizer(text, return_tensors="pt")["input_ids"].to(torch.int32).to(dev)
            # qwen.py/llama.py prepare_model_inputs: the prompt in one step, then the replies.
            cache = fresh_cache(models)
            logits, cache = run(models, emb(ids), cache, 0, prepared)
            replies(models, emb, prepared, cache, logits, ids.shape[1])
        return len(rows)

    def retarget(graph):
        """The graph on ``dev``: torch.export records ops' device arguments (the CPU it traced on),
        so they are rewritten as well as the parameters moved."""
        if dev.type != "cpu":
            for node in graph.graph.nodes:
                if "device" in node.kwargs:
                    node.update_kwarg("device", dev)
            graph.recompile()
        return graph.to(dev)

    def prepare(models, precision):
        """Observers on every chunk, as the scripts put them (exported on the CPU, where the
        example inputs are)."""
        prepared = []
        for model in models:
            if lfm:
                prepared.append(script.prepare_chunk(model, precision, ntok, window)[0])
                continue
            example_inputs, dynamic_shapes = model.get_example_inputs(ntok, window, True)
            graph = torch.export.export(model, example_inputs, dynamic_shapes=dynamic_shapes, strict=True).module()
            quantizer = NeuropilotQuantizer()
            quantizer.setup_precision(getattr(Precision, precision))
            prepared.append(prepare_pt2e(graph, quantizer))
        return prepared

    def to_dev(model):
        """An eager chunk on ``dev``; MediaTek's chunks also move their inputs to device_list."""
        model.to(dev)
        if hasattr(model, "device_list"):
            model.device_list = [str(dev)] * len(model.device_list)
        return model

    staged = []
    for name, precision, weights in variants:
        if weights:
            _, _, _, models, emb, _ = build(weights)
        else:
            models, emb = ref_models, embedding
        staged.append((name, precision, weights, models, emb, prepare(models, precision)))
    for model in ref_models:
        to_dev(model)
    embedding.to(dev)
    quantized = {}
    for name, precision, weights, models, emb, prepared in staged:
        for model in models:
            to_dev(model)
        emb.to(dev)
        prepared = [retarget(g) for g in prepared]
        started = time.time()
        count = calibrate(models, emb, prepared)
        seconds = time.time() - started
        graphs = [retarget(convert_pt2e(g, fold_quantize=False)) for g in prepared]
        quantized[name] = {
            "graphs": graphs,
            "embedding": emb,
            "precision": precision,
            "weights": weights,
            "calibration_samples": count,
            "calibration_s": round(seconds, 1),
        }
        print(f"{name}: {precision} calibrated on {count} samples in {seconds:.0f} s", flush=True)
        del models, prepared
    del staged

    def feed(ids, every_position: bool):
        """Teacher-force ``ids`` through the reference and every variant; yields per block the
        reference's and each variant's log-probabilities (all positions, or the last only)."""
        caches = {"ref": fresh_cache(ref_models), **{v: fresh_cache(ref_models) for v in quantized}}
        pos = 0
        for start in range(0, len(ids), ntok):
            block = torch.tensor([ids[start : start + ntok]], dtype=torch.int32, device=dev)
            out = {}
            hidden, caches["ref"] = run(ref_models, embedding(block), caches["ref"], pos)
            out["ref"] = hidden
            for v, q in quantized.items():
                hidden, caches[v] = run(q["graphs"], q["embedding"](block), caches[v], pos)
                out[v] = hidden
            pos += block.shape[1]
            yield (
                start,
                {k: torch.log_softmax((x[0] if every_position else x[0, -1:]).float(), -1) for k, x in out.items()},
            )

    # Held-out text: a chat-template user turn holding a book passage (finding 35's harness), so
    # the positions are what a long prompt occupies on the phone.
    text = open(args.book, encoding="utf-8").read().replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    marker = "\x00passage\x00"
    head = chat.apply_chat_template(
        [{"role": "user", "content": f"Here is a passage from a book.\n\n{marker}"}], tokenize=False
    ).split(marker)[0]
    body = chat(text[len(text) // 3 :], add_special_tokens=False)["input_ids"]
    ids = (chat(head, add_special_tokens=False)["input_ids"] + body)[: args.book_tokens or window - 1]
    per = {v: {"kl": [], "nll_ref": [], "nll_q": [], "agree": []} for v in quantized}
    started = time.time()
    for start, lp in feed(ids, every_position=True):
        targets = ids[start + 1 : start + lp["ref"].shape[0] + 1]
        m = len(targets)  # the last block's last position has no next token
        target = torch.tensor(targets, device=lp["ref"].device).unsqueeze(1)
        ref = lp["ref"][:m]
        ref_top = ref.argmax(-1)
        for v in quantized:
            q = lp[v][:m]
            per[v]["kl"] += (ref.exp() * (ref - q)).sum(-1).tolist()
            per[v]["nll_ref"] += (-ref.gather(1, target)[:, 0]).tolist()
            per[v]["nll_q"] += (-q.gather(1, target)[:, 0]).tolist()
            per[v]["agree"] += (q.argmax(-1) == ref_top).int().tolist()
    book_s = time.time() - started

    def bucketed(values):
        edges = [0, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
        out = {}
        for lo, hi in zip(edges, edges[1:], strict=False):
            part = values[lo:hi]
            if part:
                out[f"{lo}-{min(hi, len(values))}"] = round(sum(part) / len(part), 5)
        return out

    result = {
        "script": args.script,
        "weights": os.path.abspath(args.weights),
        "window": window,
        "chunks": n,
        "prompt_tokens": ntok,
        "device": args.device,
        "book": {
            "file": os.path.basename(args.book),
            "positions": len(per[variants[0][0]]["kl"]),
            "seconds": round(book_s),
        },
        "variants": {},
        "per_position": per,
    }
    for v, q in quantized.items():
        p = per[v]
        m = len(p["kl"])
        result["variants"][v] = {
            "precision": q["precision"],
            "weights": q["weights"],
            "calibration_samples": q["calibration_samples"],
            "calibration_s": q["calibration_s"],
            "kl": round(sum(p["kl"]) / m, 5),
            "ppl_ref": round(math.exp(sum(p["nll_ref"]) / m), 3),
            "ppl_q": round(math.exp(sum(p["nll_q"]) / m), 3),
            "top1_agree": round(sum(p["agree"]) / m, 4),
            "kl_by_position": bucketed(p["kl"]),
        }

    if not args.no_gate:
        sys.path.insert(0, os.environ.get("EXECUPACK", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from pipeline import gate

        ref_rows = {}
        measured = {v: {} for v in quantized}
        first_kl = {v: [] for v in quantized}
        for question, prompt in gate.prompts(chat).items():
            row_ids = chat(prompt, add_special_tokens=False)["input_ids"]
            if len(row_ids) >= window:
                raise SystemExit(f"gate row of {len(row_ids)} tokens does not fit a {window}-token cache")
            for _, lp in feed(row_ids, every_position=False):
                last = lp
            ref = last["ref"][0]
            token = int(ref.argmax())
            ref_rows[question] = {"token": token, "p": float(ref[token].exp()), "text": chat.decode([token])}
            for v in quantized:
                q = last[v][0]
                measured[v][question] = {
                    "p": float(q[token].exp()),
                    "top": int(q.argmax()),
                    "text": chat.decode([int(q.argmax())]),
                }
                first_kl[v].append(float((ref.exp() * (ref - q)).sum()))
        reference = {"rows": ref_rows, "confident": sum(1 for r in ref_rows.values() if r["p"] > gate.CONFIDENT)}
        result["gate_reference"] = reference
        for v in quantized:
            verdict = gate.check(reference, measured[v])
            verdict["first_token_kl"] = round(sum(first_kl[v]) / len(first_kl[v]), 5)
            verdict["measured"] = measured[v]
            result["variants"][v]["gate"] = verdict

    with open(args.out, "w") as f:
        json.dump(result, f)
    for v, r in result["variants"].items():
        g = r.get("gate", {})
        print(
            f"{v:>14}  KL {r['kl']:.4f}  ppl {r['ppl_q']:.2f} (fp32 {r['ppl_ref']:.2f})  top1 {r['top1_agree']:.3f}  "
            f"gate {'PASS' if g.get('passed') else 'FAIL'} {g.get('agreed')}/{g.get('graded')} "
            f"mean {g.get('export_mean')} (fp32 {g.get('fp32_mean')}) first-token KL {g.get('first_token_kl')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
