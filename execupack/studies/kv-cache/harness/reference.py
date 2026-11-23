"""KV study data: PG-19 token streams with needles, and the fp32 Hugging Face reference the phone is scored against.

    modal run scripts/kvstudy/reference.py [--models llama,lfm,qwen,gemma]

For each model and each of the three PG-19 test books (work/kvstudy/pg19, chosen by a fixed rule: test split, at
least 300,000 bytes, first three by file name), on one GPU in fp32 (TF32 off):

- the stream: the model's chat template around RULER's niah instruction and the book, 32,500 tokens, with nine
  needle sentences ("One of the special magic numbers for KEY is: NUMBER.") spliced in at sentence ends near
  10/50/90% of each window's length (1,900, 7,900 and 32,500 tokens for the 2k, 8k and 32k files). Keys and numbers
  are the same for every model; positions differ only by tokenizer.
- the reference: for the 128 targets ending at 1,900, 7,900 and 32,500, the fp32 model's top-64 log-probabilities,
  the true token's log-probability and its top-1 (one causal pass over the stream).
- the needle baseline: for each window, the fp32 model's greedy answer (10 tokens) to every needle that window
  holds, asked after the stream is cut at the window's length.

Writes work/kvstudy/data/<model>/book<id>.{stream.i32,ref.bin,json}; the phone harness reads them as they are.
ref.bin, per scored target in order: int32 t, int32 target id, float32 target logp, int32[64] top ids,
float32[64] top logps (little-endian).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import modal

BOOKS = ("10321", "10356", "10762")
STREAM = 32500
CHECKPOINTS = (1900, 7900, 32500)  # each window's stream length: 2k, 8k and 32k files
WINDOWS = {"2048": 1900, "8192": 7900, "32768": 32500}
SCORED = 128
TOP = 64
ANSWER_TOKENS = 10
MODELS = {
    "llama": ("meta-llama/Llama-3.2-1B-Instruct", "9213176726f574b556790deb65791e0c5aa438b6"),
    "lfm": ("LiquidAI/LFM2.5-1.2B-Instruct", "0f604ada3f766f9f257460c4c9f0b5d6f69d431b"),
    "qwen": ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
    "gemma": ("google/gemma-3-1b-it", "dcc83ea841ab6100d6b47a070329e1ba4cf78752"),
}
# RULER's niah template (NVIDIA/RULER scripts/data/synthetic/constants.py), single-key needles.
INTRO = (
    "Some special magic numbers are hidden within the following text. Make sure to memorize it. "
    "I will quiz you about the numbers afterwards.\n"
)
NEEDLE = " One of the special magic numbers for {key} is: {number}."
QUESTION = "\nWhat is the special magic number for {key} mentioned in the provided text?"
ANSWER_PREFIX = " The special magic number for {key} mentioned in the provided text is"
KEYS = (
    "amber-lantern", "quiet-harbor", "silver-orchard", "copper-meadow", "velvet-canyon",
    "marble-falcon", "crimson-willow", "hollow-summit", "golden-thistle",
)
TARGETS = sorted({round(f * n) for n in CHECKPOINTS for f in (0.1, 0.5, 0.9)})

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.14.0", "transformers==5.22.1", "tokenizers==0.23.3", "huggingface_hub==1.31.0", "safetensors==0.8.0",
    "numpy",
)
app = modal.App("execupack-kvstudy-reference", image=image)


@app.function(gpu="H100", timeout=3 * 3600, memory=65536)
def build(key: str, books: dict[str, str], hf_token: str) -> dict[str, bytes]:
    import os
    import struct

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ["HF_TOKEN"] = hf_token
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model_id, revision = MODELS[key]
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, dtype=torch.float32, attn_implementation="sdpa"
    ).to("cuda").eval()

    sentinel = "⁣SENTINEL⁣"
    # date_string pins Llama 3's template, which otherwise writes today's date into the system turn; the other
    # templates ignore it. enable_thinking=False is Qwen3's no-thinking turn, as the app sends it.
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": sentinel}], tokenize=False, add_generation_prompt=True, enable_thinking=False,
        date_string="26 Jul 2024",
    )
    head_s, tail_s = rendered.split(sentinel)
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
    head = enc(head_s) + enc(INTRO)
    softcap = getattr(model.config, "final_logit_softcapping", None) or getattr(
        getattr(model.config, "text_config", None), "final_logit_softcapping", None
    )
    out: dict[str, bytes] = {}
    for book_id, text in books.items():
        rng = random.Random(int(book_id))
        numbers = [str(rng.randint(1_000_000, 9_999_999)) for _ in KEYS]
        body = enc(text.lstrip())
        pieces = [tok.decode([t]) for t in body[: STREAM + 4000]]
        # Sentence ends only: PG-19 keeps Gutenberg's hard line wraps, so a newline is often mid-sentence.
        boundary = [p.rstrip().endswith((".", '."', ".'", "!", "?", '?"', '!"')) for p in pieces]
        stream, needles, j = list(head), [], 0
        for n, target in enumerate(TARGETS):
            want = target - len(stream)  # body tokens still to copy before this needle
            k = j + max(want, 0)
            while k < len(boundary) - 1 and not boundary[k]:
                k += 1
            stream += body[j : k + 1]
            j = k + 1
            ids = enc(NEEDLE.format(key=KEYS[n], number=numbers[n]))
            needles.append({"key": KEYS[n], "number": numbers[n], "pos": len(stream), "end": len(stream) + len(ids)})
            stream += ids
        stream += body[j:]
        stream = stream[:STREAM]
        assert len(stream) == STREAM and all(nd["end"] < STREAM for nd in needles)

        # Reference: one causal pass, logits only at the scored targets' previous positions.
        x = torch.tensor([stream], device="cuda")
        targets = [t for d in CHECKPOINTS for t in range(d - SCORED, d)]
        with torch.no_grad():
            hidden = model.model(input_ids=x).last_hidden_state[0]
            logits = model.lm_head(hidden[[t - 1 for t in targets]]).double()
            if softcap:
                logits = torch.tanh(logits / softcap) * softcap
            logp = torch.log_softmax(logits, -1)
        del hidden
        top_v, top_i = torch.topk(logp, TOP, dim=-1)
        ref = bytearray()
        for row, t in enumerate(targets):
            ref += struct.pack("<iif", t, stream[t], float(logp[row, stream[t]]))
            ref += struct.pack(f"<{TOP}i", *top_i[row].tolist()) + struct.pack(f"<{TOP}f", *top_v[row].float().tolist())

        # Needle baseline per window.
        questions = {}
        for window, length in WINDOWS.items():
            asked = []
            for nd in needles:
                if nd["end"] > length - 4:
                    continue
                tail = enc(QUESTION.format(key=nd["key"])) + enc(tail_s) + enc(ANSWER_PREFIX.format(key=nd["key"]))
                prompt = torch.tensor([stream[:length] + tail], device="cuda")
                with torch.no_grad():
                    gen = model.generate(
                        prompt, max_new_tokens=ANSWER_TOKENS, do_sample=False, pad_token_id=tok.eos_token_id
                    )[0, prompt.shape[1] :].tolist()
                answer = tok.decode(gen, skip_special_tokens=True)
                asked.append({**nd, "tail": tail, "hf_ids": gen, "hf_answer": answer,
                              "hf_correct": nd["number"] in answer})
            questions[window] = asked
        meta = {
            "model": model_id, "revision": revision, "book": book_id, "len": STREAM, "head_len": len(head),
            "checkpoints": list(CHECKPOINTS), "scored": SCORED, "top": TOP, "needles": needles,
            "questions": questions, "answer_tokens": ANSWER_TOKENS,
            "hf_nll": {str(d): float(-sum(logp[i, stream[t]] for i, t in enumerate(targets) if d - SCORED <= t < d)
                                     / SCORED) for d in CHECKPOINTS},
        }
        out[f"book{book_id}.stream.i32"] = struct.pack(f"<{STREAM}i", *stream)
        out[f"book{book_id}.ref.bin"] = bytes(ref)
        out[f"book{book_id}.json"] = json.dumps(meta).encode()
        print(key, book_id, "needles", [q["hf_correct"] for w in questions for q in questions[w]], flush=True)
    return out


@app.local_entrypoint()
def main(models: str = "llama,lfm,qwen,gemma"):
    # Resolved here, on the Mac: inside the container this file is /root/reference.py.
    WORK = Path(__file__).resolve().parents[2] / "work" / "kvstudy"  # noqa: N806
    token = (Path.home() / ".cache" / "huggingface" / "token").read_text().strip()
    books = {b: (WORK / "pg19" / f"{b}.txt").read_text(encoding="utf-8") for b in BOOKS}
    keys = models.split(",")
    for key, files in zip(keys, build.map(keys, kwargs={"books": books, "hf_token": token})):
        d = WORK / "data" / key
        d.mkdir(parents=True, exist_ok=True)
        for name, blob in files.items():
            (d / name).write_bytes(blob)
        print("wrote", d, sorted(files))
