"""Does the published Qwen3-4B-Thinking-2507 CPU file still call tools after it reasons?

    python think_check.py <execupack checkout> <work dir> [max_new_tokens]

The one-token gate cannot see this model's decision (it opens every reply with "Okay"), so
this generates through the reasoning instead: the gate's 16 questions in the app's prompt
with thinking on (the app turns it on whenever tools are offered, ChatViewModel), greedy,
through the same TextLLMRunner the smoke test uses, and counts the replies that reach a
<tool_call> on the 12 search rows against the 4 quiet ones.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
work = Path(sys.argv[2])
max_new = int(sys.argv[3]) if len(sys.argv) > 3 else 600
from huggingface_hub import hf_hub_download  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from executorch.extension.pybindings import portable_lib  # noqa: E402,F401
from executorch.extension.llm.custom_ops import custom_ops  # noqa: E402,F401
from executorch.kernels import quantized  # noqa: E402,F401
from executorch.extension.llm.runner import GenerationConfig, TextLLMRunner  # noqa: E402
from pipeline import calibration, gate  # noqa: E402

REPO, SOURCE = "experimentalmachines/Qwen3-4B-Thinking-2507-ExecuTorch", "Qwen/Qwen3-4B-Thinking-2507"
pte = hf_hub_download(REPO, "xnnpack/Qwen3-4B-Thinking-2507-8da4w-gptq-4k.pte", local_dir=work)
tok_file = hf_hub_download(REPO, "tokenizer.json", local_dir=work)
tok = AutoTokenizer.from_pretrained(SOURCE)
rows = []
for q, subject in gate.SEARCH:
    rows.append(("search", q, calibration.render_app(tok, f"{q}\n\n{calibration.TRAILER.format(subject=subject)}")))
rows += [("quiet", q, calibration.render_app(tok, q)) for q in gate.QUIET]
results = []
for kind, q, prompt in rows:
    pieces = []
    runner = TextLLMRunner(pte, tok_file)  # a fresh KV cache per question
    runner.generate(prompt, GenerationConfig(echo=False, max_new_tokens=max_new, temperature=0.0, num_bos=0, num_eos=0),
                    token_callback=pieces.append)
    text = "".join(pieces)
    called = "<tool_call>" in text
    closed = "</think>" in text
    results.append({"kind": kind, "q": q, "called": called, "think_closed": closed, "chars": len(text),
                    "tail": text[-160:]})
    print(kind, "CALL" if called else ("answer" if closed else "still thinking"), q, flush=True)
summary = {k: {"rows": sum(r["kind"] == k for r in results), "called": sum(r["called"] for r in results if r["kind"] == k),
               "still_thinking": sum(not r["think_closed"] for r in results if r["kind"] == k)}
           for k in ("search", "quiet")}
(work / "think_check.json").write_text(json.dumps({"summary": summary, "rows": results}, indent=1))
print(json.dumps(summary))
