"""Gemma 3 (text) for ExecuTorch's ``export_llm``: the block ExecuTorch 1.5.1's transformer lacks.

ExecuTorch 1.5.1 ships Gemma 3 1B's params (examples/models/gemma3/config/1b_config.json) for
Qualcomm's own model definition, but the shared transformer that ``export_llm`` builds
(examples/models/llama/llama_transformer.py, model_args.py) reads few of them:

* ``post_attention_norm`` and ``post_ffn_norm`` are declared and never applied (the block's
  ``post_ffn_norm`` is a different, scale-only norm behind ``use_ffn_learnable_scales``).
  Gemma normalises each sublayer's output before the residual add ("sandwich" norms).
* ``act_fn`` is declared and never applied: ``FeedForward`` hard-codes SiLU, Gemma gates with
  GELU (tanh approximation, HF's ``gelu_pytorch_tanh``).
* ``local_rope_theta`` is declared and never read: one RoPE table serves every layer, while
  Gemma's sliding layers rotate at theta 1e4 and its global layers at 1e6.
* ``embedding_scale_factor`` is declared and never applied. That one is folded into the
  checkpoint instead (pipeline/convert.py scales ``tok_embeddings`` by sqrt(dim) and keeps
  ``output`` unscaled), so the int8 embedding quantization sees the scaled table.

Everything else is upstream's, on purpose, so the KV-cache source transformations reach
Gemma exactly as they reach every other model: attention is ``AttentionMHA`` (its q/k norms
and the block norms take Gemma's ``1 + weight`` through ``rms_norm_add_unit_offset``), and the
sliding window is ``export_llm``'s own ``model.local_global_attention``
(source_transformation/custom_kv_cache.replace_kv_cache_with_ring_kv_cache), which turns each
sliding layer's cache into a ring of 2 x window slots: ``CustomRingKVCache`` for the fp32 and
fp16 caches, ``QuantizedRingKVCache`` for the int8 one. A ring takes at most ``window`` tokens
per update, so Gemma's prefill chunk is the window (families.XnnpackPlan.prefill_chunk).

Without a KV cache (the GPTQ solve) upstream attention is causal over the whole row; the
sliding layers' mask buffer is banded here, so the solve sees the window too.

``apply()`` makes ``construct_transformer`` build this model for Gemma 3 params (GELU and a
``sliding_attention`` layer type) and leaves every other model to upstream. It must run in
any process that builds the model: ``pipeline.export_with_codes --gemma3`` and the solve.
Reference: transformers' models/gemma3/modeling_gemma3.py (Gemma3DecoderLayer, Gemma3MLP).
"""

from __future__ import annotations

import dataclasses

import torch
import torch.nn.functional as F
from torch import nn

LAYER_TYPES = ("sliding_attention", "full_attention")


def is_gemma3(args) -> bool:
    from executorch.examples.models.llama.model_args import ActFn

    return (
        bool(args.layer_types)
        and "sliding_attention" in args.layer_types
        and args.act_fn == ActFn.GELU_APPROX
        and bool(args.post_attention_norm and args.post_ffn_norm)
    )


class GeluFeedForward(nn.Module):
    """Gemma3MLP: down(gelu_tanh(gate(x)) * up(x)), in upstream FeedForward's names (w1 gate,
    w3 up, w2 down) so the converter, the solve and the 8da4w quantizer see the usual linears."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x), approximate="tanh") * self.w3(x))


class Block(nn.Module):
    """Gemma3DecoderLayer on upstream's attention: x + post_attn(attn(norm(x))), then
    h + post_ffn(ffn(norm(h))). A sliding layer rotates with its own (local theta) table."""

    def __init__(self, args, layer_id: int, layer_type: str, global_rope, local_rope):
        super().__init__()
        from executorch.examples.models.llama.attention import AttentionMHA
        from executorch.examples.models.llama.norm import RMSNorm

        self.layer_id = layer_id
        self.sliding = layer_type == "sliding_attention"
        rope = local_rope if self.sliding else global_rope
        # Kept as an attribute (not only inside attention) so forward can read its table.
        self.local_rope = local_rope if self.sliding else None
        self.attention = AttentionMHA(args, layer_id, rope)
        if self.sliding:
            # The window for every path that reads the attention's own mask buffer: no KV cache
            # (the GPTQ solve, whose app-shaped rows run past 512 tokens) and a full-length cache.
            # The rings export_llm installs build their own masks (CustomRingKVCache), and the
            # exported fp32/int8 SDPA reads those. A query sees the window's last ``window``
            # positions, itself included (HF's sliding_window_overlay: kv > q - window).
            n, window = self.attention.mask.shape[0], int(args.sliding_window)
            band = torch.ones(n, n, dtype=torch.bool, device="cpu").tril().triu(-(window - 1))
            self.attention.register_buffer("mask", band, persistent=False)
        self.feed_forward = GeluFeedForward(args.dim, args.hidden_dim)

        def norm():
            return RMSNorm(args.dim, eps=args.norm_eps, add_unit_offset=args.rms_norm_add_unit_offset)

        self.attention_norm = norm()
        self.post_attention_norm = norm()
        self.ffn_norm = norm()
        self.post_ffn_norm = norm()

    def forward(self, x, freqs_cos, freqs_sin, attn_options):
        if self.sliding:
            # Transformer passes the global table's slice; a sliding layer takes the same
            # positions from the local one (upstream Rope.get_freqs, dynamic-shape narrow).
            freqs_cos, freqs_sin = self.local_rope.get_freqs(attn_options.get("input_pos"), x.shape[1])
        attention_output, update = self.attention(self.attention_norm(x), freqs_cos, freqs_sin, **attn_options)
        h = x + self.post_attention_norm(attention_output)
        out = h + self.post_ffn_norm(self.feed_forward(self.ffn_norm(h)))
        return out, update


def construct(args):
    from executorch.examples.models.llama.llama_transformer import Transformer
    from executorch.examples.models.llama.rope import Rope

    unknown = set(args.layer_types) - set(LAYER_TYPES)
    if unknown:
        raise ValueError(f"Gemma 3 layer types {sorted(unknown)} are not implemented")
    if len(args.layer_types) != args.n_layers:
        raise ValueError(f"{len(args.layer_types)} layer types for {args.n_layers} layers")
    if args.local_rope_theta is None:
        raise ValueError("Gemma 3 params need local_rope_theta for the sliding layers")
    global_rope = Rope(args)
    local_rope = Rope(dataclasses.replace(args, rope_theta=float(args.local_rope_theta)))
    layers = nn.ModuleList(
        Block(args, i, layer_type, global_rope, local_rope) for i, layer_type in enumerate(args.layer_types)
    )
    return Transformer(args, layers, global_rope)


_APPLIED = False


def apply() -> None:
    """Route ``construct_transformer`` to Gemma 3 for Gemma 3 params. Idempotent."""
    global _APPLIED
    if _APPLIED:
        return
    from executorch.examples.models.llama import llama_transformer
    from executorch.examples.models.llama import model as llama_model

    upstream = llama_transformer.construct_transformer

    def construct_transformer(model_args):
        return construct(model_args) if is_gemma3(model_args) else upstream(model_args)

    llama_transformer.construct_transformer = construct_transformer
    # model.py imported the name, so its binding is the one Llama2Model calls.
    llama_model.construct_transformer = construct_transformer
    _APPLIED = True
