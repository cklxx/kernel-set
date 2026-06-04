"""Shared definitions for the two kernel-set benchmark harnesses.

``bench.py`` (intra-kernel correctness + roofline) and ``bench_sota.py``
(kernel-set vs SOTA providers) historically each defined their own copies of the
representative LLM shape tables and the NeoX RoPE reference. This module is the
single source of truth for the parts that are *genuinely identical* between the
two harnesses, so adding a new representative shape only touches one place.

What lives here (safe to share — byte-identical in both harnesses today):
  * the LLM shape tables that both harnesses drive over the SAME set of shapes
    (``ATTN_PREFILL_SHAPES``, ``ATTN_DECODE_SHAPES``, ``ROPE_SHAPES``,
    ``CE_SHAPES``), and
  * the NeoX / rotate-half RoPE reference (``ref_rope_neox``).

What is intentionally NOT shared (the harnesses differ on purpose — do not
fold these together, it would change benchmark behavior):
  * ``_NORM_SHAPES`` / ``_SWIGLU_SHAPES`` / ``_GEMM_SHAPES`` / ``_MOE_SHAPES`` —
    bench_sota deliberately drives a *subset* of bench.py's shapes for the
    cross-impl sweep, and carries its own extra tables (MLA, SSM) that bench.py
    does not.
  * the correctness tolerance / gate. bench.py uses tighter *intra-kernel*
    tolerances (kernel vs its own fp32 math reference); bench_sota uses looser
    *cross-impl* tolerances (two different fast kernels can disagree a bit more
    than a kernel vs its own reference). See ``Ctx.tol`` / ``SotaCtx.tol``.

No third-party imports at module scope: this file is import-safe with no torch
so ``--list`` / ``ast.parse`` checks work on a CPU host. ``ref_rope_neox`` uses
``torch`` only when actually called (the harnesses already require torch to run).
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Representative LLM shape tables shared verbatim by both harnesses.
# Each entry is (label, *dims); the label is also the row key in reports.
# --------------------------------------------------------------------------- #

# Attention prefill: (label, batch, seqlen, qheads, kvheads, head_dim)
ATTN_PREFILL_SHAPES = [
    ("b=1,seq=2048,qh=32,kvh=8,hd=128", 1, 2048, 32, 8, 128),
    ("b=4,seq=1024,qh=32,kvh=8,hd=128", 4, 1024, 32, 8, 128),
]

# Attention decode: (label, num_seqs, ctx_len, qheads, kvheads, head_dim, block_size)
ATTN_DECODE_SHAPES = [
    ("seqs=64,ctx=2048,qh=32,kvh=8,hd=128", 64, 2048, 32, 8, 128, 16),
    ("seqs=256,ctx=1024,qh=32,kvh=8,hd=128", 256, 1024, 32, 8, 128, 16),
]

# RoPE: (label, tokens, qheads, kvheads, head_dim)
ROPE_SHAPES = [
    ("tokens=4096,qh=32,kvh=8,hd=128", 4096, 32, 8, 128),
    ("tokens=1,qh=32,kvh=8,hd=128",    1,    32, 8, 128),   # decode
]

# Cross-entropy: (label, num_tokens, vocab)
CE_SHAPES = [
    ("tokens=4096,vocab=32000", 4096, 32000),
    ("tokens=8192,vocab=128256", 8192, 128256),
]


# --------------------------------------------------------------------------- #
# Reference math shared by both harnesses.
# --------------------------------------------------------------------------- #
def ref_rope_neox(x, cos, sin, cast: bool = True):
    """NeoX / rotate-half RoPE reference (fp32 angle math).

    ``x``: (tokens, heads, hd); ``cos``/``sin``: (tokens, hd/2).

    ``cast=True`` returns the result in ``x.dtype`` (bench.py's intra-kernel
    gate compares kernel-output dtype vs a same-dtype reference). ``cast=False``
    returns the raw fp32 result (bench_sota compares fp32 references across
    impls). ``rel_err`` upcasts to fp32 internally either way, so the two paths
    differ only by an intermediate round-trip — preserved exactly as before.
    """
    import torch

    hd = x.shape[-1]
    half = hd // 2
    x1 = x[..., :half].float()
    x2 = x[..., half:].float()
    c = cos.float().unsqueeze(1)   # [tokens,1,hd/2]
    s = sin.float().unsqueeze(1)
    o1 = x1 * c - x2 * s
    o2 = x2 * c + x1 * s
    out = torch.cat([o1, o2], dim=-1)
    return out.to(x.dtype) if cast else out
