#!/usr/bin/env python3
"""Generator for the kernel-set model registry.

This is the single source of truth for the model catalog. Running it emits two
mirror files next to it:

  * registry.json  — machine-readable; consumed by ksctl / select.py (no deps).
  * registry.yaml  — human-readable mirror (hand-editing should go through here
                     by re-running this generator).

The registry records, per model family (and per notable concrete model), the
architecture facts and the kernel-set C ABI entry points each logical op maps
to, with a preferred dtype per op. The selection engine (select.py) layers GPU
capability rules on top of these facts to pick the *strongest* available kernel.

Run:  python3 models/_gen_registry.py
"""

import json
import os

# ---------------------------------------------------------------------------
# Logical op vocabulary. These are the columns of the model->kernel table.
# Each maps (at the registry level) to a *default* kernel-set ABI symbol; the
# selection engine may upgrade/downgrade the dtype or pick a paged vs dense
# variant based on (gpu, dtype, mode).
# ---------------------------------------------------------------------------
LOGICAL_OPS = [
    "embedding",
    "attn_norm",       # the pre-attention norm
    "rope",
    "attn_prefill",
    "attn_decode",
    "qkv_proj",
    "o_proj",
    "ffn_norm",        # the pre-MLP norm
    "mlp_gate_up",     # up/gate projection GEMM(s)
    "mlp_act",         # SwiGLU / GeGLU / GeLU activation
    "mlp_down",        # down projection GEMM
    "moe_gate",        # router (only for MoE models)
    "moe_grouped_gemm",# grouped expert GEMM (only for MoE)
    "lm_head",
    "sampling",
]

# Training-only logical ops (appended when mode=training)
TRAINING_OPS = [
    "attn_backward",
    "cross_entropy",
    "optimizer",
]


def base_dense(**over):
    """A standard dense Llama-style decoder op->kernel map (defaults)."""
    m = {
        "embedding":        {"fn": "ks_embedding_lookup",   "dtype": "model"},
        "attn_norm":        {"fn": "ks_rms_norm_residual",  "dtype": "model"},
        "rope":             {"fn": "ks_rope_gather",        "dtype": "model"},
        "attn_prefill":     {"fn": "ks_flash_attn_varlen",  "dtype": "model"},
        "attn_decode":      {"fn": "ks_paged_attn_decode",  "dtype": "model"},
        "qkv_proj":         {"fn": "ks_gemm",               "dtype": "model"},
        "o_proj":           {"fn": "ks_gemm",               "dtype": "model"},
        "ffn_norm":         {"fn": "ks_rms_norm_residual",  "dtype": "model"},
        "mlp_gate_up":      {"fn": "ks_gemm",               "dtype": "model"},
        "mlp_act":          {"fn": "ks_swiglu",             "dtype": "model"},
        "mlp_down":         {"fn": "ks_gemm",               "dtype": "model"},
        "moe_gate":         None,
        "moe_grouped_gemm": None,
        "lm_head":          {"fn": "ks_gemm",               "dtype": "model"},
        "sampling":         {"fn": "ks_sample",             "dtype": "f32"},
    }
    m.update(over)
    return m


def with_moe(m, gate="softmax"):
    """Convert a dense op-map into a MoE op-map.

    gate: "softmax" (Mixtral/Qwen/GPT-OSS style) or "sigmoid_group"
          (DeepSeek-V3 group-limited aux-loss-free sigmoid gating).
    """
    m = dict(m)
    if gate == "sigmoid_group":
        m["moe_gate"] = {"fn": "ks_moe_gate_sigmoid_group_topk", "dtype": "f32"}
    else:
        m["moe_gate"] = {"fn": "ks_moe_gate_softmax_topk", "dtype": "f32"}
    m["moe_grouped_gemm"] = {"fn": "ks_moe_grouped_gemm", "dtype": "model"}
    # MoE experts replace the dense MLP gate/up/down GEMMs with the grouped path;
    # we keep mlp_act for the per-expert activation and drop the dense GEMMs.
    m["mlp_gate_up"] = {"fn": "ks_moe_grouped_gemm", "dtype": "model"}
    m["mlp_down"] = {"fn": "ks_moe_grouped_gemm", "dtype": "model"}
    return m


def layernorm(m):
    """Switch the norms from RMSNorm to LayerNorm (Falcon/MPT/NeoX/Command-R)."""
    m = dict(m)
    m["attn_norm"] = {"fn": "ks_layer_norm", "dtype": "model"}
    m["ffn_norm"] = {"fn": "ks_layer_norm", "dtype": "model"}
    return m


def geglu(m):
    m = dict(m)
    m["mlp_act"] = {"fn": "ks_geglu", "dtype": "model"}
    return m


def gelu_mlp(m):
    """Dense GeLU MLP (no gate): up-proj -> gelu -> down. Falcon/MPT/NeoX."""
    m = dict(m)
    m["mlp_act"] = {"fn": "ks_gelu", "dtype": "model"}
    return m


def mla(m, gate="sigmoid_group"):
    """DeepSeek MLA attention + DeepSeekMoE."""
    m = dict(m)
    m["attn_decode"] = {"fn": "ks_mla_decode", "dtype": "model"}
    # prefill of MLA still uses flash varlen on the up-projected q/k/v.
    m["attn_prefill"] = {"fn": "ks_flash_attn_varlen", "dtype": "model"}
    m = with_moe(m, gate=gate)
    return m


# ---------------------------------------------------------------------------
# Model family + concrete-model catalog.
#
# arch fields:
#   hidden, n_layers, n_heads, n_kv_heads, head_dim, vocab,
#   attention (MHA|GQA|MQA|MLA), rope {style, theta, scaling},
#   norm (rmsnorm|layernorm|rmsnorm_prepost), norm_positions,
#   mlp {type: SwiGLU|GeGLU|GeLU|MoE, inter, n_experts, top_k, n_shared,
#        gating, n_group, topk_group},
#   quant (list of supported schemes), extras (list of notable features)
# ---------------------------------------------------------------------------

def fam(name, kind, family, arch, ops, notes, examples=None):
    return {
        "id": name,
        "kind": kind,             # "family" or "model"
        "family": family,
        "arch": arch,
        "ops": ops,
        "notes": notes,
        "examples": examples or [],
    }


def rope(style="neox", theta=10000, scaling=None, interleaved=0):
    return {"style": style, "theta": theta, "scaling": scaling,
            "interleaved": interleaved}


MODELS = []

# ===== Llama 2 ==============================================================
MODELS.append(fam(
    "llama-2", "family", "Llama 2",
    {"hidden": 4096, "n_layers": 32, "n_heads": 32, "n_kv_heads": 32,
     "head_dim": 128, "vocab": 32000, "attention": "MHA",
     "rope": rope("neox", 10000), "norm": "rmsnorm", "norm_positions": "pre",
     "mlp": {"type": "SwiGLU", "inter": 11008}, "quant": ["bf16", "fp16", "w4a16", "fp8"],
     "extras": []},
    base_dense(),
    "Baseline modern Llama recipe: RMSNorm(pre) + RoPE(theta=10k) + SwiGLU + GQA at scale.",
    ["Llama-2-7B", "Llama-2-13B", "Llama-2-70B"]))

MODELS.append(fam(
    "llama-2-7b", "model", "Llama 2",
    {"hidden": 4096, "n_layers": 32, "n_heads": 32, "n_kv_heads": 32,
     "head_dim": 128, "vocab": 32000, "attention": "MHA",
     "rope": rope("neox", 10000), "norm": "rmsnorm", "norm_positions": "pre",
     "mlp": {"type": "SwiGLU", "inter": 11008}, "quant": ["bf16", "fp16", "w4a16", "fp8"],
     "extras": []},
    base_dense(),
    "Llama-2 7B: dense MHA (no GQA at this size).", ["Llama-2-7B"]))

MODELS.append(fam(
    "llama-2-70b", "model", "Llama 2",
    {"hidden": 8192, "n_layers": 80, "n_heads": 64, "n_kv_heads": 8,
     "head_dim": 128, "vocab": 32000, "attention": "GQA",
     "rope": rope("neox", 10000), "norm": "rmsnorm", "norm_positions": "pre",
     "mlp": {"type": "SwiGLU", "inter": 28672}, "quant": ["bf16", "fp16", "w4a16", "fp8"],
     "extras": []},
    base_dense(),
    "Llama-2 70B: GQA (8 kv heads), SwiGLU.", ["Llama-2-70B"]))

# ===== Llama 3 / 3.1 / 3.3 ==================================================
MODELS.append(fam(
    "llama-3", "family", "Llama 3 / 3.1 / 3.3",
    {"hidden": 4096, "n_layers": 32, "n_heads": 32, "n_kv_heads": 8,
     "head_dim": 128, "vocab": 128256, "attention": "GQA",
     "rope": rope("neox", 500000, "llama3"), "norm": "rmsnorm", "norm_positions": "pre",
     "mlp": {"type": "SwiGLU", "inter": 14336}, "quant": ["bf16", "fp8", "w4a16", "w8a8"],
     "extras": ["high-theta-rope", "rope-scaling-128k"]},
    base_dense(),
    "High-theta RoPE (500k) + Llama3 rope-scaling for 128K ctx; vocab 128k.",
    ["Llama-3-8B", "Llama-3.1-70B", "Llama-3.1-405B"]))

MODELS.append(fam(
    "llama-3-8b", "model", "Llama 3 / 3.1 / 3.3",
    {"hidden": 4096, "n_layers": 32, "n_heads": 32, "n_kv_heads": 8,
     "head_dim": 128, "vocab": 128256, "attention": "GQA",
     "rope": rope("neox", 500000, "llama3"), "norm": "rmsnorm", "norm_positions": "pre",
     "mlp": {"type": "SwiGLU", "inter": 14336}, "quant": ["bf16", "fp8", "w4a16", "w8a8"],
     "extras": ["high-theta-rope", "rope-scaling-128k"]},
    base_dense(),
    "Llama-3 8B: GQA(8 kv), 128k vocab, theta=500k.", ["Llama-3-8B", "Llama-3.1-8B"]))

MODELS.append(fam(
    "llama-3-70b", "model", "Llama 3 / 3.1 / 3.3",
    {"hidden": 8192, "n_layers": 80, "n_heads": 64, "n_kv_heads": 8,
     "head_dim": 128, "vocab": 128256, "attention": "GQA",
     "rope": rope("neox", 500000, "llama3"), "norm": "rmsnorm", "norm_positions": "pre",
     "mlp": {"type": "SwiGLU", "inter": 28672}, "quant": ["bf16", "fp8", "w4a16", "w8a8"],
     "extras": ["high-theta-rope", "rope-scaling-128k"]},
    base_dense(),
    "Llama-3 70B: GQA(8 kv), theta=500k.", ["Llama-3-70B", "Llama-3.1-70B", "Llama-3.3-70B"]))

MODELS.append(fam(
    "llama-3.1-405b", "model", "Llama 3 / 3.1 / 3.3",
    {"hidden": 16384, "n_layers": 126, "n_heads": 128, "n_kv_heads": 8,
     "head_dim": 128, "vocab": 128256, "attention": "GQA",
     "rope": rope("neox", 500000, "llama3"), "norm": "rmsnorm", "norm_positions": "pre",
     "mlp": {"type": "SwiGLU", "inter": 53248}, "quant": ["bf16", "fp8", "w4a16", "w8a8"],
     "extras": ["high-theta-rope", "rope-scaling-128k"]},
    base_dense(),
    "Llama-3.1 405B: largest dense Llama; native FP8 serving common.",
    ["Llama-3.1-405B"]))

# ===== Llama 4 (MoE + iRoPE) ===============================================
MODELS.append(fam(
    "llama-4", "family", "Llama 4",
    {"hidden": 5120, "n_layers": 48, "n_heads": 40, "n_kv_heads": 8,
     "head_dim": 128, "vocab": 202048, "attention": "GQA",
     "rope": rope("neox", 500000, "irope"), "norm": "rmsnorm", "norm_positions": "pre",
     "mlp": {"type": "MoE", "inter": 8192, "n_experts": 16, "top_k": 1, "n_shared": 1,
             "gating": "softmax"},
     "quant": ["bf16", "fp8", "w4a16"],
     "extras": ["irope", "nope-layers", "moe-interleaved-dense", "early-fusion-multimodal"]},
    with_moe(base_dense(), gate="softmax"),
    "First Llama MoE; iRoPE (interleaved RoPE / NoPE blocks) for length generalization; routed+shared experts interleaved with dense.",
    ["Llama-4-Scout", "Llama-4-Maverick"]))

MODELS.append(fam(
    "llama-4-scout", "model", "Llama 4",
    {"hidden": 5120, "n_layers": 48, "n_heads": 40, "n_kv_heads": 8,
     "head_dim": 128, "vocab": 202048, "attention": "GQA",
     "rope": rope("neox", 500000, "irope"), "norm": "rmsnorm", "norm_positions": "pre",
     "mlp": {"type": "MoE", "inter": 8192, "n_experts": 16, "top_k": 1, "n_shared": 1,
             "gating": "softmax"},
     "quant": ["bf16", "fp8", "w4a16"],
     "extras": ["irope", "nope-layers", "moe-interleaved-dense"]},
    with_moe(base_dense(), gate="softmax"),
    "Llama-4 Scout: 16 experts, top-1 + 1 shared, iRoPE.", ["Llama-4-Scout"]))

# ===== Qwen 1.5 / 2 ========================================================
MODELS.append(fam(
    "qwen2", "family", "Qwen 1.5 / 2",
    {"hidden": 3584, "n_layers": 28, "n_heads": 28, "n_kv_heads": 4,
     "head_dim": 128, "vocab": 152064, "attention": "GQA",
     "rope": rope("neox", 1000000), "norm": "rmsnorm", "norm_positions": "pre",
     "mlp": {"type": "SwiGLU", "inter": 18944}, "quant": ["bf16", "w4a16", "fp8"],
     "extras": ["qkv-bias"]},
    base_dense(),
    "Distinctive QKV bias term in attention projections; RoPE theta=1e6.",
    ["Qwen1.5-72B", "Qwen2-7B", "Qwen2-57B-A14B"]))

MODELS.append(fam(
    "qwen2-7b", "model", "Qwen 1.5 / 2",
    {"hidden": 3584, "n_layers": 28, "n_heads": 28, "n_kv_heads": 4,
     "head_dim": 128, "vocab": 152064, "attention": "GQA",
     "rope": rope("neox", 1000000), "norm": "rmsnorm", "norm_positions": "pre",
     "mlp": {"type": "SwiGLU", "inter": 18944}, "quant": ["bf16", "w4a16", "fp8"],
     "extras": ["qkv-bias"]},
    base_dense(),
    "Qwen2 7B: GQA(4 kv) + QKV bias.", ["Qwen2-7B"]))

MODELS.append(fam(
    "qwen2-57b-a14b", "model", "Qwen 1.5 / 2",
    {"hidden": 3584, "n_layers": 28, "n_heads": 28, "n_kv_heads": 4,
     "head_dim": 128, "vocab": 151936, "attention": "GQA",
     "rope": rope("neox", 1000000), "norm": "rmsnorm", "norm_positions": "pre",
     "mlp": {"type": "MoE", "inter": 2560, "n_experts": 64, "top_k": 8, "n_shared": 8,
             "gating": "softmax"},
     "quant": ["bf16", "w4a16", "fp8"], "extras": ["qkv-bias", "shared-experts"]},
    with_moe(base_dense(), gate="softmax"),
    "Qwen2-MoE: 64 experts top-8 + shared experts; QKV bias.", ["Qwen2-57B-A14B"]))

# ===== Qwen 2.5 ============================================================
MODELS.append(fam(
    "qwen2.5", "family", "Qwen 2.5",
    {"hidden": 3584, "n_layers": 28, "n_heads": 28, "n_kv_heads": 4,
     "head_dim": 128, "vocab": 152064, "attention": "GQA",
     "rope": rope("neox", 1000000, "yarn"), "norm": "rmsnorm", "norm_positions": "pre",
     "mlp": {"type": "SwiGLU", "inter": 18944}, "quant": ["bf16", "w4a16", "fp8", "w8a8"],
     "extras": ["qkv-bias", "yarn-128k"]},
    base_dense(),
    "GQA + QKV bias; high-theta RoPE + YaRN long-context (128K+).",
    ["Qwen2.5-7B", "Qwen2.5-72B", "Qwen2.5-Coder"]))

MODELS.append(fam(
    "qwen2.5-7b", "model", "Qwen 2.5",
    {"hidden": 3584, "n_layers": 28, "n_heads": 28, "n_kv_heads": 4,
     "head_dim": 128, "vocab": 152064, "attention": "GQA",
     "rope": rope("neox", 1000000, "yarn"), "norm": "rmsnorm", "norm_positions": "pre",
     "mlp": {"type": "SwiGLU", "inter": 18944}, "quant": ["bf16", "w4a16", "fp8", "w8a8"],
     "extras": ["qkv-bias", "yarn-128k"]},
    base_dense(),
    "Qwen2.5 7B: GQA(4 kv) + QKV bias + YaRN.", ["Qwen2.5-7B"]))

MODELS.append(fam(
    "qwen2.5-72b", "model", "Qwen 2.5",
    {"hidden": 8192, "n_layers": 80, "n_heads": 64, "n_kv_heads": 8,
     "head_dim": 128, "vocab": 152064, "attention": "GQA",
     "rope": rope("neox", 1000000, "yarn"), "norm": "rmsnorm", "norm_positions": "pre",
     "mlp": {"type": "SwiGLU", "inter": 29568}, "quant": ["bf16", "w4a16", "fp8", "w8a8"],
     "extras": ["qkv-bias", "yarn-128k"]},
    base_dense(),
    "Qwen2.5 72B: GQA(8 kv) + QKV bias + YaRN.", ["Qwen2.5-72B"]))

# ===== Qwen 3 / 3-Next =====================================================
MODELS.append(fam(
    "qwen3", "family", "Qwen 3 / 3-Next",
    {"hidden": 4096, "n_layers": 36, "n_heads": 32, "n_kv_heads": 8,
     "head_dim": 128, "vocab": 151936, "attention": "GQA",
     "rope": rope("neox", 1000000, "yarn"), "norm": "rmsnorm", "norm_positions": "pre",
     "mlp": {"type": "SwiGLU", "inter": 12288}, "quant": ["bf16", "fp8", "w4a16"],
     "extras": ["qk-norm", "yarn", "dca"]},
    base_dense(),
    "Adds QK-norm; RoPE theta=1e6, YaRN/DCA. 3-Next uses hybrid linear/full attention.",
    ["Qwen3-8B", "Qwen3-235B-A22B", "Qwen3-Next-80B-A3B"]))

MODELS.append(fam(
    "qwen3-8b", "model", "Qwen 3 / 3-Next",
    {"hidden": 4096, "n_layers": 36, "n_heads": 32, "n_kv_heads": 8,
     "head_dim": 128, "vocab": 151936, "attention": "GQA",
     "rope": rope("neox", 1000000, "yarn"), "norm": "rmsnorm", "norm_positions": "pre",
     "mlp": {"type": "SwiGLU", "inter": 12288}, "quant": ["bf16", "fp8", "w4a16"],
     "extras": ["qk-norm", "yarn"]},
    base_dense(),
    "Qwen3 8B dense: GQA(8 kv) + QK-norm.", ["Qwen3-8B"]))

MODELS.append(fam(
    "qwen3-235b-a22b", "model", "Qwen 3 / 3-Next",
    {"hidden": 4096, "n_layers": 94, "n_heads": 64, "n_kv_heads": 4,
     "head_dim": 128, "vocab": 151936, "attention": "GQA",
     "rope": rope("neox", 1000000, "yarn"), "norm": "rmsnorm", "norm_positions": "pre",
     "mlp": {"type": "MoE", "inter": 1536, "n_experts": 128, "top_k": 8, "n_shared": 0,
             "gating": "softmax"},
     "quant": ["bf16", "fp8", "w4a16"], "extras": ["qk-norm", "ultra-sparse-moe"]},
    with_moe(base_dense(), gate="softmax"),
    "Qwen3 235B-A22B: 128 experts top-8, no shared; QK-norm; ultra-sparse MoE.",
    ["Qwen3-235B-A22B"]))

MODELS.append(fam(
    "qwen3-30b-a3b", "model", "Qwen 3 / 3-Next",
    {"hidden": 2048, "n_layers": 48, "n_heads": 32, "n_kv_heads": 4,
     "head_dim": 128, "vocab": 151936, "attention": "GQA",
     "rope": rope("neox", 1000000, "yarn"), "norm": "rmsnorm", "norm_positions": "pre",
     "mlp": {"type": "MoE", "inter": 768, "n_experts": 128, "top_k": 8, "n_shared": 0,
             "gating": "softmax"},
     "quant": ["bf16", "fp8", "w4a16"], "extras": ["qk-norm", "ultra-sparse-moe"]},
    with_moe(base_dense(), gate="softmax"),
    "Qwen3 30B-A3B: 128 experts top-8; QK-norm.", ["Qwen3-30B-A3B"]))

# ===== Mistral / NeMo (SWA) ================================================
MODELS.append(fam(
    "mistral-7b", "family", "Mistral 7B / NeMo",
    {"hidden": 4096, "n_layers": 32, "n_heads": 32, "n_kv_heads": 8,
     "head_dim": 128, "vocab": 32000, "attention": "GQA",
     "rope": rope("neox", 1000000), "norm": "rmsnorm", "norm_positions": "pre",
     "mlp": {"type": "SwiGLU", "inter": 14336}, "quant": ["bf16", "w4a16", "fp8"],
     "extras": ["sliding-window-4096"]},
    base_dense(),
    "GQA + sliding-window attention (4096); RoPE theta=1e6.",
    ["Mistral-7B", "Mistral-NeMo-12B"]))

MODELS.append(fam(
    "mistral-nemo-12b", "model", "Mistral 7B / NeMo",
    {"hidden": 5120, "n_layers": 40, "n_heads": 32, "n_kv_heads": 8,
     "head_dim": 128, "vocab": 131072, "attention": "GQA",
     "rope": rope("neox", 1000000), "norm": "rmsnorm", "norm_positions": "pre",
     "mlp": {"type": "SwiGLU", "inter": 14336}, "quant": ["bf16", "w4a16", "fp8"],
     "extras": []},
    base_dense(),
    "Mistral-NeMo 12B: GQA(8 kv), 128k vocab.", ["Mistral-NeMo-12B"]))

# ===== Mixtral (MoE) =======================================================
MODELS.append(fam(
    "mixtral-8x7b", "family", "Mixtral 8x7B / 8x22B",
    {"hidden": 4096, "n_layers": 32, "n_heads": 32, "n_kv_heads": 8,
     "head_dim": 128, "vocab": 32000, "attention": "GQA",
     "rope": rope("neox", 1000000), "norm": "rmsnorm", "norm_positions": "pre",
     "mlp": {"type": "MoE", "inter": 14336, "n_experts": 8, "top_k": 2, "n_shared": 0,
             "gating": "softmax"},
     "quant": ["bf16", "w4a16", "fp8"], "extras": ["sliding-window"]},
    with_moe(base_dense(), gate="softmax"),
    "Sparse MoE top-2 of 8, SwiGLU experts; GQA (+SWA on 8x7B).",
    ["Mixtral-8x7B", "Mixtral-8x22B"]))

MODELS.append(fam(
    "mixtral-8x22b", "model", "Mixtral 8x7B / 8x22B",
    {"hidden": 6144, "n_layers": 56, "n_heads": 48, "n_kv_heads": 8,
     "head_dim": 128, "vocab": 32768, "attention": "GQA",
     "rope": rope("neox", 1000000), "norm": "rmsnorm", "norm_positions": "pre",
     "mlp": {"type": "MoE", "inter": 16384, "n_experts": 8, "top_k": 2, "n_shared": 0,
             "gating": "softmax"},
     "quant": ["bf16", "w4a16", "fp8"], "extras": []},
    with_moe(base_dense(), gate="softmax"),
    "Mixtral 8x22B: 8 experts top-2.", ["Mixtral-8x22B"]))

# ===== DeepSeek V2 (MLA) ===================================================
MODELS.append(fam(
    "deepseek-v2", "family", "DeepSeek V2",
    {"hidden": 5120, "n_layers": 60, "n_heads": 128, "n_kv_heads": 128,
     "head_dim": 128, "vocab": 102400, "attention": "MLA",
     "rope": rope("neox", 10000, "yarn"), "norm": "rmsnorm", "norm_positions": "pre",
     "mlp": {"type": "MoE", "inter": 1536, "n_experts": 160, "top_k": 6, "n_shared": 2,
             "gating": "softmax", "n_group": 8, "topk_group": 3},
     "quant": ["bf16", "fp8", "w4a16"],
     "extras": ["mla", "kv-lora-rank-512", "rope-dim-64", "decoupled-rope-key",
                "deepseek-moe", "shared-experts"]},
    mla(base_dense(), gate="softmax"),
    "Introduced MLA (low-rank KV compression + decoupled RoPE key) and shared-expert DeepSeekMoE; YaRN.",
    ["DeepSeek-V2", "DeepSeek-V2-Lite"]))

# ===== DeepSeek V3 / R1 (MLA + sigmoid-group MoE) ==========================
DSV3_ARCH = {
    "hidden": 7168, "n_layers": 61, "n_heads": 128, "n_kv_heads": 128,
    "head_dim": 128, "vocab": 129280, "attention": "MLA",
    "rope": rope("neox", 10000, "yarn"), "norm": "rmsnorm", "norm_positions": "pre",
    "mlp": {"type": "MoE", "inter": 2048, "n_experts": 256, "top_k": 8, "n_shared": 1,
            "gating": "sigmoid_group", "n_group": 8, "topk_group": 4},
    "quant": ["fp8", "bf16", "w4a16"],
    "extras": ["mla", "kv-lora-rank-512", "rope-dim-64", "matrix-absorption-decode",
               "deepseek-moe", "aux-loss-free-sigmoid-gating", "mtp",
               "native-fp8-block-scaled"],
}
MODELS.append(fam(
    "deepseek-v3", "family", "DeepSeek V3 / R1", dict(DSV3_ARCH),
    mla(base_dense(), gate="sigmoid_group"),
    "671B total / 37B active; native FP8 (block-scaled, DeepGEMM); MLA decode w/ matrix absorption; "
    "256 routed + 1 shared top-8 with group-limited aux-loss-free sigmoid gating; MTP.",
    ["DeepSeek-V3", "DeepSeek-R1"]))

MODELS.append(fam(
    "deepseek-r1", "model", "DeepSeek V3 / R1", dict(DSV3_ARCH),
    mla(base_dense(), gate="sigmoid_group"),
    "R1: same arch as V3 (reasoning RL fine-tune); native FP8 block-scaled.",
    ["DeepSeek-R1"]))

# ===== Gemma 2 (pre+post RMSNorm, GeGLU, SWA, soft-cap) ====================
MODELS.append(fam(
    "gemma2", "family", "Gemma 2",
    {"hidden": 3584, "n_layers": 42, "n_heads": 16, "n_kv_heads": 8,
     "head_dim": 256, "vocab": 256000, "attention": "GQA",
     "rope": rope("neox", 10000), "norm": "rmsnorm_prepost", "norm_positions": "pre+post",
     "mlp": {"type": "GeGLU", "inter": 14336}, "quant": ["bf16", "fp8", "w4a16", "w8a8"],
     "extras": ["pre-post-norm", "local-global-swa", "logit-soft-cap", "head-dim-256"]},
    geglu(base_dense()),
    "Pre+post RMSNorm; local/global interleaved SWA; attention/final logit soft-capping; GeGLU(gelu-tanh).",
    ["Gemma-2-9B", "Gemma-2-27B"]))

MODELS.append(fam(
    "gemma2-9b", "model", "Gemma 2",
    {"hidden": 3584, "n_layers": 42, "n_heads": 16, "n_kv_heads": 8,
     "head_dim": 256, "vocab": 256000, "attention": "GQA",
     "rope": rope("neox", 10000), "norm": "rmsnorm_prepost", "norm_positions": "pre+post",
     "mlp": {"type": "GeGLU", "inter": 14336}, "quant": ["bf16", "fp8", "w4a16", "w8a8"],
     "extras": ["pre-post-norm", "local-global-swa", "logit-soft-cap", "head-dim-256"]},
    geglu(base_dense()),
    "Gemma-2 9B: GQA(8 kv), head_dim 256, GeGLU, pre+post norm.", ["Gemma-2-9B"]))

MODELS.append(fam(
    "gemma2-27b", "model", "Gemma 2",
    {"hidden": 4608, "n_layers": 46, "n_heads": 32, "n_kv_heads": 16,
     "head_dim": 128, "vocab": 256000, "attention": "GQA",
     "rope": rope("neox", 10000), "norm": "rmsnorm_prepost", "norm_positions": "pre+post",
     "mlp": {"type": "GeGLU", "inter": 36864}, "quant": ["bf16", "fp8", "w4a16", "w8a8"],
     "extras": ["pre-post-norm", "local-global-swa", "logit-soft-cap"]},
    geglu(base_dense()),
    "Gemma-2 27B: GQA(16 kv), GeGLU, pre+post norm.", ["Gemma-2-27B"]))

# ===== Gemma 3 (QK-norm replaces soft-cap, 5:1 SWA) ========================
MODELS.append(fam(
    "gemma3", "family", "Gemma 3",
    {"hidden": 3840, "n_layers": 48, "n_heads": 16, "n_kv_heads": 8,
     "head_dim": 256, "vocab": 262144, "attention": "GQA",
     "rope": rope("neox", 1000000), "norm": "rmsnorm_prepost", "norm_positions": "pre+post",
     "mlp": {"type": "GeGLU", "inter": 15360}, "quant": ["bf16", "fp8", "w4a16", "w8a8"],
     "extras": ["pre-post-norm", "qk-norm", "swa-5local-1global-w1024", "multimodal",
                "global-theta-1e6", "head-dim-256"]},
    geglu(base_dense()),
    "QK-norm replaces Gemma-2 soft-cap; 5 local : 1 global SWA (window 1024); GeGLU; 128K ctx.",
    ["Gemma-3-4B", "Gemma-3-27B"]))

MODELS.append(fam(
    "gemma3-27b", "model", "Gemma 3",
    {"hidden": 5376, "n_layers": 62, "n_heads": 32, "n_kv_heads": 16,
     "head_dim": 128, "vocab": 262144, "attention": "GQA",
     "rope": rope("neox", 1000000), "norm": "rmsnorm_prepost", "norm_positions": "pre+post",
     "mlp": {"type": "GeGLU", "inter": 21504}, "quant": ["bf16", "fp8", "w4a16", "w8a8"],
     "extras": ["pre-post-norm", "qk-norm", "swa-5local-1global-w1024", "multimodal"]},
    geglu(base_dense()),
    "Gemma-3 27B: GQA(16 kv), QK-norm, GeGLU.", ["Gemma-3-27B"]))

# ===== Phi-3 ===============================================================
MODELS.append(fam(
    "phi-3", "family", "Phi-3",
    {"hidden": 3072, "n_layers": 32, "n_heads": 32, "n_kv_heads": 32,
     "head_dim": 96, "vocab": 32064, "attention": "MHA",
     "rope": rope("neox", 10000, "longrope"), "norm": "rmsnorm", "norm_positions": "pre",
     "mlp": {"type": "SwiGLU", "inter": 8192}, "quant": ["bf16", "w4a16", "fp8"],
     "extras": ["longrope-su-scaling-128k", "head-dim-96"]},
    base_dense(),
    "LongRoPE (SU-scaling) for 128K; Phi-3.5 adds a MoE variant.",
    ["Phi-3-mini", "Phi-3-medium", "Phi-3.5-MoE"]))

MODELS.append(fam(
    "phi-3.5-moe", "model", "Phi-3",
    {"hidden": 4096, "n_layers": 32, "n_heads": 32, "n_kv_heads": 8,
     "head_dim": 128, "vocab": 32064, "attention": "GQA",
     "rope": rope("neox", 10000, "longrope"), "norm": "rmsnorm", "norm_positions": "pre",
     "mlp": {"type": "MoE", "inter": 6400, "n_experts": 16, "top_k": 2, "n_shared": 0,
             "gating": "softmax"},
     "quant": ["bf16", "w4a16", "fp8"], "extras": ["longrope"]},
    with_moe(base_dense(), gate="softmax"),
    "Phi-3.5-MoE: 16 experts top-2; LongRoPE.", ["Phi-3.5-MoE"]))

# ===== Phi-4 ===============================================================
MODELS.append(fam(
    "phi-4", "family", "Phi-4",
    {"hidden": 5120, "n_layers": 40, "n_heads": 40, "n_kv_heads": 10,
     "head_dim": 128, "vocab": 100352, "attention": "GQA",
     "rope": rope("neox", 250000), "norm": "rmsnorm", "norm_positions": "pre",
     "mlp": {"type": "SwiGLU", "inter": 17920}, "quant": ["bf16", "w4a16", "fp8"],
     "extras": []},
    base_dense(),
    "Llama-style dense recipe (GQA, RMSNorm, SwiGLU) with curated/synthetic-data training.",
    ["Phi-4", "Phi-4-mini"]))

# ===== GPT-OSS (MoE + attention sinks + alt SWA + MXFP4) ===================
MODELS.append(fam(
    "gpt-oss", "family", "GPT-OSS (20B/120B)",
    {"hidden": 2880, "n_layers": 24, "n_heads": 64, "n_kv_heads": 8,
     "head_dim": 64, "vocab": 201088, "attention": "GQA",
     "rope": rope("neox", 150000, "yarn"), "norm": "rmsnorm", "norm_positions": "pre",
     "mlp": {"type": "MoE", "inter": 2880, "n_experts": 32, "top_k": 4, "n_shared": 0,
             "gating": "softmax"},
     "quant": ["mxfp4", "fp8", "bf16"],
     "extras": ["attention-sinks", "alternating-full-swa128", "native-mxfp4-experts",
                "head-dim-64"]},
    with_moe(base_dense(), gate="softmax"),
    "OpenAI open-weight; GQA(group 8) + learned per-head attention sinks; alternating full / 128-tok SWA; "
    "MoE (32 experts/20B, 128/120B; top-4; no shared); native MXFP4 experts.",
    ["gpt-oss-20b", "gpt-oss-120b"]))

MODELS.append(fam(
    "gpt-oss-20b", "model", "GPT-OSS (20B/120B)",
    {"hidden": 2880, "n_layers": 24, "n_heads": 64, "n_kv_heads": 8,
     "head_dim": 64, "vocab": 201088, "attention": "GQA",
     "rope": rope("neox", 150000, "yarn"), "norm": "rmsnorm", "norm_positions": "pre",
     "mlp": {"type": "MoE", "inter": 2880, "n_experts": 32, "top_k": 4, "n_shared": 0,
             "gating": "softmax"},
     "quant": ["mxfp4", "fp8", "bf16"],
     "extras": ["attention-sinks", "alternating-full-swa128", "native-mxfp4-experts"]},
    with_moe(base_dense(), gate="softmax"),
    "gpt-oss-20b: 32 experts top-4, attention sinks, native MXFP4.", ["gpt-oss-20b"]))

MODELS.append(fam(
    "gpt-oss-120b", "model", "GPT-OSS (20B/120B)",
    {"hidden": 2880, "n_layers": 36, "n_heads": 64, "n_kv_heads": 8,
     "head_dim": 64, "vocab": 201088, "attention": "GQA",
     "rope": rope("neox", 150000, "yarn"), "norm": "rmsnorm", "norm_positions": "pre",
     "mlp": {"type": "MoE", "inter": 2880, "n_experts": 128, "top_k": 4, "n_shared": 0,
             "gating": "softmax"},
     "quant": ["mxfp4", "fp8", "bf16"],
     "extras": ["attention-sinks", "alternating-full-swa128", "native-mxfp4-experts"]},
    with_moe(base_dense(), gate="softmax"),
    "gpt-oss-120b: 128 experts top-4, attention sinks, native MXFP4.", ["gpt-oss-120b"]))

# ===== Yi ==================================================================
MODELS.append(fam(
    "yi", "family", "Yi",
    {"hidden": 7168, "n_layers": 60, "n_heads": 56, "n_kv_heads": 8,
     "head_dim": 128, "vocab": 64000, "attention": "GQA",
     "rope": rope("neox", 5000000), "norm": "rmsnorm", "norm_positions": "pre",
     "mlp": {"type": "SwiGLU", "inter": 20480}, "quant": ["bf16", "w4a16", "fp8"],
     "extras": ["long-context-200k"]},
    base_dense(),
    "Llama architecture clone with extended-context (200K) variants; RoPE high-theta.",
    ["Yi-6B", "Yi-34B", "Yi-1.5"]))

MODELS.append(fam(
    "yi-34b", "model", "Yi",
    {"hidden": 7168, "n_layers": 60, "n_heads": 56, "n_kv_heads": 8,
     "head_dim": 128, "vocab": 64000, "attention": "GQA",
     "rope": rope("neox", 5000000), "norm": "rmsnorm", "norm_positions": "pre",
     "mlp": {"type": "SwiGLU", "inter": 20480}, "quant": ["bf16", "w4a16", "fp8"],
     "extras": ["long-context-200k"]},
    base_dense(),
    "Yi-34B: GQA(8 kv), Llama-like.", ["Yi-34B"]))

# ===== Baichuan 2 (7B RoPE / 13B ALiBi) ====================================
MODELS.append(fam(
    "baichuan2", "family", "Baichuan 2",
    {"hidden": 4096, "n_layers": 32, "n_heads": 32, "n_kv_heads": 32,
     "head_dim": 128, "vocab": 125696, "attention": "MHA",
     "rope": rope("neox", 10000), "norm": "rmsnorm", "norm_positions": "pre",
     "mlp": {"type": "SwiGLU", "inter": 11008}, "quant": ["w4a16", "bf16"],
     "extras": ["7b-rope-13b-alibi"]},
    base_dense(),
    "MHA; 7B uses RoPE, 13B uses ALiBi (no rotary).",
    ["Baichuan2-7B", "Baichuan2-13B"]))

MODELS.append(fam(
    "baichuan2-13b", "model", "Baichuan 2",
    {"hidden": 5120, "n_layers": 40, "n_heads": 40, "n_kv_heads": 40,
     "head_dim": 128, "vocab": 125696, "attention": "MHA",
     "rope": rope("alibi", 0), "norm": "rmsnorm", "norm_positions": "pre",
     "mlp": {"type": "SwiGLU", "inter": 13696}, "quant": ["w4a16", "bf16"],
     "extras": ["alibi"]},
    # ALiBi: no rope op; attention uses additive bias (still flash/paged).
    dict(base_dense(), rope=None),
    "Baichuan2-13B: ALiBi positional bias instead of rotary.", ["Baichuan2-13B"]))

# ===== ChatGLM / GLM-4 / GLM-4.5 ===========================================
MODELS.append(fam(
    "glm-4", "family", "ChatGLM / GLM-4 / GLM-4.5",
    {"hidden": 4096, "n_layers": 40, "n_heads": 32, "n_kv_heads": 2,
     "head_dim": 128, "vocab": 151552, "attention": "GQA",
     "rope": rope("gptj", 10000, "partial"), "norm": "rmsnorm", "norm_positions": "post",
     "mlp": {"type": "SwiGLU", "inter": 13696}, "quant": ["w4a16", "bf16", "fp8"],
     "extras": ["partial-2d-rope", "qkv-bias", "post-norm"]},
    base_dense(rope={"fn": "ks_rope_gather", "dtype": "model"}),
    "MQA/GQA; rotary on half-dim (2D/partial RoPE, GPT-J interleaved); QKV bias; GLM-4.5+ are MoE.",
    ["ChatGLM3-6B", "GLM-4-9B", "GLM-4.5", "GLM-4.6"]))

MODELS.append(fam(
    "glm-4-9b", "model", "ChatGLM / GLM-4 / GLM-4.5",
    {"hidden": 4096, "n_layers": 40, "n_heads": 32, "n_kv_heads": 2,
     "head_dim": 128, "vocab": 151552, "attention": "GQA",
     "rope": rope("gptj", 10000, "partial"), "norm": "rmsnorm", "norm_positions": "post",
     "mlp": {"type": "SwiGLU", "inter": 13696}, "quant": ["w4a16", "bf16", "fp8"],
     "extras": ["partial-2d-rope", "qkv-bias"]},
    base_dense(),
    "GLM-4-9B: GQA(2 kv), partial RoPE (GPT-J interleaved), QKV bias.", ["GLM-4-9B"]))

MODELS.append(fam(
    "glm-4.5", "model", "ChatGLM / GLM-4 / GLM-4.5",
    {"hidden": 5120, "n_layers": 92, "n_heads": 96, "n_kv_heads": 8,
     "head_dim": 128, "vocab": 151552, "attention": "GQA",
     "rope": rope("gptj", 10000, "partial"), "norm": "rmsnorm", "norm_positions": "post",
     "mlp": {"type": "MoE", "inter": 1536, "n_experts": 160, "top_k": 8, "n_shared": 1,
             "gating": "sigmoid_group", "n_group": 1, "topk_group": 1},
     "quant": ["bf16", "fp8", "w4a16"], "extras": ["partial-2d-rope", "moe", "qk-norm"]},
    with_moe(base_dense(), gate="sigmoid_group"),
    "GLM-4.5: MoE (160 experts top-8 + 1 shared), sigmoid gating; partial RoPE.",
    ["GLM-4.5", "GLM-4.6"]))

# ===== Falcon (MQA/parallel attn+MLP, LayerNorm, GeLU) =====================
MODELS.append(fam(
    "falcon", "family", "Falcon",
    {"hidden": 8192, "n_layers": 60, "n_heads": 64, "n_kv_heads": 8,
     "head_dim": 128, "vocab": 65024, "attention": "MQA",
     "rope": rope("neox", 10000), "norm": "layernorm", "norm_positions": "pre-parallel",
     "mlp": {"type": "GeLU", "inter": 32768}, "quant": ["w4a16", "int8"],
     "extras": ["parallel-attn-mlp", "multi-query"]},
    gelu_mlp(layernorm(base_dense())),
    "MQA (40B/180B) / MHA (7B); parallel attention+MLP block; LayerNorm; dense GeLU MLP.",
    ["Falcon-7B", "Falcon-40B", "Falcon-180B"]))

MODELS.append(fam(
    "falcon-40b", "model", "Falcon",
    {"hidden": 8192, "n_layers": 60, "n_heads": 128, "n_kv_heads": 8,
     "head_dim": 64, "vocab": 65024, "attention": "MQA",
     "rope": rope("neox", 10000), "norm": "layernorm", "norm_positions": "pre-parallel",
     "mlp": {"type": "GeLU", "inter": 32768}, "quant": ["w4a16", "int8"],
     "extras": ["parallel-attn-mlp", "multi-query", "head-dim-64"]},
    gelu_mlp(layernorm(base_dense())),
    "Falcon-40B: MQA (8 kv-group), parallel attn+MLP, LayerNorm.", ["Falcon-40B"]))

# ===== Command-R / R+ (LayerNorm, tied emb, GQA) ===========================
MODELS.append(fam(
    "command-r", "family", "Command-R / R+",
    {"hidden": 8192, "n_layers": 40, "n_heads": 64, "n_kv_heads": 64,
     "head_dim": 128, "vocab": 256000, "attention": "MHA",
     "rope": rope("neox", 10000), "norm": "layernorm", "norm_positions": "pre",
     "mlp": {"type": "SwiGLU", "inter": 22528}, "quant": ["bf16", "w4a16", "fp8", "int8"],
     "extras": ["tied-embeddings", "no-bias"]},
    layernorm(base_dense()),
    "Cohere RAG/tool-use; GQA/MHA, RoPE, LayerNorm, tied embeddings, no bias.",
    ["Command-R-35B", "Command-R-Plus-104B"]))

MODELS.append(fam(
    "command-r-plus", "model", "Command-R / R+",
    {"hidden": 12288, "n_layers": 64, "n_heads": 96, "n_kv_heads": 8,
     "head_dim": 128, "vocab": 256000, "attention": "GQA",
     "rope": rope("neox", 10000), "norm": "layernorm", "norm_positions": "pre",
     "mlp": {"type": "SwiGLU", "inter": 33792}, "quant": ["bf16", "w4a16", "fp8", "int8"],
     "extras": ["tied-embeddings", "no-bias"]},
    layernorm(base_dense()),
    "Command-R+ 104B: GQA(8 kv), LayerNorm, tied embeddings.", ["Command-R-Plus-104B"]))

# ===== StableLM (partial RoPE, LayerNorm/RMSNorm) ==========================
MODELS.append(fam(
    "stablelm", "family", "StableLM",
    {"hidden": 2560, "n_layers": 32, "n_heads": 32, "n_kv_heads": 32,
     "head_dim": 80, "vocab": 100352, "attention": "MHA",
     "rope": rope("neox", 10000, "partial"), "norm": "layernorm", "norm_positions": "pre",
     "mlp": {"type": "SwiGLU", "inter": 6912}, "quant": ["w4a16", "bf16"],
     "extras": ["partial-rotary", "optional-qk-layernorm", "head-dim-80"]},
    layernorm(base_dense()),
    "Partial-rotary RoPE (rotary_pct); LayerNorm or RMSNorm by variant; optional QK-LayerNorm.",
    ["StableLM-2-1.6B", "StableLM-2-12B"]))

# ===== MPT (ALiBi, LayerNorm no-bias, GeLU) ================================
MODELS.append(fam(
    "mpt", "family", "MPT",
    {"hidden": 4096, "n_layers": 32, "n_heads": 32, "n_kv_heads": 32,
     "head_dim": 128, "vocab": 50432, "attention": "MHA",
     "rope": rope("alibi", 0), "norm": "layernorm", "norm_positions": "pre",
     "mlp": {"type": "GeLU", "inter": 16384}, "quant": ["w4a16", "int8"],
     "extras": ["alibi", "no-bias"]},
    gelu_mlp(layernorm(dict(base_dense(), rope=None))),
    "ALiBi positional bias instead of rotary; LayerNorm (no bias); dense GeLU.",
    ["MPT-7B", "MPT-30B"]))

# ===== GPT-NeoX (partial GPT-J RoPE, parallel, LayerNorm, GeLU) ============
MODELS.append(fam(
    "gpt-neox", "family", "GPT-NeoX",
    {"hidden": 6144, "n_layers": 44, "n_heads": 64, "n_kv_heads": 64,
     "head_dim": 96, "vocab": 50432, "attention": "MHA",
     "rope": rope("gptj", 10000, "partial"), "norm": "layernorm", "norm_positions": "pre-parallel",
     "mlp": {"type": "GeLU", "inter": 24576}, "quant": ["w4a16", "int8"],
     "extras": ["partial-rope", "parallel-attn-mlp", "head-dim-96"]},
    gelu_mlp(layernorm(base_dense())),
    "Partial/GPT-J RoPE (interleaved) + parallel residual; LayerNorm; ancestor of many open recipes.",
    ["GPT-NeoX-20B", "Pythia"]))

# ===== OLMo / OLMo-2 (QK-norm, reordered post-norm in OLMo-2) ==============
MODELS.append(fam(
    "olmo-2", "family", "OLMo / OLMo-2",
    {"hidden": 4096, "n_layers": 32, "n_heads": 32, "n_kv_heads": 32,
     "head_dim": 128, "vocab": 100352, "attention": "MHA",
     "rope": rope("neox", 500000), "norm": "rmsnorm", "norm_positions": "post",
     "mlp": {"type": "SwiGLU", "inter": 11008}, "quant": ["bf16", "fp8", "w4a16"],
     "extras": ["qk-norm", "reordered-post-norm"]},
    base_dense(),
    "Fully-open AllenAI; OLMo-2 uses RMSNorm + QK-norm + post-norm (reordered) for stability. "
    "OLMo-1 used non-parametric LayerNorm.",
    ["OLMo-7B", "OLMo-2-13B"]))

MODELS.append(fam(
    "olmo-2-13b", "model", "OLMo / OLMo-2",
    {"hidden": 5120, "n_layers": 40, "n_heads": 40, "n_kv_heads": 40,
     "head_dim": 128, "vocab": 100352, "attention": "MHA",
     "rope": rope("neox", 500000), "norm": "rmsnorm", "norm_positions": "post",
     "mlp": {"type": "SwiGLU", "inter": 13824}, "quant": ["bf16", "fp8", "w4a16"],
     "extras": ["qk-norm", "reordered-post-norm"]},
    base_dense(),
    "OLMo-2 13B: MHA, RMSNorm, QK-norm, post-norm.", ["OLMo-2-13B"]))


REGISTRY = {
    "schema_version": 1,
    "abi_header": "include/kernel_set/kernel_set.h",
    "logical_ops": LOGICAL_OPS,
    "training_ops": TRAINING_OPS,
    "op_descriptions": {
        "embedding": "token embedding lookup",
        "attn_norm": "pre-attention normalization",
        "rope": "rotary position embedding",
        "attn_prefill": "prefill / context attention (full sequence)",
        "attn_decode": "autoregressive decode attention (paged KV)",
        "qkv_proj": "fused Q/K/V projection GEMM",
        "o_proj": "attention output projection GEMM",
        "ffn_norm": "pre-MLP normalization",
        "mlp_gate_up": "MLP up/gate projection GEMM (or MoE grouped GEMM)",
        "mlp_act": "gated MLP activation (SwiGLU/GeGLU) or dense act",
        "mlp_down": "MLP down projection GEMM (or MoE grouped GEMM)",
        "moe_gate": "MoE router / gating",
        "moe_grouped_gemm": "grouped per-expert GEMM",
        "lm_head": "final logits projection",
        "sampling": "token sampling / decode head",
        "attn_backward": "attention backward (training)",
        "cross_entropy": "fused (linear) cross-entropy loss (training)",
        "optimizer": "fused optimizer step (training)",
    },
    "models": MODELS,
}


def to_yaml(obj, indent=0):
    """Tiny deterministic YAML emitter (dicts, lists, scalars only)."""
    sp = "  " * indent
    lines = []
    if isinstance(obj, dict):
        if not obj:
            return sp + "{}\n"
        for k, v in obj.items():
            if isinstance(v, (dict, list)) and v:
                lines.append(f"{sp}{k}:")
                lines.append(to_yaml(v, indent + 1))
            else:
                lines.append(f"{sp}{k}: {scalar(v)}")
        return "\n".join(lines) + "\n"
    if isinstance(obj, list):
        if not obj:
            return sp + "[]\n"
        for item in obj:
            if isinstance(item, (dict, list)) and item:
                # render the first key inline after the dash
                rendered = to_yaml(item, indent + 1)
                rendered_lines = rendered.rstrip("\n").split("\n")
                first = rendered_lines[0].lstrip()
                lines.append(f"{sp}- {first}")
                for rl in rendered_lines[1:]:
                    lines.append(rl)
            else:
                lines.append(f"{sp}- {scalar(item)}")
        return "\n".join(lines) + "\n"
    return sp + scalar(obj) + "\n"


def scalar(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v)
    if s == "" or any(c in s for c in ":#{}[],&*!|>'\"%@`") or s != s.strip():
        return json.dumps(s)
    return s


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(here, "registry.json")
    yaml_path = os.path.join(here, "registry.yaml")
    with open(json_path, "w") as f:
        json.dump(REGISTRY, f, indent=2)
        f.write("\n")
    with open(yaml_path, "w") as f:
        f.write("# kernel-set model registry (human-readable mirror).\n")
        f.write("# SOURCE OF TRUTH IS registry.json — regenerate via "
                "`python3 models/_gen_registry.py`.\n")
        f.write(to_yaml(REGISTRY))
    n_fam = sum(1 for m in MODELS if m["kind"] == "family")
    n_mod = sum(1 for m in MODELS if m["kind"] == "model")
    print(f"wrote {json_path} and {yaml_path}: {len(MODELS)} entries "
          f"({n_fam} families, {n_mod} concrete models)")


if __name__ == "__main__":
    main()
