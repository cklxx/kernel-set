#!/usr/bin/env python3
"""Qwen3 smoke comparison for composed kernel-set decode paths.

This runner is intentionally small and auditable. It is not a production
serving engine: scheduling, request batching, CUDA graphs, allocator policy, and
paginated block management are still Python. The default path is the
best-practice composition: dense linears stay on torch/cuBLAS, while kernel-set
is used for the memory/attention/sample kernels where this repo has useful
coverage. The all-kernel GEMM path remains available as a coverage smoke via
``--run-full-kernels``.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_PROMPT = "用户：今天下班有点累，想晚上吃得简单一点，你有什么轻松的建议？\n助手："


def _run(cmd: List[str], cwd: Optional[pathlib.Path] = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _run_capture(cmd: List[str], cwd: Optional[pathlib.Path] = None) -> str:
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.check_output(
        cmd, cwd=str(cwd) if cwd else None, text=True, stderr=subprocess.STDOUT
    )


def _install_deps(include_vllm: bool) -> None:
    pkgs = [
        "cmake>=3.24",
        "transformers>=4.51.0",
        "accelerate",
        "sentencepiece",
        "safetensors",
    ]
    if include_vllm:
        pkgs.append("vllm==0.10.2")
    _run([sys.executable, "-m", "pip", "install", "-q", *pkgs])


def _prepare_repo(repo: pathlib.Path, clone_url: str, ref: Optional[str]) -> pathlib.Path:
    if repo.exists() and (repo / "CMakeLists.txt").exists():
        return repo
    if repo.exists():
        shutil.rmtree(repo)
    _run(["git", "clone", "--depth", "1", clone_url, str(repo)])
    if ref:
        _run(["git", "fetch", "--depth", "1", "origin", ref], cwd=repo)
        _run(["git", "checkout", "FETCH_HEAD"], cwd=repo)
    return repo


def _detect_sm() -> Optional[str]:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability(0)
        return f"{major}{minor}"
    except Exception:
        return None


def _build_kernel_set(repo: pathlib.Path, arch: Optional[str]) -> pathlib.Path:
    build = repo / "build-qwen3-engine"
    cmd = [
        "cmake",
        "-S",
        str(repo),
        "-B",
        str(build),
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    if arch:
        cmd.append(f"-DCMAKE_CUDA_ARCHITECTURES={arch}")
    _run(cmd)
    _run(["cmake", "--build", str(build), "-j", str(os.cpu_count() or 4)])
    lib = build / "libkernel_set.so"
    if not lib.exists():
        raise FileNotFoundError(lib)
    return lib


def _torch_sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timer(fn, repeat: int = 1) -> Tuple[float, Any]:
    _torch_sync()
    t0 = time.perf_counter()
    out = None
    for _ in range(repeat):
        out = fn()
    _torch_sync()
    return (time.perf_counter() - t0) / repeat, out


def _rms_eps(mod) -> float:
    return float(getattr(mod, "variance_epsilon", getattr(mod, "eps", 1e-6)))


def _make_rope_cache(max_pos: int, head_dim: int, theta: float, device, dtype):
    import torch

    inv_freq = 1.0 / (
        theta
        ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(max_pos, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos().to(dtype).contiguous(), freqs.sin().to(dtype).contiguous()


@dataclass
class KernelStats:
    ks_embedding_lookup_calls: int = 0
    ks_gemm_calls: int = 0
    ks_rmsnorm_calls: int = 0
    ks_rope_calls: int = 0
    ks_flash_attn_calls: int = 0
    ks_reshape_and_cache_calls: int = 0
    ks_paged_attn_decode_calls: int = 0
    ks_swiglu_calls: int = 0
    ks_argmax_calls: int = 0
    torch_embedding_calls: int = 0
    torch_linear_calls: int = 0
    torch_rmsnorm_calls: int = 0
    torch_rope_calls: int = 0
    torch_cache_write_calls: int = 0
    torch_attention_prefill_calls: int = 0
    torch_attention_decode_calls: int = 0
    torch_swiglu_calls: int = 0
    torch_argmax_calls: int = 0


class KernelSetQwen3FullPath:
    """Single-request Qwen3 path using kernel-set kernels where available."""

    def __init__(self, model, ks, max_total_tokens: int, block_size: int = 16):
        import torch

        self.torch = torch
        self.model = model.eval()
        self.core = model.model
        self.layers = list(self.core.layers)
        self.config = model.config
        self.ks = ks
        self.stats = KernelStats()
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.hidden = int(self.config.hidden_size)
        self.intermediate = int(self.config.intermediate_size)
        self.n_heads = int(self.config.num_attention_heads)
        self.n_kv_heads = int(getattr(self.config, "num_key_value_heads", self.n_heads))
        self.head_dim = int(getattr(self.config, "head_dim", self.hidden // self.n_heads))
        self.attn_width = self.n_heads * self.head_dim
        self.attn_scale = self.head_dim ** -0.5
        self.rope_theta = float(getattr(self.config, "rope_theta", 1000000.0))
        self.block_size = int(block_size)
        self.max_total_tokens = int(max_total_tokens)
        self.max_blocks = (self.max_total_tokens + self.block_size - 1) // self.block_size
        self.block_tables = torch.arange(
            self.max_blocks, device=self.device, dtype=torch.int32
        ).view(1, self.max_blocks)
        self.k_caches = [
            torch.empty(
                self.max_blocks,
                self.n_kv_heads,
                self.block_size,
                self.head_dim,
                device=self.device,
                dtype=self.dtype,
            )
            for _ in self.layers
        ]
        self.v_caches = [torch.empty_like(k) for k in self.k_caches]
        max_rope = max(
            int(getattr(self.config, "max_position_embeddings", 4096)),
            self.max_total_tokens + 8,
        )
        self.cos, self.sin = _make_rope_cache(
            max_rope, self.head_dim, self.rope_theta, self.device, self.dtype
        )

    def reset_stats(self) -> None:
        self.stats = KernelStats()

    def _embedding(self, input_ids):
        ids = input_ids.reshape(-1).contiguous()
        out = self.torch.empty(
            (int(ids.numel()), self.hidden), device=self.device, dtype=self.dtype
        )
        self.ks.embedding.embedding_lookup(
            out,
            self.core.embed_tokens.weight.contiguous(),
            ids,
            int(ids.numel()),
            self.hidden,
        )
        self.stats.ks_embedding_lookup_calls += 1
        return out

    def _rms(self, x, norm_mod):
        out = self.torch.empty_like(x)
        self.ks.norm.rms_norm(
            out, x.contiguous(), norm_mod.weight.contiguous(), eps=_rms_eps(norm_mod)
        )
        self.stats.ks_rmsnorm_calls += 1
        return out

    def _linear(self, mod, x):
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        out_features, in_features = mod.weight.shape
        out = self.torch.empty(
            (x2.shape[0], out_features), device=x2.device, dtype=x2.dtype
        )
        self.ks.gemm.gemm(
            out,
            x2,
            mod.weight.contiguous(),
            m=int(x2.shape[0]),
            n=int(out_features),
            k=int(in_features),
            trans_b=True,
        )
        if mod.bias is not None:
            out.add_(mod.bias)
        self.stats.ks_gemm_calls += 1
        return out.view(*x.shape[:-1], out_features)

    def _qk_norm(self, attn, q, k):
        if hasattr(attn, "q_norm") and attn.q_norm is not None:
            q = self._rms(q.reshape(-1, self.head_dim), attn.q_norm).view_as(q)
        if hasattr(attn, "k_norm") and attn.k_norm is not None:
            k = self._rms(k.reshape(-1, self.head_dim), attn.k_norm).view_as(k)
        return q, k

    def _rope(self, q, k, start_pos: int):
        positions = self.torch.arange(
            start_pos,
            start_pos + q.shape[0],
            device=self.device,
            dtype=self.torch.int32,
        )
        q = q.contiguous()
        k = k.contiguous()
        self.ks.rope.rope_gather(
            q,
            k,
            self.cos,
            self.sin,
            positions,
            int(q.shape[0]),
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
            interleaved=False,
        )
        self.stats.ks_rope_calls += 1
        return q, k

    def _cache_write(self, layer_idx: int, k, v, start_pos: int):
        slot_mapping = self.torch.arange(
            start_pos,
            start_pos + k.shape[0],
            device=self.device,
            dtype=self.torch.int32,
        )
        self.ks.attention.reshape_and_cache(
            self.k_caches[layer_idx],
            self.v_caches[layer_idx],
            k.contiguous(),
            v.contiguous(),
            slot_mapping,
            int(k.shape[0]),
            self.n_kv_heads,
            self.head_dim,
            self.block_size,
        )
        self.stats.ks_reshape_and_cache_calls += 1

    def _attention_prefill(self, q, k, v):
        seq = int(q.shape[0])
        out = self.torch.empty((1, seq, self.n_heads, self.head_dim), device=self.device, dtype=self.dtype)
        self.ks.attention.flash_attn(
            out,
            q.view(1, seq, self.n_heads, self.head_dim),
            k.view(1, seq, self.n_kv_heads, self.head_dim),
            v.view(1, seq, self.n_kv_heads, self.head_dim),
            1,
            seq,
            seq,
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
            softmax_scale=self.attn_scale,
            causal=True,
        )
        self.stats.ks_flash_attn_calls += 1
        return out.view(seq, self.attn_width)

    def _attention_decode(self, layer_idx: int, q, seq_len: int):
        out = self.torch.empty((1, self.n_heads, self.head_dim), device=self.device, dtype=self.dtype)
        seq_lens = self.torch.tensor([seq_len], device=self.device, dtype=self.torch.int32)
        self.ks.attention.paged_attn_decode(
            out,
            q.view(1, self.n_heads, self.head_dim).contiguous(),
            self.k_caches[layer_idx],
            self.v_caches[layer_idx],
            self.block_tables,
            seq_lens,
            1,
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
            self.block_size,
            self.max_blocks,
            softmax_scale=self.attn_scale,
        )
        self.stats.ks_paged_attn_decode_calls += 1
        return out.view(1, self.attn_width)

    def _mlp(self, layer, h):
        gate = self._linear(layer.mlp.gate_proj, h)
        up = self._linear(layer.mlp.up_proj, h)
        inter = self.torch.empty_like(gate)
        self.ks.activation.swiglu(inter, gate.contiguous(), up.contiguous())
        self.stats.ks_swiglu_calls += 1
        return self._linear(layer.mlp.down_proj, inter)

    def _next_token(self, x_last):
        x_last = self._rms(x_last, self.core.norm)
        logits = self._linear(self.model.lm_head, x_last)
        out = self.torch.empty(1, device=self.device, dtype=self.torch.int32)
        self.ks.sampling.argmax(out, logits.contiguous(), 1, int(logits.shape[-1]))
        self.stats.ks_argmax_calls += 1
        return int(out.item()), logits

    def prefill(self, input_ids):
        x = self._embedding(input_ids)
        seq = int(x.shape[0])
        for li, layer in enumerate(self.layers):
            attn = layer.self_attn
            residual = x
            h = self._rms(x, layer.input_layernorm)
            q = self._linear(attn.q_proj, h).view(seq, self.n_heads, self.head_dim)
            k = self._linear(attn.k_proj, h).view(seq, self.n_kv_heads, self.head_dim)
            v = self._linear(attn.v_proj, h).view(seq, self.n_kv_heads, self.head_dim)
            q, k = self._qk_norm(attn, q, k)
            q, k = self._rope(q, k, 0)
            self._cache_write(li, k, v, 0)
            ctx = self._attention_prefill(q, k, v)
            x = residual + self._linear(attn.o_proj, ctx)
            residual = x
            h = self._rms(x, layer.post_attention_layernorm)
            x = residual + self._mlp(layer, h)
        return self._next_token(x[-1:].contiguous())

    def decode_one(self, token_id: int, pos: int):
        ids = self.torch.tensor([[token_id]], device=self.device, dtype=self.torch.long)
        x = self._embedding(ids)
        for li, layer in enumerate(self.layers):
            attn = layer.self_attn
            residual = x
            h = self._rms(x, layer.input_layernorm)
            q = self._linear(attn.q_proj, h).view(1, self.n_heads, self.head_dim)
            k = self._linear(attn.k_proj, h).view(1, self.n_kv_heads, self.head_dim)
            v = self._linear(attn.v_proj, h).view(1, self.n_kv_heads, self.head_dim)
            q, k = self._qk_norm(attn, q, k)
            q, k = self._rope(q, k, pos)
            self._cache_write(li, k, v, pos)
            ctx = self._attention_decode(li, q, seq_len=pos + 1)
            x = residual + self._linear(attn.o_proj, ctx)
            residual = x
            h = self._rms(x, layer.post_attention_layernorm)
            x = residual + self._mlp(layer, h)
        return self._next_token(x)

    def generate(self, input_ids, max_new_tokens: int):
        tokens = [int(t) for t in input_ids.reshape(-1).tolist()]
        next_id, logits = self.prefill(input_ids)
        generated: List[int] = []
        for i in range(max_new_tokens):
            generated.append(next_id)
            if i + 1 < max_new_tokens:
                next_id, logits = self.decode_one(next_id, pos=len(tokens) + i)
        return tokens + generated, logits


BEST_PRACTICE_MODES = {
    "embedding": "ks",
    "linear": "torch",
    "norm": "ks",
    "rope": "ks",
    "cache": "ks",
    "attention": "ks",
    "swiglu": "ks",
    "argmax": "ks",
}

TORCH_MANUAL_MODES = {
    "embedding": "torch",
    "linear": "torch",
    "norm": "torch",
    "rope": "torch",
    "cache": "torch",
    "attention": "torch",
    "swiglu": "torch",
    "argmax": "torch",
}

ABLATION_VARIANTS = [
    (
        "torch_embedding",
        {"embedding": "torch"},
        "replace ks embedding lookup with torch embedding",
    ),
    (
        "torch_norm",
        {"norm": "torch"},
        "replace ks hidden/QK RMSNorm with HF torch modules",
    ),
    (
        "torch_rope",
        {"rope": "torch"},
        "replace ks RoPE gather with torch rotate-half RoPE",
    ),
    (
        "torch_cache_write",
        {"cache": "torch"},
        "replace ks reshape_and_cache with torch cache scatter",
    ),
    (
        "torch_attention",
        {"attention": "torch"},
        "replace ks prefill/decode attention with torch SDPA/manual decode",
    ),
    (
        "torch_swiglu",
        {"swiglu": "torch"},
        "replace ks SwiGLU with torch silu(gate)*up",
    ),
    (
        "torch_argmax",
        {"argmax": "torch"},
        "replace ks argmax with torch argmax",
    ),
    (
        "manual_torch_ops",
        TORCH_MANUAL_MODES,
        "manual Python engine with torch ops for every replaceable component",
    ),
]


def _merge_modes(overrides: Dict[str, str]) -> Dict[str, str]:
    modes = dict(BEST_PRACTICE_MODES)
    modes.update(overrides)
    return modes


class KernelSetQwen3ConfigurablePath(KernelSetQwen3FullPath):
    """Single-request Qwen3 path with per-component ks/torch switches."""

    def __init__(
        self,
        model,
        ks,
        max_total_tokens: int,
        block_size: int = 16,
        op_modes: Optional[Dict[str, str]] = None,
    ):
        super().__init__(model, ks, max_total_tokens=max_total_tokens, block_size=block_size)
        self.op_modes = _merge_modes(op_modes or {})

    def _mode(self, name: str) -> str:
        return self.op_modes.get(name, "ks")

    def _embedding(self, input_ids):
        if self._mode("embedding") != "torch":
            return super()._embedding(input_ids)
        self.stats.torch_embedding_calls += 1
        return self.core.embed_tokens(input_ids).reshape(-1, self.hidden).contiguous()

    def _rms(self, x, norm_mod):
        if self._mode("norm") != "torch":
            return super()._rms(x, norm_mod)
        self.stats.torch_rmsnorm_calls += 1
        return norm_mod(x)

    def _linear(self, mod, x):
        if self._mode("linear") != "torch":
            return super()._linear(mod, x)
        self.stats.torch_linear_calls += 1
        return mod(x)

    def _rotate_half(self, x):
        half = x.shape[-1] // 2
        return self.torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    def _rope(self, q, k, start_pos: int):
        if self._mode("rope") != "torch":
            return super()._rope(q, k, start_pos)
        positions = self.torch.arange(
            start_pos,
            start_pos + q.shape[0],
            device=self.device,
            dtype=self.torch.long,
        )
        cos = self.cos.index_select(0, positions)
        sin = self.sin.index_select(0, positions)
        cos = self.torch.cat((cos, cos), dim=-1).unsqueeze(1)
        sin = self.torch.cat((sin, sin), dim=-1).unsqueeze(1)
        q = (q * cos + self._rotate_half(q) * sin).contiguous()
        k = (k * cos + self._rotate_half(k) * sin).contiguous()
        self.stats.torch_rope_calls += 1
        return q, k

    def _cache_write(self, layer_idx: int, k, v, start_pos: int):
        if self._mode("cache") != "torch":
            return super()._cache_write(layer_idx, k, v, start_pos)
        slots = self.torch.arange(
            start_pos,
            start_pos + k.shape[0],
            device=self.device,
            dtype=self.torch.long,
        )
        blocks = slots // self.block_size
        offsets = slots % self.block_size
        self.k_caches[layer_idx][blocks, :, offsets, :] = k.contiguous()
        self.v_caches[layer_idx][blocks, :, offsets, :] = v.contiguous()
        self.stats.torch_cache_write_calls += 1

    def _repeat_kv(self, x):
        if self.n_heads == self.n_kv_heads:
            return x
        return x.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)

    def _attention_prefill(self, q, k, v):
        if self._mode("attention") != "torch":
            return super()._attention_prefill(q, k, v)
        import torch.nn.functional as F

        seq = int(q.shape[0])
        q_t = q.transpose(0, 1).unsqueeze(0)
        k_t = self._repeat_kv(k.transpose(0, 1).unsqueeze(0))
        v_t = self._repeat_kv(v.transpose(0, 1).unsqueeze(0))
        out = F.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            dropout_p=0.0,
            is_causal=True,
            scale=self.attn_scale,
        )
        self.stats.torch_attention_prefill_calls += 1
        return out.squeeze(0).transpose(0, 1).contiguous().view(seq, self.attn_width)

    def _cache_read(self, layer_idx: int, seq_len: int):
        slots = self.torch.arange(seq_len, device=self.device, dtype=self.torch.long)
        blocks = slots // self.block_size
        offsets = slots % self.block_size
        k = self.k_caches[layer_idx][blocks, :, offsets, :].contiguous()
        v = self.v_caches[layer_idx][blocks, :, offsets, :].contiguous()
        return k, v

    def _attention_decode(self, layer_idx: int, q, seq_len: int):
        if self._mode("attention") != "torch":
            return super()._attention_decode(layer_idx, q, seq_len)
        import torch.nn.functional as F

        k, v = self._cache_read(layer_idx, seq_len)
        q_t = q.transpose(0, 1).unsqueeze(0)
        k_t = self._repeat_kv(k.transpose(0, 1).unsqueeze(0))
        v_t = self._repeat_kv(v.transpose(0, 1).unsqueeze(0))
        out = F.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            dropout_p=0.0,
            is_causal=False,
            scale=self.attn_scale,
        )
        self.stats.torch_attention_decode_calls += 1
        return out.squeeze(0).transpose(0, 1).contiguous().view(1, self.attn_width)

    def _mlp(self, layer, h):
        if self._mode("swiglu") != "torch":
            return super()._mlp(layer, h)
        gate = self._linear(layer.mlp.gate_proj, h)
        up = self._linear(layer.mlp.up_proj, h)
        inter = layer.mlp.act_fn(gate) * up
        self.stats.torch_swiglu_calls += 1
        return self._linear(layer.mlp.down_proj, inter)

    def _next_token(self, x_last):
        x_last = self._rms(x_last, self.core.norm)
        logits = self._linear(self.model.lm_head, x_last)
        if self._mode("argmax") != "torch":
            out = self.torch.empty(1, device=self.device, dtype=self.torch.int32)
            self.ks.sampling.argmax(out, logits.contiguous(), 1, int(logits.shape[-1]))
            self.stats.ks_argmax_calls += 1
            return int(out.item()), logits
        self.stats.torch_argmax_calls += 1
        return int(logits.argmax(dim=-1).item()), logits


class KernelSetQwen3BestPracticePath(KernelSetQwen3ConfigurablePath):
    """Kernel-set best-practice path: torch linears, ks memory/attention ops."""

    def __init__(self, model, ks, max_total_tokens: int, block_size: int = 16):
        super().__init__(
            model,
            ks,
            max_total_tokens=max_total_tokens,
            block_size=block_size,
            op_modes=BEST_PRACTICE_MODES,
        )


def _token_match(a: List[int], b: List[int]) -> Dict[str, Any]:
    prefix = 0
    for x, y in zip(a, b):
        if x != y:
            break
        prefix += 1
    return {
        "exact_same_as_reference": a == b,
        "token_match_prefix": prefix,
        "token_overlap": min(len(a), len(b)),
    }


def _run_hf(model, tokenizer, input_ids, attention_mask, new_tokens: int, repeat: int):
    import torch

    def generate():
        with torch.no_grad():
            return model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )

    generate()
    seconds, out = _timer(generate, repeat=repeat)
    tokens = [int(t) for t in out[0].tolist()]
    return {
        "seconds": seconds,
        "tokens_per_s_new": new_tokens / seconds,
        "prompt_tokens": int(input_ids.shape[-1]),
        "new_tokens": new_tokens,
        "tokens": tokens,
        "text": tokenizer.decode(tokens, skip_special_tokens=False),
        "exact_same_as_reference": True,
        "token_match_prefix": len(tokens),
        "token_overlap": len(tokens),
        "scope": "HuggingFace generate",
        "note": "reference",
    }


def _kernel_set_reference_row(
    engine: Dict[str, Any], source_run_id: Optional[str]
) -> Dict[str, Any]:
    row = copy.deepcopy(engine)
    row["historical_baseline"] = True
    source = row.get("source_run_id") or source_run_id
    if source:
        row["source_run_id"] = source
    note = str(row.get("note") or "").strip()
    suffix = (
        f"historical baseline from {source}"
        if source
        else "historical baseline"
    )
    row["note"] = note if suffix in note else (f"{note}; {suffix}" if note else suffix)
    return row


def _load_reference_json(path_or_url: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path_or_url:
        return None
    if path_or_url.startswith(("http://", "https://")):
        with urllib.request.urlopen(path_or_url, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    path = pathlib.Path(path_or_url)
    return json.loads(path.read_text(encoding="utf-8"))


def _run_kernel_set_full(model, tokenizer, input_ids, new_tokens: int, repeat: int, block_size: int):
    import kernel_set as ks

    max_total_tokens = int(input_ids.shape[-1]) + new_tokens

    def generate():
        engine = KernelSetQwen3FullPath(
            model, ks, max_total_tokens=max_total_tokens, block_size=block_size
        )
        with engine.torch.inference_mode():
            tokens, _ = engine.generate(input_ids, max_new_tokens=new_tokens)
        return tokens, engine.stats

    # Warmup once to pay lazy allocations outside the measured run.
    generate()
    seconds, (tokens, stats) = _timer(generate, repeat=repeat)
    return {
        "seconds": seconds,
        "tokens_per_s_new": new_tokens / seconds,
        "prompt_tokens": int(input_ids.shape[-1]),
        "new_tokens": new_tokens,
        "tokens": tokens,
        "text": tokenizer.decode(tokens, skip_special_tokens=False),
        "scope": (
            "single-request Python engine; ks embedding/GEMM/RMSNorm/RoPE/"
            "FlashAttn/reshape_cache/paged_decode/SwiGLU/argmax"
        ),
        "note": "kernel coverage smoke, not production serving",
        "stats": asdict(stats),
        "kernel_coverage": {
            "covered": [
                "ks_embedding_lookup",
                "ks_gemm",
                "ks_rms_norm",
                "ks_rope_gather",
                "ks_flash_attn",
                "ks_reshape_and_cache",
                "ks_paged_attn_decode",
                "ks_swiglu",
                "ks_argmax",
            ],
            "torch_fallback": [
                "residual add",
                "tensor reshape/view/allocation",
                "Python request/decode loop",
                "paged block scheduler",
            ],
        },
    }


def _kernel_coverage_for_modes(modes: Dict[str, str]) -> Dict[str, List[str]]:
    covered: List[str] = []
    fallbacks: List[str] = []
    if modes.get("embedding") == "ks":
        covered.append("ks_embedding_lookup")
    else:
        fallbacks.append("embedding=torch")
    if modes.get("linear") == "ks":
        covered.append("ks_gemm")
    else:
        fallbacks.append("linear=torch/cuBLAS")
    if modes.get("norm") == "ks":
        covered.append("ks_rms_norm")
    else:
        fallbacks.append("norm=torch")
    if modes.get("rope") == "ks":
        covered.append("ks_rope_gather")
    else:
        fallbacks.append("rope=torch")
    if modes.get("attention") == "ks":
        covered.extend(["ks_flash_attn", "ks_paged_attn_decode"])
    else:
        fallbacks.append("attention=torch SDPA/manual decode")
    if modes.get("cache") == "ks":
        covered.append("ks_reshape_and_cache")
    else:
        fallbacks.append("KV write=torch scatter")
    if modes.get("swiglu") == "ks":
        covered.append("ks_swiglu")
    else:
        fallbacks.append("SwiGLU=torch silu*mul")
    if modes.get("argmax") == "ks":
        covered.append("ks_argmax")
    else:
        fallbacks.append("argmax=torch")
    fallbacks.extend(
        [
            "residual add",
            "tensor reshape/view/allocation",
            "Python request/decode loop",
            "paged block scheduler",
        ]
    )
    return {"covered": covered, "torch_fallback": fallbacks}


def _run_kernel_set_variant(
    model,
    tokenizer,
    input_ids,
    new_tokens: int,
    repeat: int,
    block_size: int,
    *,
    modes: Dict[str, str],
    scope: str,
    note: str,
):
    import kernel_set as ks
    import torch

    max_total_tokens = int(input_ids.shape[-1]) + new_tokens

    def make_engine():
        return KernelSetQwen3ConfigurablePath(
            model,
            ks,
            max_total_tokens=max_total_tokens,
            block_size=block_size,
            op_modes=modes,
        )

    # Warm up module loading, cuBLAS algorithm selection, and kernel-set bindings
    # outside the timed region. The measured loop still includes Python decode
    # orchestration and per-layer tensor allocations.
    warm = make_engine()
    with torch.inference_mode():
        warm.generate(input_ids, max_new_tokens=new_tokens)

    engine = make_engine()

    def generate():
        engine.reset_stats()
        with torch.inference_mode():
            tokens, _ = engine.generate(input_ids, max_new_tokens=new_tokens)
        return tokens, engine.stats

    seconds, (tokens, stats) = _timer(generate, repeat=repeat)
    return {
        "seconds": seconds,
        "tokens_per_s_new": new_tokens / seconds,
        "prompt_tokens": int(input_ids.shape[-1]),
        "new_tokens": new_tokens,
        "tokens": tokens,
        "text": tokenizer.decode(tokens, skip_special_tokens=False),
        "scope": scope,
        "note": note,
        "op_modes": modes,
        "stats": asdict(stats),
        "kernel_coverage": _kernel_coverage_for_modes(modes),
    }


def _run_kernel_set_best_practice(
    model, tokenizer, input_ids, new_tokens: int, repeat: int, block_size: int
):
    return _run_kernel_set_variant(
        model,
        tokenizer,
        input_ids,
        new_tokens,
        repeat,
        block_size,
        modes=BEST_PRACTICE_MODES,
        scope=(
            "single-request Python engine; torch/cuBLAS linears + ks "
            "RMSNorm/RoPE/FlashAttn/KV write/paged decode/SwiGLU/argmax"
        ),
        note=(
            "best-practice kernel-set composition; Python loop/allocation and "
            "unfused Q/K/V + gate/up remain"
        ),
    )


def _ablation_row(name: str, note: str, engine: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": name,
        "seconds": engine.get("seconds"),
        "tokens_per_s_new": engine.get("tokens_per_s_new"),
        "exact_same_as_reference": engine.get("exact_same_as_reference"),
        "token_match_prefix": engine.get("token_match_prefix"),
        "token_overlap": engine.get("token_overlap"),
        "op_modes": engine.get("op_modes"),
        "stats": engine.get("stats"),
        "note": note,
    }


def _run_composition_ablation(
    model,
    tokenizer,
    input_ids,
    new_tokens: int,
    repeat: int,
    block_size: int,
    reference_tokens: Optional[List[int]],
    baseline_engine: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    variants: List[Dict[str, Any]] = []
    if baseline_engine is None:
        baseline_engine = _run_kernel_set_best_practice(
            model, tokenizer, input_ids, new_tokens, repeat, block_size
        )
        if reference_tokens is not None:
            baseline_engine.update(_token_match(reference_tokens, baseline_engine["tokens"]))
    variants.append(
        _ablation_row(
            "kernel_set_best_practice",
            "baseline: torch/cuBLAS linears plus ks non-linear/attention/sample kernels",
            baseline_engine,
        )
    )
    baseline_tps = float(baseline_engine.get("tokens_per_s_new") or 0.0)

    for name, overrides, note in ABLATION_VARIANTS:
        modes = _merge_modes(overrides)
        engine = _run_kernel_set_variant(
            model,
            tokenizer,
            input_ids,
            new_tokens,
            repeat,
            block_size,
            modes=modes,
            scope=f"composition ablation: {name}",
            note=note,
        )
        if reference_tokens is not None:
            engine.update(_token_match(reference_tokens, engine["tokens"]))
        row = _ablation_row(name, note, engine)
        if baseline_tps:
            row["vs_best_practice_pct"] = (
                (float(engine["tokens_per_s_new"]) / baseline_tps - 1.0) * 100.0
            )
        variants.append(row)

    for row in variants:
        if "vs_best_practice_pct" not in row:
            row["vs_best_practice_pct"] = 0.0
    return {
        "baseline": "kernel_set_best_practice",
        "variants": variants,
        "note": (
            "Each row changes one component from the best-practice path unless "
            "the name says manual_torch_ops; same prompt, greedy 4-token decode."
        ),
    }


def _run_vllm_subprocess(args, input_token_count: int) -> Optional[Dict[str, Any]]:
    if not args.run_vllm:
        return None
    inner = pathlib.Path("/content/qwen3_vllm_inner.py")
    output = pathlib.Path("/content/qwen3_vllm_result.json")
    vllm_dtype = "bfloat16" if args.dtype == "bf16" else "float16"
    inner.write_text(
        f"""
import json, time, torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from vllm import LLM, SamplingParams

model_id = {args.model!r}
prompt = {args.prompt!r}
new_tokens = {args.new_tokens}
out_path = {str(output)!r}
if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
    PreTrainedTokenizerBase.all_special_tokens_extended = property(
        lambda self: self.all_special_tokens)
tokenizer = AutoTokenizer.from_pretrained(model_id)
llm = LLM(model=model_id, dtype={vllm_dtype!r}, max_model_len=1024, gpu_memory_utilization=0.85)
params = SamplingParams(temperature=0.0, max_tokens=new_tokens)
llm.generate([prompt], params)
torch.cuda.synchronize()
t0 = time.perf_counter()
outs = llm.generate([prompt], params)
torch.cuda.synchronize()
seconds = time.perf_counter() - t0
seq = outs[0].outputs[0]
prompt_ids = tokenizer(prompt, return_tensors='pt').input_ids[0].tolist()
tokens = prompt_ids + list(seq.token_ids)
json.dump({{
  "seconds": seconds,
  "tokens_per_s_new": new_tokens / seconds,
  "prompt_tokens": len(prompt_ids),
  "new_tokens": new_tokens,
  "tokens": tokens,
  "generated_tokens": list(seq.token_ids),
  "text": tokenizer.decode(tokens, skip_special_tokens=False),
  "scope": "vLLM LLM.generate",
  "note": ""
}}, open(out_path, "w"), indent=2)
""",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, str(inner)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        timeout=args.vllm_timeout_s,
    )
    print(proc.stdout[-4000:], flush=True)
    if proc.returncode != 0 or not output.exists():
        print(f"vLLM failed rc={proc.returncode}; omitted from result table", flush=True)
        return None
    return json.loads(output.read_text(encoding="utf-8"))


def run(args) -> Dict[str, Any]:
    _install_deps(include_vllm=args.run_vllm)
    reference = _load_reference_json(args.reference_json_url or args.reference_json)
    repo = _prepare_repo(pathlib.Path(args.repo), args.clone_url, args.repo_ref)
    arch = args.arch or _detect_sm()
    lib = _build_kernel_set(repo, arch)
    os.environ["KERNEL_SET_LIB"] = str(lib)
    os.environ["KERNEL_SET_LIB_DIR"] = str(lib.parent)
    sys.path.insert(0, str(repo / "bindings" / "python"))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map={"": "cuda"},
    ).eval()
    model.requires_grad_(False)

    enc = tokenizer(args.prompt, return_tensors="pt").to("cuda")
    input_ids = enc.input_ids[:, : args.prompt_tokens].contiguous()
    attention_mask = enc.attention_mask[:, : args.prompt_tokens].contiguous()
    prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)

    reference_tokens: Optional[List[int]] = None
    reference_run_id = (
        str(reference.get("run_id"))
        if reference and reference.get("run_id") is not None
        else None
    )
    reference_engines = reference.get("engines") if reference else {}
    if isinstance(reference_engines, dict):
        ref_transformers = reference_engines.get("transformers")
        if isinstance(ref_transformers, dict) and "tokens" in ref_transformers:
            reference_tokens = [int(t) for t in ref_transformers["tokens"]]
    else:
        reference_engines = {}

    engines: Dict[str, Any] = {}
    run_best_practice = bool(args.run_best_practice)
    run_full_kernels = bool(args.run_full_kernels and not args.skip_full_kernels)
    if args.skip_hf:
        if args.merge_reference_engines:
            reference_names = ["transformers", "vllm", "sglang"]
            if not run_full_kernels:
                reference_names.append("kernel_set_full_kernels")
            for name in reference_names:
                engine = reference_engines.get(name)
                if isinstance(engine, dict):
                    engines[name] = _kernel_set_reference_row(engine, reference_run_id)
    else:
        engines["transformers"] = _run_hf(
            model, tokenizer, input_ids, attention_mask, args.new_tokens, args.hf_repeat
        )
        reference_tokens = [int(t) for t in engines["transformers"]["tokens"]]

    def apply_match(name: str) -> None:
        if reference_tokens is not None and name in engines:
            engines[name].update(_token_match(reference_tokens, engines[name]["tokens"]))

    if run_best_practice:
        engines["kernel_set_best_practice"] = _run_kernel_set_best_practice(
            model, tokenizer, input_ids, args.new_tokens, args.ks_repeat, args.block_size
        )
        apply_match("kernel_set_best_practice")

    if run_full_kernels:
        engines["kernel_set_full_kernels"] = _run_kernel_set_full(
            model, tokenizer, input_ids, args.new_tokens, args.ks_repeat, args.block_size
        )
        apply_match("kernel_set_full_kernels")

    optimization_ablation = None
    if args.run_ablation_suite:
        optimization_ablation = _run_composition_ablation(
            model,
            tokenizer,
            input_ids,
            args.new_tokens,
            args.ks_repeat,
            args.block_size,
            reference_tokens,
            engines.get("kernel_set_best_practice"),
        )

    if args.run_vllm:
        del model
        torch.cuda.empty_cache()
        vllm = _run_vllm_subprocess(args, int(input_ids.shape[-1]))
        if vllm is not None:
            if reference_tokens is not None:
                vllm.update(_token_match(reference_tokens, vllm["tokens"]))
            engines["vllm"] = vllm

    props = torch.cuda.get_device_properties(0)
    result = {
        "schema_version": 1,
        "run_id": args.run_id
        or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-qwen3-8b-best-practice"),
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": args.model,
        "prompt": prompt_text,
        "prompt_kind": args.prompt_kind,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_sm": props.major * 10 + props.minor,
        "dtype": args.dtype,
        "new_tokens": args.new_tokens,
        "engines": engines,
        "reference_run_id": reference_run_id,
    }
    if optimization_ablation is not None:
        result["optimization_ablation"] = optimization_ablation
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="/content/kernel-set")
    parser.add_argument("--clone-url", default="https://github.com/cklxx/kernel-set.git")
    parser.add_argument("--repo-ref", default=None)
    parser.add_argument("--arch", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-kind", default="daily Chinese chat")
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--new-tokens", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--hf-repeat", type=int, default=1)
    parser.add_argument("--ks-repeat", type=int, default=1)
    parser.add_argument("--skip-hf", action="store_true")
    parser.add_argument("--run-best-practice", dest="run_best_practice", action="store_true")
    parser.add_argument("--skip-best-practice", dest="run_best_practice", action="store_false")
    parser.set_defaults(run_best_practice=True)
    parser.add_argument("--run-full-kernels", action="store_true")
    parser.add_argument("--skip-full-kernels", action="store_true")
    parser.add_argument("--run-ablation-suite", action="store_true")
    parser.add_argument("--run-vllm", action="store_true")
    parser.add_argument("--vllm-timeout-s", type=float, default=1200.0)
    parser.add_argument("--reference-json", default=None)
    parser.add_argument("--reference-json-url", default=None)
    parser.add_argument("--merge-reference-engines", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output", default="/content/qwen3_engine_compare.json")
    args = parser.parse_args()

    result = run(args)
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("RESULT_JSON_BEGIN", flush=True)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    print("RESULT_JSON_END", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
