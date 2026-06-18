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


def _install_deps(
    include_vllm: bool, include_sglang: bool, sglang_package: str
) -> None:
    pkgs = [
        "cmake>=3.24",
        "transformers>=4.51.0",
        "accelerate",
        "sentencepiece",
        "safetensors",
    ]
    if include_vllm:
        pkgs.append("vllm==0.10.2")
    if include_sglang:
        pkgs.append(sglang_package)
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


def _bench_cuda(fn, warmup: int = 20, iters: int = 100) -> Tuple[float, Any]:
    import torch

    out = None
    with torch.inference_mode():
        for _ in range(warmup):
            out = fn()
        _torch_sync()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            out = fn()
        end.record()
        torch.cuda.synchronize()
    return float(start.elapsed_time(end) * 1000.0 / iters), out


def _rms_eps(mod) -> float:
    return float(getattr(mod, "variance_epsilon", getattr(mod, "eps", 1e-6)))


def _rel_err(a, b) -> float:
    a = a.float()
    b = b.float()
    denom = b.abs().max().clamp_min(1e-12)
    return float(((a - b).abs().max() / denom).item())


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
        self.positions_i32 = torch.arange(max_rope, device=self.device, dtype=torch.int32)
        self.positions_long = torch.arange(max_rope, device=self.device, dtype=torch.long)
        self.seq_lens_i32 = torch.arange(
            1, self.max_total_tokens + 1, device=self.device, dtype=torch.int32
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
        positions = self.positions_i32[start_pos : start_pos + q.shape[0]]
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
        slot_mapping = self.positions_i32[start_pos : start_pos + k.shape[0]]
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
        seq_lens = self.seq_lens_i32[seq_len - 1 : seq_len]
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
    "embedding": "auto",
    "linear": "torch",
    "norm": "ks",
    "rope": "ks",
    "cache": "ks",
    "attention": "auto",
    "swiglu": "ks",
    "argmax": "torch",
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
        "ks_embedding",
        {"embedding": "ks"},
        "force kernel-set embedding lookup for every token shape",
    ),
    (
        "torch_embedding",
        {"embedding": "torch"},
        "force torch embedding lookup for every token shape",
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
        "ks_attention",
        {"attention": "ks"},
        "force kernel-set FlashAttn/paged decode for every attention shape",
    ),
    (
        "torch_attention",
        {"attention": "torch"},
        "force torch SDPA/manual decode for every attention shape",
    ),
    (
        "torch_swiglu",
        {"swiglu": "torch"},
        "replace ks SwiGLU with torch silu(gate)*up",
    ),
    (
        "ks_argmax",
        {"argmax": "ks"},
        "force kernel-set argmax instead of torch argmax",
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
        self.auto_decode_torch_min_ctx = 512

    def _mode(self, name: str) -> str:
        return self.op_modes.get(name, "ks")

    def _embedding(self, input_ids):
        embedding_mode = self._mode("embedding")
        if embedding_mode == "ks" or (
            embedding_mode == "auto" and int(input_ids.numel()) > 1
        ):
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
        positions = self.positions_long[start_pos : start_pos + q.shape[0]]
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
        slots = self.positions_long[start_pos : start_pos + k.shape[0]]
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
        if self._mode("attention") == "ks":
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
        slots = self.positions_long[:seq_len]
        blocks = slots // self.block_size
        offsets = slots % self.block_size
        k = self.k_caches[layer_idx][blocks, :, offsets, :].contiguous()
        v = self.v_caches[layer_idx][blocks, :, offsets, :].contiguous()
        return k, v

    def _attention_decode(self, layer_idx: int, q, seq_len: int):
        attention_mode = self._mode("attention")
        if attention_mode == "ks" or (
            attention_mode == "auto" and seq_len < self.auto_decode_torch_min_ctx
        ):
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
    """Kernel-set best-practice path with shape-aware provider selection."""

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
    elif modes.get("embedding") == "auto":
        covered.append("ks_embedding_lookup(auto multi-token)")
        fallbacks.append("embedding single-token=torch")
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
    elif modes.get("attention") == "auto":
        covered.append("ks_paged_attn_decode(auto short-context)")
        fallbacks.append("attention prefill/long-context=torch SDPA")
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
            "RMSNorm/RoPE/KV write/short-decode/SwiGLU + shape-aware embedding/attention"
        ),
        note=(
            "shape-aware best-practice composition from Qwen3 kernel microbench; "
            "Python loop/allocation and unfused Q/K/V + gate/up remain"
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
            "baseline: torch/cuBLAS linears plus ks non-linear/cache kernels and shape-aware embedding/attention",
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


def _run_kernel_microbench(model, input_ids, args) -> Dict[str, Any]:
    import kernel_set as ks
    import torch
    import torch.nn.functional as F

    torch.manual_seed(1234)
    config = model.config
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    hidden = int(config.hidden_size)
    inter = int(config.intermediate_size)
    qh = int(config.num_attention_heads)
    kvh = int(getattr(config, "num_key_value_heads", qh))
    hd = int(getattr(config, "head_dim", hidden // qh))
    block = int(args.block_size)
    prompt_len = int(input_ids.shape[-1])
    vocab = int(config.vocab_size)
    scale = hd ** -0.5
    first_attn = model.model.layers[0].self_attn
    q_norm_w = first_attn.q_norm.weight.contiguous()
    k_norm_w = first_attn.k_norm.weight.contiguous()
    hidden_norm_w = model.model.layers[0].input_layernorm.weight.contiguous()
    max_pos = max(4096, max(args.kernel_ctx_sweep or [prompt_len + args.new_tokens]) + 8)
    cos, sin = _make_rope_cache(
        max_pos,
        hd,
        float(getattr(config, "rope_theta", 1000000.0)),
        device,
        dtype,
    )

    rows: List[Dict[str, Any]] = []

    def bench_pair(
        *,
        op: str,
        shape: str,
        ks_fn,
        ref_fn,
        ref_impl: str,
        note: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            with torch.inference_mode():
                ks_out = ks_fn()
                ref_out = ref_fn()
                _torch_sync()
            exact = False
            if isinstance(ks_out, tuple):
                rel = max(_rel_err(a, b) for a, b in zip(ks_out, ref_out))
            elif ks_out.dtype in (torch.int32, torch.int64, torch.long):
                exact = bool(torch.equal(ks_out, ref_out))
                rel = 0.0 if exact else 1.0
            else:
                rel = _rel_err(ks_out, ref_out)
            ks_us, _ = _bench_cuda(ks_fn, args.kernel_bench_warmup, args.kernel_bench_iters)
            ref_us, _ = _bench_cuda(ref_fn, args.kernel_bench_warmup, args.kernel_bench_iters)
            speedup = ref_us / ks_us if ks_us else None
            row = {
                "op": op,
                "shape": shape,
                "ks_us": ks_us,
                "ref_us": ref_us,
                "ref_impl": ref_impl,
                "winner": "kernel-set" if speedup is not None and speedup >= 1.0 else ref_impl,
                "speedup_ref_over_ks": speedup,
                "rel_err": rel,
                "exact": exact,
                "status": "ok",
                "note": note,
            }
            if extra:
                row.update(extra)
            rows.append(row)
        except Exception as exc:
            rows.append({
                "op": op,
                "shape": shape,
                "ref_impl": ref_impl,
                "status": "error",
                "note": f"{type(exc).__name__}: {exc}",
            })

    def torch_rms(x, w, eps):
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + eps)
        return (y.to(x.dtype) * w).contiguous()

    def rotate_half(x):
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    def torch_rope(q, k, positions):
        c = cos.index_select(0, positions)
        s = sin.index_select(0, positions)
        c = torch.cat((c, c), dim=-1).unsqueeze(1)
        s = torch.cat((s, s), dim=-1).unsqueeze(1)
        return (
            (q * c + rotate_half(q) * s).contiguous(),
            (k * c + rotate_half(k) * s).contiguous(),
        )

    def repeat_kv(x):
        if kvh == qh:
            return x
        return x.repeat_interleave(qh // kvh, dim=1)

    def torch_prefill(q, k, v, causal=True):
        qt = q.transpose(1, 2)
        kt = repeat_kv(k.transpose(1, 2))
        vt = repeat_kv(v.transpose(1, 2))
        out = F.scaled_dot_product_attention(
            qt, kt, vt, dropout_p=0.0, is_causal=causal, scale=scale
        )
        return out.transpose(1, 2).contiguous()

    def gather_cache(k_cache, v_cache, num_seqs: int, ctx_len: int, blocks_per_seq: int):
        k = k_cache.permute(0, 2, 1, 3).reshape(
            num_seqs, blocks_per_seq * block, kvh, hd
        )[:, :ctx_len].contiguous()
        v = v_cache.permute(0, 2, 1, 3).reshape(
            num_seqs, blocks_per_seq * block, kvh, hd
        )[:, :ctx_len].contiguous()
        return k, v

    def torch_decode(q, k_dense, v_dense):
        qt = q.unsqueeze(1).transpose(1, 2)
        kt = repeat_kv(k_dense.transpose(1, 2))
        vt = repeat_kv(v_dense.transpose(1, 2))
        out = F.scaled_dot_product_attention(
            qt, kt, vt, dropout_p=0.0, is_causal=False, scale=scale
        )
        return out.transpose(1, 2).squeeze(1).contiguous()

    for tokens in [1, prompt_len]:
        ids = torch.randint(0, vocab, (tokens,), device=device, dtype=torch.long)
        out = torch.empty(tokens, hidden, device=device, dtype=dtype)
        bench_pair(
            op="embedding_lookup",
            shape=f"tokens={tokens},hidden={hidden}",
            ks_fn=lambda ids=ids, out=out, tokens=tokens: (
                ks.embedding.embedding_lookup(
                    out, model.model.embed_tokens.weight.contiguous(), ids, tokens, hidden
                ),
                out,
            )[1],
            ref_fn=lambda ids=ids: model.model.embed_tokens(ids).contiguous(),
            ref_impl="torch_embedding",
        )

    for label, rows_n, width, weight in [
        ("decode_hidden", 1, hidden, hidden_norm_w),
        ("prefill_hidden", prompt_len, hidden, hidden_norm_w),
        ("decode_q_norm", qh, hd, q_norm_w),
        ("decode_k_norm", kvh, hd, k_norm_w),
        ("prefill_q_norm", prompt_len * qh, hd, q_norm_w),
        ("prefill_k_norm", prompt_len * kvh, hd, k_norm_w),
    ]:
        x = torch.randn(rows_n, width, device=device, dtype=dtype)
        out = torch.empty_like(x)
        eps = _rms_eps(first_attn.q_norm if width == hd else model.model.layers[0].input_layernorm)
        bench_pair(
            op="rms_norm",
            shape=f"{label},rows={rows_n},hidden={width}",
            ks_fn=lambda x=x, out=out, weight=weight, eps=eps: (
                ks.norm.rms_norm(out, x.contiguous(), weight, eps=eps),
                out,
            )[1],
            ref_fn=lambda x=x, weight=weight, eps=eps: torch_rms(x, weight, eps),
            ref_impl="torch_rms",
        )

    for tokens in [1, prompt_len]:
        q = torch.randn(tokens, qh, hd, device=device, dtype=dtype)
        k = torch.randn(tokens, kvh, hd, device=device, dtype=dtype)
        q_ks = q.clone()
        k_ks = k.clone()
        positions = torch.arange(tokens, device=device, dtype=torch.int32)
        bench_pair(
            op="rope_gather",
            shape=f"tokens={tokens},qh={qh},kvh={kvh},hd={hd}",
            ks_fn=lambda q=q_ks, k=k_ks, positions=positions, tokens=tokens: (
                ks.rope.rope_gather(
                    q, k, cos, sin, positions, tokens, qh, kvh, hd, interleaved=False
                ),
                (q, k),
            )[1],
            ref_fn=lambda q=q, k=k, positions=positions: torch_rope(q, k, positions.long()),
            ref_impl="torch_rope",
        )

    for tokens in [1, prompt_len]:
        max_blocks = (tokens + block - 1) // block
        k_cache = torch.empty(max_blocks, kvh, block, hd, device=device, dtype=dtype)
        v_cache = torch.empty_like(k_cache)
        k_cache_ref = torch.empty_like(k_cache)
        v_cache_ref = torch.empty_like(k_cache)
        key = torch.randn(tokens, kvh, hd, device=device, dtype=dtype)
        value = torch.randn(tokens, kvh, hd, device=device, dtype=dtype)
        slots = torch.arange(tokens, device=device, dtype=torch.int32)
        slots_long = slots.long()

        def written_slots(kc, vc, slots=slots_long):
            flat_k = kc.permute(0, 2, 1, 3).reshape(-1, kvh, hd)
            flat_v = vc.permute(0, 2, 1, 3).reshape(-1, kvh, hd)
            return (
                flat_k[slots].contiguous(),
                flat_v[slots].contiguous(),
            )

        def ref_cache_write(
            kc=k_cache_ref,
            vc=v_cache_ref,
            key=key,
            value=value,
            slots=slots_long,
        ):
            blocks = slots // block
            offsets = slots % block
            kc[blocks, :, offsets, :] = key
            vc[blocks, :, offsets, :] = value
            return written_slots(kc, vc, slots)

        bench_pair(
            op="reshape_and_cache",
            shape=f"tokens={tokens},kvh={kvh},hd={hd},block={block}",
            ks_fn=lambda kc=k_cache, vc=v_cache, key=key, value=value, slots=slots, tokens=tokens: (
                ks.attention.reshape_and_cache(
                    kc, vc, key.contiguous(), value.contiguous(), slots, tokens, kvh, hd, block
                ),
                written_slots(kc, vc),
            )[1],
            ref_fn=ref_cache_write,
            ref_impl="torch_scatter",
        )

    for seq in [prompt_len, 128, 512]:
        q = torch.randn(1, seq, qh, hd, device=device, dtype=dtype)
        k = torch.randn(1, seq, kvh, hd, device=device, dtype=dtype)
        v = torch.randn(1, seq, kvh, hd, device=device, dtype=dtype)
        out = torch.empty(1, seq, qh, hd, device=device, dtype=dtype)
        bench_pair(
            op="flash_attn_prefill",
            shape=f"b=1,seq={seq},qh={qh},kvh={kvh},hd={hd}",
            ks_fn=lambda out=out, q=q, k=k, v=v, seq=seq: (
                ks.attention.flash_attn(
                    out, q, k, v, 1, seq, seq, qh, kvh, hd,
                    softmax_scale=scale, causal=True
                ),
                out,
            )[1],
            ref_fn=lambda q=q, k=k, v=v: torch_prefill(q, k, v, causal=True),
            ref_impl="torch_sdpa",
        )

    ctx_sweep = args.kernel_ctx_sweep or [prompt_len + 1, prompt_len + args.new_tokens, 128, 512, 2048]
    decode_shapes = [(1, int(ctx)) for ctx in ctx_sweep]
    if args.kernel_include_batch_decode:
        decode_shapes.append((64, 2048))
    for num_seqs, ctx_len in decode_shapes:
        blocks_per_seq = (ctx_len + block - 1) // block
        total_blocks = num_seqs * blocks_per_seq
        q = torch.randn(num_seqs, qh, hd, device=device, dtype=dtype)
        k_cache = torch.randn(total_blocks, kvh, block, hd, device=device, dtype=dtype)
        v_cache = torch.randn(total_blocks, kvh, block, hd, device=device, dtype=dtype)
        block_tables = torch.arange(total_blocks, device=device, dtype=torch.int32).view(
            num_seqs, blocks_per_seq
        )
        seq_lens = torch.full((num_seqs,), ctx_len, device=device, dtype=torch.int32)
        out = torch.empty(num_seqs, qh, hd, device=device, dtype=dtype)
        k_dense, v_dense = gather_cache(k_cache, v_cache, num_seqs, ctx_len, blocks_per_seq)

        def ref_gather_sdpa(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            num_seqs=num_seqs,
            ctx_len=ctx_len,
            blocks_per_seq=blocks_per_seq,
        ):
            kd, vd = gather_cache(k_cache, v_cache, num_seqs, ctx_len, blocks_per_seq)
            return torch_decode(q, kd, vd)

        dense_us, _ = _bench_cuda(
            lambda q=q, kd=k_dense, vd=v_dense: torch_decode(q, kd, vd),
            args.kernel_bench_warmup,
            args.kernel_bench_iters,
        )
        bench_pair(
            op="paged_attn_decode",
            shape=f"seqs={num_seqs},ctx={ctx_len},qh={qh},kvh={kvh},hd={hd},block={block}",
            ks_fn=lambda out=out, q=q, kc=k_cache, vc=v_cache, bt=block_tables, sl=seq_lens, bps=blocks_per_seq, ns=num_seqs: (
                ks.attention.paged_attn_decode(
                    out, q, kc, vc, bt, sl, ns, qh, kvh, hd, block, bps, softmax_scale=scale
                ),
                out,
            )[1],
            ref_fn=ref_gather_sdpa,
            ref_impl="torch_gather_sdpa",
            extra={"torch_dense_sdpa_us": dense_us},
        )

    for rows_n in [1, prompt_len]:
        gate = torch.randn(rows_n, inter, device=device, dtype=dtype)
        up = torch.randn(rows_n, inter, device=device, dtype=dtype)
        out = torch.empty_like(gate)
        bench_pair(
            op="swiglu",
            shape=f"rows={rows_n},inter={inter}",
            ks_fn=lambda out=out, gate=gate, up=up: (
                ks.activation.swiglu(out, gate.contiguous(), up.contiguous()),
                out,
            )[1],
            ref_fn=lambda gate=gate, up=up: (F.silu(gate) * up).contiguous(),
            ref_impl="torch_silu_mul",
        )

    logits = torch.randn(1, vocab, device=device, dtype=dtype)
    out_i32 = torch.empty(1, device=device, dtype=torch.int32)
    bench_pair(
        op="argmax",
        shape=f"rows=1,vocab={vocab}",
        ks_fn=lambda out=out_i32, logits=logits: (
            ks.sampling.argmax(out, logits.contiguous(), 1, vocab),
            out,
        )[1],
        ref_fn=lambda logits=logits: logits.argmax(dim=-1).to(torch.int32),
        ref_impl="torch_argmax",
    )

    return {
        "schema_version": 1,
        "timing": "cuda_events",
        "warmup": args.kernel_bench_warmup,
        "iters": args.kernel_bench_iters,
        "model_shape": {
            "prompt_tokens": prompt_len,
            "hidden": hidden,
            "intermediate": inter,
            "num_heads": qh,
            "num_kv_heads": kvh,
            "head_dim": hd,
            "block_size": block,
            "vocab_size": vocab,
        },
        "rows": rows,
    }


def _subprocess_work_dir(args) -> pathlib.Path:
    out = pathlib.Path(args.output)
    work_dir = out.parent if str(out.parent) else pathlib.Path("/content")
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


def _subprocess_env_with_cuda_libs() -> Dict[str, str]:
    env = dict(os.environ)
    candidates: List[pathlib.Path] = []
    try:
        import site

        site_roots = site.getsitepackages()
        user_site = site.getusersitepackages()
        if user_site:
            site_roots.append(user_site)
    except Exception:
        site_roots = []
    for root in site_roots:
        nvidia = pathlib.Path(root) / "nvidia"
        if nvidia.exists():
            candidates.extend(nvidia.glob("*/lib"))
            candidates.extend(nvidia.glob("*/lib64"))
    candidates.extend(
        pathlib.Path(path)
        for path in (
            "/usr/local/cuda/lib64",
            "/usr/local/cuda/targets/x86_64-linux/lib",
            "/usr/local/cuda-12.8/lib64",
            "/usr/local/cuda-12.8/targets/x86_64-linux/lib",
        )
    )
    existing = [str(path) for path in candidates if path.exists()]
    current = env.get("LD_LIBRARY_PATH")
    if current:
        existing.append(current)
    if existing:
        env["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(existing))
    return env


def _write_text_only_sitecustomize(work_dir: pathlib.Path) -> pathlib.Path:
    stub_dir = work_dir / "sglang_text_only_site"
    stub_dir.mkdir(parents=True, exist_ok=True)
    (stub_dir / "sitecustomize.py").write_text(
        r'''
import enum
import importlib.machinery
import sys
import types


def _missing_vision_fn(name):
    def fn(*_args, **_kwargs):
        raise RuntimeError(f"torchvision.{name} is unavailable in this text-only run")

    fn.__name__ = name.rsplit(".", 1)[-1]
    return fn


def _module_getattr(prefix):
    def get(name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _missing_vision_fn(f"{prefix}.{name}")

    return get


class ImageReadMode(enum.Enum):
    UNCHANGED = "UNCHANGED"
    GRAY = "GRAY"
    GRAY_ALPHA = "GRAY_ALPHA"
    RGB = "RGB"
    RGB_ALPHA = "RGB_ALPHA"


class InterpolationMode(enum.Enum):
    NEAREST = "nearest"
    NEAREST_EXACT = "nearest-exact"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    BOX = "box"
    HAMMING = "hamming"
    LANCZOS = "lanczos"


tv = types.ModuleType("torchvision")
tv_io = types.ModuleType("torchvision.io")
tv_transforms = types.ModuleType("torchvision.transforms")
tv_transforms_functional = types.ModuleType("torchvision.transforms.functional")
tv_transforms_v2 = types.ModuleType("torchvision.transforms.v2")
tv_transforms_v2_functional = types.ModuleType("torchvision.transforms.v2.functional")

tv_io.decode_jpeg = _missing_vision_fn("io.decode_jpeg")
tv_io.decode_image = _missing_vision_fn("io.decode_image")
tv_io.ImageReadMode = ImageReadMode
tv_io.__getattr__ = _module_getattr("io")

tv_transforms.InterpolationMode = InterpolationMode
tv_transforms.functional = tv_transforms_functional
tv_transforms.v2 = tv_transforms_v2
tv_transforms.__getattr__ = _module_getattr("transforms")
tv_transforms_functional.__getattr__ = _module_getattr("transforms.functional")
tv_transforms_v2.functional = tv_transforms_v2_functional
tv_transforms_v2.__getattr__ = _module_getattr("transforms.v2")
tv_transforms_v2_functional.__getattr__ = _module_getattr("transforms.v2.functional")
for name in ("pil_to_tensor", "to_pil_image", "to_tensor", "resize", "center_crop", "normalize"):
    setattr(tv_transforms_functional, name, _missing_vision_fn(name))

tv.io = tv_io
tv.transforms = tv_transforms
tv.__version__ = "0.0.text_only_stub"
tv.__path__ = []
tv_transforms.__path__ = []
tv_transforms_v2.__path__ = []

for module in (tv, tv_io, tv_transforms, tv_transforms_functional, tv_transforms_v2, tv_transforms_v2_functional):
    module.__file__ = "<torchvision_text_only_stub>"
tv.__spec__ = importlib.machinery.ModuleSpec("torchvision", loader=None)
tv_io.__spec__ = importlib.machinery.ModuleSpec("torchvision.io", loader=None)
tv_transforms.__spec__ = importlib.machinery.ModuleSpec("torchvision.transforms", loader=None)
tv_transforms_functional.__spec__ = importlib.machinery.ModuleSpec("torchvision.transforms.functional", loader=None)
tv_transforms_v2.__spec__ = importlib.machinery.ModuleSpec("torchvision.transforms.v2", loader=None)
tv_transforms_v2_functional.__spec__ = importlib.machinery.ModuleSpec("torchvision.transforms.v2.functional", loader=None)

sys.modules["torchvision"] = tv
sys.modules["torchvision.io"] = tv_io
sys.modules["torchvision.transforms"] = tv_transforms
sys.modules["torchvision.transforms.functional"] = tv_transforms_functional
sys.modules["torchvision.transforms.v2"] = tv_transforms_v2
sys.modules["torchvision.transforms.v2.functional"] = tv_transforms_v2_functional

tc = types.ModuleType("torchcodec")
tc_decoders = types.ModuleType("torchcodec.decoders")


class VideoDecoder:
    def __init__(self, *_args, **_kwargs):
        raise RuntimeError("torchcodec video decode is unavailable in this text-only run")


tc_decoders.VideoDecoder = VideoDecoder
tc.decoders = tc_decoders
tc.__path__ = []
tc.__file__ = "<torchcodec_text_only_stub>"
tc_decoders.__file__ = "<torchcodec_text_only_stub>"
tc.__spec__ = importlib.machinery.ModuleSpec("torchcodec", loader=None)
tc_decoders.__spec__ = importlib.machinery.ModuleSpec("torchcodec.decoders", loader=None)
sys.modules["torchcodec"] = tc
sys.modules["torchcodec.decoders"] = tc_decoders
''',
        encoding="utf-8",
    )
    return stub_dir


def _run_vllm_subprocess(args, prompt_text: str) -> Optional[Dict[str, Any]]:
    if not args.run_vllm:
        return None
    work_dir = _subprocess_work_dir(args)
    inner = work_dir / "qwen3_vllm_inner.py"
    output = work_dir / "qwen3_vllm_result.json"
    vllm_dtype = "bfloat16" if args.dtype == "bf16" else "float16"
    inner.write_text(
        f"""
import json, time, torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from vllm import LLM, SamplingParams

model_id = {args.model!r}
prompt = {prompt_text!r}
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


def _run_sglang_subprocess(args, prompt_text: str) -> Optional[Dict[str, Any]]:
    if not args.run_sglang:
        return None
    work_dir = _subprocess_work_dir(args)
    stub_dir = _write_text_only_sitecustomize(work_dir)
    inner = work_dir / "qwen3_sglang_inner.py"
    output = work_dir / "qwen3_sglang_result.json"
    sglang_dtype = "bfloat16" if args.dtype == "bf16" else "float16"
    inner.write_text(
        f"""
import json, signal, time, torch
from transformers import AutoTokenizer

model_id = {args.model!r}
prompt = {prompt_text!r}
new_tokens = {args.new_tokens}
dtype = {sglang_dtype!r}
out_path = {str(output)!r}
tokenizer = AutoTokenizer.from_pretrained(model_id)
prompt_ids = tokenizer(prompt, return_tensors='pt').input_ids[0].tolist()


def _engine_class():
    import sglang as sgl
    if hasattr(sgl, "Engine"):
        return sgl.Engine
    from sglang.srt.entrypoints.engine import Engine
    return Engine


def _make_engine():
    engine_cls = _engine_class()
    attempts = [
        {{"model_path": model_id, "dtype": dtype, "mem_fraction_static": 0.85, "context_length": 1024}},
        {{"model_path": model_id, "dtype": dtype, "mem_fraction_static": 0.85}},
        {{"model_path": model_id, "dtype": dtype}},
        {{"model_path": model_id}},
        {{"model": model_id, "dtype": dtype, "mem_fraction_static": 0.85, "context_length": 1024}},
        {{"model": model_id, "dtype": dtype, "mem_fraction_static": 0.85}},
        {{"model": model_id, "dtype": dtype}},
        {{"model": model_id}},
    ]
    last_error = None
    for kwargs in attempts:
        try:
            return engine_cls(**kwargs)
        except TypeError as exc:
            last_error = exc
    raise last_error


def _generate(llm):
    sampling_params = {{"temperature": 0.0, "max_new_tokens": new_tokens}}
    attempts = [
        lambda: llm.generate([prompt], sampling_params),
        lambda: llm.generate(prompt, sampling_params),
        lambda: llm.generate([prompt], sampling_params=sampling_params),
        lambda: llm.generate(prompt, sampling_params=sampling_params),
    ]
    last_error = None
    for fn in attempts:
        try:
            return fn()
        except TypeError as exc:
            last_error = exc
    raise last_error


def _get(obj, key):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _first(obj):
    if isinstance(obj, (list, tuple)) and obj:
        return obj[0]
    return obj


def _as_int_list(value):
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        value = value[0]
    if not isinstance(value, (list, tuple)):
        return None
    try:
        return [int(x) for x in value]
    except Exception:
        return None


def _extract_output(result):
    item = _first(result)
    nested = _first(_get(item, "outputs"))
    meta = _get(item, "meta_info")
    nested_meta = _get(nested, "meta_info")
    text = (
        _get(item, "text")
        or _get(nested, "text")
        or _get(item, "output")
        or _get(nested, "output")
        or ""
    )
    for source in (item, nested, meta, nested_meta):
        ids = _as_int_list(_get(source, "output_ids"))
        if ids is None:
            ids = _as_int_list(_get(source, "token_ids"))
        if ids is None:
            ids = _as_int_list(_get(source, "output_token_ids"))
        if ids is None:
            ids = _as_int_list(_get(source, "completion_token_ids"))
        if ids is not None:
            return text, ids
    return text, None


class _ShutdownTimeout(Exception):
    pass


def _shutdown_timeout_handler(_signum, _frame):
    raise _ShutdownTimeout()


def _shutdown(llm):
    old_handler = signal.signal(signal.SIGALRM, _shutdown_timeout_handler)
    signal.alarm(30)
    try:
        for name in ("shutdown", "release", "close"):
            fn = getattr(llm, name, None)
            if callable(fn):
                try:
                    fn()
                except _ShutdownTimeout:
                    print("SGLang shutdown timed out; continuing", flush=True)
                except Exception:
                    pass
                return
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def main():
    llm = _make_engine()
    try:
        _generate(llm)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outs = _generate(llm)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        generated_text, generated_ids = _extract_output(outs)
        if generated_ids is not None:
            if generated_ids[:len(prompt_ids)] == prompt_ids:
                tokens = generated_ids
                completion_ids = generated_ids[len(prompt_ids):]
            else:
                completion_ids = generated_ids
                tokens = prompt_ids + completion_ids
        else:
            full_text = generated_text if generated_text.startswith(prompt) else prompt + generated_text
            tokens = tokenizer(full_text, return_tensors='pt').input_ids[0].tolist()
            completion_ids = tokens[len(prompt_ids):]
        json.dump({{
          "seconds": seconds,
          "tokens_per_s_new": new_tokens / seconds,
          "prompt_tokens": len(prompt_ids),
          "new_tokens": new_tokens,
          "tokens": tokens,
          "generated_tokens": completion_ids,
          "text": tokenizer.decode(tokens, skip_special_tokens=False),
          "generated_text": generated_text,
          "scope": "SGLang Engine.generate",
          "note": ""
        }}, open(out_path, "w"), indent=2)
    finally:
        _shutdown(llm)


if __name__ == "__main__":
    main()
""",
        encoding="utf-8",
    )
    env = _subprocess_env_with_cuda_libs()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(stub_dir)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    proc = subprocess.run(
        [sys.executable, str(inner)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        timeout=args.sglang_timeout_s,
        env=env,
    )
    print(proc.stdout[-4000:], flush=True)
    if proc.returncode != 0 or not output.exists():
        print(f"SGLang failed rc={proc.returncode}; omitted from result table", flush=True)
        return None
    return json.loads(output.read_text(encoding="utf-8"))


def run(args) -> Dict[str, Any]:
    run_best_practice = bool(args.run_best_practice)
    run_full_kernels = bool(args.run_full_kernels and not args.skip_full_kernels)
    needs_kernel_set = (
        run_best_practice
        or run_full_kernels
        or bool(args.run_ablation_suite)
        or bool(args.run_kernel_microbench)
    )
    needs_hf_model = (not args.skip_hf) or needs_kernel_set

    _install_deps(
        include_vllm=args.run_vllm,
        include_sglang=args.run_sglang,
        sglang_package=args.sglang_package,
    )
    reference = _load_reference_json(args.reference_json_url or args.reference_json)
    if needs_kernel_set:
        repo = _prepare_repo(pathlib.Path(args.repo), args.clone_url, args.repo_ref)
        arch = args.arch or _detect_sm()
        lib = _build_kernel_set(repo, arch)
        os.environ["KERNEL_SET_LIB"] = str(lib)
        os.environ["KERNEL_SET_LIB_DIR"] = str(lib.parent)
        sys.path.insert(0, str(repo / "bindings" / "python"))

    import torch
    from transformers import AutoTokenizer

    torch.manual_seed(0)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    enc = tokenizer(args.prompt, return_tensors="pt")
    input_ids = enc.input_ids[:, : args.prompt_tokens].contiguous()
    attention_mask = enc.attention_mask[:, : args.prompt_tokens].contiguous()
    prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    model = None
    if needs_hf_model:
        from transformers import AutoModelForCausalLM

        dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map={"": "cuda"},
        ).eval()
        model.requires_grad_(False)
        input_ids = input_ids.to("cuda")
        attention_mask = attention_mask.to("cuda")

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
    if args.skip_hf:
        if args.merge_reference_engines:
            reference_names = ["transformers", "vllm", "sglang"]
            if not run_best_practice:
                reference_names.append("kernel_set_best_practice")
            if not run_full_kernels:
                reference_names.append("kernel_set_full_kernels")
            for name in reference_names:
                engine = reference_engines.get(name)
                if isinstance(engine, dict):
                    engines[name] = _kernel_set_reference_row(engine, reference_run_id)
    else:
        assert model is not None
        engines["transformers"] = _run_hf(
            model, tokenizer, input_ids, attention_mask, args.new_tokens, args.hf_repeat
        )
        reference_tokens = [int(t) for t in engines["transformers"]["tokens"]]

    def apply_match(name: str) -> None:
        if reference_tokens is not None and name in engines:
            engines[name].update(_token_match(reference_tokens, engines[name]["tokens"]))

    if run_best_practice:
        assert model is not None
        engines["kernel_set_best_practice"] = _run_kernel_set_best_practice(
            model, tokenizer, input_ids, args.new_tokens, args.ks_repeat, args.block_size
        )
        apply_match("kernel_set_best_practice")

    if run_full_kernels:
        assert model is not None
        engines["kernel_set_full_kernels"] = _run_kernel_set_full(
            model, tokenizer, input_ids, args.new_tokens, args.ks_repeat, args.block_size
        )
        apply_match("kernel_set_full_kernels")

    optimization_ablation = None
    if args.run_ablation_suite:
        assert model is not None
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

    kernel_microbench = None
    if args.run_kernel_microbench:
        assert model is not None
        kernel_microbench = _run_kernel_microbench(model, input_ids, args)
    elif args.merge_reference_engines and isinstance(reference, dict):
        ref_microbench = reference.get("kernel_microbench")
        if isinstance(ref_microbench, dict):
            kernel_microbench = copy.deepcopy(ref_microbench)

    if (
        optimization_ablation is None
        and args.merge_reference_engines
        and isinstance(reference, dict)
    ):
        ref_ablation = reference.get("optimization_ablation")
        if isinstance(ref_ablation, dict):
            optimization_ablation = copy.deepcopy(ref_ablation)

    def release_model() -> None:
        nonlocal model
        if model is not None:
            del model
            model = None
            torch.cuda.empty_cache()

    if args.run_vllm:
        release_model()
        vllm = _run_vllm_subprocess(args, prompt_text)
        if vllm is not None:
            if reference_tokens is not None:
                vllm.update(_token_match(reference_tokens, vllm["tokens"]))
            engines["vllm"] = vllm

    if args.run_sglang:
        release_model()
        sglang = _run_sglang_subprocess(args, prompt_text)
        if sglang is not None:
            if reference_tokens is not None:
                sglang.update(_token_match(reference_tokens, sglang["tokens"]))
            engines["sglang"] = sglang

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
    if kernel_microbench is not None:
        result["kernel_microbench"] = kernel_microbench
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
    parser.add_argument("--run-kernel-microbench", action="store_true")
    parser.add_argument("--kernel-bench-warmup", type=int, default=20)
    parser.add_argument("--kernel-bench-iters", type=int, default=100)
    parser.add_argument("--kernel-ctx-sweep", type=int, nargs="*", default=None)
    parser.add_argument("--kernel-include-batch-decode", action="store_true")
    parser.add_argument("--run-vllm", action="store_true")
    parser.add_argument("--vllm-timeout-s", type=float, default=1200.0)
    parser.add_argument("--run-sglang", action="store_true")
    parser.add_argument("--sglang-timeout-s", type=float, default=1800.0)
    parser.add_argument("--sglang-package", default="sglang")
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
