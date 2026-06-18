#!/usr/bin/env python3
"""Single-request causal-LM greedy decode engine built from kernel-set primitives.

This is intentionally a small engine, not a serving runtime. It owns the
causal attention/KV-cache/decode loop so benchmark scripts can stay focused on
measurement and reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


def rms_eps(mod) -> float:
    return float(getattr(mod, "variance_epsilon", getattr(mod, "eps", 1e-6)))


def make_rope_cache(max_pos: int, head_dim: int, theta: float, device, dtype):
    import torch

    inv_freq = 1.0 / (
        theta
        ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(max_pos, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos().to(dtype).contiguous(), freqs.sin().to(dtype).contiguous()


def _resolve_core(model):
    for name in ("model", "language_model", "text_model"):
        core = getattr(model, name, None)
        if core is not None and hasattr(core, "layers") and hasattr(core, "embed_tokens"):
            return core
    if hasattr(model, "layers") and hasattr(model, "embed_tokens"):
        return model
    raise TypeError(
        "KernelSetCausalLMEngine expects a HF causal-LM core with layers and embed_tokens"
    )


def _first_tensor(model, *, floating: bool = False):
    import torch

    for tensor in list(model.parameters()) + list(model.buffers()):
        if floating and not torch.is_floating_point(tensor):
            continue
        return tensor
    raise ValueError("model has no tensors")


def _activation_dtype(model, core):
    import torch

    weight = getattr(getattr(core, "embed_tokens", None), "weight", None)
    if weight is not None and torch.is_floating_point(weight):
        return weight.dtype
    for tensor in list(model.parameters()) + list(model.buffers()):
        if torch.is_floating_point(tensor):
            return tensor.dtype
    raise ValueError("model has no floating-point tensor for activation dtype")


def _module_attr(module, names: tuple[str, ...]):
    for name in names:
        value = getattr(module, name, None)
        if value is not None:
            return value
    raise AttributeError(f"{type(module).__name__} missing any of {names}")


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


def merge_modes(overrides: Dict[str, str]) -> Dict[str, str]:
    modes = dict(BEST_PRACTICE_MODES)
    modes.update(overrides)
    return modes


def kernel_coverage_for_modes(modes: Dict[str, str]) -> Dict[str, List[str]]:
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


class KernelSetCausalLMFullPath:
    """Single-request Llama-family causal-LM path using kernel-set kernels."""

    def __init__(self, model, ks, max_total_tokens: int, block_size: int = 16):
        import torch

        self.torch = torch
        self.model = model.eval()
        self.core = _resolve_core(model)
        self.layers = list(self.core.layers)
        self.config = model.config
        self.ks = ks
        self.stats = KernelStats()
        self.device = _first_tensor(model).device
        self.dtype = _activation_dtype(model, self.core)
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
        self.cos, self.sin = make_rope_cache(
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
            out,
            x.contiguous(),
            norm_mod.weight.to(dtype=x.dtype).contiguous(),
            eps=rms_eps(norm_mod),
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
        out = self.torch.empty(
            (1, seq, self.n_heads, self.head_dim),
            device=self.device,
            dtype=self.dtype,
        )
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
        out = self.torch.empty(
            (1, self.n_heads, self.head_dim), device=self.device, dtype=self.dtype
        )
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

    def _final_norm(self):
        return _module_attr(self.core, ("norm", "final_layernorm"))

    def _input_norm(self, layer):
        return _module_attr(layer, ("input_layernorm", "pre_attention_layernorm"))

    def _post_attention_norm(self, layer):
        return _module_attr(
            layer,
            (
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_attention_norm",
            ),
        )

    def _next_token(self, x_last):
        x_last = self._rms(x_last, self._final_norm())
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
            h = self._rms(x, self._input_norm(layer))
            q = self._linear(attn.q_proj, h).view(seq, self.n_heads, self.head_dim)
            k = self._linear(attn.k_proj, h).view(seq, self.n_kv_heads, self.head_dim)
            v = self._linear(attn.v_proj, h).view(seq, self.n_kv_heads, self.head_dim)
            q, k = self._qk_norm(attn, q, k)
            q, k = self._rope(q, k, 0)
            self._cache_write(li, k, v, 0)
            ctx = self._attention_prefill(q, k, v)
            x = residual + self._linear(attn.o_proj, ctx)
            residual = x
            h = self._rms(x, self._post_attention_norm(layer))
            x = residual + self._mlp(layer, h)
        return self._next_token(x[-1:].contiguous())

    def decode_one(self, token_id: int, pos: int):
        ids = self.torch.tensor([[token_id]], device=self.device, dtype=self.torch.long)
        x = self._embedding(ids)
        for li, layer in enumerate(self.layers):
            attn = layer.self_attn
            residual = x
            h = self._rms(x, self._input_norm(layer))
            q = self._linear(attn.q_proj, h).view(1, self.n_heads, self.head_dim)
            k = self._linear(attn.k_proj, h).view(1, self.n_kv_heads, self.head_dim)
            v = self._linear(attn.v_proj, h).view(1, self.n_kv_heads, self.head_dim)
            q, k = self._qk_norm(attn, q, k)
            q, k = self._rope(q, k, pos)
            self._cache_write(li, k, v, pos)
            ctx = self._attention_decode(li, q, seq_len=pos + 1)
            x = residual + self._linear(attn.o_proj, ctx)
            residual = x
            h = self._rms(x, self._post_attention_norm(layer))
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


class KernelSetCausalLMConfigurablePath(KernelSetCausalLMFullPath):
    """Single-request causal-LM path with per-component ks/torch switches."""

    def __init__(
        self,
        model,
        ks,
        max_total_tokens: int,
        block_size: int = 16,
        op_modes: Optional[Dict[str, str]] = None,
    ):
        super().__init__(model, ks, max_total_tokens=max_total_tokens, block_size=block_size)
        self.op_modes = merge_modes(op_modes or {})
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
        if hasattr(layer.mlp, "act_fn"):
            inter = layer.mlp.act_fn(gate) * up
        else:
            import torch.nn.functional as F

            inter = F.silu(gate) * up
        self.stats.torch_swiglu_calls += 1
        return self._linear(layer.mlp.down_proj, inter)

    def _next_token(self, x_last):
        x_last = self._rms(x_last, self._final_norm())
        logits = self._linear(self.model.lm_head, x_last)
        if self._mode("argmax") != "torch":
            out = self.torch.empty(1, device=self.device, dtype=self.torch.int32)
            self.ks.sampling.argmax(out, logits.contiguous(), 1, int(logits.shape[-1]))
            self.stats.ks_argmax_calls += 1
            return int(out.item()), logits
        self.stats.torch_argmax_calls += 1
        return int(logits.argmax(dim=-1).item()), logits


class KernelSetCausalLMBestPracticePath(KernelSetCausalLMConfigurablePath):
    """Kernel-set best-practice path with shape-aware provider selection."""

    def __init__(self, model, ks, max_total_tokens: int, block_size: int = 16):
        super().__init__(
            model,
            ks,
            max_total_tokens=max_total_tokens,
            block_size=block_size,
            op_modes=BEST_PRACTICE_MODES,
        )


__all__ = [
    "ABLATION_VARIANTS",
    "BEST_PRACTICE_MODES",
    "KernelSetCausalLMBestPracticePath",
    "KernelSetCausalLMConfigurablePath",
    "KernelSetCausalLMFullPath",
    "KernelStats",
    "TORCH_MANUAL_MODES",
    "kernel_coverage_for_modes",
    "make_rope_cache",
    "merge_modes",
    "rms_eps",
]
