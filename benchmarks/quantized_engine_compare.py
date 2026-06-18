#!/usr/bin/env python3
"""Real quantized-checkpoint smoke for the generic LLM engine."""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional


DEFAULT_MODEL = "Qwen/Qwen3-8B"
KERNEL_SET_ENGINE = "kernel_set_engine"
DEFAULT_PROMPT = (
    "用户：我在准备一次关于推理引擎优化的内部分享，听众里有模型同学、平台同学，"
    "也有刚加入项目的新同学。请用自然的日常对话方式解释：为什么同一个长一点的"
    "问题，在 prefill、decode、KV cache、RMSNorm、RoPE、MLP 和采样这些阶段会卡在"
    "不同地方；如果要比较不同量化模型，应该怎么避免把加载时间、Python 调度和真实"
    "kernel 时间混在一起。助手："
)


def _run(cmd: List[str], cwd: Optional[pathlib.Path] = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _install_deps() -> None:
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "cmake>=3.24",
            "transformers>=4.51.0",
            "accelerate",
            "sentencepiece",
            "safetensors",
            "bitsandbytes>=0.43.0",
        ]
    )


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
    build = repo / "build-quantized-engine"
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


def _timer(fn, repeat: int = 1):
    _torch_sync()
    t0 = time.perf_counter()
    out = None
    for _ in range(repeat):
        out = fn()
    _torch_sync()
    return (time.perf_counter() - t0) / repeat, out


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


def _gpu_peak_gb() -> Optional[float]:
    import torch

    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated() / (1024**3))


def _reset_peak() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _run_hf(model, tokenizer, input_ids, attention_mask, new_tokens: int, repeat: int):
    import torch

    def generate():
        with torch.inference_mode():
            return model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )

    generate()
    _reset_peak()
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
        "note": "same quantized checkpoint reference",
        "peak_memory_gb": _gpu_peak_gb(),
    }


def _run_kernel_set_engine(
    model,
    tokenizer,
    input_ids,
    new_tokens: int,
    repeat: int,
    block_size: int,
    modes: Dict[str, str],
    scope: str,
    note: str,
):
    import kernel_set as ks
    import torch
    from engines.llm_greedy_engine import (
        KernelSetLLMConfigurablePath,
        kernel_coverage_for_modes,
    )

    max_total_tokens = int(input_ids.shape[-1]) + new_tokens

    def make_engine():
        return KernelSetLLMConfigurablePath(
            model,
            ks,
            max_total_tokens=max_total_tokens,
            block_size=block_size,
            op_modes=modes,
        )

    warm = make_engine()
    with torch.inference_mode():
        warm.generate(input_ids, max_new_tokens=new_tokens)

    engine = make_engine()

    def generate():
        engine.reset_stats()
        with torch.inference_mode():
            tokens, _ = engine.generate(input_ids, max_new_tokens=new_tokens)
        return tokens, engine.stats

    _reset_peak()
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
        "kernel_coverage": kernel_coverage_for_modes(modes),
        "peak_memory_gb": _gpu_peak_gb(),
    }


def _quant_engine_modes(args):
    from engines.llm_greedy_engine import KERNEL_SET_ENGINE_MODES, TORCH_MANUAL_MODES

    variants: Dict[str, Dict[str, str]] = {
        KERNEL_SET_ENGINE: dict(KERNEL_SET_ENGINE_MODES)
    }
    if args.run_torch_attention_engine:
        modes = dict(KERNEL_SET_ENGINE_MODES)
        modes["attention"] = "torch"
        variants["kernel_set_torch_attention"] = modes
    if args.run_manual_torch_engine:
        variants["manual_torch_ops"] = dict(TORCH_MANUAL_MODES)
    return variants


def _quant_config(mode: str, dtype):
    if mode == "bf16":
        return None
    from transformers import BitsAndBytesConfig

    if mode == "bnb_int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    if mode in ("bnb_nf4", "bnb_fp4"):
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4" if mode == "bnb_nf4" else "fp4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    raise ValueError(f"unknown quant mode: {mode}")


def _load_model(model_id: str, mode: str, dtype):
    from transformers import AutoModelForCausalLM

    kwargs: Dict[str, Any] = {
        "low_cpu_mem_usage": True,
        "device_map": {"": "cuda"},
    }
    qc = _quant_config(mode, dtype)
    if qc is None:
        kwargs["torch_dtype"] = dtype
    else:
        kwargs["quantization_config"] = qc
        kwargs["torch_dtype"] = dtype
    try:
        kwargs["attn_implementation"] = "sdpa"
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except TypeError:
        kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    try:
        model.requires_grad_(False)
    except Exception:
        pass
    return model


def _run_one_mode(args, tokenizer, input_ids_cpu, attention_mask_cpu, mode: str):
    import torch

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    input_ids = input_ids_cpu.to("cuda")
    attention_mask = attention_mask_cpu.to("cuda")
    item: Dict[str, Any] = {
        "quant_mode": mode,
        "status": "ok",
        "engines": {},
    }
    try:
        _reset_peak()
        t0 = time.perf_counter()
        model = _load_model(args.model, mode, dtype)
        _torch_sync()
        item["load_seconds"] = time.perf_counter() - t0
        item["load_peak_memory_gb"] = _gpu_peak_gb()
        item["engines"]["transformers"] = _run_hf(
            model,
            tokenizer,
            input_ids,
            attention_mask,
            args.new_tokens,
            args.hf_repeat,
        )
        ref = item["engines"]["transformers"]["tokens"]
        for engine_name, modes in _quant_engine_modes(args).items():
            attention = modes.get("attention")
            all_torch = all(value == "torch" for value in modes.values())
            item["engines"][engine_name] = _run_kernel_set_engine(
                model,
                tokenizer,
                input_ids,
                args.new_tokens,
                args.ks_repeat,
                args.block_size,
                modes,
                scope=(
                    "generic LLM Python engine; quantized/dense linears stay "
                    "on model modules"
                    + (
                        "; torch ops for exactness/control"
                        if all_torch
                        else ", ks covers norm/RoPE/KV write/SwiGLU"
                    )
                    + (
                        "/short decode"
                        if attention == "auto"
                        else "; attention uses torch SDPA for exactness"
                    )
                ),
                note=(
                    "manual torch-op control path, not serving runtime"
                    if all_torch
                    else (
                        "kernel_set_engine provider selection, not serving runtime"
                        if attention == "auto"
                        else "exactness check: torch attention, ks non-attention kernels"
                    )
                ),
            )
            item["engines"][engine_name].update(
                _token_match(ref, item["engines"][engine_name]["tokens"])
            )
        return item
    except Exception as exc:
        item["status"] = "error"
        item["error"] = f"{type(exc).__name__}: {exc}"
        return item
    finally:
        try:
            del model
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()


def run(args) -> Dict[str, Any]:
    _install_deps()
    repo = _prepare_repo(pathlib.Path(args.repo), args.clone_url, args.repo_ref)
    arch = args.arch or _detect_sm()
    lib = _build_kernel_set(repo, arch)
    os.environ["KERNEL_SET_LIB"] = str(lib)
    os.environ["KERNEL_SET_LIB_DIR"] = str(lib.parent)
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "bindings" / "python"))

    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    enc = tokenizer(args.prompt, return_tensors="pt")
    input_ids = enc.input_ids[:, : args.prompt_tokens].contiguous()
    attention_mask = enc.attention_mask[:, : args.prompt_tokens].contiguous()
    prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)

    variants: Dict[str, Any] = {}
    for mode in args.quant_modes:
        variants[mode] = _run_one_mode(args, tokenizer, input_ids, attention_mask, mode)

    props = torch.cuda.get_device_properties(0)
    return {
        "schema_version": 1,
        "kind": "quantized_engine_compare",
        "run_id": args.run_id
        or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-qwen3-8b-quantized-engine"),
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": args.model,
        "prompt": prompt_text,
        "prompt_kind": args.prompt_kind,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_sm": props.major * 10 + props.minor,
        "dtype": args.dtype,
        "new_tokens": args.new_tokens,
        "engine": "KernelSetLLMConfigurablePath",
        "quant_modes": args.quant_modes,
        "engine_variants": list(_quant_engine_modes(args)),
        "variants": variants,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="/content/kernel-set")
    parser.add_argument("--clone-url", default="https://github.com/cklxx/kernel-set.git")
    parser.add_argument("--repo-ref", default=None)
    parser.add_argument("--arch", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-kind", default="long daily Chinese chat")
    parser.add_argument("--prompt-tokens", type=int, default=160)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument(
        "--quant-modes",
        nargs="+",
        default=["bf16", "bnb_int8", "bnb_nf4", "bnb_fp4"],
        choices=["bf16", "bnb_int8", "bnb_nf4", "bnb_fp4"],
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--hf-repeat", type=int, default=1)
    parser.add_argument("--ks-repeat", type=int, default=1)
    parser.add_argument("--run-torch-attention-engine", action="store_true", default=True)
    parser.add_argument("--skip-torch-attention-engine", dest="run_torch_attention_engine", action="store_false")
    parser.add_argument("--run-manual-torch-engine", action="store_true", default=True)
    parser.add_argument("--skip-manual-torch-engine", dest="run_manual_torch_engine", action="store_false")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output", default="/content/quantized_engine_compare.json")
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
