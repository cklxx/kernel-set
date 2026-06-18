#!/usr/bin/env python3
"""Stdlib smoke for the tiny cross-family engine loop."""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engines import TinyEngine  # noqa: E402


PROMPT = (
    "用户：我下周要在团队会上做一次关于推理引擎优化的分享，听众里有做模型的、"
    "有做平台的，也有刚加入项目的新同学。我希望开场不要太像论文汇报，而是先用"
    "一个真实的线上请求讲清楚问题：为什么同样是一个日常对话，prefill、decode、"
    "KV cache、MoE 路由和采样这些环节会分别卡在不同地方。请帮我组织一段自然、"
    "不夸张、但足够具体的回答。助手："
)


class ScriptedGreedySpec:
    def __init__(self, name: str, script: str):
        self.name = name
        self.script = script

    def encode(self, text: str) -> list[int]:
        return [ord(ch) for ch in text]

    def decode(self, tokens) -> str:
        return "".join(chr(int(t)) for t in tokens)

    def prefill(self, tokens, *, image_embeds=()):
        return {"cursor": 0, "images": len(image_embeds)}

    def decode_one(self, state, tokens):
        prefix = "先把图像摘要并入上下文，再" if state["images"] else ""
        script = prefix + self.script
        token = ord(script[state["cursor"] % len(script)])
        state = dict(state)
        state["cursor"] += 1
        return token, state


class ScriptedDiffusionSpec:
    name = "toy-diffusion"

    def condition(self, prompt: str):
        return prompt[:32]

    def init_latents(self, conditioning):
        return ["noise", f"cond={len(conditioning)}"]

    def denoise_step(self, latents, conditioning, step: int, steps: int):
        return [*latents, f"{step + 1}/{steps}"]

    def decode_latents(self, latents):
        return " -> ".join(latents)


def main() -> None:
    engine = TinyEngine()
    assert len(PROMPT) > 120

    answer = (
        "可以从一个用户问题进入：先说明首 token 前主要在搬权重、算 attention 和写 KV，"
        "随后每个 token 都是在读缓存、跑少量矩阵和做采样。这样模型同学能看到算子形状，"
        "平台同学能看到调度边界，新同学也能把一次请求拆成可测的几段。"
    )
    for name in ("qwen3.5-llm", "gemma-llm"):
        out = engine.greedy(ScriptedGreedySpec(name, answer), PROMPT, max_new_tokens=96)
        assert out.prompt_tokens == len(PROMPT)
        assert out.new_tokens == 96
        assert out.generated_text.startswith("可以从一个用户问题")

    vlm = engine.greedy(
        ScriptedGreedySpec("gemma-vlm", answer),
        PROMPT,
        max_new_tokens=96,
        image_embeds=[("slide-preview", 4)],
    )
    assert vlm.generated_text.startswith("先把图像摘要并入上下文")

    image = engine.diffuse(ScriptedDiffusionSpec(), PROMPT, steps=6)
    assert image.artifact.endswith("6/6")
    print("tiny_engine_smoke ok")


if __name__ == "__main__":
    main()
