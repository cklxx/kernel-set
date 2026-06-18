#!/usr/bin/env python3
"""Small engine loops shared by AR text, VLM prefill, and diffusion smoke specs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class GreedyResult:
    model: str
    prompt_tokens: int
    new_tokens: int
    tokens: tuple[int, ...]
    generated_tokens: tuple[int, ...]
    text: str
    generated_text: str


@dataclass(frozen=True)
class DiffusionResult:
    model: str
    steps: int
    artifact: Any


class GreedySpec(Protocol):
    name: str

    def encode(self, text: str) -> list[int]: ...

    def decode(self, tokens: Sequence[int]) -> str: ...

    def prefill(self, tokens: Sequence[int], *, image_embeds: Sequence[Any] = ()) -> Any: ...

    def decode_one(self, state: Any, tokens: Sequence[int]) -> tuple[int, Any]: ...


class DiffusionSpec(Protocol):
    name: str

    def condition(self, prompt: str) -> Any: ...

    def init_latents(self, conditioning: Any) -> Any: ...

    def denoise_step(self, latents: Any, conditioning: Any, step: int, steps: int) -> Any: ...

    def decode_latents(self, latents: Any) -> Any: ...


class TinyEngine:
    """ponytail: one-loop prototype; replace specs before adding runtime features."""

    def greedy(
        self,
        spec: GreedySpec,
        prompt: str,
        *,
        max_new_tokens: int,
        image_embeds: Sequence[Any] = (),
    ) -> GreedyResult:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        prompt_tokens = spec.encode(prompt)
        tokens = list(prompt_tokens)
        generated: list[int] = []
        state = spec.prefill(prompt_tokens, image_embeds=image_embeds)
        for _ in range(max_new_tokens):
            token, state = spec.decode_one(state, tokens)
            token = int(token)
            tokens.append(token)
            generated.append(token)
        return GreedyResult(
            model=spec.name,
            prompt_tokens=len(prompt_tokens),
            new_tokens=len(generated),
            tokens=tuple(tokens),
            generated_tokens=tuple(generated),
            text=spec.decode(tokens),
            generated_text=spec.decode(generated),
        )

    def diffuse(self, spec: DiffusionSpec, prompt: str, *, steps: int) -> DiffusionResult:
        if steps <= 0:
            raise ValueError("steps must be positive")
        conditioning = spec.condition(prompt)
        latents = spec.init_latents(conditioning)
        for step in range(steps):
            latents = spec.denoise_step(latents, conditioning, step, steps)
        return DiffusionResult(spec.name, steps, spec.decode_latents(latents))
