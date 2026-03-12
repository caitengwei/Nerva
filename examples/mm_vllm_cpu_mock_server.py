"""CPU mock server for e2e Nerva vs Triton benchmark.

Replaces the vLLM backend with a CPU-only mock that simulates token
generation latency via asyncio.sleep.  All three pipeline stages run
in-process on CPU (backend='pytorch'), so no GPU is required.

Latency model: delay_s = max_tokens * MOCK_TOKEN_LATENCY_MS / 1000
Configure MOCK_TOKEN_LATENCY_MS env var (default: 0.5 ms/token).
At 256 tokens this produces ~128 ms of simulated generation delay,
which is realistic enough to stress the serving layer without a GPU.
"""

from __future__ import annotations

import asyncio
import math
import os
from typing import Any

from nerva import Model, build_nerva_app, model, trace


class MMPreprocessModel(Model):
    """Normalize multimodal inputs into a prompt for the mock LLM."""

    def load(self) -> None:
        pass

    async def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        text = str(inputs.get("text", ""))
        image_bytes = inputs.get("image_bytes", b"")
        image_size = len(image_bytes) if isinstance(image_bytes, bytes) else 0
        max_tokens = max(int(inputs.get("max_tokens", 256)), 1)
        temperature = float(inputs.get("temperature", 1.0))
        top_p = float(inputs.get("top_p", 1.0))
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and >= 0")
        if not math.isfinite(top_p) or top_p <= 0 or top_p > 1:
            raise ValueError("top_p must be finite and in (0, 1]")
        prompt = f"[image_bytes={image_size}]\n{text}".strip()
        return {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }


class MockCPULLMModel(Model):
    """Mock LLM: simulates autoregressive token generation via asyncio.sleep.

    Delay is proportional to max_tokens so the workload behaves like a real
    LLM from the serving layer's perspective (variable latency, not instant).
    Output is a deterministic dummy string of max_tokens 'tok' tokens.
    """

    def load(self) -> None:
        self._token_latency_ms = float(os.environ.get("MOCK_TOKEN_LATENCY_MS", "0.5"))

    async def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        max_tokens = max(int(inputs.get("max_tokens", 256)), 1)
        delay_s = (max_tokens * self._token_latency_ms) / 1000.0
        await asyncio.sleep(delay_s)
        text = " ".join(["tok"] * max_tokens)
        return {"text": text}


class MMPostprocessModel(Model):
    """Normalize mock LLM output schema for benchmark consumers."""

    def load(self) -> None:
        pass

    async def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        text = str(inputs.get("text", "")).strip()
        return {
            "output_text": text,
            "raw": text,
        }


mm_preprocess = model(
    "mm_preprocess",
    MMPreprocessModel,
    backend="pytorch",
    device="cpu",
)
mm_mock_llm = model(
    "mm_mock_llm",
    MockCPULLMModel,
    backend="pytorch",
    device="cpu",
)
mm_postprocess = model(
    "mm_postprocess",
    MMPostprocessModel,
    backend="pytorch",
    device="cpu",
)


def _pipeline(request: Any) -> Any:
    pre_out = mm_preprocess(request)
    llm_out = mm_mock_llm(
        {
            "prompt": pre_out["prompt"],
            "max_tokens": pre_out["max_tokens"],
            "temperature": pre_out["temperature"],
            "top_p": pre_out["top_p"],
        }
    )
    return mm_postprocess({"text": llm_out["text"]})


graph = trace(_pipeline)
app = build_nerva_app({"mm_vllm": graph})
