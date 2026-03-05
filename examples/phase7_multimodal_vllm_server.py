"""Phase 7 benchmark server: multimodal preprocess -> vLLM -> postprocess."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from nerva import Model, build_nerva_app, model, trace

if TYPE_CHECKING:
    from nerva.core.graph import Graph


class MMPreprocessModel(Model):
    """Normalize multimodal inputs into a prompt for vLLM."""

    def load(self) -> None:
        pass

    async def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        text = str(inputs.get("text", ""))
        image_bytes = inputs.get("image_bytes", b"")
        image_size = len(image_bytes) if isinstance(image_bytes, bytes) else 0
        max_tokens = max(int(inputs.get("max_tokens", 256)), 1)
        temperature = max(float(inputs.get("temperature", 1.0)), 0.0)
        top_p = max(float(inputs.get("top_p", 1.0)), 1e-6)
        prompt = f"[image_bytes={image_size}]\n{text}".strip()
        return {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }


class MMVLLMPlaceholderModel(Model):
    """Placeholder class for vLLM backend declaration."""

    def load(self) -> None:
        pass

    async def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("MMVLLMPlaceholderModel should run with backend='vllm'")


class MMPostprocessModel(Model):
    """Normalize vLLM output schema for benchmark consumers."""

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
mm_vllm = model(
    "mm_vllm",
    MMVLLMPlaceholderModel,
    backend="vllm",
    device="cuda:0",
    model_path=os.environ.get("PHASE7_VLLM_MODEL_PATH", "Qwen/Qwen2.5-0.5B-Instruct"),
)
mm_postprocess = model(
    "mm_postprocess",
    MMPostprocessModel,
    backend="pytorch",
    device="cpu",
)


def _phase7_pipeline(request: Any) -> Any:
    pre_out = mm_preprocess(request)
    llm_out = mm_vllm(
        {
            "prompt": pre_out["prompt"],
            "max_tokens": pre_out["max_tokens"],
            "temperature": pre_out["temperature"],
            "top_p": pre_out["top_p"],
        }
    )
    return mm_postprocess({"text": llm_out["text"]})


def build_phase7_graph() -> Graph:
    return trace(_phase7_pipeline)


graph = build_phase7_graph()
app = build_nerva_app({"phase7_mm_vllm": graph})
