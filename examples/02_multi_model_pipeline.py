"""Example: Tokenizer → LLM → Detokenizer pipeline.

Demonstrates a text generation pipeline using Nerva's DAG execution.
Uses a toy LLM stub — replace with VLLMBackend for production use.

Run (CPU, no GPU):
    uv run python examples/02_multi_model_pipeline.py

Production usage with vLLM:
    llm = model(LLMModel, name="llm", backend="vllm", device="cuda:0",
                backend_options={"model_path": "/path/to/model"})
"""

from __future__ import annotations

import asyncio
from typing import Any

from nerva import model, trace
from nerva.core.model import Model
from nerva.observability.logging import configure_logging

configure_logging(dev=True)


class TokenizerModel(Model):
    def load(self) -> None:
        pass

    async def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        text: str = inputs["text"]
        return {"tokens": text.split(), "prompt": text}


class LLMModel(Model):
    """Toy LLM stub. Replace with VLLMBackend in production."""

    def load(self) -> None:
        pass

    async def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        prompt: str = inputs["prompt"]
        return {"raw_output": f"[LLM response to: {prompt[:40]}]"}


class DetokenizerModel(Model):
    def load(self) -> None:
        pass

    async def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"text": inputs["raw_output"].strip()}


tokenizer_h = model("tokenizer", TokenizerModel, device="cpu")
llm_h = model("llm", LLMModel, device="cpu")
detokenizer_h = model("detokenizer", DetokenizerModel, device="cpu")


def pipeline(text_input: Any) -> Any:
    tok_out = tokenizer_h(text_input)
    llm_out = llm_h(tok_out)
    return detokenizer_h(llm_out)


if __name__ == "__main__":
    async def main() -> None:
        graph = trace(pipeline)
        print(f"Pipeline graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        # To serve:
        # import nerva
        # nerva.serve({"text_gen": graph}, host="0.0.0.0", port=8080)

    asyncio.run(main())
