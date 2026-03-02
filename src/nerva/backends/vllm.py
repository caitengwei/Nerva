"""VLLMBackend — thin wrapper around vllm.AsyncLLMEngine.

Import of this module succeeds without vllm installed.
ImportError is deferred to load_model() to support optional dependency.

Inputs dict keys:
    prompt (str): Input prompt.
    max_tokens (int, optional): Default 256.
    temperature (float, optional): Default 1.0.
    top_p (float, optional): Default 1.0.

Unary output dict keys:
    text (str): Final generated text.

Streaming output dict keys (per chunk):
    text (str): Cumulative text so far.
    finished (bool): True on final chunk.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import structlog

from nerva.backends.base import Backend, BatchMeta, InferContext, ModelConfig
from nerva.backends.registry import register_backend

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = structlog.get_logger(__name__)


@register_backend("vllm")
class VLLMBackend(Backend):
    """Backend wrapping vllm.AsyncLLMEngine for LLM text generation.

    Design decisions:
    - Bypasses Nerva DynamicBatcher; vLLM has continuous batching internally.
    - Deadline enforced via asyncio.wait_for().
    - SHM not used; vLLM manages its own GPU memory (PagedAttention).
    """

    def __init__(self) -> None:
        self._engine: Any | None = None
        self._config: ModelConfig | None = None

    @property
    def is_loaded(self) -> bool:
        return self._engine is not None

    def health_check(self) -> bool:
        return self._engine is not None

    async def load_model(self, config: ModelConfig) -> None:
        """Initialize vllm.AsyncLLMEngine.

        Args:
            config: backend_options must contain 'model_path' (str).
                    Optional: 'tensor_parallel_size' (int), 'max_model_len' (int).

        Raises:
            ImportError: vllm not installed.
            ValueError: model_path missing from backend_options.
            RuntimeError: model already loaded.
        """
        try:
            import vllm
        except ImportError as exc:
            raise ImportError(
                "vllm is required for VLLMBackend. "
                "Install it with: pip install 'nerva[vllm]'"
            ) from exc

        if self._engine is not None:
            name = self._config.model_name if self._config else "?"
            raise RuntimeError(f"Model already loaded: {name}. Call unload_model() first.")

        model_path: str | None = config.backend_options.get("model_path")
        if not model_path:
            raise ValueError("VLLMBackend requires 'model_path' in backend_options.")

        engine_args: dict[str, Any] = {
            "model": model_path,
            "tensor_parallel_size": config.backend_options.get("tensor_parallel_size", 1),
        }
        max_model_len: int | None = config.backend_options.get("max_model_len")
        if max_model_len is not None:
            engine_args["max_model_len"] = max_model_len

        logger.info("Loading VLLMBackend", model=config.model_name, device=config.device)
        self._engine = vllm.AsyncLLMEngine.from_engine_args(vllm.AsyncEngineArgs(**engine_args))
        self._config = config
        logger.info("VLLMBackend loaded", model=config.model_name)

    async def unload_model(self) -> None:
        if self._engine is None:
            return
        name = self._config.model_name if self._config else "?"
        logger.info("Unloading VLLMBackend", model=name)
        self._engine = None
        self._config = None

    async def infer(
        self,
        inputs: dict[str, Any],
        context: InferContext,
        batch_meta: BatchMeta | None = None,
    ) -> dict[str, Any]:
        """Unary inference: consume all tokens, return final text."""
        engine = self._ensure_loaded()
        prompt: str = inputs.get("prompt", "")
        sampling_params = self._build_sampling_params(inputs)

        async def _consume() -> dict[str, Any]:
            final_text = ""
            async for output in engine.generate(prompt, sampling_params, context.request_id):
                if output.outputs:
                    final_text = output.outputs[0].text
            return {"text": final_text}

        try:
            return await asyncio.wait_for(_consume(), timeout=context.deadline_ms / 1000.0)
        except TimeoutError as exc:
            raise RuntimeError("DEADLINE_EXCEEDED") from exc

    async def infer_stream(
        self,
        inputs: dict[str, Any],
        context: InferContext,
        batch_meta: BatchMeta | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Streaming inference: yield chunks as they arrive.

        Deadline is enforced: if total elapsed time exceeds context.deadline_ms,
        a RuntimeError("DEADLINE_EXCEEDED") is raised.
        """
        engine = self._ensure_loaded()
        prompt: str = inputs.get("prompt", "")
        sampling_params = self._build_sampling_params(inputs)
        deadline_s = context.deadline_ms / 1000.0

        try:
            async with asyncio.timeout(deadline_s):
                async for output in engine.generate(
                    prompt, sampling_params, context.request_id
                ):
                    if output.outputs:
                        yield {
                            "text": output.outputs[0].text,
                            "finished": output.finished,
                        }
        except TimeoutError as exc:
            raise RuntimeError("DEADLINE_EXCEEDED") from exc

    def _ensure_loaded(self) -> Any:
        if self._engine is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        return self._engine

    def _build_sampling_params(self, inputs: dict[str, Any]) -> Any:
        """Build vllm.SamplingParams from inputs dict."""
        import vllm
        return vllm.SamplingParams(
            max_tokens=inputs.get("max_tokens", 256),
            temperature=inputs.get("temperature", 1.0),
            top_p=inputs.get("top_p", 1.0),
        )
