"""Tests for VLLMBackend — all tests run without real vllm (GPU not required).

Strategy: mock sys.modules["vllm"] with MagicMock to simulate vllm API.
VLLMBackend uses conditional import in load_model() only.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from nerva.backends.base import InferContext, ModelConfig


def _ctx(request_id: str = "req-1", deadline_ms: int = 30000) -> InferContext:
    return InferContext(request_id=request_id, deadline_ms=deadline_ms)


def _config(model_path: str = "/models/llm", device: str = "cuda:0") -> ModelConfig:
    return ModelConfig(
        model_name="test-llm",
        model_class=object,
        device=device,
        backend_options={"model_path": model_path},
    )


def _make_vllm_mock(text_chunks: list[str] | None = None) -> Any:
    """Build a vllm mock that streams the given text chunks."""
    chunks = text_chunks or ["Hello", " world!"]

    async def _generate(prompt: str, sampling_params: Any, request_id: str):  # type: ignore[no-untyped-def]
        for i, _text in enumerate(chunks):
            output = MagicMock()
            output.outputs = [MagicMock()]
            output.outputs[0].text = "".join(chunks[: i + 1])
            output.finished = i == len(chunks) - 1
            yield output

    engine = MagicMock()
    engine.generate = _generate

    vllm_mod = MagicMock()
    vllm_mod.AsyncLLMEngine.from_engine_args = MagicMock(return_value=engine)
    vllm_mod.AsyncEngineArgs = MagicMock(return_value=MagicMock())
    vllm_mod.SamplingParams = MagicMock(return_value=MagicMock())
    return vllm_mod


class TestVLLMBackendImport:
    def test_import_does_not_crash_without_vllm(self) -> None:
        """Importing VLLMBackend must not crash even if vllm is absent.

        Since vllm is only imported inside load_model(), the module-level import
        is unconditionally safe. We verify this by checking the class is accessible.
        """
        from nerva.backends.vllm import VLLMBackend
        assert VLLMBackend is not None

    def test_vllm_backend_class_accessible(self) -> None:
        from nerva.backends.vllm import VLLMBackend
        assert VLLMBackend is not None


class TestVLLMBackendLifecycle:
    async def test_is_loaded_false_before_load(self) -> None:
        from nerva.backends.vllm import VLLMBackend
        assert not VLLMBackend().is_loaded

    async def test_health_check_false_before_load(self) -> None:
        from nerva.backends.vllm import VLLMBackend
        assert not VLLMBackend().health_check()

    async def test_load_model_sets_is_loaded(self) -> None:
        from nerva.backends.vllm import VLLMBackend
        with patch.dict("sys.modules", {"vllm": _make_vllm_mock()}):
            backend = VLLMBackend()
            await backend.load_model(_config())
            assert backend.is_loaded

    async def test_load_model_without_vllm_raises_import_error(self) -> None:
        from nerva.backends.vllm import VLLMBackend
        with patch.dict("sys.modules", {"vllm": None}):  # type: ignore[arg-type]
            backend = VLLMBackend()
            with pytest.raises((ImportError, RuntimeError)):
                await backend.load_model(_config())

    async def test_unload_model_clears_is_loaded(self) -> None:
        from nerva.backends.vllm import VLLMBackend
        with patch.dict("sys.modules", {"vllm": _make_vllm_mock()}):
            backend = VLLMBackend()
            await backend.load_model(_config())
            await backend.unload_model()
            assert not backend.is_loaded

    async def test_load_model_missing_model_path_raises(self) -> None:
        from nerva.backends.vllm import VLLMBackend
        with patch.dict("sys.modules", {"vllm": _make_vllm_mock()}):
            backend = VLLMBackend()
            bad_config = ModelConfig(
                model_name="bad", model_class=object, device="cpu",
                backend_options={},  # no model_path
            )
            with pytest.raises(ValueError, match="model_path"):
                await backend.load_model(bad_config)


class TestVLLMBackendInfer:
    async def test_infer_returns_text_key(self) -> None:
        from nerva.backends.vllm import VLLMBackend
        with patch.dict("sys.modules", {"vllm": _make_vllm_mock(["Hello", " world!"])}):
            backend = VLLMBackend()
            await backend.load_model(_config())
            result = await backend.infer({"prompt": "Hi"}, _ctx())
            assert "text" in result
            assert isinstance(result["text"], str)

    async def test_infer_without_load_raises_runtime_error(self) -> None:
        from nerva.backends.vllm import VLLMBackend
        with pytest.raises(RuntimeError, match="No model loaded"):
            await VLLMBackend().infer({"prompt": "test"}, _ctx())

    async def test_infer_deadline_exceeded_raises(self) -> None:
        from nerva.backends.vllm import VLLMBackend

        async def _slow_generate(prompt: str, params: Any, req_id: str):  # type: ignore[no-untyped-def]
            await asyncio.sleep(10)
            yield MagicMock()

        vllm_mock = _make_vllm_mock()
        vllm_mock.AsyncLLMEngine.from_engine_args.return_value.generate = _slow_generate
        with patch.dict("sys.modules", {"vllm": vllm_mock}):
            backend = VLLMBackend()
            await backend.load_model(_config())
            with pytest.raises((RuntimeError, asyncio.TimeoutError, TimeoutError)):
                await backend.infer({"prompt": "test"}, _ctx(deadline_ms=50))

    async def test_infer_stream_yields_chunks(self) -> None:
        from nerva.backends.vllm import VLLMBackend
        with patch.dict("sys.modules", {"vllm": _make_vllm_mock(["A", "B", "C"])}):
            backend = VLLMBackend()
            await backend.load_model(_config())
            chunks = [
                chunk async for chunk in backend.infer_stream({"prompt": "Hi"}, _ctx())
            ]
            assert len(chunks) == 3
            assert all("text" in c for c in chunks)
            assert chunks[-1].get("finished") is True


class TestVLLMBackendRegistered:
    def test_registered_as_vllm(self) -> None:
        import nerva.backends.vllm  # noqa: F401 — triggers @register_backend
        from nerva.backends.registry import list_backends
        assert "vllm" in list_backends()
