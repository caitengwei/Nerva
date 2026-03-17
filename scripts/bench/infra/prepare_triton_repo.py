from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


PREPROCESS_MODEL = "mm_preprocess"
INFER_MODEL = "mm_infer"
POSTPROCESS_MODEL = "mm_postprocess"


def _python_backend_config(
    model_name: str,
    *,
    inputs: list[tuple[str, str]],
    outputs: list[tuple[str, str]],
    instance_count: int | None = None,
) -> str:
    input_entries = ",\n".join(
        f'  {{ name: "{name}" data_type: {dtype} dims: [ 1 ] }}' for name, dtype in inputs
    )
    output_entries = ",\n".join(
        f'  {{ name: "{name}" data_type: {dtype} dims: [ 1 ] }}' for name, dtype in outputs
    )

    instance_group = ""
    if instance_count is not None:
        instance_group = (
            "instance_group [\n"
            f"  {{ kind: KIND_CPU count: {instance_count} }}\n"
            "]\n"
        )

    return (
        f'name: "{model_name}"\n'
        'backend: "python"\n'
        # Keep batching disabled for full-e2e comparability and to match the scalar
        # request handling implemented in generated Python backend stages.
        'max_batch_size: 0\n'
        f"{instance_group}"
        'input [\n'
        f"{input_entries}\n"
        ']\n'
        'output [\n'
        f"{output_entries}\n"
        ']\n'
    )


def _infer_model_config(model_name: str) -> str:
    return (
        f'name: "{model_name}"\n'
        'backend: "python"\n'
        'max_batch_size: 0\n'
        "model_transaction_policy {\n"
        "  decoupled: true\n"
        "}\n"
        'input [\n'
        '  { name: "PROMPT" data_type: TYPE_STRING dims: [ 1 ] },\n'
        '  { name: "MAX_TOKENS" data_type: TYPE_INT32 dims: [ 1 ] },\n'
        '  { name: "TEMPERATURE" data_type: TYPE_FP32 dims: [ 1 ] },\n'
        '  { name: "TOP_P" data_type: TYPE_FP32 dims: [ 1 ] },\n'
        '  { name: "DEADLINE_MS" data_type: TYPE_INT32 dims: [ 1 ] },\n'
        '  { name: "STREAM" data_type: TYPE_BOOL dims: [ 1 ] optional: true }\n'
        ']\n'
        'output [\n'
        '  { name: "TEXT" data_type: TYPE_STRING dims: [ 1 ] }\n'
        ']\n'
    )


def _ensemble_config(model_name: str) -> str:
    input_entries = ",\n".join(
        [
            '  { name: "TEXT" data_type: TYPE_STRING dims: [ 1 ] }',
            '  { name: "IMAGE_BYTES" data_type: TYPE_STRING dims: [ 1 ] }',
            '  { name: "MAX_TOKENS" data_type: TYPE_INT32 dims: [ 1 ] }',
            '  { name: "TEMPERATURE" data_type: TYPE_FP32 dims: [ 1 ] }',
            '  { name: "TOP_P" data_type: TYPE_FP32 dims: [ 1 ] }',
            '  { name: "DEADLINE_MS" data_type: TYPE_INT32 dims: [ 1 ] }',
            '  { name: "STREAM" data_type: TYPE_BOOL dims: [ 1 ] optional: true }',
        ]
    )
    output_entries = ",\n".join(
        [
            '  { name: "OUTPUT_TEXT" data_type: TYPE_STRING dims: [ 1 ] }',
            '  { name: "RAW" data_type: TYPE_STRING dims: [ 1 ] }',
        ]
    )
    return (
        f'name: "{model_name}"\n'
        'platform: "ensemble"\n'
        # Keep batching disabled for full-e2e comparability and to match the scalar
        # request handling implemented in generated Python backend stages.
        'max_batch_size: 0\n'
        "model_transaction_policy {\n"
        "  decoupled: true\n"
        "}\n"
        'input [\n'
        f"{input_entries}\n"
        ']\n'
        'output [\n'
        f"{output_entries}\n"
        ']\n'
        "ensemble_scheduling {\n"
        "  step [\n"
        "    {\n"
        f'      model_name: "{PREPROCESS_MODEL}"\n'
        "      model_version: -1\n"
        '      input_map { key: "TEXT" value: "TEXT" }\n'
        '      input_map { key: "IMAGE_BYTES" value: "IMAGE_BYTES" }\n'
        '      input_map { key: "MAX_TOKENS" value: "MAX_TOKENS" }\n'
        '      input_map { key: "TEMPERATURE" value: "TEMPERATURE" }\n'
        '      input_map { key: "TOP_P" value: "TOP_P" }\n'
        '      input_map { key: "DEADLINE_MS" value: "DEADLINE_MS" }\n'
        '      output_map { key: "PROMPT" value: "PHASE7_PROMPT" }\n'
        '      output_map { key: "MAX_TOKENS" value: "PHASE7_MAX_TOKENS" }\n'
        '      output_map { key: "TEMPERATURE" value: "PHASE7_TEMPERATURE" }\n'
        '      output_map { key: "TOP_P" value: "PHASE7_TOP_P" }\n'
        '      output_map { key: "DEADLINE_MS" value: "PHASE7_DEADLINE_MS" }\n'
        "    },\n"
        "    {\n"
        f'      model_name: "{INFER_MODEL}"\n'
        "      model_version: -1\n"
        '      input_map { key: "PROMPT" value: "PHASE7_PROMPT" }\n'
        '      input_map { key: "MAX_TOKENS" value: "PHASE7_MAX_TOKENS" }\n'
        '      input_map { key: "TEMPERATURE" value: "PHASE7_TEMPERATURE" }\n'
        '      input_map { key: "TOP_P" value: "PHASE7_TOP_P" }\n'
        '      input_map { key: "DEADLINE_MS" value: "PHASE7_DEADLINE_MS" }\n'
        '      input_map { key: "STREAM" value: "STREAM" }\n'
        '      output_map { key: "TEXT" value: "PHASE7_TEXT" }\n'
        "    },\n"
        "    {\n"
        f'      model_name: "{POSTPROCESS_MODEL}"\n'
        "      model_version: -1\n"
        '      input_map { key: "TEXT" value: "PHASE7_TEXT" }\n'
        '      output_map { key: "OUTPUT_TEXT" value: "OUTPUT_TEXT" }\n'
        '      output_map { key: "RAW" value: "RAW" }\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )


def _write_python_model(model_root: Path, *, source: str) -> None:
    version_dir = model_root / "1"
    version_dir.mkdir(parents=True, exist_ok=True)
    (version_dir / "model.py").write_text(source)


def _preprocess_model_py() -> str:
    return (
        "from __future__ import annotations\n"
        "\n"
        "import numpy as np\n"
        "import triton_python_backend_utils as pb_utils\n"
        "\n"
        "\n"
        "def _to_str(value: object) -> str:\n"
        "    if isinstance(value, bytes):\n"
        "        return value.decode('utf-8', errors='ignore')\n"
        "    return str(value)\n"
        "\n"
        "\n"
        "class TritonPythonModel:\n"
        "    def initialize(self, args):\n"
        "        del args\n"
        "\n"
        "    def execute(self, requests):\n"
        "        responses = []\n"
        "        for request in requests:\n"
        "            text_in = pb_utils.get_input_tensor_by_name(request, 'TEXT')\n"
        "            image_bytes_in = pb_utils.get_input_tensor_by_name(request, 'IMAGE_BYTES')\n"
        "            max_tokens_in = pb_utils.get_input_tensor_by_name(request, 'MAX_TOKENS')\n"
        "            temperature_in = pb_utils.get_input_tensor_by_name(request, 'TEMPERATURE')\n"
        "            top_p_in = pb_utils.get_input_tensor_by_name(request, 'TOP_P')\n"
        "            deadline_in = pb_utils.get_input_tensor_by_name(request, 'DEADLINE_MS')\n"
        "            text_raw = text_in.as_numpy().reshape(-1)[0]\n"
        "            image_raw = image_bytes_in.as_numpy().reshape(-1)[0]\n"
        "            max_tokens_raw = max_tokens_in.as_numpy().reshape(-1)[0]\n"
        "            temperature_raw = temperature_in.as_numpy().reshape(-1)[0]\n"
        "            top_p_raw = top_p_in.as_numpy().reshape(-1)[0]\n"
        "            deadline_raw = deadline_in.as_numpy().reshape(-1)[0]\n"
        "            text = _to_str(text_raw)\n"
        "            # Triton HTTP/REST frontend base64-decodes BYTES tensors before passing\n"
        "            # to the Python backend; image_raw is already the raw image bytes.\n"
        "            image_bytes = image_raw if isinstance(image_raw, bytes) else str(image_raw).encode()\n"
        "            image_size = len(image_bytes)\n"
        "            max_tokens = int(max_tokens_raw)\n"
        "            temperature = float(temperature_raw)\n"
        "            top_p = float(top_p_raw)\n"
        "            deadline_ms = max(int(deadline_raw), 1)\n"
        "            prompt = f'[image_bytes={image_size}]\\n{text}'.strip()\n"
        "            out_prompt = pb_utils.Tensor('PROMPT', np.array([prompt], dtype=object))\n"
        "            out_max_tokens = pb_utils.Tensor('MAX_TOKENS', np.array([max_tokens], dtype=np.int32))\n"
        "            out_temperature = pb_utils.Tensor('TEMPERATURE', np.array([temperature], dtype=np.float32))\n"
        "            out_top_p = pb_utils.Tensor('TOP_P', np.array([top_p], dtype=np.float32))\n"
        "            out_deadline_ms = pb_utils.Tensor('DEADLINE_MS', np.array([deadline_ms], dtype=np.int32))\n"
        "            responses.append(\n"
        "                pb_utils.InferenceResponse(\n"
        "                    output_tensors=[\n"
        "                        out_prompt,\n"
        "                        out_max_tokens,\n"
        "                        out_temperature,\n"
        "                        out_top_p,\n"
        "                        out_deadline_ms,\n"
        "                    ]\n"
        "                )\n"
        "            )\n"
        "        return responses\n"
    )


def _infer_model_py(*, vllm_model_name: str) -> str:
    vllm_model_literal = repr(vllm_model_name)
    return (
        "from __future__ import annotations\n"
        "\n"
        "import asyncio\n"
        "import threading\n"
        "import uuid\n"
        "\n"
        "import numpy as np\n"
        "import triton_python_backend_utils as pb_utils\n"
        "\n"
        "\n"
        "def _to_str(value: object) -> str:\n"
        "    if isinstance(value, bytes):\n"
        "        return value.decode('utf-8', errors='ignore')\n"
        "    return str(value)\n"
        "\n"
        "\n"
        "class TritonPythonModel:\n"
        "    def initialize(self, args):\n"
        "        del args\n"
        "        import vllm\n"
        f"        engine_args = vllm.AsyncEngineArgs(model={vllm_model_literal})\n"
        "        self._engine = vllm.AsyncLLMEngine.from_engine_args(engine_args)\n"
        "        self._loop = asyncio.new_event_loop()\n"
        "        self._thread = threading.Thread(\n"
        "            target=self._loop.run_forever, daemon=True\n"
        "        )\n"
        "        self._thread.start()\n"
        "\n"
        "    def finalize(self):\n"
        "        loop = getattr(self, '_loop', None)\n"
        "        thread = getattr(self, '_thread', None)\n"
        "        self._engine = None\n"
        "        self._loop = None\n"
        "        self._thread = None\n"
        "        if loop is not None:\n"
        "            try:\n"
        "                loop.call_soon_threadsafe(loop.stop)\n"
        "            except Exception:\n"
        "                pass\n"
        "        if (\n"
        "            thread is not None\n"
        "            and thread.is_alive()\n"
        "            and thread is not threading.current_thread()\n"
        "        ):\n"
        "            thread.join(timeout=5.0)\n"
        "\n"
        "    def execute(self, requests):\n"
        "        for request in requests:\n"
            "            sender = request.get_response_sender()\n"
        "            asyncio.run_coroutine_threadsafe(\n"
        "                self._handle(request, sender), self._loop\n"
        "            )\n"
        "        return None\n"
        "\n"
        "    async def _handle(self, request, sender):\n"
        "        import vllm\n"
        "        deadline_ms = 0\n"
        "        timeout_s = 0.0\n"
        "        try:\n"
            "            prompt_t = pb_utils.get_input_tensor_by_name(request, 'PROMPT')\n"
            "            max_tokens_t = pb_utils.get_input_tensor_by_name(request, 'MAX_TOKENS')\n"
            "            temperature_t = pb_utils.get_input_tensor_by_name(request, 'TEMPERATURE')\n"
            "            top_p_t = pb_utils.get_input_tensor_by_name(request, 'TOP_P')\n"
        "            deadline_t = pb_utils.get_input_tensor_by_name(request, 'DEADLINE_MS')\n"
        "            stream_t = pb_utils.get_input_tensor_by_name(request, 'STREAM')\n"
        "\n"
        "            prompt = _to_str(prompt_t.as_numpy().reshape(-1)[0])\n"
        "            max_tokens = int(max_tokens_t.as_numpy().reshape(-1)[0])\n"
        "            temperature = float(temperature_t.as_numpy().reshape(-1)[0])\n"
        "            top_p = float(top_p_t.as_numpy().reshape(-1)[0])\n"
        "            deadline_ms = max(int(deadline_t.as_numpy().reshape(-1)[0]), 1)\n"
        "            stream = (\n"
        "                bool(stream_t.as_numpy().reshape(-1)[0])\n"
        "                if stream_t is not None\n"
        "                else False\n"
        "            )\n"
        "\n"
        "            sampling_params = vllm.SamplingParams(\n"
        "                max_tokens=max_tokens,\n"
        "                temperature=temperature,\n"
        "                top_p=top_p,\n"
        "            )\n"
        "            request_id = str(uuid.uuid4())\n"
        "            timeout_s = deadline_ms / 1000.0\n"
        "\n"
        "            final_text = ''\n"
        "            async with asyncio.timeout(timeout_s):\n"
        "                async for output in self._engine.generate(\n"
        "                    prompt, sampling_params, request_id\n"
        "                ):\n"
        "                    if output.outputs:\n"
        "                        final_text = output.outputs[0].text\n"
        "                    if stream:\n"
        "                        if output.outputs:\n"
        "                            text_chunk = output.outputs[0].text\n"
        "                            out = pb_utils.Tensor(\n"
        "                                'TEXT',\n"
        "                                np.array([text_chunk], dtype=object),\n"
        "                            )\n"
        "                            flags = (\n"
        "                                pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL\n"
        "                                if output.finished\n"
        "                                else 0\n"
        "                            )\n"
        "                            sender.send(\n"
        "                                pb_utils.InferenceResponse(output_tensors=[out]),\n"
        "                                flags=flags,\n"
        "                            )\n"
        "                        elif output.finished:\n"
        "                            # finished but no outputs: send empty FINAL to unblock client\n"
        "                            out = pb_utils.Tensor(\n"
        "                                'TEXT',\n"
        "                                np.array([''], dtype=object),\n"
        "                            )\n"
        "                            sender.send(\n"
        "                                pb_utils.InferenceResponse(output_tensors=[out]),\n"
        "                                flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL,\n"
        "                            )\n"
        "                    elif not stream and output.finished:\n"
        "                        out = pb_utils.Tensor(\n"
        "                            'TEXT',\n"
        "                            np.array([final_text], dtype=object),\n"
        "                        )\n"
        "                        sender.send(\n"
        "                            pb_utils.InferenceResponse(output_tensors=[out]),\n"
        "                            flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL,\n"
        "                        )\n"
        "        except TimeoutError:\n"
        "            sender.send(\n"
        "                pb_utils.InferenceResponse(\n"
        "                    error=pb_utils.TritonError(\n"
        "                        f'DEADLINE_EXCEEDED: deadline_ms={deadline_ms}, timeout_s={timeout_s:.3f}'\n"
        "                    )\n"
        "                ),\n"
        "                flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL,\n"
        "            )\n"
        "        except Exception as exc:\n"
            "            sender.send(\n"
                "                pb_utils.InferenceResponse(\n"
                    "                    error=pb_utils.TritonError(str(exc))\n"
                "                ),\n"
        "                flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL,\n"
        "            )\n"
    )


def _infer_model_py_cpu_mock(*, token_latency_ms: float = 0.5) -> str:
    """Generate mm_infer model.py that simulates LLM latency without vLLM/GPU.

    Preserves the decoupled sender pattern of the real mm_infer so the ensemble
    routing and response-handling path stay identical to production.
    Delay = max_tokens * token_latency_ms milliseconds.
    """
    latency_literal = repr(float(token_latency_ms))
    return (
        "from __future__ import annotations\n"
        "\n"
        "import asyncio\n"
        "import threading\n"
        "\n"
        "import numpy as np\n"
        "import triton_python_backend_utils as pb_utils\n"
        "\n"
        f"_TOKEN_LATENCY_MS: float = {latency_literal}\n"
        "\n"
        "\n"
        "class TritonPythonModel:\n"
        "    def initialize(self, args):\n"
        "        del args\n"
        "        self._loop = asyncio.new_event_loop()\n"
        "        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)\n"
        "        self._thread.start()\n"
        "\n"
        "    def finalize(self):\n"
        "        loop = getattr(self, '_loop', None)\n"
        "        thread = getattr(self, '_thread', None)\n"
        "        self._loop = None\n"
        "        self._thread = None\n"
        "        if loop is not None:\n"
        "            try:\n"
        "                loop.call_soon_threadsafe(loop.stop)\n"
        "            except Exception:\n"
        "                pass\n"
        "        if (\n"
        "            thread is not None\n"
        "            and thread.is_alive()\n"
        "            and thread is not threading.current_thread()\n"
        "        ):\n"
        "            thread.join(timeout=5.0)\n"
        "\n"
        "    def execute(self, requests):\n"
        "        for request in requests:\n"
        "            sender = request.get_response_sender()\n"
        "            asyncio.run_coroutine_threadsafe(self._handle(request, sender), self._loop)\n"
        "        return None\n"
        "\n"
        "    async def _handle(self, request, sender):\n"
        "        try:\n"
        "            max_tokens_t = pb_utils.get_input_tensor_by_name(request, 'MAX_TOKENS')\n"
        "            max_tokens = max(int(max_tokens_t.as_numpy().reshape(-1)[0]), 1)\n"
        "            await asyncio.sleep(max_tokens * _TOKEN_LATENCY_MS / 1000.0)\n"
        "            text = ' '.join(['tok'] * max_tokens)\n"
        "            out = pb_utils.Tensor('TEXT', np.array([text], dtype=object))\n"
        "            sender.send(\n"
        "                pb_utils.InferenceResponse(output_tensors=[out]),\n"
        "                flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL,\n"
        "            )\n"
        "        except Exception as exc:\n"
        "            sender.send(\n"
        "                pb_utils.InferenceResponse(error=pb_utils.TritonError(str(exc))),\n"
        "                flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL,\n"
        "            )\n"
    )


def _postprocess_model_py() -> str:
    return (
        "from __future__ import annotations\n"
        "\n"
        "import numpy as np\n"
        "import triton_python_backend_utils as pb_utils\n"
        "\n"
        "\n"
        "def _to_str(value: object) -> str:\n"
        "    if isinstance(value, bytes):\n"
        "        return value.decode('utf-8', errors='ignore')\n"
        "    return str(value)\n"
        "\n"
        "\n"
        "class TritonPythonModel:\n"
        "    def initialize(self, args):\n"
        "        del args\n"
        "\n"
        "    def execute(self, requests):\n"
        "        responses = []\n"
        "        for request in requests:\n"
        "            text_in = pb_utils.get_input_tensor_by_name(request, 'TEXT')\n"
        "            text_raw = text_in.as_numpy().reshape(-1)[0]\n"
        "            raw_text = _to_str(text_raw)\n"
        "            normalized = raw_text.strip()\n"
        "            out_text = pb_utils.Tensor('OUTPUT_TEXT', np.array([normalized], dtype=object))\n"
        "            out_raw = pb_utils.Tensor('RAW', np.array([raw_text], dtype=object))\n"
        "            responses.append(pb_utils.InferenceResponse(output_tensors=[out_text, out_raw]))\n"
        "        return responses\n"
    )


def prepare_triton_repo(
    output: Path,
    *,
    model_name: str = "mm_vllm",
    vllm_model_name: str = "/models",
    cpu_mock: bool = False,
    mock_token_latency_ms: float = 0.5,
) -> Path:
    if model_name in {PREPROCESS_MODEL, INFER_MODEL, POSTPROCESS_MODEL}:
        raise ValueError(f"model_name '{model_name}' conflicts with reserved stage model names")
    if not cpu_mock and not vllm_model_name:
        raise ValueError("vllm_model_name must not be empty")

    output.mkdir(parents=True, exist_ok=True)

    preprocess_root = output / PREPROCESS_MODEL
    infer_root = output / INFER_MODEL
    postprocess_root = output / POSTPROCESS_MODEL
    ensemble_root = output / model_name

    preprocess_root.mkdir(parents=True, exist_ok=True)
    infer_root.mkdir(parents=True, exist_ok=True)
    postprocess_root.mkdir(parents=True, exist_ok=True)
    ensemble_root.mkdir(parents=True, exist_ok=True)

    (preprocess_root / "config.pbtxt").write_text(
        _python_backend_config(
            PREPROCESS_MODEL,
            inputs=[
                ("TEXT", "TYPE_STRING"),
                ("IMAGE_BYTES", "TYPE_STRING"),
                ("MAX_TOKENS", "TYPE_INT32"),
                ("TEMPERATURE", "TYPE_FP32"),
                ("TOP_P", "TYPE_FP32"),
                ("DEADLINE_MS", "TYPE_INT32"),
            ],
            outputs=[
                ("PROMPT", "TYPE_STRING"),
                ("MAX_TOKENS", "TYPE_INT32"),
                ("TEMPERATURE", "TYPE_FP32"),
                ("TOP_P", "TYPE_FP32"),
                ("DEADLINE_MS", "TYPE_INT32"),
            ],
        )
    )
    _write_python_model(preprocess_root, source=_preprocess_model_py())

    if cpu_mock:
        (infer_root / "config.pbtxt").write_text(_infer_model_config(INFER_MODEL))
        _write_python_model(
            infer_root,
            source=_infer_model_py_cpu_mock(token_latency_ms=mock_token_latency_ms),
        )
    else:
        (infer_root / "config.pbtxt").write_text(_infer_model_config(INFER_MODEL))
        _write_python_model(infer_root, source=_infer_model_py(vllm_model_name=vllm_model_name))

    (postprocess_root / "config.pbtxt").write_text(
        _python_backend_config(
            POSTPROCESS_MODEL,
            inputs=[("TEXT", "TYPE_STRING")],
            outputs=[("OUTPUT_TEXT", "TYPE_STRING"), ("RAW", "TYPE_STRING")],
        )
    )
    _write_python_model(postprocess_root, source=_postprocess_model_py())

    (ensemble_root / "config.pbtxt").write_text(_ensemble_config(model_name))
    # Ensemble model also needs a concrete version directory for Triton loading.
    (ensemble_root / "1").mkdir(parents=True, exist_ok=True)
    return output


def _cli(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Triton model repository for the mm_vllm ensemble")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default="mm_vllm")
    parser.add_argument("--vllm-model", default="/models")
    parser.add_argument(
        "--cpu-mock",
        action="store_true",
        help="replace mm_infer vLLM backend with a CPU mock (asyncio.sleep); no GPU required",
    )
    parser.add_argument(
        "--mock-token-latency-ms",
        type=float,
        default=0.5,
        help="per-token latency injected by the CPU mock mm_infer (ms, default: 0.5)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _cli(argv)
    output = Path(args.output)
    repo = prepare_triton_repo(
        output,
        model_name=args.model_name,
        vllm_model_name=args.vllm_model,
        cpu_mock=args.cpu_mock,
        mock_token_latency_ms=args.mock_token_latency_ms,
    )
    print(repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
