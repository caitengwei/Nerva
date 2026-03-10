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


def _ensemble_config(model_name: str) -> str:
    input_entries = ",\n".join(
        [
            '  { name: "TEXT" data_type: TYPE_STRING dims: [ 1 ] }',
            '  { name: "IMAGE_SIZE" data_type: TYPE_INT32 dims: [ 1 ] }',
            '  { name: "MAX_TOKENS" data_type: TYPE_INT32 dims: [ 1 ] }',
            '  { name: "TEMPERATURE" data_type: TYPE_FP32 dims: [ 1 ] }',
            '  { name: "TOP_P" data_type: TYPE_FP32 dims: [ 1 ] }',
            '  { name: "DEADLINE_MS" data_type: TYPE_INT32 dims: [ 1 ] }',
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
        '      input_map { key: "IMAGE_SIZE" value: "IMAGE_SIZE" }\n'
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
        "            size_in = pb_utils.get_input_tensor_by_name(request, 'IMAGE_SIZE')\n"
        "            max_tokens_in = pb_utils.get_input_tensor_by_name(request, 'MAX_TOKENS')\n"
        "            temperature_in = pb_utils.get_input_tensor_by_name(request, 'TEMPERATURE')\n"
        "            top_p_in = pb_utils.get_input_tensor_by_name(request, 'TOP_P')\n"
        "            deadline_in = pb_utils.get_input_tensor_by_name(request, 'DEADLINE_MS')\n"
        "            text_raw = text_in.as_numpy().reshape(-1)[0]\n"
        "            size_raw = size_in.as_numpy().reshape(-1)[0]\n"
        "            max_tokens_raw = max_tokens_in.as_numpy().reshape(-1)[0]\n"
        "            temperature_raw = temperature_in.as_numpy().reshape(-1)[0]\n"
        "            top_p_raw = top_p_in.as_numpy().reshape(-1)[0]\n"
        "            deadline_raw = deadline_in.as_numpy().reshape(-1)[0]\n"
        "            text = _to_str(text_raw)\n"
        "            image_size = int(size_raw)\n"
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


def _infer_model_py(*, vllm_base_url: str, vllm_model_name: str) -> str:
    vllm_url = f"{vllm_base_url.rstrip('/')}/v1/completions"
    vllm_url_literal = repr(vllm_url)
    vllm_model_literal = repr(vllm_model_name)
    return (
        "from __future__ import annotations\n"
        "\n"
        "import json\n"
        "import urllib.error\n"
        "import urllib.request\n"
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
        f"        self._vllm_url = {vllm_url_literal}\n"
        f"        self._vllm_model = {vllm_model_literal}\n"
        "\n"
        "    def _request_vllm(self, *, prompt: str, max_tokens: int, temperature: float, top_p: float, deadline_ms: int) -> str:\n"
        "        body = {\n"
        "            'model': self._vllm_model,\n"
        "            'prompt': prompt,\n"
        "            'max_tokens': max_tokens,\n"
        "            'temperature': temperature,\n"
        "            'top_p': top_p,\n"
        "            'stream': False,\n"
        "        }\n"
        "        encoded = json.dumps(body).encode('utf-8')\n"
        "        req = urllib.request.Request(\n"
        "            self._vllm_url,\n"
        "            data=encoded,\n"
        "            headers={'Content-Type': 'application/json'},\n"
        "            method='POST',\n"
        "        )\n"
        "        timeout_s = max(float(deadline_ms) / 1000.0, 0.001)\n"
        "        with urllib.request.urlopen(req, timeout=timeout_s) as response:\n"
        "            payload = response.read().decode('utf-8', errors='ignore')\n"
        "        data = json.loads(payload)\n"
        "        choices = data.get('choices') if isinstance(data, dict) else None\n"
        "        if not isinstance(choices, list) or not choices:\n"
        "            raise RuntimeError('missing choices in vLLM response')\n"
        "        first = choices[0]\n"
        "        if not isinstance(first, dict):\n"
        "            raise RuntimeError('invalid first choice in vLLM response')\n"
        "        text = first.get('text')\n"
        "        if not isinstance(text, str):\n"
        "            raise RuntimeError('missing text in vLLM response')\n"
        "        return text\n"
        "\n"
        "    def execute(self, requests):\n"
        "        responses = []\n"
        "        for request in requests:\n"
        "            prompt_in = pb_utils.get_input_tensor_by_name(request, 'PROMPT')\n"
        "            max_tokens_in = pb_utils.get_input_tensor_by_name(request, 'MAX_TOKENS')\n"
        "            temperature_in = pb_utils.get_input_tensor_by_name(request, 'TEMPERATURE')\n"
        "            top_p_in = pb_utils.get_input_tensor_by_name(request, 'TOP_P')\n"
        "            deadline_in = pb_utils.get_input_tensor_by_name(request, 'DEADLINE_MS')\n"
        "            prompt_raw = prompt_in.as_numpy().reshape(-1)[0]\n"
        "            max_tokens_raw = max_tokens_in.as_numpy().reshape(-1)[0]\n"
        "            temperature_raw = temperature_in.as_numpy().reshape(-1)[0]\n"
        "            top_p_raw = top_p_in.as_numpy().reshape(-1)[0]\n"
        "            deadline_raw = deadline_in.as_numpy().reshape(-1)[0]\n"
        "            max_tokens = int(max_tokens_raw)\n"
        "            temperature = float(temperature_raw)\n"
        "            top_p = float(top_p_raw)\n"
        "            deadline_ms = max(int(deadline_raw), 1)\n"
        "            prompt = _to_str(prompt_raw)\n"
        "            try:\n"
        "                text = self._request_vllm(\n"
        "                    prompt=prompt,\n"
        "                    max_tokens=max_tokens,\n"
        "                    temperature=temperature,\n"
        "                    top_p=top_p,\n"
        "                    deadline_ms=deadline_ms,\n"
        "                )\n"
        "            except Exception as exc:\n"
        "                responses.append(\n"
        "                    pb_utils.InferenceResponse(error=pb_utils.TritonError(str(exc)))\n"
        "                )\n"
        "                continue\n"
        "            out = pb_utils.Tensor('TEXT', np.array([text], dtype=object))\n"
        "            responses.append(pb_utils.InferenceResponse(output_tensors=[out]))\n"
        "        return responses\n"
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
    vllm_base_url: str = "http://127.0.0.1:8001",
    vllm_model_name: str = "/models",
    infer_instance_count: int = 1,
) -> Path:
    if model_name in {PREPROCESS_MODEL, INFER_MODEL, POSTPROCESS_MODEL}:
        raise ValueError(f"model_name '{model_name}' conflicts with reserved stage model names")
    if not vllm_base_url:
        raise ValueError("vllm_base_url must not be empty")
    if not vllm_model_name:
        raise ValueError("vllm_model_name must not be empty")
    if infer_instance_count <= 0:
        raise ValueError("infer_instance_count must be > 0")

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
                ("IMAGE_SIZE", "TYPE_INT32"),
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

    (infer_root / "config.pbtxt").write_text(
        _python_backend_config(
            INFER_MODEL,
            inputs=[
                ("PROMPT", "TYPE_STRING"),
                ("MAX_TOKENS", "TYPE_INT32"),
                ("TEMPERATURE", "TYPE_FP32"),
                ("TOP_P", "TYPE_FP32"),
                ("DEADLINE_MS", "TYPE_INT32"),
            ],
            outputs=[("TEXT", "TYPE_STRING")],
            instance_count=infer_instance_count if infer_instance_count > 1 else None,
        )
    )
    _write_python_model(
        infer_root,
        source=_infer_model_py(vllm_base_url=vllm_base_url, vllm_model_name=vllm_model_name),
    )

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
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8001")
    parser.add_argument("--vllm-model", default="/models")
    parser.add_argument("--infer-instance-count", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _cli(argv)
    output = Path(args.output)
    repo = prepare_triton_repo(
        output,
        model_name=args.model_name,
        vllm_base_url=args.vllm_url,
        vllm_model_name=args.vllm_model,
        infer_instance_count=args.infer_instance_count,
    )
    print(repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
