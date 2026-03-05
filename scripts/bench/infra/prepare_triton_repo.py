from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


PREPROCESS_MODEL = "phase7_preprocess"
INFER_MODEL = "phase7_infer"
POSTPROCESS_MODEL = "phase7_postprocess"


def _python_backend_config(
    model_name: str,
    *,
    inputs: list[tuple[str, str]],
    outputs: list[tuple[str, str]],
) -> str:
    input_entries = "\n".join(
        f'  {{ name: "{name}" data_type: {dtype} dims: [ 1 ] }}' for name, dtype in inputs
    )
    output_entries = "\n".join(
        f'  {{ name: "{name}" data_type: {dtype} dims: [ 1 ] }}' for name, dtype in outputs
    )

    return (
        f'name: "{model_name}"\n'
        'backend: "python"\n'
        'max_batch_size: 8\n'
        'input [\n'
        f"{input_entries}\n"
        ']\n'
        'output [\n'
        f"{output_entries}\n"
        ']\n'
    )


def _ensemble_config(model_name: str) -> str:
    return (
        f'name: "{model_name}"\n'
        'platform: "ensemble"\n'
        'max_batch_size: 8\n'
        'input [\n'
        '  { name: "TEXT" data_type: TYPE_STRING dims: [ 1 ] }\n'
        '  { name: "IMAGE_SIZE" data_type: TYPE_INT32 dims: [ 1 ] }\n'
        '  { name: "MAX_TOKENS" data_type: TYPE_INT32 dims: [ 1 ] }\n'
        '  { name: "TEMPERATURE" data_type: TYPE_FP32 dims: [ 1 ] }\n'
        '  { name: "TOP_P" data_type: TYPE_FP32 dims: [ 1 ] }\n'
        ']\n'
        'output [\n'
        '  { name: "OUTPUT_TEXT" data_type: TYPE_STRING dims: [ 1 ] }\n'
        '  { name: "RAW" data_type: TYPE_STRING dims: [ 1 ] }\n'
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
        '      output_map { key: "PROMPT" value: "PHASE7_PROMPT" }\n'
        '      output_map { key: "MAX_TOKENS" value: "PHASE7_MAX_TOKENS" }\n'
        '      output_map { key: "TEMPERATURE" value: "PHASE7_TEMPERATURE" }\n'
        '      output_map { key: "TOP_P" value: "PHASE7_TOP_P" }\n'
        "    },\n"
        "    {\n"
        f'      model_name: "{INFER_MODEL}"\n'
        "      model_version: -1\n"
        '      input_map { key: "PROMPT" value: "PHASE7_PROMPT" }\n'
        '      input_map { key: "MAX_TOKENS" value: "PHASE7_MAX_TOKENS" }\n'
        '      input_map { key: "TEMPERATURE" value: "PHASE7_TEMPERATURE" }\n'
        '      input_map { key: "TOP_P" value: "PHASE7_TOP_P" }\n'
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
        "            text_raw = text_in.as_numpy().reshape(-1)[0]\n"
        "            size_raw = size_in.as_numpy().reshape(-1)[0]\n"
        "            max_tokens_raw = max_tokens_in.as_numpy().reshape(-1)[0]\n"
        "            temperature_raw = temperature_in.as_numpy().reshape(-1)[0]\n"
        "            top_p_raw = top_p_in.as_numpy().reshape(-1)[0]\n"
        "            text = _to_str(text_raw)\n"
        "            image_size = int(size_raw)\n"
        "            max_tokens = int(max_tokens_raw)\n"
        "            temperature = float(temperature_raw)\n"
        "            top_p = float(top_p_raw)\n"
        "            prompt = f'[image_bytes={image_size}]\\n{text}'.strip()\n"
        "            out_prompt = pb_utils.Tensor('PROMPT', np.array([prompt], dtype=object))\n"
        "            out_max_tokens = pb_utils.Tensor('MAX_TOKENS', np.array([max_tokens], dtype=np.int32))\n"
        "            out_temperature = pb_utils.Tensor('TEMPERATURE', np.array([temperature], dtype=np.float32))\n"
        "            out_top_p = pb_utils.Tensor('TOP_P', np.array([top_p], dtype=np.float32))\n"
        "            responses.append(\n"
        "                pb_utils.InferenceResponse(\n"
        "                    output_tensors=[out_prompt, out_max_tokens, out_temperature, out_top_p]\n"
        "                )\n"
        "            )\n"
        "        return responses\n"
    )


def _infer_model_py() -> str:
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
        "            prompt_in = pb_utils.get_input_tensor_by_name(request, 'PROMPT')\n"
        "            max_tokens_in = pb_utils.get_input_tensor_by_name(request, 'MAX_TOKENS')\n"
        "            temperature_in = pb_utils.get_input_tensor_by_name(request, 'TEMPERATURE')\n"
        "            top_p_in = pb_utils.get_input_tensor_by_name(request, 'TOP_P')\n"
        "            prompt_raw = prompt_in.as_numpy().reshape(-1)[0]\n"
        "            max_tokens_raw = max_tokens_in.as_numpy().reshape(-1)[0]\n"
        "            temperature_raw = temperature_in.as_numpy().reshape(-1)[0]\n"
        "            top_p_raw = top_p_in.as_numpy().reshape(-1)[0]\n"
        "            max_tokens = int(max_tokens_raw)\n"
        "            temperature = float(temperature_raw)\n"
        "            top_p = float(top_p_raw)\n"
        "            text = _to_str(prompt_raw)\n"
        "            del max_tokens, temperature, top_p\n"
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
        "            normalized = _to_str(text_raw).strip()\n"
        "            out_text = pb_utils.Tensor('OUTPUT_TEXT', np.array([normalized], dtype=object))\n"
        "            out_raw = pb_utils.Tensor('RAW', np.array([normalized], dtype=object))\n"
        "            responses.append(pb_utils.InferenceResponse(output_tensors=[out_text, out_raw]))\n"
        "        return responses\n"
    )


def prepare_triton_repo(output: Path, *, model_name: str = "phase7_mm_vllm") -> Path:
    if model_name in {PREPROCESS_MODEL, INFER_MODEL, POSTPROCESS_MODEL}:
        raise ValueError(f"model_name '{model_name}' conflicts with reserved stage model names")

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
            ],
            outputs=[
                ("PROMPT", "TYPE_STRING"),
                ("MAX_TOKENS", "TYPE_INT32"),
                ("TEMPERATURE", "TYPE_FP32"),
                ("TOP_P", "TYPE_FP32"),
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
            ],
            outputs=[("TEXT", "TYPE_STRING")],
        )
    )
    _write_python_model(infer_root, source=_infer_model_py())

    (postprocess_root / "config.pbtxt").write_text(
        _python_backend_config(
            POSTPROCESS_MODEL,
            inputs=[("TEXT", "TYPE_STRING")],
            outputs=[("OUTPUT_TEXT", "TYPE_STRING"), ("RAW", "TYPE_STRING")],
        )
    )
    _write_python_model(postprocess_root, source=_postprocess_model_py())

    (ensemble_root / "config.pbtxt").write_text(_ensemble_config(model_name))
    return output


def _cli(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Triton model repository for Phase 7")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default="phase7_mm_vllm")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _cli(argv)
    output = Path(args.output)
    repo = prepare_triton_repo(output, model_name=args.model_name)
    print(repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
