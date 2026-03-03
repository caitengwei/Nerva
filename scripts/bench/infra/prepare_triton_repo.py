from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def _default_config_pbtxt(model_name: str) -> str:
    return (
        f'name: "{model_name}"\n'
        'backend: "python"\n'
        'max_batch_size: 8\n'
        'input [\n'
        '  { name: "PROMPT" data_type: TYPE_STRING dims: [ 1 ] }\n'
        ']\n'
        'output [\n'
        '  { name: "TEXT" data_type: TYPE_STRING dims: [ 1 ] }\n'
        ']\n'
    )


def prepare_triton_repo(output: Path, *, model_name: str = "phase7_mm_vllm") -> Path:
    model_root = output / model_name
    version_dir = model_root / "1"
    version_dir.mkdir(parents=True, exist_ok=True)

    (model_root / "config.pbtxt").write_text(_default_config_pbtxt(model_name))
    (version_dir / "model.py").write_text(
        "from __future__ import annotations\n"
        "\n"
        "import numpy as np\n"
        "import triton_python_backend_utils as pb_utils\n"
        "\n"
        "\n"
        "class TritonPythonModel:\n"
        "    def initialize(self, args):\n"
        "        del args\n"
        "\n"
        "    def execute(self, requests):\n"
        "        responses = []\n"
        "        for request in requests:\n"
        "            inp = pb_utils.get_input_tensor_by_name(request, 'PROMPT')\n"
        "            data = inp.as_numpy()\n"
        "            output = pb_utils.Tensor('TEXT', np.array(data, dtype=object))\n"
        "            responses.append(pb_utils.InferenceResponse(output_tensors=[output]))\n"
        "        return responses\n"
    )
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
