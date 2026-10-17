# SPDX-License-Identifier: Apache-2.0
"""仅 CPU 复现 Torch schema 参数上限，并检查 Tensor[] 边界表达。"""

import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"torch_version": torch.__version__, "device_execution": False,
              "scope": "Torch dispatcher CPU 小例子；没有修改 PyPTO 注册实现"}
    lib = torch.library.Library("csa_schema_capacity_probe", "FRAGMENT")
    for count in (64, 65, 132):
        arguments = ", ".join(f"Tensor t{i}" for i in range(count))
        try:
            lib.define(f"flat_{count}({arguments}) -> ()")
            result[f"flat_{count}"] = {"defined": True}
        except RuntimeError as exc:
            result[f"flat_{count}"] = {"defined": False, "error": str(exc)}

    lib.define("packed_mutate(Tensor[] inputs, Tensor(a!)[] outputs) -> ()")

    def mutate(inputs, outputs):
        for index, output in enumerate(outputs):
            output.add_(inputs[index % len(inputs)])

    lib.impl("packed_mutate", mutate, "CPU")
    lib.impl("packed_mutate", lambda inputs, outputs: None, "Meta")
    lib.define("packed(Tensor[] inputs, Tensor(a!)[] outputs) -> Tensor(a!)[]")

    def packed(inputs, outputs):
        torch.ops.csa_schema_capacity_probe.packed_mutate(inputs, outputs)
        return outputs

    lib.impl("packed", packed, "CompositeImplicitAutograd")
    op = torch.ops.csa_schema_capacity_probe.packed
    for device in ("cpu", "meta"):
        inputs = [torch.full((1,), float(index + 1), device=device) for index in range(56)]
        outputs = [torch.zeros(1, device=device) for _ in range(76)]
        actual = op(inputs, outputs)
        assert len(actual) == 76 and all(a is b for a, b in zip(actual, outputs))
        if device == "cpu":
            for index, output in enumerate(outputs):
                torch.testing.assert_close(output, inputs[index % len(inputs)], atol=0, rtol=0)
        result[f"packed_{device}"] = {"passed": True, "tensors": 132, "schema_arguments": 2,
                                     "mutated_tensors": 76, "original_objects_returned": True}
    try:
        inputs = [torch.ones(1) for _ in range(56)]
        outputs = [torch.zeros(1) for _ in range(76)]
        actual = torch.func.functionalize(packed)(inputs, outputs)
        assert all(torch.equal(tensor, torch.ones(1)) for tensor in actual)
        result["functionalize"] = {"passed": True}
    except Exception as exc:
        result["functionalize"] = {"passed": False, "error": str(exc)}
    (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
