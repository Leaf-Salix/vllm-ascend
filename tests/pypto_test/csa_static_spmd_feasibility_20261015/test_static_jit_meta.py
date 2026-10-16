# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
# 必须先激活隔离组件路径，再导入 Torch/PyPTO 与准备对象。
"""完整静态 CSA 入口的 CPU Meta 注册验证；不编译设备二进制、不运行 NPU。"""

import argparse
import ast
import importlib.util
import json
import sys
import traceback
from pathlib import Path

import dsv4_csa_env

dsv4_csa_env.activate()

import pypto.torch
import torch
from prepared_arguments import DTYPES, PreparedStaticArguments
from pypto.runtime import CompileOptions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"scope": "完整 CSA 单 MIX JIT 的 CPU Meta 注册",
              "passed": False, "device_execution": False, "full_csa_validated": False}
    try:
        source = args.case / "entry_jit.py"
        snapshot = args.output / "source.py"
        snapshot.write_text(source.read_text())
        spec = importlib.util.spec_from_file_location("csa_static_meta_probe", snapshot)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        op = pypto.torch.register(
            module.entry, "csa_static_full_probe::entry",
            config=CompileOptions(platform="a2a3", output_dir=str(args.output / "kernel")),
        )
        schedule = json.loads(
            (args.case.parent.parent / "bindings" / f"{args.case.name}_prepared.json").read_text()
        )
        native = {
            name: torch.empty_strided(ref["shape"], ref["strides"], dtype=DTYPES[ref["dtype"]], device="meta")
            for name, ref in schedule["root_arguments"].items()
        }
        prepared = PreparedStaticArguments(args.case, native)
        prepared.require_core_capacity(256)
        actual = op(*prepared.core_args)
        entry = next(node for node in ast.parse(snapshot.read_text()).body
                     if isinstance(node, ast.FunctionDef) and node.name == "entry")
        returned = entry.body[-1]
        assert isinstance(returned, ast.Return) and isinstance(returned.value, ast.Tuple)
        names = [node.id for node in returned.value.elts]
        assert len(actual) == len(names)
        for tensor, name in zip(actual, names):
            assert tensor is prepared.core_args[prepared.param_names.index(name)]
        result.update(passed=True, tensor_arguments=len(prepared.core_args),
                      returned_aliases=len(names), native_inputs=len(native),
                      schema=str(op._schema), source_snapshot=str(snapshot.resolve()))
    except Exception as exc:
        result.update(error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({key: value for key, value in result.items() if key not in {"schema", "traceback"}},
                         ensure_ascii=False))


if __name__ == "__main__":
    main()
