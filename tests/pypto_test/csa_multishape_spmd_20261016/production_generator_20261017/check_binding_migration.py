# SPDX-License-Identifier: Apache-2.0
"""纯 CPU 复现生产绑定/合同迁移差分；不导入 PyPTO、不编译、不使用 NPU。

默认检查已执行过的七个规格；--cases 可限定追加检查的规格。只忽略因文件
收敛必然改变的源码位置及新增 Host 上界描述，保留 task.scope 等实际语义。
"""

import argparse
import ast
import copy
import importlib
import importlib.util
import json
import sys
import types
from datetime import datetime, timezone
from pathlib import Path


CASES = (
    "h131072_b4", "h131072_b8", "h131072_b16", "h131072_b24",
    "h8192_b16", "h8192_b24", "h8192_b32",
)


def normalize(value):
    """保留任务作用域、别名、依赖、分支和值；仅去除来源/新增描述标记。"""
    ignored = {"source_line", "source_ir", "metadata_source", "max_seq_len"}
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items() if key not in ignored}
    if isinstance(value, list):
        return [normalize(item) for item in value]
    return value


def load(path, name):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def expect_value_error(call, text):
    try:
        call()
    except ValueError as error:
        if text and text not in str(error):
            raise AssertionError(f"错误类型符合，但错误说明不符：{error}") from error
    else:
        raise AssertionError("应拒绝的输入被接受")


def check(cases):
    # 本文件位于 tests/pypto_test/<实验>/production_generator_20261017。
    repository = Path(__file__).resolve().parents[4]
    production = repository / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"
    frozen = repository / "tests/pypto_test/csa_static_spmd_feasibility_20261015"
    # 绕开 vllm_ascend 的应用初始化；生产纯 CPU 模块仍通过包内相对导入使用。
    package_name = "_csa_binding_migration_package"
    package = types.ModuleType(package_name)
    package.__path__ = [str(production)]
    sys.modules[package_name] = package
    binding = importlib.import_module(package_name + ".binding")
    ir = importlib.import_module(package_name + ".ir")
    catalog = importlib.import_module(package_name + ".catalog").catalog()
    # 只有测试脚本导入旧绑定器；生产模块不得读取此测试目录。
    load(frozen / "build_ir_catalog.py", "build_ir_catalog")
    legacy = load(frozen / "bind_orchestration.py", "_csa_legacy_binding")

    logical_before = frozen / "references/symbolic_ir/10_after_OutlineIncoreScopes.py"
    logical_after = production / "orchestration.py"
    old_functions = ir.functions_from_ir(logical_before)[1]
    new_functions = ir.functions_from_ir(logical_after)[1]
    if old_functions.keys() != new_functions.keys():
        raise AssertionError("逻辑函数集合变化")
    original_bodies, alias_stubs = [], []
    for name, function in old_functions.items():
        current = new_functions[name]
        nz = any(parameter.annotation is not None and "NZ" in ast.unparse(parameter.annotation)
                 for parameter in function.args.args)
        if name == "_decode_csa_tp1_layer" or nz:
            assert ast.dump(function, include_attributes=False) == ast.dump(current, include_attributes=False), name
            original_bodies.append(name)
        else:
            assert ast.dump(function.args, include_attributes=False) == ast.dump(current.args, include_attributes=False), name
            return_before = ast.dump(function.returns, include_attributes=False) if function.returns else None
            return_after = ast.dump(current.returns, include_attributes=False) if current.returns else None
            assert return_before == return_after, name
            assert legacy.Recorder.resolve_returns(function) == binding.Recorder.resolve_returns(current), name
            alias_stubs.append(name)
    assert len(original_bodies) == 11 and len(alias_stubs) == 57

    old_catalog = json.loads((frozen / "generated/core_body_catalog.json").read_text())
    assert len(old_catalog["kernels"]) == len(catalog["kernels"]) == 73
    for before, after in zip(old_catalog["kernels"], catalog["kernels"]):
        name = before["name"]
        for key in ("name", "core_type", "func_id", "logical_task", "inline_parameter_order"):
            assert before[key] == after[key], (name, key)
        parameter_map = lambda parameters: {p["name"]: p["annotation"] for p in parameters}
        assert parameter_map(before["parameters"]) == parameter_map(after["parameters"]), name
        assert normalize(before["system_operations"]) == normalize(after["system_operations"]), name

    reverse_types = {value: key for key, value in binding.DTYPES.items()}
    case_results = []
    for case in cases:
        history = int(case.split("_")[0][1:])
        reference = json.loads((frozen / "bindings" / (case + "_prepared.json")).read_text())
        descriptors = {
            name: {"shape": value["shape"], "stride": value["strides"],
                   "dtype": reverse_types[value["dtype"]]}
            for name, value in reference["root_arguments"].items()
        }
        old_descriptors = copy.deepcopy(descriptors)
        old_descriptors["kv_seq_lens"]["metadata_values"] = [history + 6] * descriptors["kv_seq_lens"]["shape"][0]
        before = legacy.bind(logical_before, old_descriptors)
        after = binding.bind(logical_after, descriptors, max_seq_len=history + 6)
        assert normalize(before) == normalize(after), case
        case_results.append({"case": case, "compute_calls": after["compute_calls"],
                             "dependency_only_calls": after["dependency_only_calls"],
                             "task_scopes_compared": True, "passed": True})

    expect_value_error(lambda: binding.bind(logical_after, old_descriptors, max_seq_len=history + 6), "metadata_values")
    expect_value_error(lambda: binding.bind(logical_after, descriptors), "max_seq_len")
    bad_dtype = copy.deepcopy(descriptors)
    bad_dtype["attn_norm_w"]["dtype"] = "torch.float32"
    expect_value_error(lambda: binding.bind(logical_after, bad_dtype, max_seq_len=history + 6), "dtype")
    recorder = binding.Recorder(new_functions, max_seq_len=history + 6)
    other = binding.TensorRef("argument:other", [4], "INT32", [1], label="other")
    kv = binding.TensorRef("argument:kv_seq_lens", [4], "INT32", [1], label="kv_seq_lens")
    expect_value_error(lambda: recorder.read(other, [0]), "kv_seq_lens")
    expect_value_error(lambda: recorder.read(kv, [4]), "kv_seq_lens")
    pypto_imported = any(name == "pypto" or name.startswith("pypto.") for name in sys.modules)
    assert not pypto_imported
    return {
        "scope": "CPU 源码、合同及绑定差分；不代表lowering或NPU正确性/性能",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "original_root_and_nz_bodies": original_bodies,
        "logical_alias_stubs": alias_stubs,
        "physical_contracts_equal": 73,
        "system_operations_equal": 73,
        "binding_equivalence": case_results,
        "guards": {"metadata_values_rejected": True, "missing_host_max_rejected": True,
                   "norm_dtype_mismatch_rejected": True, "other_tensor_read_rejected": True,
                   "out_of_bounds_kv_read_rejected": True},
        "pypto_imported": False, "npu_run": False, "passed": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = check(arguments.cases)
    arguments.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": result["passed"], "cases": arguments.cases,
                      "pypto_imported": False, "npu_run": False}, ensure_ascii=False))
