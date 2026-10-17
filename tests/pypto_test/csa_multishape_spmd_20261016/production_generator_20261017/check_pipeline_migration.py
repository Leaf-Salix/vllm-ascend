# SPDX-License-Identifier: Apache-2.0
"""生产生成链的 CPU 差分；读取历史证据只发生在此测试，不启动 NPU。"""

import argparse
import ast
import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

HERE = Path(__file__).resolve().parent
N = HERE.parent
F = N.parent / "csa_static_spmd_feasibility_20261015"
C = N.parents[2]
P = C / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"


def load(name, path, *, package=False):
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[str(path.parent)] if package else None
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def same_ast(left, right):
    return ast.dump(ast.parse(left), include_attributes=False) == ast.dump(ast.parse(right), include_attributes=False)


def strip_source_lines(value):
    if isinstance(value, dict):
        return {k: strip_source_lines(v) for k, v in value.items() if k != "source_line"}
    if isinstance(value, list):
        return [strip_source_lines(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    load("_csa_pipeline_check", P / "__init__.py", package=True)
    from _csa_pipeline_check.compiler import build_csa, GeneratedCSA
    from _csa_pipeline_check.binding import DTYPES
    from _csa_pipeline_check.catalog import catalog
    from _csa_pipeline_check.codegen import generate, jit_source
    from _csa_pipeline_check.layout import prepare
    from _csa_pipeline_check.qk_indices import cache_qk_indices
    from _csa_pipeline_check.row_dependencies import refine
    from _csa_pipeline_check.wait_codegen import rewrite_wait_source

    sys.path[:0] = [str(N), str(F)]
    import generate_static_entry as old_codegen
    import prepare_static_layout as old_layout
    import qk_index_cache as old_qk
    import row_dependencies as old_rows
    from validate_static_protocol import validate

    report = {"scope": "CPU源码/布局/别名/协议差分，非设备精度或性能证明", "checks": {}}
    checks = report["checks"]
    bodies = catalog()
    before = N / "production_migration_20261017/cpu_case"
    schedule = json.loads((before / "bindings/h65536_b17_prepared.json").read_text())
    legacy_catalog = json.loads((F / "generated/core_body_catalog.json").read_text())
    legacy_catalog.update(
        body_bundle=str(P / "kernels.py"),
        source_ir=str(F / "references/symbolic_ir/25_after_ExpandMixedKernel.py"),
        logical_source_ir=str(F / "references/symbolic_ir/10_after_OutlineIncoreScopes.py"),
    )
    old_l = old_layout.prepare(copy.deepcopy(schedule), legacy_catalog)
    new_l = prepare(copy.deepcopy(schedule), bodies)
    assert strip_source_lines({k: v for k, v in old_l.items() if k not in ("abi", "scope")}) == strip_source_lines(
        {k: v for k, v in new_l.items() if k not in ("abi", "scope")}
    )
    a, ab = old_codegen.generate(
        schedule, old_l, legacy_catalog, workspace_binding="external", native_subviews_in_entry=True
    )
    b, bb = generate(schedule, new_l, bodies)
    assert same_ast(a, b) and same_ast(old_codegen.jit_source(a), jit_source(b))
    checks["same_schedule_layout_and_full_math_ast"] = True
    previous = json.loads((before / "variants/h65536_b17/preparation.json").read_text())
    costs = {
        (schedule["call_bindings"][r["task"]]["ordinal"], r["slot"]): r["model_cost"] + 1
        for r in previous["source_schedule_audit"]["cost_sources"]
    }
    current = build_csa(
        previous["descriptors"],
        max_seq_len=65542,
        workspace_directory=args.output / "b17_math",
        core_tensor_capacity=256,
        member_costs=costs,
    )
    current.save(args.output / "b17_explicit_costs")
    assert same_ast(current.source, (before / "variants/h65536_b17/entry_jit.py").read_text())
    checks["full_production_b17_same_cost_ast_identical"] = True
    checks["b17_protocol"] = validate(current.schedule, current.layout)
    roots = json.loads((F / "bindings/h131072_b16_prepared.json").read_text())["root_arguments"]
    inverse = {v: k for k, v in DTYPES.items()}
    descriptors = {
        k: {"shape": v["shape"], "stride": v["strides"], "dtype": inverse[v["dtype"]]} for k, v in roots.items()
    }
    b16 = build_csa(
        descriptors, max_seq_len=131078, workspace_directory=args.output / "b16_math", core_tensor_capacity=256
    )
    b16.save(args.output / "b16_default_model")
    checks["b16_protocol"] = validate(b16.schedule, b16.layout)
    qk_source, _ = cache_qk_indices(b16.schedule, bodies)
    old_catalog, _ = old_qk.cache_qk_indices(b16.schedule, bodies, args.output / "old_qk")
    assert same_ast(qk_source, Path(old_catalog["body_bundle"]).read_text())
    checks["qk_selected_math_transform_identical"] = True
    # 从同一完整成员图重复施加行依赖策略，比较真正下发的成员顺序/等待/发布。
    left, right = copy.deepcopy(b16.schedule), copy.deepcopy(b16.schedule)
    row_catalog = dict(bodies, body_bundle=str(args.output / "b16_math/qk_indices.py"))
    old_rows.refine(left, row_catalog, args.output / "old_rows")
    refine(right, row_catalog)
    assert left["per_core_sequences"] == right["per_core_sequences"]
    assert left["wait_plan"] == right["wait_plan"]
    checks["row_dependency_default_branch_identical"] = True
    # 在独立 CPU 进程安装旧编译器替身，比对真正应用的 helper 文本。
    source = "using namespace pto;\npto::comm::TWAIT(a,b,c);\npto::comm::TWAIT(d,e,f);\n"
    backend = SimpleNamespace(_compile_pto_module=lambda *a, **k: source)
    package = ModuleType("pypto")
    backend_package = ModuleType("pypto.backend")
    backend_package.pto_backend = backend
    sys.modules["pypto"] = package
    sys.modules["pypto.backend"] = backend_package
    old_wait = load("_old_wait_for_compare", HERE / "before/signal_polling.py")
    with old_wait.bypass_signal_wait(args.output / "old_wait", 100, cached=True):
        expected = backend._compile_pto_module("", "csa_static_group", "")
    actual, _ = rewrite_wait_source(source)
    assert actual == expected
    checks["cached_100_wait_cpp_identical"] = True
    sys.modules.pop("pypto")
    sys.modules.pop("pypto.backend")
    import torch
    from _csa_pipeline_check.arguments import PreparedCSAArguments, validate_root_aliases

    dtypes = {
        "BF16": torch.bfloat16,
        "FP16": torch.float16,
        "FP32": torch.float32,
        "INT8": torch.int8,
        "INT32": torch.int32,
        "INT64": torch.int64,
    }
    native = {
        k: torch.empty_strided(v["shape"], v["strides"], dtype=dtypes[v["dtype"]], device="meta")
        for k, v in current.schedule["root_arguments"].items()
    }
    prepared = PreparedCSAArguments(current, native)
    aliases = nz = 0
    for item, tensor in zip(prepared.contract["ordered_inputs"], prepared.core_args):
        ref = item["reference"]
        allocation = ref.get("allocation", "")
        if allocation.startswith("argument:"):
            original = native[allocation[9:]]
            assert original.untyped_storage()._cdata == tensor.untyped_storage()._cdata
            aliases += 1
            if ref["layout"] == "NZ":
                assert tensor is original
                nz += 1
    checks["meta_native_aliases"] = {"views": aliases, "nz_objects": nz, "inputs": len(prepared.core_args)}
    backing = torch.empty(16)
    roots = {"a": {"layout": "ND"}, "b": {"layout": "ND"}}
    contract = {
        "ordered_inputs": [
            {"reference": {"allocation": "argument:a"}, "mutable": True},
            {"reference": {"allocation": "argument:b"}, "mutable": False},
        ]
    }
    try:
        validate_root_aliases({"a": backing[:8], "b": backing[4:12]}, roots, contract)
    except ValueError:
        pass
    else:
        raise AssertionError("未拒绝跨root写侧重叠")
    validate_root_aliases({"a": backing[:8], "b": backing[8:]}, roots, contract)
    contract["ordered_inputs"][0]["mutable"] = False
    validate_root_aliases({"a": backing[:8], "b": backing[4:12]}, roots, contract)
    checks["cross_root_write_alias_rejected_read_alias_preserved"] = True
    report.update(passed=True, device_execution=False, performance_validated=False)
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": True, "checks": list(checks), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
