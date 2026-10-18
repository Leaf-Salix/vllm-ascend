# SPDX-License-Identifier: Apache-2.0
"""完整动态CSA编译：不传入样例shape，不运行设备，不把CPU成功作为精度或性能证据。"""

import argparse
import ast
import importlib
import json
import re
import traceback
from pathlib import Path


def audit_codegen(kernel, report):
    """核对已有生成物；根入参动态轴与内部派生视图须分别检查。"""
    contract = json.loads((Path(__file__).parent / "source_interface.json").read_text())
    root_names = set(re.findall(r"\b[A-Z][A-Z0-9_]*\b", " ".join(contract["runtime_tensors"].values())))
    root_dims = {"CSA_FULL_" + name for name in root_names.intersection(contract["dynamic_axes"])}
    orchestration = "\n".join(p.read_text() for p in (kernel / "orchestration").glob("*.cpp"))
    submissions = re.findall(r"\brt_submit_\w+\s*\(", orchestration)
    report["submissions"] = submissions
    report["single_mix"] = len(submissions) == 1 and "MixedKernels" in orchestration
    report["sync_start"] = "set_require_sync_start(true)" in orchestration
    assert report["single_mix"] and report["sync_start"]
    cpp_dims = {}
    for path in (kernel / "kernels").rglob("*.cpp"):
        cpp_dims[str(path.relative_to(kernel))] = re.findall(
            r"// Extract dynamic dim: (CSA_FULL_\w+)", path.read_text()
        )
    report["cpp_dimensions"] = cpp_dims
    report["required_root_dimensions"] = sorted(root_dims)
    assert root_dims <= {dim for values in cpp_dims.values() for dim in values}
    # KEY0/KEY64的内部helper维度在内联时替换成原cache尺寸表达式，无须再有独立Extract。
    report["derived_key_views"] = {
        name: bool(
            re.search(
                r"uint32_t " + name + r"_shapes\[2\] = .*CSA_FULL_ICACHE_PAGES.*CSA_FULL_ICACHE_BYTES", orchestration
            )
        )
        for name in ("idx_key0", "idx_key64")
    }
    assert all(report["derived_key_views"].values())
    assert not re.search(r"\brt_alloc\w*\s*\(", orchestration)
    report["compiled_abi"] = json.loads((kernel / "compiled_meta.json").read_text())["params"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strategy", choices=("long", "medium"), default="long")
    parser.add_argument("--hc-row-dependencies", choices=("on", "off"), default="on")
    parser.add_argument("--hc-consumer-reduce", choices=("on", "off"), default="off")
    parser.add_argument("--attention-query-metadata", choices=("on", "off"), default="on")
    parser.add_argument("--spmd-source", type=Path, default=Path(__file__).resolve().parent / "source_candidate")
    parser.add_argument("--controls-json", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    package = args.spmd_source.resolve() if args.spmd_source else here.parents[3] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"
    controls = dict(
        QUERY_GROUP=6,
        QK_PANEL=64,
        SCORE_TILE=1024,
        PREFETCH=1,
        TO_L0=True,
        BALANCED=True,
        ROOTS=1,
        OA_COLS=256,
        OB_ROWS=128,
        OUTPUT_GROUP_PIPELINE=True,
        HC_ROW_DEPENDENCIES=args.hc_row_dependencies == "on",
        ATTENTION_QUERY_METADATA=args.attention_query_metadata == "on",
        HC_CONSUMER_REDUCE=args.hc_consumer_reduce == "on",
    )
    if args.strategy == "medium":
        controls.update(
            QUERY_GROUP=2, QK_PANEL=128, SCORE_TILE=768, PREFETCH=0, BALANCED=False, ROOTS=2, OA_COLS=128, OB_ROWS=96
        )
    if args.controls_json:
        overrides = json.loads(args.controls_json.read_text())
        if not isinstance(overrides, dict) or any(type(v) not in (bool, int) for v in overrides.values()):
            raise ValueError("CPU策略覆盖只允许constexpr名称到int/bool的JSON对象")
        controls.update(overrides)
    report = dict(
        passed=False,
        scope="完整源级动态CSA单SPMD",
        controls=controls,
        sample_shapes_supplied=False,
        device_execution=False,
        device_precision_validated=False,
        performance_measured=False,
    )
    try:
        cube_source = ast.parse((package / "cube_schedule.py").read_text())
        cube = next(n for n in cube_source.body if isinstance(n, ast.FunctionDef) and n.name == "cube_schedule")
        attributes = [
            ast.literal_eval(n.value.args[0])
            for n in cube.body
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == "pl.func_attr"
        ]
        policies = next(a["cache_policy"] for a in attributes if "cache_policy" in a)
        names = [arg.arg for arg in cube.args.args]
        report["weight_cache_policy"] = {names[index]: value for index, value in policies}
        assert report["weight_cache_policy"] == {n: 1 for n in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b")}
        (output / "source").mkdir()
        for path in package.glob("*.py"):
            (output / "source" / path.name).write_text(path.read_text())
        for path in here.glob("*.py"):
            (output / path.name).write_text(path.read_text())
        import dsv4_csa_env

        dsv4_csa_env.activate()
        from pypto import passes
        from pypto.ir.printer import python_print
        from pypto.runtime import CompileOptions

        from source_loader import load_package

        isolated = load_package(args.spmd_source, "cpu_candidate")
        decode_csa_spmd = importlib.import_module(isolated.name + ".entry").decode_csa_spmd
        report["source_package"] = str(isolated.source)

        logical = python_print(decode_csa_spmd.specialize(**controls))
        (output / "symbolic.py").write_text(logical)
        report["symbolic_parse_passed"] = True
        dimensions = set(re.findall(r'pl.dynamic\("(CSA_FULL_\w+)"\)', logical))
        report["symbolic_dimensions"] = sorted(dimensions)
        assert dimensions, "没有提供样例shape时仍必须保留动态Tensor维度"
        print(json.dumps({"symbolic_parse_passed": True, "dimensions": len(dimensions)}), flush=True)
        artifact = decode_csa_spmd._resolve_kernel_artifact(
            (),
            {
                **controls,
                "config": CompileOptions(
                    platform="a2a3",
                    output_dir=str(output / "kernel"),
                    dump_passes=True,
                    memory_planner=passes.MemoryPlanner.PYPTO,
                ),
            },
            allow_signature_mode=True,
        )
        report["artifact_execution_mode"] = "kernel"
        kernel = Path(artifact.output_dir)
        report["artifact"] = str(kernel)
        report["full_codegen_passed"] = True
        audit_codegen(kernel, report)
        report["passed"] = True
    except BaseException as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
    finally:
        (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(
            json.dumps({k: v for k, v in report.items() if k not in ("traceback", "compiled_abi")}, ensure_ascii=False),
            flush=True,
        )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
