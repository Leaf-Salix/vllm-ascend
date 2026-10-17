# SPDX-License-Identifier: Apache-2.0
"""复现生产 CSA DAG、分核和等待迁移的 CPU 对照，不编译或运行 NPU。

历史 JSON 只作为测试输入/参考；生产构造器仍只使用本轮绑定、生产源码合同
和调用方显式给出的模型成本。默认将本次运行结果打印到 stdout，不覆盖已存
graph_migration.json。输出文件须通过 --output 指定一个不存在的路径。
"""

import argparse
import copy
import importlib
import importlib.util
import json
import sys
from pathlib import Path

MEMBER_FIELDS = (
    "task",
    "original_subslot",
    "logical_block",
    "logical_block_num",
    "logical_block_stride",
    "dispatch_seq",
    "wait_for_core_progress",
    "publish_completion",
)
TORCH_DTYPES = {
    "FP32": "torch.float32",
    "BF16": "torch.bfloat16",
    "FP16": "torch.float16",
    "INT8": "torch.int8",
    "INT32": "torch.int32",
    "INT64": "torch.int64",
}


def load_production(package):
    """与原终端检查相同，按文件导入包，避免加载 vLLM 插件注册。"""
    name = "csa_graph_migration_cpu"
    spec = importlib.util.spec_from_file_location(
        name, package / "__init__.py", submodule_search_locations=[str(package)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return {part: importlib.import_module(name + "." + part) for part in ("graph", "wait_plan", "catalog", "binding")}


def task_graph(schedule):
    calls = schedule["call_bindings"]
    return {
        calls[task["task_id"]]["ordinal"]: {calls[parent]["ordinal"] for parent in task["predecessors"]}
        for task in schedule["tasks"]
    }


def members(schedule):
    return {
        core: [{field: row[field] for field in MEMBER_FIELDS} for row in rows]
        for core, rows in schedule["per_core_sequences"].items()
        if rows
    }


def expect_value_error(function, label):
    try:
        function()
    except ValueError:
        return {"check": label, "passed": True, "expected_exception": "ValueError"}
    raise AssertionError("未拒绝非法输入：" + label)


def check(repo):
    package = repo / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"
    history = repo / "tests/pypto_test/csa_static_spmd_feasibility_20261015/bindings"
    dynamic = repo / "tests/pypto_test/csa_multishape_spmd_20261016"
    api = load_production(package)
    graph, wait = api["graph"], api["wait_plan"]
    catalog = api["catalog"].catalog()
    contracts = graph.SourceContracts(catalog)
    report = {
        "scope": "CPU 源码 DAG/分核/等待差分，无 lowering/NPU/性能测量",
        "production_package": str(package),
        "seven_shapes": [],
        "passed": False,
    }

    for path in sorted(history.glob("*_prepared.json")):
        name = path.stem.removesuffix("_prepared")
        old = json.loads(path.read_text())
        bound = json.loads((history / (name + ".json")).read_text())
        schedule, audit = graph.build_source_schedule(bound, catalog, contracts=contracts)
        if graph.closure(task_graph(old)) != graph.closure(task_graph(schedule)):
            raise AssertionError("七档数学偏序不一致：" + name)
        if not audit["wait_plan"]["identical_dependency_partial_order"]:
            raise AssertionError("等待表改变成员偏序：" + name)
        report["seven_shapes"].append(
            {
                "case": name,
                "passed": True,
                "tasks": len(schedule["tasks"]),
                "reachable_pairs": audit["task_graph_reachable_pairs"],
                "members": schedule["summary"]["physical_members"],
            }
        )
    if len(report["seven_shapes"]) != 7:
        raise AssertionError("七档历史输入缺失或额外增加，需要明确更新测试集合")

    case = dynamic / "anchor_gap_design_v1/builder_check_v3_war/extra_h65536_b17"
    old = json.loads((case / "prepared.json").read_text())
    old_audit = json.loads((case / "source_audit.json").read_text())
    descriptors = {
        name: {"shape": ref["shape"], "stride": ref["strides"], "dtype": TORCH_DTYPES[ref["dtype"]]}
        for name, ref in old["root_arguments"].items()
    }
    bound = api["binding"].bind(Path(catalog["logical_source_ir"]), descriptors, max_seq_len=65542)
    # 旧调度器在记录的 model_cost 上加1；这里显式还原这项，生产算法不再偷偷加1。
    costs = {
        (old["call_bindings"][row["task"]]["ordinal"], row["slot"]): row["model_cost"] + 1
        for row in old_audit["cost_sources"]
    }
    schedule, _ = graph.build_source_schedule(bound, catalog, contracts=contracts, member_costs=costs)
    if members(old) != members(schedule):
        raise AssertionError("B17 显式同成本时分核/等待/发布不一致")
    if graph.closure(task_graph(old)) != graph.closure(task_graph(schedule)):
        raise AssertionError("B17 数学任务偏序不一致")
    report["b17_exact_cost"] = {
        "case": "h65536_b17",
        "passed": True,
        "host_max_seq_len": 65542,
        "cost_key": "(call_ordinal,slot)",
        "cost_value": "historical model_cost + 1",
        "compared_member_fields": list(MEMBER_FIELDS),
        "compared_core_keys_and_sequence_order": True,
        "task_closure_equal": True,
        "excluded_fields": ["model timestamps", "kernel_id", "source_line"],
    }

    short = json.loads((dynamic / "anchor_gap_design_v1/class_0_h8191_b1_bindings.json").read_text())
    schedule, audit = graph.build_source_schedule(short, catalog, contracts=contracts, policy="critical_path")
    if len(schedule["tasks"]) != 65 or audit["task_graph_reachable_pairs"] != 1333:
        raise AssertionError("短 Vector B1 源码 DAG 与记录不符")
    report["short_vector"] = {
        "case": "class_0_h8191_b1",
        "passed": True,
        "policy": "critical_path",
        "tasks": 65,
        "reachable_pairs": 1333,
    }

    checks = []
    for label, invalid in (
        ("unknown_cost_kernel", {"unknown": 1}),
        ("zero_cost", {"hc_widen_rms": 0}),
        ("nan_cost", {"hc_widen_rms": float("nan")}),
        ("boolean_cost", {"hc_widen_rms": True}),
    ):
        checks.append(
            expect_value_error(
                lambda values=invalid: graph.build_source_schedule(
                    short, catalog, contracts=contracts, member_costs=values
                ),
                label,
            )
        )
    for label, parents in (("member_cycle", [{1}, {0}]), ("unknown_member_parent", [{2}, set()])):
        checks.append(expect_value_error(lambda edges=parents: wait.ancestor_sets(edges), label))
    view = {"allocation": "x", "dtype": "FP32", "offset": 0, "shape": [8], "strides": [1], "manual_dep": False}
    if graph._overlap(view, {**view, "offset": 4, "shape": [0]}):
        raise AssertionError("空 view 被当成非空重叠")
    checks.append({"check": "empty_view_has_no_overlap", "passed": True})

    def task(ordinal, role):
        return {
            "ordinal": ordinal,
            "kind": "compute",
            "scope": [],
            "options": {},
            "logical_function": role,
            "arguments": [{"parameter": "x", "value": view}],
        }

    synthetic = {
        "tasks": [
            task(0, "read"),
            task(1, "write"),
            {
                "ordinal": 2,
                "kind": "dependency_only",
                "scope": [],
                "options": {"deps": [{"task_ordinal": 1}]},
                "arguments": [],
            },
            task(3, "read"),
        ],
        "scopes": [],
    }
    synthetic["tasks"][3]["options"] = {"deps": [{"task_ordinal": 2}]}
    access = {"read": {"x": {"R"}}, "write": {"x": {"R", "W"}}}
    parents, explicit, war = graph.infer_dependencies(synthetic, access)
    if not (0 in parents[1] and 1 in parents[3] and 1 in explicit[3] and (0, 1) in war):
        raise AssertionError("WAR/RAW/dummy 展开不符")
    checks.append({"check": "war_raw_and_dummy_expansion", "passed": True})
    manual = copy.deepcopy(synthetic)
    manual["scopes"] = [{"id": 1, "options": {"mode": "manual"}}]
    manual["tasks"][1]["scope"] = [1]
    parents, _, _ = graph.infer_dependencies(manual, access)
    if parents[1]:
        raise AssertionError("manual scope 被强行添加自动依赖")
    checks.append({"check": "manual_scope_skips_automatic_edges", "passed": True})
    report.update(edge_checks=checks, passed=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--output", type=Path, help="可选的新结果文件，不覆盖已有文件")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("输出路径已存在，禁止覆盖已有记录")
    report = check(args.repo.resolve())
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        with args.output.open("x") as handle:
            handle.write(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
