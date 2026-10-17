# SPDX-License-Identifier: Apache-2.0
"""从实际记录汇总准备状态；不调用编译器或设备，不抄录旧的排队状态。"""

import json
from datetime import datetime
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def all_pass(value):
    if isinstance(value, dict):
        return value.get("status", "PASS") == "PASS" and all(all_pass(v) for v in value.values())
    if isinstance(value, list):
        return all(all_pass(v) for v in value)
    return True


def main():
    here = Path(__file__).resolve().parent
    result_root = here.parent / "results" / here.name
    reports = []
    for path in sorted((here / "bindings").glob("*_prepared.json")):
        case = path.name.removesuffix("_prepared.json")
        schedule = read(path)
        layout = read(here / "static" / case / "layout.json")
        model = read(here / "static" / case / "protocol_model.json")
        parsed = read(here / "static" / case / "parse_result.json")
        captured = read(Path(schedule["source"]).parent / "report.json")
        reports.append({"case": case, "schedule_capture": schedule["source"],
                        "capture_status": captured["status"],
                        "original_graph_replay_passed": all_pass(captured["timing"]["eager_comparison"]),
                        "guards_passed": all_pass(captured["initial_guards"]) and all_pass(captured["timing"]["guards"]),
                        **schedule["summary"],
                        "logical_block_mapping_complete": schedule["logical_block_mapping_complete"],
                        "tensor_bindings_checked": sum(v["tensor_arguments_checked"] for v in schedule["binding_checks"]),
                        "all_call_bindings_verified": schedule["all_call_bindings_verified"],
                        "mixed_logical_groups_checked": schedule["mixed_logical_groups_checked"],
                        "source_batches": {role: len(rows) for role, rows in layout["batches"].items()},
                        "workspace_bytes": layout["workspace"]["bytes"],
                        "protocol_model_passed": all(v["status"] == "PASS" for v in model["progress_generation_model"]),
                        "parse_passed": parsed["parse_passed"], "entry_contract": {
                            k: v for k, v in parsed["entry_contract"].items() if k != "ordered_inputs"}})
    probes = []
    for name in ("static_b4_cpu_v15", "static_b4_cpu_v16_dsa_rp", "static_b4_cpu_v17_ptoas", "static_b4_cpu_v18_external", "static_b4_cpu_v19_valid_view", "static_b4_cpu_v21_buffer_ir", "static_b4_cpu_v22_buffer_ir", "static_b4_cpu_v23_flat_phi", "static_b4_jit_cpu_v1", "static_b4_jit_cpu_v2", "static_b4_jit_cpu_v3"):
        path = result_root / name / "result.json"
        if path.exists():
            probe = read(path)
            probes.append({"run": name, "record": str(path),
                           **{k: v for k, v in probe.items() if k not in ("traceback", "source")}})
    blockers = [
        {"name": "自动 MIX 拆分 while 状态丢失", "component": "PyPTO",
         "issue": "https://github.com/hw-native-sys/pypto/issues/3009",
         "scope": "独立复现仍有效；当前显式 Group 路径的 pass25 保留轮询 carry，不能混用两条路径的结论"},
        {"name": "CORE Tensor ABI 容量", "component": "Simpler/PyPTO 接口",
         "current_limit": 32, "prototype_leaf_counts": [132, 133],
         "evidence": "SIMPLER_CORE_ARGUMENT_CAPACITY.md", "local_dependency_modification": False},
        {"name": "完整融合核片上布局及视图 lowering 尚未通过", "component": "PyPTO/PTOAS 编译链，需进一步归因",
         "evidence": "COMPILE_AND_RESUME.md",
         "scope": "默认 PYPTO Left 81920 > 65536；DSA_RP 无布局；PTOAS 路径生成的 qk_pv AIV MLIR 存在不支配使用点的 SSA 定义。没有证明硬件不可实现"},
    ]
    result = {"updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
              "objective": "把原动态 CSA 排程固化到单 MIX SPMD；完成组件问题修复前可以独立推进的准备",
              "status": "PREPARED_WITH_TOOLCHAIN_BLOCKERS", "cases": reports,
              "compute_body_catalog": read(here / "generated/body_parse_result.json"),
              "compile_probes": probes, "blockers": blockers,
              "static_device_execution": False, "static_performance_measured": False,
              "production_or_dependency_changes_this_preparation": False,
              "next_steps": ["解决 CORE 参数承载方式与完整融合核编译限制，保留现有 tile 和流水语义",
                             "确认单 MIX 提交、while 状态、片上容量及内部事件生命周期",
                             "task-submit 最小协议验证，随后 B4 功能、七档精度，最后原方式性能比较"],
              "do_not_repeat": ["七档动态映射补采", "完整 73 核体导出", "原实参和分片匹配", "已有 CPU 协议模型检查"]}
    binding_file = here / "static/argument_binding_result.json"
    if binding_file.exists():
        result["test_adapter_meta_bindings"] = read(binding_file)
    audit_file = here / "static/b4_ir_audit.json"
    if audit_file.exists():
        result["explicit_group_ir_audit"] = read(audit_file)
    result["focused_compile_probes"] = []
    for name in ("memory_indexer_PYPTO", "memory_oproj_PYPTO", "memory_pair_compact_PYPTO", "memory_pair_compact_DSA_RP_v2", "jit_group_wrapper_v1"):
        path = result_root / name / "result.json"
        if path.exists():
            result["focused_compile_probes"].append({"run": name, "record": str(path),
                                                     **{k: v for k, v in read(path).items() if k != "traceback"}})
    (here / "preparation_progress.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    historical = here / "assessment_initial.json"
    if not historical.exists():
        historical.write_text((here / "assessment.json").read_text())
    assessment = {"updated_at": result["updated_at"],
                  "conclusion": "七档结构准备和 CPU 绑定已建立；不能声称只修复 #3009 即可部署，也尚无静态运行/性能结论",
                  "current_record": "preparation_progress.json", "historical_record": historical.name,
                  "resolved_gaps": ["七档同轮 logical block/subslot 映射", "完整核体和原参数绑定", "静态表、工作区及代际模型"],
                  "remaining_blockers": blockers,
                  "performance_limit": "DEBUG 映射补采不代表历史最佳排程；这里只固定工作与依赖，不固定墙钟时间"}
    (here / "assessment.json").write_text(json.dumps(assessment, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "cases": len(reports), "compile_probes": len(probes)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
