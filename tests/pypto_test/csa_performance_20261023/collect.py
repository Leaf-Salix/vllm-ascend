# SPDX-License-Identifier: Apache-2.0
"""依据八档规范复用既有数据；只读原始计时，不执行设备测试或源码hash扫描。"""

import ast
import copy
import importlib.util
import json
import statistics
import subprocess
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OLD = HERE.parent / "csa_performance_20261022_aicpu"
MPMD = HERE.parent / "csa_mpmd_aicpu_20261022"
SPMD_REPO = REPO.parent / ".worktrees/csa-spmd-tnd-20261020"
SPMD = SPMD_REPO / "tests/pypto_test/csa_spmd_tnd_eight_20261022"
COLUMNS = ("Native", "BSH-MPMD-L1", "BSH-SPMD", "TND-MPMD-AICPU", "TND-MPMD-L1", "TND-SPMD")


def read(path):
    return json.loads(Path(path).read_text())


def write(name, data):
    (HERE / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def require_checks(report):
    assert report["stage"] == "complete" and report["passed"]
    result = {}
    for name, check in report["checks"].items():
        guards = all(check["guards"].values())
        numerical = all(row["status"] == "PASS" for row in check["numerical"].values())
        assert guards and numerical, name
        result[name] = dict(
            guards=guards,
            numerical=numerical,
            bit_equal=all(row.get("bit_equal", False) for row in check["numerical"].values()),
        )
    return result


def require_sdk(actual, expected):
    assert actual["status"] == expected["status"] == "validated"
    assert actual["release"] == expected["release"]
    for name in ("pypto", "ptoas", "simpler"):
        if isinstance(actual.get(name), dict):
            assert actual[name]["commit"] == expected[name]["commit"]


def l1(report, side, single):
    windows = [window for window in report["windows"] if window["side"] == side]
    assert len(windows) >= 10
    samples, paths = [], []
    for window in windows:
        raw = read(window["raw"])
        rows = raw["aicore_tasks"]
        assert raw["chip_swimlane_level"] == 1 and rows
        assert all(len(row) == 6 and row[3] > 0 and row[4] >= row[3] for row in rows)
        if single:
            assert len(rows) == len({row[0] for row in rows}) == 72
            assert len({row[1] for row in rows}) == 1
        else:
            assert len({row[1] for row in rows}) > 1
        value = (max(row[4] for row in rows) - min(row[3] for row in rows)) * 1e6 / raw["metadata"]["clock_freq_hz"]
        assert abs(value - window["incore_span_us"]) < 1e-7
        samples.append(value)
        paths.append(window["raw"])
    return samples, paths


def main():
    previous = read(OLD / "PERFORMANCE.json")
    shared = read(HERE.parent / "csa_shared_toolchain/manifest.json")
    require_sdk(previous["manifest"]["shared_toolchain"], shared)
    cases = read(HERE / "cases.json")["cases"]
    case_by_name = {case["name"]: case for case in cases}
    cells = copy.deepcopy(previous["cells"])
    for row in cells.values():
        for cell in row.values():
            cell["reuse"] = True
    provenance = {}
    rejected = []

    # 复用原正式测量器的全设备边界；此函数本身不导入torch或分配设备。
    source = HERE.parent / "csa_five_way_latest_20261020"
    tree = ast.parse((source / "csa_measure.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "layer_device_interval")
    namespace = {"statistics": statistics}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "original_device_interval", "exec"), namespace)
    interval = namespace["layer_device_interval"]
    spec = importlib.util.spec_from_file_location("native_checks", source / "summarize.py")
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)

    # 固定v22整包来源；纳入全部已发现的同版合格窗口，包括后续消融的控制组。
    target = str(MPMD / "experiments/aicpu_v22b_bsh_order_live_workers/operator")
    fragments = {case["id"]: [] for case in cases}
    seen = set()
    for job in read(MPMD / "jobs.json"):
        path = MPMD / job["output"] / "result.json"
        if not path.is_file():
            continue
        data = read(path)
        case = case_by_name.get(data["case"]["name"])
        if case is None or not data.get("windows"):
            continue
        for side, actual_source in data.get("sources", {}).items():
            if actual_source != target or (str(path), side) in seen:
                continue
            seen.add((str(path), side))
            checks = require_checks(data)
            assert data["case"] == case
            # 此壳的实际公共绑定在日志中保留；冻结manifest记录完整SDK提交。
            require_sdk(
                read(MPMD / "experiments/aicpu_v22b_bsh_order_live_workers/manifest.json")["shared_toolchain"], shared
            )
            samples, raw = l1(data, side, single=False)
            fragments[case["id"]].append(
                dict(
                    source_report=str(path),
                    task_id=job["task_id"],
                    side=side,
                    source_date=job["task_id"][5:13],
                    device=data.get("device_id"),
                    samples_us=samples,
                    raw=raw,
                )
            )
            provenance[str(path)] = dict(
                checks=checks,
                environment=data.get("environment"),
                actual_sources=data["sources"],
                policies=data.get("policies"),
                weights=data.get("weights"),
                fixture_checks=data.get("fixture_checks"),
                sampling=dict(warmup=3, l1_windows_per_side=10, replays_per_window=1),
            )
    for case in cases:
        parts = fragments[case["id"]]
        assert parts
        samples = [x for part in parts for x in part["samples_us"]]
        cells[case["id"]]["TND-MPMD-L1"] = dict(
            status="PASS",
            reuse=True,
            input_contract=case,
            selected_snapshot="v22 / d86c95f69",
            source=target,
            samples_us=samples,
            median_us=statistics.median(samples),
            valid_windows=len(samples),
            raw=[x for part in parts for x in part["raw"]],
            fragments=parts,
            source_reports=[part["source_report"] for part in parts],
            actual_toolchain_selection=shared,
            timing="完整链路全部incore首尾，包含task间等待",
            sampling="每窗一次graph replay；每个已纳入来源的全部有效样本",
        )

    qualified = read(MPMD / "scheduling_v22_qualified.json")
    for row in qualified["results"]:
        case = row["case"]
        path = REPO / row["source"]
        data = read(path)
        assert data["case"] == case_by_name[case["name"]]
        require_checks(data)
        actual = interval(path.parent / "profile_mpmd")
        samples = actual["span_us_all"]
        assert actual["span_includes_all_streams"] and len(samples) >= 10
        assert samples == data["outer_profile"]["span_us_all"]
        cells[case["id"]]["TND-MPMD-AICPU"] = dict(
            status="PASS",
            reuse=True,
            input_contract=case,
            selected_snapshot="v22 / d86c95f69",
            source=target,
            source_report=str(path),
            source_reports=[str(path)],
            task_id=row["task_id"],
            source_date=row["task_id"][5:13],
            device=data["device_id"],
            samples_us=samples,
            valid_windows=len(samples),
            median_us=statistics.median(samples),
            raw_csv=actual["raw_csv"],
            span_includes_all_streams=True,
            task_types=data["outer_profile"]["task_types"],
            actual_toolchain_selection=shared,
            runtime_environment=data["environment"],
            paired_l1_note=data["outer_profile"]["paired_l1_note"],
            timing="全部相关stream的AICPU+AICore device跨度；排除测试专用reset",
        )

    # 选定一个已通过设备门禁的冻结快照；七个关闭档用现有精确代码等价证据复用。
    eq = read(HERE / "evidence/output_rectangle_v3_codegen_validation.json")
    runtime = read(HERE / "evidence/output_rectangle_v3_runtime_validation.json")
    rect = read(HERE / "evidence/output_rectangle_v3.json")
    assert eq["passed"] and runtime["passed"]
    for case in cases:
        gate = eq["cases"][case["id"]]
        assert gate["kernel_abi_equal"]
        if case["id"] != "C05":
            assert gate["all_four_cpp_equal_to_retained"]
    assert read(HERE / "evidence/physical_capacity_production_v1_codegen.json")["passed"]
    measurements = {
        "C02": read(HERE / "evidence/physical_capacity_2080_v1_C02_result.json"),
        "C04": read(HERE / "evidence/leaf_scale_one_row_v1_C04_result.json"),
        "C05": read(HERE / "evidence/output_rectangle_v3_results.json")["cases"]["C05"],
        "C06": read(HERE / "evidence/critical_prefix_v2_expansion_results.json")["cases"]["C06"],
    }
    for case in cases:
        key = case["id"]
        cell = cells[key]["TND-SPMD"]
        cell["selected_snapshot"] = "output_rectangle_v3"
        cell["selected_source"] = rect["source"]
        cell["source_mapping"] = "manifest.json:selected_versions.tnd_spmd"
        cell["code_equivalence_evidence"] = dict(
            case=key,
            final="evidence/output_rectangle_v3_codegen_validation.json",
            production="evidence/physical_capacity_production_v1_codegen.json",
            parent="evidence/physical_capacity_tier_v1_codegen.json",
            earlier="evidence/merged_schedule_v1_codegen.json",
        )
        if key not in measurements:
            continue
        measure = measurements[key]
        path = Path(measure.get("source_report", measure.get("report", "")))
        if not path.is_absolute():
            path = SPMD_REPO / path
        data = read(path)
        assert data["case"] == case
        checks = require_checks(data)
        require_sdk(data["shared_toolchain"], shared)
        samples, raw = l1(data, "candidate", single=True)
        assert all(
            abs(a - b) < 1e-7 for a, b in zip(samples, data["statistics_us"]["candidate"]["samples"], strict=True)
        )
        if key == "C05":
            assert data["source_hashes"]["candidate"] == rect["files"]  # 比较已有收据，不计算文件hash。
        cell.update(
            samples_us=samples,
            median_us=statistics.median(samples),
            valid_windows=len(samples),
            raw=raw,
            source_report=str(path),
            source_reports=[str(path)],
            source_tasks=[measure["task_id"]],
            task_id=measure["task_id"],
            device=data["device_id"],
            source_date=measure["task_id"][5:13],
            actual_toolchain_selection=data["shared_toolchain"],
            actual_sources=data["sources"],
            actual_policies=data["policies"],
            task_log=f"task-submit --log {measure['task_id']}",
            task_log_evidence=f"TASK_LOG_EVIDENCE.json:{key}",
            sampling=data["sampling"],
            reuse=True,
            measured_candidate=data["sources"]["candidate"],
            source_manifest=measure.get("source_manifest"),
            runtime_environment=dict(
                source="同一公共bootstrap与虚拟环境，CANN9.2由任务日志确认",
                comparison_environment=previous["cells"][key]["TND-SPMD"]["runtime_environment"],
            ),
        )
        provenance[str(path)] = dict(
            checks=checks,
            shared_toolchain=data["shared_toolchain"],
            weights=data["weights"],
            fixture_checks=data["fixture_checks"],
            descriptors=data["descriptors"],
            source_hashes_receipt=data.get("source_hashes"),
            sampling=data["sampling"],
        )

    # 逐格确认原始记录可用；旧列同样读取实际边界，不能只相信Markdown数字。
    verified = []
    for case in cases:
        row = cells[case["id"]]
        assert tuple(row) == COLUMNS
        for column, cell in row.items():
            if case["id"] == "C08" and column.startswith("BSH"):
                assert cell["status"] == "N/A"
                continue
            assert cell["status"] == "PASS" and cell["input_contract"] == case
            assert cell["valid_windows"] == len(cell["samples_us"]) >= 10
            cell["median_us"] = statistics.median(cell["samples_us"])
            if column == "Native":
                path = Path(cell["source_report"])
                data = read(path)
                assert "9.2.0-beta.2" in data["cann"]
                _, evidence = native.outer(path, data, native=True)
                actual = interval(Path(evidence["profile"]["raw_csv"]).parents[2])
                assert actual["span_us_all"] == cell["samples_us"]
                provenance[str(path)] = dict(
                    compile=data["torch_compile_options"],
                    backend=data["backend_options"],
                    multistream=data["multistream"],
                    formats=data["native_weight_formats"],
                    framework=data["versions"],
                    path_evidence=evidence,
                )
            elif column not in ("TND-MPMD-AICPU", "TND-MPMD-L1"):
                require_sdk(cell["actual_toolchain_selection"], shared)
                actual = []
                for raw_path in cell["raw"]:
                    raw = read(raw_path)
                    records = raw["aicore_tasks"]
                    assert raw["chip_swimlane_level"] == 1 and records
                    if column.endswith("SPMD"):
                        assert len(records) == len({r[0] for r in records}) == 72
                        assert len({r[1] for r in records}) == 1
                    actual.append(
                        (max(r[4] for r in records) - min(r[3] for r in records))
                        * 1e6
                        / raw["metadata"]["clock_freq_hz"]
                    )
                assert len(actual) == len(cell["samples_us"])
                assert all(abs(a - b) < 1e-7 for a, b in zip(actual, cell["samples_us"], strict=True))
                for path in cell.get("source_reports", [cell["source_report"]]):
                    checks = require_checks(read(path))
                    provenance.setdefault(path, dict(checks=checks))
            verified.append(
                dict(case=case["id"], column=column, windows=len(cell["samples_us"]), median_us=cell["median_us"])
            )
    assert len(verified) == 46
    manifest = dict(
        generated_at=datetime.now().astimezone().isoformat(),
        report_source_commit=subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        spec="SPEC.md",
        shared_toolchain=shared,
        device_execution=False,
        new_device_tasks=0,
        selected_versions=dict(
            native=previous["manifest"]["native"],
            bsh_mpmd=previous["manifest"]["bsh_mpmd"],
            bsh_spmd=previous["manifest"]["bsh_spmd"],
            tnd_mpmd=dict(commit="d86c95f69", source=target),
            tnd_spmd=dict(
                snapshot="output_rectangle_v3",
                source=rect["source"],
                evidence_commit="bb37e1cce",
                default_service_adopted=False,
                selection="已完成完整链路十窗、16+4状态及service门禁的冻结候选；非未验证HEAD",
            ),
        ),
        timing="完整HC_pre→input RMSNorm→CSA→HC_post；按规范保留不同计时边界",
        history_rule="同v22来源的已发现合格L1窗口全部纳入，包含后续控制组；SPMD按单一快照和现有逐档代码等价证据复用",
        large_files="原始trace/CSV使用逐格可访问路径；本次不复制或重生成大型trace",
        optimization_goal="保持paused；本次仅按用户指令汇总数据，不恢复优化、不push",
    )
    write("manifest.json", manifest)
    write(
        "PERFORMANCE.json",
        dict(
            manifest=manifest,
            cells=cells,
            invalid=rejected,
            provenance=provenance,
            native_precision_note=previous["native_precision_note"],
        ),
    )
    write(
        "VALIDATION.json",
        dict(
            passed=True,
            device_execution=False,
            valid_cells=46,
            not_applicable_cells=2,
            total_samples=sum(v["windows"] for v in verified),
            minimum_windows=min(v["windows"] for v in verified),
            cells=verified,
            source_equivalence="复用已保存的编译等价收据；未重新计算源码/工具链hash",
            denominator="median_tnd_spmd",
        ),
    )
    lines = [
        "| 场景 | Native (μs) | BSH-MPMD-L1 (μs) | BSH-SPMD (μs) | TND-MPMD-AICPU (μs) "
        "| TND-MPMD-L1 (μs) | TND-SPMD (μs) | (Native − TND-SPMD) / TND-SPMD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        row = cells[case["id"]]
        label = f"{case['history_label'] // 1024}K/B{case['active_requests']}/" + (
            "混长 T56" if case["id"] == "C08" else "S6"
        )
        values = [f"{row[c]['median_us']:.2f}" if row[c]["status"] == "PASS" else row[c]["status"] for c in COLUMNS]
        pct = 100 * (row["Native"]["median_us"] - row["TND-SPMD"]["median_us"]) / row["TND-SPMD"]["median_us"]
        lines.append("| " + " | ".join([label, *values, f"{pct:.2f}%"]) + " |")
    (HERE / "PERFORMANCE.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print({"validated_cells": 46, "samples": sum(v["windows"] for v in verified), "device_tasks": 0})


if __name__ == "__main__":
    main()
