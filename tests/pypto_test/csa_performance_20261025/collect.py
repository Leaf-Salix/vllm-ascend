# SPDX-License-Identifier: Apache-2.0
"""按八档六列规范复算已有原始样本，并合并本轮必要补测；不占卡、不扫描源码hash。"""

import ast
import copy
import importlib.util
import json
import math
import statistics
from collections import Counter
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TESTS = HERE.parent
SPMD_TREE = REPO.parent / ".worktrees/csa-spmd-tnd-20261020"
FOCUS = SPMD_TREE / "tests/pypto_test/csa_spmd_tnd_focus_20261025"
RECOVERY = SPMD_TREE / "tests/pypto_test/csa_spmd_tnd_workspace_recovery_20261025"
COLUMNS = ("Native", "BSH-MPMD-L1", "BSH-SPMD", "TND-MPMD-AICPU", "TND-MPMD-L1", "TND-SPMD")


def read(path):
    return json.loads(Path(path).read_text())


def checks(data):
    assert data["stage"] == "complete" and data["passed"]
    assert data["checks"] and all(data["fixture_checks"].values())
    for key, row in data["checks"].items():
        assert all(row["guards"].values()), key
        assert all(v["status"] == "PASS" for v in row["numerical"].values()), key
    return {
        key: dict(
            guards=True, numerical=True, bit_equal=all(v.get("bit_equal", False) for v in row["numerical"].values())
        )
        for key, row in data["checks"].items()
    }


def sdk_matches(actual, expected):
    assert actual["status"] == expected["status"] == "validated"
    for name in ("pypto", "ptoas", "simpler"):
        assert actual[name]["commit"] == expected[name]["commit"], name


def l1(data, side, single):
    windows = [w for w in data["windows"] if w["side"] == side]
    assert len(windows) >= 10
    samples, provenance = [], []
    for window in windows:
        raw = read(window["raw"])
        meta = raw["metadata"]
        rows, hz = raw["aicore_tasks"], meta["clock_freq_hz"]
        assert raw["chip_swimlane_level"] == 1 and rows and math.isfinite(hz) and hz > 0
        assert all(len(r) == 6 and r[3] > 0 and r[4] >= r[3] for r in rows)
        assert len(meta["run_boundaries"]) == 1 and meta.get("dropped_run_boundaries", 0) == 0
        assert Counter(meta["core_types"]) == {"aic": 24, "aiv": 48}
        if single:
            assert len(rows) == len({r[0] for r in rows}) == 72 and len({r[1] for r in rows}) == 1
        else:
            assert len({r[1] for r in rows}) > 1
        value = (max(r[4] for r in rows) - min(r[3] for r in rows)) * 1e6 / hz
        assert abs(value - window["incore_span_us"]) < 1e-7
        samples.append(value)
        provenance.append(
            dict(
                raw=window["raw"],
                sample_us=value,
                valid=True,
                cores=len({r[0] for r in rows}),
                task_groups=len({r[1] for r in rows}),
            )
        )
    return samples, provenance


def main():
    manifest = read(HERE / "manifest.json")
    cases = read(HERE / "cases.json")["cases"]
    case_map = {c["id"]: c for c in cases}
    old = read(TESTS / "csa_performance_20261023/PERFORMANCE.json")
    cells = {c["id"]: {column: dict(status="待采集") for column in COLUMNS} for c in cases}
    for column in ("BSH-MPMD-L1", "BSH-SPMD"):
        cells["C08"][column] = dict(status="N/A", reason="原BSH接口不支持混长T56，按规范不测")
    excluded, evidence = [], {}
    # 仅提取既有CPU汇总函数；不导入Torch或初始化任何设备。
    module_path = TESTS / "csa_five_way_latest_20261020/summarize.py"
    spec = importlib.util.spec_from_file_location("native_summary", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tree = ast.parse((TESTS / "csa_five_way_latest_20261020/csa_measure.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "layer_device_interval")
    namespace = {"statistics": statistics}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "original_device_interval", "exec"), namespace)
    interval = namespace["layer_device_interval"]
    exact_namespace = {"statistics": statistics, "float": Decimal}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "exact_device_interval", "exec"), exact_namespace)

    def exact_interval(directory):
        # 同一过滤/归组规则；先用Decimal对约1.8e15微秒的绝对时间戳做差，再转普通时长。
        value = exact_namespace["layer_device_interval"](directory)
        value["span_us_all"] = [float(x) for x in value["span_us_all"]]
        value["span_us"] = float(value["span_us"])
        value["timestamp_arithmetic"] = "Decimal；过滤、stream/model归组及完整首尾边界不变"
        return value

    for case in cases:
        row = copy.deepcopy(old["cells"][case["id"]]["Native"])
        assert row["input_contract"] == case
        source = Path(row["source_report"])
        data = read(source)
        assert (data["history"], data["batch"], data["tokens"], data["layer_index"], data["seed"]) == (
            case["history_label"],
            case["active_requests"],
            case["token_capacity"],
            4,
            case["seed"],
        )
        if "case" in data:
            assert data["case"] == case
        _, proof = module.outer(source, data, native=True)
        row["raw_csv"] = proof["profile"]["raw_csv"]
        measured = interval(Path(row["raw_csv"]).parents[2])
        assert measured["span_us_all"] == row["samples_us"]
        exact = exact_interval(Path(row["raw_csv"]).parents[2])
        row["legacy_float_samples_us"] = row["samples_us"]
        row["samples_us"] = exact["span_us_all"]
        row["timestamp_arithmetic"] = exact["timestamp_arithmetic"]
        assert len(row["samples_us"]) >= 10
        row.update(
            median_us=statistics.median(row["samples_us"]),
            reuse=True,
            reuse_reason="Native不依赖PTO SDK；已验证CANN9.2/static/SuperKernel/多流/NZ2路径与本规范输入保持一致",
            actual_source_version=row.get("source_commit"),
            current_pto_sdk_applicable=False,
        )
        cells[case["id"]]["Native"] = row
        evidence[str(source)] = {"path_checks": proof, "timing_boundary_verified": True}

    # 正式八档验收的最终单task版本；前一版有硬件回退分支，不混入本列。
    adopted = read(FOCUS / "single_spmd_v2_eight_case_validation.json")
    bridge = read(FOCUS / "production_integration.json")
    assert adopted["adopted"] and bridge["adopted"] and bridge["source_equivalence_passed"]
    fragments = {c["id"]: [] for c in cases}
    seen = set()

    def add_spmd(path, data, side, task, source_reason):
        cid = data["case"]["id"]
        assert data["case"] == case_map[cid]
        sdk_matches(data["shared_toolchain"], manifest["shared_toolchain"])
        assert data["ring_config"] == {"heap_mb": [16, 256, 16, 16], "task_window": 4096}
        proof = checks(data)
        samples, windows = l1(data, side, True)
        keys = [w["raw"] for w in windows]
        assert not (set(keys) & seen), "不能重复统计同一窗口"
        seen.update(keys)
        fragments[cid].append(
            dict(
                source_report=str(path),
                task_id=task,
                side=side,
                selected_snapshot="f72a1f752 / init_coeff_single_spmd_v2",
                device=data["device_id"],
                sampling=data.get("sampling"),
                prepare_order=data.get("prepare_order"),
                source_reason=source_reason,
                samples_us=samples,
                windows=windows,
            )
        )
        evidence[str(path)] = dict(
            checks=proof,
            fixture_checks=data["fixture_checks"],
            weights=data["weights"],
            policies=data.get("policies"),
            sources=data.get("sources"),
            ring_config=data["ring_config"],
        )

    for case in adopted["cases"]:
        summary = read(FOCUS / case["result"])
        path = FOCUS / summary["raw_report"]
        data = read(path)
        add_spmd(path, data, "candidate", summary["task_id"], "已采用正式单task版本八档完整设备验收")

    initial_path = RECOVERY / "evidence/initial_triple_C04_r1.json"
    initial = read(initial_path)
    assert Path(initial["sources"]["candidate"]).name == "internal_single_f72a"
    add_spmd(
        initial_path,
        initial,
        "candidate",
        read(RECOVERY / "initial_triple_C04_r1.json")["task_id"],
        "迁移三侧复测中的正式f72版本；保留全部10窗",
    )

    # 后续对照只纳入未改动f72控制侧，不取未采用候选，也排除插桩/GM扰动/改变预热节奏。
    for path in sorted((RECOVERY / "evidence").glob("*.json")):
        data = read(path)
        if not isinstance(data, dict) or not data.get("windows") or "current" not in data.get("sources", {}):
            continue
        if data.get("performance_eligible") is False or data.get("diagnostic_only"):
            excluded.append(dict(source=str(path), reason="插桩、L2/GM扰动或改变采样节奏，非正式性能"))
            continue
        if Path(data["sources"]["current"]).name != "internal_single_f72a":
            excluded.append(dict(source=str(path), reason="非所选f72版本"))
            continue
        if not data.get("passed") or data.get("stage") != "complete":
            excluded.append(dict(source=str(path), reason="未完整通过设备检查"))
            continue
        summaries = [RECOVERY / (path.stem + "_result.json")]
        summaries += list(RECOVERY.glob("*_result.json"))
        task = None
        for summary_path in dict.fromkeys(summaries):
            if not summary_path.is_file():
                continue
            summary = read(summary_path)
            if summary.get("attempt") == path.stem:
                task = summary["task_id"]
                break
        if task is None:
            # 早期三侧诊断采用独立摘要命名，从归档任务清单定位，不猜task ID。
            for summary_path in (RECOVERY / "current_state.json", RECOVERY / "initial_triple_C04_r1.json"):
                if not summary_path.is_file():
                    continue
                summary = read(summary_path)
                entries = summary.get("completed_tasks", []) if isinstance(summary, dict) else []
                for item in entries:
                    if item.get("attempt") == path.stem:
                        task = item["task_id"]
        if task is None:
            excluded.append(dict(source=str(path), reason="缺少可核对的task关联；保留记录，未拼接样本"))
            continue
        add_spmd(path, data, "current", task, "同SDK/输入/正常采样的f72控制侧；全部窗口保留")
        if data["sources"].get("candidate") == data["sources"]["current"]:
            add_spmd(path, data, "candidate", task, "同源A/A另一实例；同口径全部有效窗口，不选择快侧")
    for case in cases:
        parts = fragments[case["id"]]
        samples = [x for p in parts for x in p["samples_us"]]
        cells[case["id"]]["TND-SPMD"] = dict(
            status="PASS",
            reuse=True,
            input_contract=case,
            selected_snapshot="f72a1f752",
            task_structure="单MIX，初始化/硬屏障/业务均在L1内，24AIC+48AIV",
            samples_us=samples,
            valid_windows=len(samples),
            median_us=statistics.median(samples),
            toolchain=manifest["shared_toolchain"],
            fragments=parts,
        )

    invalid = []
    for job in read(HERE / "jobs.json") if (HERE / "jobs.json").exists() else []:
        path = Path(job["output"]) / "result.json"
        if job.get("status") != "completed (exit=0)":
            continue
        if not path.exists():
            continue
        data = read(path)
        if data.get("stage") != "complete" or not data.get("passed"):
            invalid.append(
                dict(task_id=job["task_id"], result=str(path), stage=data.get("stage"), error=data.get("error"))
            )
            continue
        case = case_map[job["case"]]
        assert data["case"] == case
        sdk_matches(data["shared_toolchain"], manifest["shared_toolchain"])
        proof = checks(data)
        for side, column in (
            (("mpmd", "TND-MPMD-L1"),) if job["family"] == "tnd" else (("mpmd", "BSH-MPMD-L1"), ("spmd", "BSH-SPMD"))
        ):
            samples, windows = l1(data, side, single=side == "spmd")
            cells[case["id"]][column] = dict(
                status="PASS",
                reuse=False,
                input_contract=case,
                source_report=str(path),
                task_id=job["task_id"],
                actual_source=(
                    str(Path(manifest["bsh"]["source"]) / "deepseek_v4_flash_csa")
                    if job["family"] == "bsh" and side == "mpmd"
                    else data["sources"][side]
                ),
                raw_source_field_note=(
                    "旧壳sources.mpmd沿用candidate参数；实际BSHEntry调用冻结BSH NativeCSACall"
                    if job["family"] == "bsh" and side == "mpmd"
                    else None
                ),
                samples_us=samples,
                windows=windows,
                valid_windows=len(samples),
                median_us=statistics.median(samples),
                toolchain=manifest["shared_toolchain"],
                environment=data["environment"],
                sampling=data["sampling"],
            )
        if job["family"] == "tnd":
            outer = data["outer_profile"]
            measured = interval(Path(outer["raw_csv"]).parents[2])
            assert measured["span_us_all"] == outer["span_us_all"] and measured["span_includes_all_streams"]
            assert any("aicpu" in t.lower() for t in outer["task_types"])
            assert any("aicore" in t.lower() for t in outer["task_types"])
            exact = exact_interval(Path(outer["raw_csv"]).parents[2])
            samples = exact["span_us_all"]
            cells[case["id"]]["TND-MPMD-AICPU"] = dict(
                status="PASS",
                reuse=False,
                input_contract=case,
                source_report=str(path),
                task_id=job["task_id"],
                actual_source=data["sources"]["mpmd"],
                samples_us=samples,
                legacy_float_samples_us=outer["span_us_all"],
                valid_windows=len(samples),
                median_us=statistics.median(samples),
                toolchain=manifest["shared_toolchain"],
                environment=data["environment"],
                sampling=dict(warmup=3, replays_per_window=1, valid_windows=30),
                profile={**outer, **exact},
                boundary="同一实现图的全部相关device stream首尾，包含AICPU/AICore；L1与Profiler分开采",
            )
        evidence[str(path)] = dict(
            checks=proof,
            fixture_checks=data["fixture_checks"],
            weights=data["weights"],
            sources=data["sources"],
            policies=data["policies"],
        )
    missing = [(c["id"], col) for c in cases for col in COLUMNS if cells[c["id"]][col]["status"] == "待采集"]
    # 完整固定场景只保存一份；逐格引用，避免重复堆放Q/L数组和同一层权重描述。
    for case in cases:
        for cell in cells[case["id"]].values():
            if "input_contract" in cell:
                cell["input_contract"] = dict(
                    case_id=case["id"],
                    definition="cases.json",
                    H=case["history_label"],
                    B=case["request_capacity"],
                    T=case["token_capacity"],
                    verified_equal=True,
                )
    for path, proof in evidence.items():
        if "weights" in proof:
            proof["weights"] = dict(
                model=read(HERE / "cases.json")["model"], layer=4, full_record=path, field="weights"
            )
    result = dict(
        manifest=manifest,
        generated_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        cells=cells,
        provenance=evidence,
        excluded=excluded,
        invalid=invalid,
        missing=missing,
        final=not missing,
        native_precision_note=old.get("native_precision_note"),
        formula="100*(median_native-median_tnd_spmd)/median_tnd_spmd",
    )
    (HERE / "PERFORMANCE.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "| 场景 | Native (μs) | BSH-MPMD-L1 (μs) | BSH-SPMD (μs) | TND-MPMD-AICPU (μs) | "
        "TND-MPMD-L1 (μs) | TND-SPMD (μs) | (Native − TND-SPMD) / TND-SPMD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        row = cells[case["id"]]
        values = [
            f"{row[col]['median_us']:.2f}" if row[col]["status"] == "PASS" else row[col]["status"] for col in COLUMNS
        ]
        native, spmd = row["Native"], row["TND-SPMD"]
        ratio = (
            f"{100 * (native['median_us'] - spmd['median_us']) / spmd['median_us']:.2f}%"
            if native["status"] == spmd["status"] == "PASS"
            else "—"
        )
        label = f"{case['history_label'] // 1024}K/B{case['active_requests']}/" + (
            "混长 T56" if case["id"] == "C08" else "S6"
        )
        lines.append("| " + label + " | " + " | ".join([*values, ratio]) + " |")
    (HERE / "PERFORMANCE.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            dict(
                final=not missing,
                missing=missing,
                reused_native=8,
                spmd_windows={c["id"]: cells[c["id"]]["TND-SPMD"]["valid_windows"] for c in cases},
                invalid_attempts=len(invalid),
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
