# SPDX-License-Identifier: Apache-2.0
"""只读核验汇总与原始计时边界，输出验证收据；不执行设备任务。"""

import ast
import importlib.util
import json
import statistics
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
COLUMNS = ("Native", "BSH-MPMD-L1", "BSH-SPMD", "TND-MPMD-AICPU", "TND-MPMD-L1", "TND-SPMD")


def read(path):
    return json.loads(Path(path).read_text())


def main():
    report = read(HERE / "PERFORMANCE.json")
    cases = read(HERE / "cases.json")["cases"]
    assert len(cases) == 8 and not report["invalid"]
    shared = report["manifest"]["shared_toolchain"]
    assert shared["status"] == "validated"
    # 原生路径检查器不导入 torch；保留静态编译/SuperKernel/NZ/多流实际生效判据。
    source = HERE.parent / "csa_five_way_latest_20261020"
    spec = importlib.util.spec_from_file_location("native_report_checks", source / "summarize.py")
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    # 直接复用原测量器的全设备边界函数，避免导入其不必要的设备测试依赖。
    tree = ast.parse((source / "csa_measure.py").read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "layer_device_interval"
    )
    namespace = {"statistics": statistics}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "original_device_interval", "exec"), namespace)
    interval = namespace["layer_device_interval"]
    cells, precision = [], {}
    for case in cases:
        row = report["cells"][case["id"]]
        assert tuple(row) == COLUMNS
        precision[case["id"]] = {}
        for column, cell in row.items():
            if case["id"] == "C08" and column.startswith("BSH"):
                assert cell["status"] == "N/A"
                continue
            assert cell["status"] == "PASS", (case["id"], column)
            samples = cell["samples_us"]
            assert len(samples) == cell["valid_windows"] >= 10
            assert statistics.median(samples) == cell["median_us"]
            assert all(value > 0 for value in samples)
            assert cell["input_contract"] == case
            path = Path(cell["source_report"])
            raw_report = read(path)
            entry = dict(
                case=case["id"],
                column=column,
                windows=len(samples),
                median_us=cell["median_us"],
                source=str(path),
                reused=cell["reuse"],
            )
            if column == "Native":
                assert "9.2.0-beta.2" in raw_report["cann"]
                assert (raw_report["history"], raw_report["batch"], raw_report["tokens"], raw_report["seed"]) == (
                    case["history_label"],
                    case["active_requests"],
                    case["token_capacity"],
                    case["seed"],
                )
                _, evidence = native.outer(path, raw_report, native=True)
                directory = Path(evidence["profile"]["raw_csv"]).parents[2]
                actual = interval(directory)
                assert actual["span_us_all"] == samples
                entry.update(
                    cann=raw_report["cann"],
                    framework=raw_report["versions"],
                    compile_options=raw_report["torch_compile_options"],
                    backend_options=raw_report["backend_options"],
                    multistream=raw_report["multistream"],
                    static_super_kernels=evidence["static_super_kernels"],
                    weight_formats=raw_report["native_weight_formats"],
                    raw_csv=actual["raw_csv"],
                    pto_sdk="不适用",
                )
            else:
                assert cell["actual_toolchain_selection"]["release"] == shared["release"]
                for check_path in cell.get("source_reports", [str(path)]):
                    checked = read(check_path)
                    assert checked["stage"] == "complete" and checked["passed"]
                    bit_equal = True
                    for check in checked["checks"].values():
                        assert all(check["guards"].values())
                        assert all(value["status"] == "PASS" for value in check["numerical"].values())
                        bit_equal &= all(value.get("bit_equal", False) for value in check["numerical"].values())
                    precision[case["id"]][str(check_path)] = dict(
                        numerical="PASS", guards="PASS", all_checked_tensors_bit_equal=bit_equal
                    )
                if column == "TND-MPMD-AICPU":
                    actual = interval(path.parent / "profile_mpmd")
                    assert actual["span_us_all"] == samples
                    entry.update(
                        raw_csv=actual["raw_csv"],
                        all_streams=actual["span_includes_all_streams"],
                        kernels_per_replay=actual["kernels_per_replay"],
                        streams_per_replay=actual["stream_ids_per_replay"],
                    )
                else:
                    actual = []
                    for raw_path in cell["raw"]:
                        raw = read(raw_path)
                        rows = raw["aicore_tasks"]
                        assert raw["chip_swimlane_level"] == 1 and rows
                        assert all(len(r) == 6 and r[3] > 0 and r[4] >= r[3] for r in rows)
                        if column.endswith("SPMD"):
                            assert len(rows) == len({r[0] for r in rows}) == 72 and len({r[1] for r in rows}) == 1
                        else:
                            assert len({r[1] for r in rows}) > 1
                        actual.append(
                            (max(r[4] for r in rows) - min(r[3] for r in rows)) * 1e6 / raw["metadata"]["clock_freq_hz"]
                        )
                    assert actual == samples
            cells.append(entry)
    assert len(cells) == 46
    receipt = dict(
        passed=True,
        checked_at=datetime.now().astimezone().isoformat(),
        device_execution=False,
        valid_cells=len(cells),
        not_applicable_cells=2,
        shared_release=shared["release"],
        minimum_windows=min(cell["windows"] for cell in cells),
        valid_samples=sum(cell["windows"] for cell in cells),
        cells=cells,
        precision=precision,
        denominator="median_tnd_spmd",
        native_precision="Native 自身 eager/graph 一致；既有跨实现浮点差异未改口径",
    )
    (HERE / "VALIDATION.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(
        {
            key: receipt[key]
            for key in ("passed", "valid_cells", "not_applicable_cells", "minimum_windows", "valid_samples")
        }
    )


if __name__ == "__main__":
    main()
