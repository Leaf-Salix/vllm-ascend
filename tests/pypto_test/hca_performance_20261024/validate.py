# SPDX-License-Identifier: Apache-2.0
"""从原始CSV/L1重新计算已采样边界，验证表格；纯CPU，不重测、不访问NPU。"""
import ast
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent


def main():
    result = json.loads((HERE / "PERFORMANCE.json").read_text())
    tree = ast.parse((HERE / "measure_hca.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "layer_device_interval")
    namespace = {"statistics": statistics}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "native_interval", "exec"), namespace)
    rows = []
    assert len(result["rows"]) == 8
    for row in result["rows"]:
        cells = row["metrics"]
        assert all(v["status"] == "MEASURED" for v in cells.values())
        native = json.loads(Path(cells["Native"]["source_report"]).read_text())
        pto = json.loads(Path(cells["HCA-MPMD-L1"]["source_report"]).read_text())
        assert native["case"] == row["case"]
        assert pto["query_lengths"] == row["case"]["query_lengths"]
        assert pto["history_lengths"] == row["case"]["history_lengths"]
        assert pto["token_capacity"] == row["case"]["token_capacity"]
        assert native["cann"] == pto["performance_spec"]["cann"]
        assert native["custom_opp"] == pto["performance_spec"]["custom_opp"]
        assert native["versions"] == pto["performance_spec"]["versions"]
        actual_native = namespace["layer_device_interval"](Path(cells["Native"]["source_report"]).parent / "profile")
        assert actual_native["span_us_all"] == cells["Native"]["samples_us"]
        actual_l1 = []
        for window in pto["performance_spec"]["l1"]["windows"]:
            raw = json.loads(Path(window["raw"]).read_text())
            records = raw["aicore_tasks"]
            assert raw["chip_swimlane_level"] == 1
            actual_l1.append((max(r[4] for r in records) - min(r[3] for r in records))
                             * 1e6 / raw["metadata"]["clock_freq_hz"])
        assert actual_l1 == cells["HCA-MPMD-L1"]["samples_us"]
        for cell in cells.values():
            assert len(cell["samples_us"]) >= 10
            assert statistics.median(cell["samples_us"]) == cell["median_us"]
        n, p = cells["Native"]["median_us"], cells["HCA-MPMD-L1"]["median_us"]
        assert row["native_over_l1_percent"] == 100 * (n-p)/p
        rows.append({"case_id": row["case"]["id"], "native_csv_recomputed": True,
                     "l1_raw_recomputed": True, "fixture_environment_equal": True,
                     "samples": {k: len(v["samples_us"]) for k, v in cells.items()}})
    value = {"passed": True, "new_device_execution": False, "cases": rows}
    (HERE / "VALIDATION.json").write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print("八档原始CSV、L1首尾、样本中位数、环境及比例复核通过；未重新执行设备。")


if __name__ == "__main__":
    main()
