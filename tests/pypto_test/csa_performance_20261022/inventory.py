# SPDX-License-Identifier: Apache-2.0
"""复核可复用Native原始记录，明确PTO旧记录缺项，保留全部样本。"""

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent


def main():
    spec = importlib.util.spec_from_file_location(
        "native_history_summary", TESTS / "csa_five_way_latest_20261020/summarize.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = json.loads((HERE / "cases.json").read_text())["cases"]
    cells = {}
    for case in cases:
        case_id = case["id"]
        cells[case_id] = {}
        folder = f"h{case['history_label']}_b{case['active_requests']}"
        attempt = "native_supplement_20261021_v1" if case_id == "C01" else "native_v1"
        report_path = TESTS / "csa_five_way_latest_20261020/results" / folder / attempt / "report.json"
        if case_id in ("C02", "C08"):
            cells[case_id]["Native"] = dict(status="待采集", reason="历史入口缺少256K或固定T56真实混长场景")
        else:
            report = json.loads(report_path.read_text())
            assert report["status"] == "MEASURED"
            assert "9.2.0-beta.2" in report["cann"]
            assert (report["history"], report["batch"], report["tokens"], report["seed"]) == (
                case["history_label"],
                case["active_requests"],
                case["token_capacity"],
                case["seed"],
            )
            median, evidence = module.outer(report_path, report, native=True)
            cells[case_id]["Native"] = dict(
                status="PASS",
                reuse=True,
                median_us=median,
                samples_us=evidence["samples_us"],
                valid_windows=len(evidence["samples_us"]),
                source_report=str(report_path),
                raw_csv=evidence["profile"]["raw_csv"],
                source_date="2026-10-21" if case_id == "C01" else "2026-10-20",
                device=report["device"],
                cann=report["cann"],
                versions=report["versions"],
                path_checks=evidence["compiler"],
                graph=evidence["graph"],
                input_contract=case,
                fixture="原Native seed1024场景；Indexer scale使用非均匀37/251公式",
            )
        for name in ("BSH-MPMD-L1", "BSH-SPMD", "TND-MPMD-AICPU", "TND-MPMD-L1", "TND-SPMD"):
            cells[case_id][name] = dict(
                status="N/A" if case_id == "C08" and name.startswith("BSH") else "待采集",
                reason="规范排除BSH混长"
                if case_id == "C08" and name.startswith("BSH")
                else "旧MPMD主矩阵是混长带padding；旧S6/SPMD多为每侧5窗或旧SDK，不能直接作为本规范样本",
            )
    (HERE / "inventory.json").write_text(json.dumps(cells, ensure_ascii=False, indent=2) + "\n")
    print({case: row["Native"].get("median_us", row["Native"]["status"]) for case, row in cells.items()})


if __name__ == "__main__":
    main()
