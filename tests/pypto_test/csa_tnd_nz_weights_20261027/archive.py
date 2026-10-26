# SPDX-License-Identifier: Apache-2.0
"""核对两种TND八档的全部原始L1窗口，归档功能证据与逐档中位数。"""

import gzip
import json
import shutil
import statistics
from pathlib import Path

from l1 import summarize


def main():
    here = Path(__file__).resolve().parent
    cases = json.loads((here.parent / "csa_performance_20261022/cases.json").read_text())["cases"]
    submitted = json.loads((here / "submitted_tasks.json").read_text())
    by_case = {(row["case"], row["variant"]): row for row in submitted}
    assert len(by_case) == len(submitted) == 16
    expected_sdk = json.loads((here / "expected_sdk.json").read_text())
    output = here / "evidence/npu"
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for case in cases:
        row = {"case": case}
        for variant in ("spmd", "mpmd"):
            task = by_case[case["id"], variant]
            assert task["passed"] and task["status"] == "completed (exit=0)"
            source = here / task["result"]
            result = json.loads(source.read_text())
            assert result["passed"] and result["device_execution"] and result["case"] == case
            for key in ("release", "status", "pypto", "ptoas", "simpler"):
                assert result["shared_toolchain"][key] == expected_sdk[key]
            for binding in result["weight_binding"].values():
                assert binding["native_format"] == binding["prepared_format"] == 29
                assert binding["native_shape"] == binding["prepared_shape"]
                assert binding["same_data_ptr"] and binding["same_storage"] and binding["same_offset"]
            assert len(result["weight_binding"]) == 7 and all(result["weights_unchanged"].values())
            required = {
                "reference_eager",
                "candidate_eager",
                "reference_graph",
                "candidate_graph",
                "reference_post_L1",
                "candidate_post_L1",
            }
            if case["id"] == "C04":
                required.update(("normal_eager", "normal_graph"))
            assert required <= result["checks"].keys()
            for check in result["checks"].values():
                assert all(check["guards"].values())
                assert all(item["bit_equal"] and item["status"] == "PASS" for item in check["numerical"].values())
            destination = output / (variant + "_" + case["id"])
            destination.mkdir()
            shutil.copyfile(source, destination / "result.json")
            raw_windows = []
            durations = {"reference": [], "candidate": []}
            structures = {"reference": set(), "candidate": set()}
            for index, window in enumerate(result["windows"]):
                raw_path = Path(window["raw"])
                raw = json.loads(raw_path.read_text())
                measured = summarize(raw, variant)
                assert all(window[key] == value for key, value in measured.items())
                side = window["side"]
                durations[side].append(measured["incore_span_us"])
                structures[side].add((measured["records"], measured["task_groups"], measured["cores"]))
                raw_windows.append({"window": index, "side": side, "source": str(raw_path), "records": raw})
                config = destination / (side + "_kernel_config.txt")
                content = (raw_path.parent / "kernel_config.py").read_bytes()
                if config.exists():
                    assert config.read_bytes() == content
                else:
                    config.write_bytes(content)
            assert structures["reference"] == structures["candidate"]
            for side, samples in durations.items():
                assert len(samples) >= 10 and samples == result["statistics_us"][side]["samples"]
                assert statistics.median(samples) == result["statistics_us"][side]["median"]
            compressed = gzip.compress(json.dumps(raw_windows).encode(), mtime=0)
            (destination / "all_l1_records.json.gz").write_bytes(compressed)
            assert json.loads(gzip.decompress(compressed)) == raw_windows
            reference = statistics.median(durations["reference"])
            candidate = statistics.median(durations["candidate"])
            row[variant] = {
                "passed": True,
                "task_id": task["task_id"],
                "device": result["device_id"],
                "reference_us": reference,
                "nz_us": candidate,
                "latency_change_pct": (candidate / reference - 1) * 100,
                "samples_us": durations,
                "structure": sorted(structures["candidate"]),
                "evidence": str(destination.relative_to(here)),
            }
        rows.append(row)
    summary = {
        "passed": True,
        "scope": "TND-SPMD和TND-MPMD，各自对照修改前三张ND权重实现；不包含BSH或Native数值重测",
        "sampling": "每侧每档10个完整L1窗口，包含各自原有初始化；同卡同进程交替配对，取中位数",
        "performance_limit": "单轮配对采样，不据此声明亚百分比收益稳定或跨调度/跨设备比较绝对耗时",
        "normal_entry_checks": "两种调度B16额外验证正式入口eager/图；所有档位正式入口编译通过",
        "shared_toolchain": expected_sdk,
        "rows": rows,
    }
    (here / "NPU_gate.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print("16个场景、320个完整L1窗口及全部逐位正确性检查通过并归档。")


if __name__ == "__main__":
    main()
