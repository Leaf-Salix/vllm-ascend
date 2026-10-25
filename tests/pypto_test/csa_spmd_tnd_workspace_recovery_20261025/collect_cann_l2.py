# SPDX-License-Identifier: Apache-2.0
"""用CPU区间与执行次序关联CANN外层MIX计数，保存全部样本及解释边界。"""

import csv
import json
import math
import shutil
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    run = here / "runs/h131072_b16_s6_cann_l2_r1"
    result = json.loads((run / "result.json").read_text())
    assert result["passed"] and len(result["windows"]) == 30 and len(result["checks"]) == 9
    csv_path = next((run / "cann_l2").rglob("kernel_details.csv"))
    with csv_path.open() as source:
        rows = list(csv.DictReader(source))
    mix = sorted([r for r in rows if r["Accelerator Core"] == "MIX_AIC"], key=lambda r: float(r["Start Time(us)"]))
    trace = json.loads(next((run / "cann_l2").rglob("trace_view.json")).read_text())
    events = trace if isinstance(trace, list) else trace["traceEvents"]
    labels = sorted([r for r in events if r.get("name", "").startswith("csa_l2/")], key=lambda r: float(r["ts"]))
    assert len(mix) == len(labels) == 30
    samples = []
    groups = defaultdict(list)
    for index, (row, label, window) in enumerate(zip(mix, labels, result["windows"])):
        side = window["side"]
        sample = int(label["name"].split("/")[1])
        assert label["name"].split("/")[2] == side
        start = float(row["Start Time(us)"])
        end = start + float(row["Duration(us)"])
        assert float(label["ts"]) <= start
        if index + 1 < len(labels):
            assert end <= float(labels[index + 1]["ts"])
        assert int(row["Block Num"]) == 24 and int(row["Mix Block Num"]) == 48
        item = {
            "sample": sample,
            "side": side,
            "label": label["name"],
            "cpu_start_us": label["ts"],
            "cann_start_us": row["Start Time(us)"].strip(),
            "outer_MIX_us": float(row["Duration(us)"]),
            "incore_L1_us": window["incore_span_us"],
            "counter_scope": "CANN外层kernel-mode MIX，包含runtime，不是仅CSA内层incore",
        }
        for core in ("aic", "aiv"):
            for key, value in row.items():
                if key.startswith(core + "_") and ("cache_" in key or key.endswith("total_cycles")):
                    item[key] = float(value)
                    assert math.isfinite(item[key]) and item[key] >= 0
            hits = sum(item[f"{core}_{port}_read_cache_hit"] for port in ("r0", "r1"))
            misses = sum(item[f"{core}_{port}_read_cache_miss_allocate"] for port in ("r0", "r1"))
            assert hits + misses > 0
            item[f"{core}_read_hits"] = hits
            item[f"{core}_read_miss_allocate"] = misses
            item[f"{core}_read_hit_fraction_pct"] = 100 * hits / (hits + misses)
        samples.append(item)
        groups[side].append(item)
    summary = {}
    for side, items in groups.items():
        assert len(items) == 10 and {x["sample"] for x in items} == set(range(10))
        summary[side] = {
            key: statistics.median([x[key] for x in items])
            for key in items[0]
            if key.startswith(("aic_", "aiv_")) or key in ("outer_MIX_us", "incore_L1_us")
        }
    evidence = here / "evidence"
    shutil.copyfile(csv_path, evidence / "cann_l2_C04_r1_kernel_details.csv")
    shutil.copyfile(run / "result.json", evidence / "h131072_b16_s6_cann_l2_r1.json")
    report = {
        "task_id": "task_20261025_174640_325114713557",
        "status": "valid_counters_and_order_mapping",
        "performance_eligible": False,
        "strict_check_groups": 9,
        "mapping": "30个CPU标签与30个MIX顺序一致，每个MIX完整落在本次标签与下次标签之间，24AIC+48AIV",
        "read_fraction_formula": "(r0_hit+r1_hit)/(r0_hit+r1_hit+r0_miss_allocate+r1_miss_allocate)",
        "summary_median": summary,
        "samples": samples,
        "interpretation": "旧外部与当前内部总体读计数比例接近，不支持明显整体L2读命中退化；内部Vector写事件较多",
        "limits": [
            "外层MIX计数包含runtime，不能与单个内层CSA阶段直接等同",
            "全算子聚合不能排除小范围关键阶段受到影响，或证明清零贡献全部差额",
            "CANN提示acl-to-npu自动flow关联失败；本报告用可验证CPU区间及顺序关联，不依赖缺失的flow",
            "插桩L1只作诊断，不能替代正式无profiler计时或与历史数值计算收益",
        ],
    }
    (here / "cann_l2_C04_r1_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
