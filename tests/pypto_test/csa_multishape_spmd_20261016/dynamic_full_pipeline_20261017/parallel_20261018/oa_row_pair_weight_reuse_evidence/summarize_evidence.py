#!/usr/bin/env python3
"""汇总已完成的 O_A 试验；只读取历史结果，不启动设备、不剔除样本。

三方 S6 试验有原多 task PTO、f222a917 动态 SPMD、候选三侧；S1/S5
边界试验只有后两侧。二者不能混称为七档性能验收。所有百分比同时记录
耗时降低与用户要求的展示提升率，避免改分母后误判 30% 验收目标。
"""

import json
from pathlib import Path
from statistics import median


HERE = Path(__file__).resolve().parent
CASES = (
    ("h131072_b24_v1", "K64首版"),
    ("h8192_b32_v1", "K64首版"),
    ("h131072_b24_v4", "K128显式交错"),
    ("h8192_b32_v4", "K128显式交错"),
    ("h8192_b22_s6_region_boundary_v4", "T132区间代表"),
    ("h8192_b129_s1_boundary_v4_fixture_v3", "T129/S1下沿"),
    ("h8192_b32_s5_boundary_v4", "T160/S5代表"),
    ("h8192_b256_s1_boundary_v4_fixture_v3", "T256/S1容量上沿"),
    ("h32760_b32_s6_history_boundary_v4", "短历史上沿"),
)


def main():
    rows = []
    for name, purpose in CASES:
        path = HERE / name / "result.json"
        data = json.loads(path.read_text())
        assert data["stage"] == "complete" and data["passed"] and data["performance_measured"], name
        stats = data.get("source_matrix", data.get("l1", {}))["statistics_us"]
        reference = stats.get("spmd_reference", stats.get("reference"))
        candidate = stats.get("spmd_candidate", stats.get("candidate"))
        assert len(reference["samples_us"]) == len(candidate["samples_us"]) == 5
        ref = median(reference["samples_us"])
        new = median(candidate["samples_us"])
        assert ref == reference["median_us"] and new == candidate["median_us"]
        multi = stats.get("dynamic_pto")
        rows.append({
            "case": name,
            "purpose": purpose,
            "source_result": str(path.resolve()),
            "history": data["history"],
            "batch": data["batch"],
            "query_tokens": data["query_tokens"],
            "tokens": data["batch"] * data["query_tokens"],
            "precision_and_protocol_passed": True,
            "all_statistics_us": stats,
            "candidate_latency_reduction_pct": (ref - new) / ref * 100,
            "candidate_display_speedup_pct": (ref - new) / new * 100,
            "candidate_faster_window_pairs": sum(c < r for c, r in zip(candidate["samples_us"], reference["samples_us"])),
            "goal_passed_against_multitask_l1": None if multi is None else new <= multi["median_us"] * 0.70,
            "candidate_adopted": False,
        })
    report = {
        "comparison": "同进程、相同根地址、正式第4层W8A8权重；完整L1五窗口中位数",
        "scope": "输出/TopK/score、完整cache/state、保护区、A→B→A图和代际；历史为既有合成输入",
        "reference": "f222a917动态单SPMD；S6另含原多task PTO",
        "goal_rule": "候选完整L1 <= 同轮原多task完整L1 * 0.70",
        "decision": "不默认采用：8K大T有小收益，但长历史与短历史上沿均退步；不继续为小收益扫描更细分界",
        "rows": rows,
    }
    (HERE / "TERMINAL_RESULTS.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for row in rows:
        s = row["all_statistics_us"]
        ref = s.get("spmd_reference", s.get("reference"))["median_us"]
        new = s.get("spmd_candidate", s.get("candidate"))["median_us"]
        print(f"{row['case']}: {ref:.2f} -> {new:.2f} us, 降低 {row['candidate_latency_reduction_pct']:.3f}%, {row['candidate_faster_window_pairs']}/5 窗口更快")


if __name__ == "__main__":
    main()
