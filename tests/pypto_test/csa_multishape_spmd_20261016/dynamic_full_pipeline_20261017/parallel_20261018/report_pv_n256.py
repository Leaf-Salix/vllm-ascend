#!/usr/bin/env python3
"""按角色字段汇总已完成的 PV N256 消融，不重新编译或启动设备任务。

三方运行顺序会旋转和反转，JSON 的字段排列也不固定；必须使用
statistics_us 的角色键，不能把打印出来的第一个/第二个数当成候选/参考。
七个目标点与额外短历史 B20 区间点分别标记，不用平均收益掩盖单档回退。
"""

import json
from pathlib import Path

from report_prefetch_quant_round import summarize


def main():
    evidence = Path(__file__).resolve().parent / "pv_n256_evidence"
    cases = []
    for registry in ("registry.json", "interval_registry.json", "long_remaining_registry.json"):
        report = summarize(evidence, registry)
        cases.extend(report["cases"])
    assert len(cases) == 8
    assert len({(case["history"], case["batch"]) for case in cases}) == 8
    report["cases"] = sorted(cases, key=lambda case: (-case["history"], case["batch"]))
    report["decision"] = (
        "不采用：完整七档及短B20区间点均功能精度PASS，但长G6内收益不一致，"
        "短区间全部回退；不制造B4/B16白名单，不开展边界补测。正式33源仍保持d253。"
    )
    report["target_cases"] = 7
    report["additional_interval_cases"] = 1
    lines = [
        "# PV N256 七档与区间补点最终结论", "", report["measurement"], "",
        report["precision_scope"], "", report["decision"], "",
        "| 历史 / B | 原分 task μs | d253 N128 μs | N256 μs | 相对 N128 降低 | 更快窗口 | 相对原分 task 降低 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in report["cases"]:
        stats = case["statistics_us"]
        lines.append(
            f"| {case['history']} / {case['batch']} | "
            f"{stats['dynamic_pto']['median_us']:.2f} | "
            f"{stats['spmd_reference']['median_us']:.2f} | "
            f"{stats['spmd_candidate']['median_us']:.2f} | "
            f"{case['candidate_vs_reference_reduction_pct']:.3f}% | "
            f"{case['candidate_faster_windows']}/5 | "
            f"{case['candidate_vs_dynamic_reduction_pct']:.3f}% |"
        )
    assert not any(case["target_met"] for case in cases)
    lines += [
        "",
        "8K/B20 是额外区间点，不计入七个目标规格。7/7 的 N256 均未达到相对原动态分 task 降低30%的目标。",
        "本轮采样按轮次旋转三方顺序、奇数轮反转；每侧重置、同步后单独采集。不是固定先后顺序对比。",
        "两侧实际 wait codegen 保持 100 tick 未就绪退避及相同协议。不能由不一致收益反推出硬件根因。",
        "PV constexpr 四文件接线的两次CPU仅证明不增加运行时代码，仍作为历史候选，不启用该策略。",
        "完整五窗、策略、任务号、输出路径见同名 JSON。", "",
    ]
    (evidence / "SEVEN_AND_INTERVAL_RESULTS.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    (evidence / "SEVEN_AND_INTERVAL_RESULTS.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
