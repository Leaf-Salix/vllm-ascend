# SPDX-License-Identifier: Apache-2.0
"""只读汇总阶段通知端点，辅助解释当前动态 CSA 的关键路径。

输入是同一调用72个核的 phase_clock.json。按角色与阶段分别统计已发布
端点、相邻通知间隔、GM TWAIT及FFTS原位时钟；没有NPU初始化或重新测量。
相邻通知间隔不一定等于一个数学helper：有些核穿插其它工作，输出组还嵌套
在整段阶段中。因此不把各阶段中位相加，也不将other称为纯计算。
原位FFTS区间记录wait_flag_dev前后Scalar时钟；其语义另附A3官方说明，
即使含等待，也无法单独区分生产者服务、同步协议和插桩引起的扰动。
这些数字不能替代完整L1性能验收，跨作业的差值更不能当作优化收益。
"""

import argparse
import json
from pathlib import Path
from statistics import median


def stats(values):
    """保留范围和中位，空集合显式返回None，不捏造零耗时。"""
    if not values:
        return None
    return {"min": min(values), "median": median(values), "max": max(values)}


def summarize(path):
    records = json.loads(path.read_text())
    groups = {}
    for row in records:
        role = "AIC" if row["core"] < 24 else "AIV"
        groups.setdefault((row["phase"], role), []).append(row)
    result = []
    for (phase, role), rows in sorted(groups.items()):
        fields = (
            "notify_start_us", "notify_end_us", "interval_us", "wait_us",
            "other_us", "ffts_issue_us", "ffts_issue_calls",
        )
        result.append({
            "phase": phase,
            "name": rows[0]["name"],
            "role": role,
            "participants": len(rows),
            "cores": [row["core"] for row in rows],
            "statistics": {
                field: stats([row[field] for row in rows if field in row])
                for field in fields
            },
            "last_publisher": max(rows, key=lambda row: row["notify_end_us"]),
        })
    return {
        "source": str(path.resolve()),
        "eligible_for_performance": False,
        "other_is_pure_compute": False,
        "stage_medians_additive": False,
        "records": len(records),
        "phases": result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clock", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.clock)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for phase in result["phases"]:
        values = phase["statistics"]
        print(json.dumps({
            "phase": phase["phase"], "name": phase["name"], "role": phase["role"],
            "end_us": values["notify_end_us"], "interval_us": values["interval_us"],
            "gm_wait_us": values["wait_us"], "ffts_issue_us": values["ffts_issue_us"],
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
