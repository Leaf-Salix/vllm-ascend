"""只读既有六档 DFX；Worker View 计一次，保留核内与排队的区别。"""

import json
import re
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT.parent.parent / "csa_six_case_profiles_20260926"


def main():
    result = {"scope": "正式 layer4 权重、合成输入/历史，单次 eager DFX；不等于整模型无 profiler 耗时",
              "cases": {}}
    for history, batch in ((131072, 4), (131072, 8), (131072, 16), (8192, 24), (8192, 32), (8192, 40)):
        name = f"h{history}_b{batch}"
        directory = BUNDLE / name
        trace = json.loads((directory / "pto_layer4_swimlane.json").read_text())["traceEvents"]
        worker_pid = next(e["pid"] for e in trace if e.get("name") == "process_name"
                          and e.get("args", {}).get("name") == "Worker View")
        raw = json.loads((directory / "dfx_source/chip_swimlane_records.json").read_text())["aicore_tasks"]
        events = [e for e in trace if e.get("ph") == "X" and e["pid"] == worker_pid
                  and "kernel-duration-us" in e.get("args", {})]
        tasks = {}
        for prefix in ("indexer_key_repack", "indexer_score_topk_leaf_aic", "indexer_score_topk_leaf_aiv",
                       "indexer_topk_query_merge", "indexer_topk_single_leaf_publish", "qk_pv_aiv"):
            subset = [e for e in events if e["name"].startswith(prefix)]
            if not subset:
                continue
            rows = []
            for e in subset:
                a = e["args"]
                core = int(re.search(r"CoreId:(\d+)", a["event-hint"])[1])
                rows.append({"core_id": core, "task_id": a["taskId"], "receive_us": e["ts"],
                             "start_us": e["ts"] + a["local_setup_us"], "end_us": e["ts"] + e["dur"],
                             "incore_us": a["kernel-duration-us"], "setup_us": a["local_setup_us"]})
            counts = Counter(row["core_id"] for row in rows)
            task_id = rows[0]["task_id"]
            # 原始记录确认重复核是真实派发，不是 Scheduler/Worker 两种视图被相加。
            raw_counts = Counter(row[0] for row in raw if row[1] == task_id and
                                 ((row[0] < 24) if "_aic" in prefix else (row[0] >= 24)))
            assert counts == raw_counts, (name, prefix, counts, raw_counts)
            tasks[prefix] = {
                "instances": len(rows), "physical_cores": len(counts),
                "repeated_core_ids": {str(k): v for k, v in counts.items() if v > 1},
                "incore_mean_us": statistics.mean(r["incore_us"] for r in rows),
                "incore_max_us": max(r["incore_us"] for r in rows),
                "compute_start_spread_us": max(r["start_us"] for r in rows) - min(r["start_us"] for r in rows),
                "compute_group_span_us": max(r["end_us"] for r in rows) - min(r["start_us"] for r in rows),
                "setup_max_us": max(r["setup_us"] for r in rows), "rows": rows,
            }
        result["cases"][name] = tasks
    (ROOT / "existing_swimlane_analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for name, tasks in result["cases"].items():
        score = tasks["indexer_score_topk_leaf_aic"]
        print(name, "AIC", score["physical_cores"], "incore", round(score["incore_mean_us"], 2),
              "group span", round(score["compute_group_span_us"], 2))


if __name__ == "__main__":
    main()
