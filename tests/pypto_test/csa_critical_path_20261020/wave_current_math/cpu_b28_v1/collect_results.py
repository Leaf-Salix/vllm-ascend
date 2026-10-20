# SPDX-License-Identifier: Apache-2.0
"""从已完成任务重建证据与七档选择，不提交任务，不重新测量。

五个启用档与 T=66 边界读取本实验候选；B4/B8 保持关闭路径，使用
上一实验的最近参考结果。二者的完整关闭 C++ 已核对一致，不取历史最小值。
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def collect():
    rows = []
    compiler = (HERE.parent / "score_key_l1_region_sync_20261019/toolchain_v3/bin/ptoas").resolve()
    for entry in read(HERE / "DEVICE_SUBMISSIONS.json"):
        if entry.get("exit_code") != 0:
            continue
        folder = Path(entry["output"])
        report = read(folder / "result.json")
        matrix = read(folder / "source_matrix/result.json")
        assert report["passed"] and matrix["passed"]
        invocations = [json.loads(line) for line in (folder / "COMPILER_INVOCATIONS.jsonl").read_text().splitlines()]
        assert invocations and all(Path(row["compiler"]).resolve() == compiler for row in invocations)
        stats = matrix["statistics_us"]
        baseline, reference, candidate = [stats[role]["median_us"] for role in
                                         ("dynamic_pto", "spmd_reference", "spmd_candidate")]
        rows.append(dict(
            batch=entry["batch"], history=entry["history"], boundary=entry.get("boundary", False),
            task=entry["task"], passed=True, exit_code=entry["exit_code"],
            multitask_us=baseline, reference_us=reference, candidate_us=candidate,
            reduction_vs_reference_pct=(reference-candidate)/reference*100,
            reduction_vs_multitask_pct=(baseline-candidate)/baseline*100,
            goal_30pct_met=candidate <= baseline*.7, statistics_us=stats,
            candidate_vs_reference=matrix["candidate_vs_reference"],
            metadata_replay=report.get("metadata_replay", {}).get("passed"),
            padding_replay=report.get("padding_replay", {}).get("passed"),
            protocol=matrix["protocol"], compiler_records=len(invocations), all_compilers_v3=True,
            source_matrix=str(folder / "source_matrix/result.json"), report=str(folder / "result.json"),
        ))
    selected = [r for r in rows if not r["boundary"]]
    for batch in (4, 8):
        path = HERE.parent / "attention_physical_plan_current_retest_20261019" / f"device_h131072_b{batch}_v1/source_matrix/result.json"
        data = read(path)
        assert data["passed"]
        stats = data["statistics_us"]
        baseline, candidate = stats["dynamic_pto"]["median_us"], stats["spmd_reference"]["median_us"]
        selected.append(dict(batch=batch, history=131072, passed=True,
                             multitask_us=baseline, candidate_us=candidate,
                             measured_role="spmd_reference", source_matrix=str(path),
                             selection="物理plan关闭；完整C++等价证据见OFF_CODEGEN.json",
                             goal_30pct_met=candidate <= baseline*.7))
    selected.sort(key=lambda r: (-r["history"], r["batch"]))
    query = []
    for entry in read(HERE / "QUERY_SUBMISSIONS.json"):
        path = Path(entry["output"]) / "result.json"
        data = read(path) if path.exists() else {}
        query.append(dict(task=entry["task"], batch=entry["batch"], query_tokens=entry["query_tokens"],
                          passed=data.get("passed", False), stage=data.get("stage"),
                          exit_code=entry.get("exit_code"), error=data.get("error"), result=str(path),
                          rounds=[dict(label=r["label"], passed=r["passed"]) for r in data.get("rounds", [])]))
    result = dict(completed=rows, selected_seven=selected, queries=query,
                  goal_30pct_met=sum(r["goal_30pct_met"] for r in selected),
                  no_new_device_tests=True, note="同轮候选/参考决定采用；跨轮数值不能累计归因。")
    (HERE / "COMPLETED_RESULTS.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(dict(rows=len(rows), selected=len(selected), queries=query,
                         goal_30pct_met=result["goal_30pct_met"]), ensure_ascii=False))


if __name__ == "__main__":
    collect()
