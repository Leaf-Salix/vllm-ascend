# SPDX-License-Identifier: Apache-2.0
"""记录FIFO分流八档，显式区分短档参考兼容补丁；不混算单SPMD总收益。"""

import json
import statistics
from pathlib import Path

from analyze_query_score import analyze
from summarize import HERE, summarize


def build():
    cases = json.loads((HERE / "CASE_STANDARD.json").read_text())["cases"]
    rows, toolchains, environments, wait_sources = [], [], [], []
    for history, batch in cases:
        attempt = "policy_v1" if history == 131072 else "policy_ref_fix_v2"
        path = HERE / "runs/query_score_fifo_policy_v11" / f"h{history}_b{batch}_{attempt}" / "result.json"
        raw = json.loads(path.read_text())
        if raw["stage"] != "complete" or not raw["passed"]:
            raise ValueError(f"未完整通过：{path}")
        if (raw["history"], raw["batch"], raw["query_tokens"]) != (history, batch, 6):
            raise ValueError("结果与档位不符")
        if raw["candidate"] != "query_score_fifo_policy_v11" or raw["current"] != "formal_4d5b2ed9":
            raise ValueError("多task候选和正式对照必须保持同一源码")
        reference = raw["spmd_reference"]
        if reference["source_commit"] != "a241fa61" or reference["backoff_ticks"] != 150:
            raise ValueError("必须明确使用a241fa61及150tick")
        patch = reference.get("private_compatibility_patch")
        if (history == 8192) != bool(patch):
            raise ValueError("仅两短档使用私有类型兼容补丁")
        if patch and patch != json.loads((HERE / "SPMD_HALF_SHAPE_COMPATIBILITY.json").read_text()):
            raise ValueError("参考补丁必须与已记录的两处尺寸修正相同")
        stats = raw["l1"]["statistics_us"]
        if set(stats) != {"multitask_original", "multitask_current", "multitask_candidate", "single_spmd"}:
            raise ValueError("必须完整保留四个角色")
        if any(len(s["samples_us"]) != 5 for s in stats.values()):
            raise ValueError("必须保留五窗，不能挑选最快窗口")
        toolchains.append(raw["toolchain_identity"])
        environments.append(raw["environment"])
        wait_sources.append(Path(raw["shared_wait_source"]).read_bytes())
        if set(raw["same_toolchain_roles"]) != set(stats):
            raise ValueError("四路未全部核对同一工具链")
        if raw["actual_ptoas_invocations"]["executables"] != [raw["toolchain_identity"]["ptoas_executable"]]:
            raise ValueError("实际PTOAS不一致")
        audit = json.loads((path.parent / "spmd_codegen/audit.json").read_text())
        if not audit["changes"] or any(c["backoff_syscnt_ticks"] != 150 for c in audit["changes"]):
            raise ValueError("实际参考等待常量不符")
        summary = summarize(path)
        summary.pop("task_windows", None)
        current, candidate = (stats[k] for k in ("multitask_current", "multitask_candidate"))
        topology = {}
        for role in ("multitask_candidate", "single_spmd"):
            window = next(w for w in raw["l1"]["windows"] if w["side"] == role)
            events = json.loads(Path(window["trace"]).read_text())["traceEvents"]
            topology[role] = len(
                {e["args"]["taskId"] for e in events if e.get("ph") == "X" and "taskId" in e.get("args", {})}
            )
        if topology["multitask_candidate"] <= 1 or topology["single_spmd"] != 1:
            raise ValueError("实际任务拓扑不符合多task对单SPMD合同")
        names = set()
        for window in raw["l1"]["windows"]:
            if window["side"] != "multitask_candidate":
                continue
            events = json.loads(Path(window["trace"]).read_text())["traceEvents"]
            names.update(
                e["name"].split("(")[0]
                for e in events
                if e.get("ph") == "X" and "indexer_score_topk_native_pair" in e["name"]
            )
        rows.append(
            dict(
                history=history,
                batch=batch,
                attempt=attempt,
                result=str(path.relative_to(HERE)),
                spmd_reference=reference,
                summary=summary,
                query_dependency=analyze(path),
                actual_score_names=sorted(names),
                active_task_ids=topology,
                mean_change_us=candidate["mean_us"] - current["mean_us"],
                max_change_us=candidate["max_us"] - current["max_us"],
                reduction_vs_current_pct=100 * (current["mean_us"] - candidate["mean_us"]) / current["mean_us"],
                faster_paired_windows=sum(a < b for a, b in zip(candidate["samples_us"], current["samples_us"])),
            )
        )
    for values in (toolchains, environments, wait_sources):
        if any(v != values[0] for v in values[1:]):
            raise ValueError("工具链、环境或等待实现不一致")
    regions = {
        str(h): statistics.mean(r["reduction_vs_current_pct"] for r in rows if r["history"] == h)
        for h in (131072, 8192)
    }
    return dict(
        passed=True,
        scope="四路同轮同卡同根输入；完整CSA L1首尾含内部间隙；单卡，非模型token或DSpark",
        reference_note="长档原a241fa61；两短档附私有不可达B1 Tile类型修正，逐行展示，不计算混合参考总收益",
        adoption_basis="只聚合共同正式多task与共同候选的配对变化；不改严格report_matrix的同参考检查",
        toolchain=toolchains[0],
        environment=environments[0],
        rows=rows,
        region_reduction_vs_current_pct=regions,
        descriptive_weighted_reduction_vs_current_pct=0.8 * regions["131072"] + 0.2 * regions["8192"],
    )


if __name__ == "__main__":
    report = build()
    destination = HERE / "QUERY_SCORE_FIFO_POLICY_EIGHT_CASE_REVIEW.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(destination)
    for row in report["rows"]:
        print(row["history"], row["batch"], row["mean_change_us"], row["max_change_us"])
