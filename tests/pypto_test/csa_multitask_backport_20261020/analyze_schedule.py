# SPDX-License-Identifier: Apache-2.0
"""从已有L1泳道分离阶段位置和整组启动离散；不新增设备测量。"""

import argparse
import json
import statistics
from pathlib import Path


def phase(name):
    name = name.lstrip("_").split("(")[0]
    if name.startswith("indexer_score_topk_native_pair"):
        return "score"
    if name.startswith(("qk_pv_", "attention_rows_cube", "attention_rows_vector", "attention_plan_rows_pipeline")):
        return "attention"
    if name.startswith("csa_slots_build_valid_qk_plan"):
        return "attention_plan"
    if name.startswith(("proj_a_mm", "proj_b_mm", "quant", "proj_b_act_hc_post", "output_cube", "output_vector")):
        return "output"
    if name.startswith(("hc_", "split_pre_post", "comb_sinkhorn", "mix_x_rms_norm")):
        return "hc_pre"
    if name.startswith("qproj_dequant"):
        return "query_dequant"
    if name.startswith("qr_proj_matmul"):
        return "qa"
    for prefix, label in (
        ("idx_qr_proj_matmul", "indexer_query_project"),
        ("qr_hadamard_matmul", "query_hadamard"),
        ("qproj_matmul", "qb"),
        ("kv_proj_matmul", "original_kv_project"),
    ):
        if name.startswith(prefix):
            return label
    if name.startswith("indexer_head_coefficients"):
        return "coefficients"
    return None


def analyze(result):
    report = json.loads(result.read_text())
    if not report.get("passed"):
        raise ValueError("只分析已经完整通过的窗口")
    rows = []
    for window in report["l1"]["windows"]:
        if window["side"] == "single_spmd":
            continue
        trace = json.loads(Path(window["trace"]).read_text())
        events = [e for e in trace["traceEvents"] if e.get("ph") == "X" and "kernel-duration-us" in e.get("args", {})]
        origin = min(e["ts"] + e["dur"] - e["args"]["kernel-duration-us"] for e in events)
        end = max(e["ts"] + e["dur"] for e in events)
        groups = {}
        for event in events:
            label = phase(event["name"])
            if label:
                finish = event["ts"] + event["dur"] - origin
                begin = finish - event["args"]["kernel-duration-us"]
                groups.setdefault(label, []).append((begin, finish))
        row = dict(side=window["side"], full_us=end - origin, trace=window["trace"], phases={})
        for label, values in groups.items():
            row["phases"][label] = dict(
                first_start_us=min(a for a, b in values),
                last_start_us=max(a for a, b in values),
                last_end_us=max(b for a, b in values),
                start_spread_us=max(a for a, b in values) - min(a for a, b in values),
            )
        score, attention, output = (row["phases"][key] for key in ("score", "attention", "output"))
        row["segments_us"] = dict(
            before_score=score["first_start_us"],
            score_to_attention=attention["first_start_us"] - score["first_start_us"],
            attention_to_output=output["first_start_us"] - attention["first_start_us"],
            output_to_end=row["full_us"] - output["first_start_us"],
        )
        rows.append(row)
    summaries = {}
    for side in dict.fromkeys(row["side"] for row in rows):
        selected = [row for row in rows if row["side"] == side]
        segments = {}
        for key in selected[0]["segments_us"]:
            values = [row["segments_us"][key] for row in selected]
            segments[key] = dict(min_us=min(values), mean_us=statistics.mean(values), max_us=max(values))
        summaries[side] = segments
    return dict(
        result=str(result),
        scope="按阶段首个incore起点分段；各段可有别的task并发，不等于阶段纯计算或精确AICPU成本",
        rows=rows,
        summaries=summaries,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.result.resolve())
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["summaries"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
