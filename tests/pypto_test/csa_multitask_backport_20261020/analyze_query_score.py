# SPDX-License-Identifier: Apache-2.0
"""读取Q/Score局部融合的已有L1，验证实际生产者并区分工作范围。"""

import argparse
import json
import re
import statistics
from pathlib import Path

TASK_LABEL = r"(?:r\d+)*t\d+"


def named(events, prefixes):
    return [e for e in events if e["name"].lstrip("_").startswith(prefixes)]


def span(events):
    if not events:
        return None
    starts = [e["ts"] + e["dur"] - e["args"]["kernel-duration-us"] for e in events]
    ends = [e["ts"] + e["dur"] for e in events]
    return dict(first_us=min(starts), last_us=max(ends), span_us=max(ends) - min(starts), records=len(events))


def analyze(path):
    raw = json.loads(path.read_text())
    if not raw.get("passed") or not raw["candidate"].startswith("query_score"):
        raise ValueError("只检查完整通过的Q/Score候选")
    expected_fusion = raw["history"] > 32768 and raw["batch"] * raw["query_tokens"] >= 32
    rows = []
    for window in raw["l1"]["windows"]:
        side = window["side"]
        if side not in ("multitask_current", "multitask_candidate"):
            continue
        trace = json.loads(Path(window["trace"]).read_text())
        events = [e for e in trace["traceEvents"] if e.get("ph") == "X" and "kernel-duration-us" in e.get("args", {})]

        qb_events = named(events, "qproj_matmul")
        q_events = named(events, "qproj_dequant_rms_nope_rope")
        score_events = named(events, "indexer_score_topk_native_pair")
        attention_events = named(events, "qk_pv_")
        qb, q_tail, score, attention = map(span, (qb_events, q_events, score_events, attention_events))
        if not all((qb, score, attention)):
            raise ValueError("缺少QB、Score或Attention阶段")
        producers = {label for event in qb_events for label in re.findall(TASK_LABEL, event["name"])}
        fusion_package = raw["candidate"] if side == "multitask_candidate" else raw.get("current") or ""
        fused = expected_fusion and fusion_package.startswith("query_score")
        if fused:
            if q_events or not producers or len(score_events) != 72:
                raise AssertionError("融合档应无独立Q尾段，并保留24AIC+48AIV Score记录")
            for event in score_events:
                fanin = set(re.findall(TASK_LABEL, event["args"].get("fanin-hint", "")))
                if not producers.issubset(fanin):
                    raise AssertionError("融合MIX缺少QB真实生产者依赖")
        elif not q_events:
            raise AssertionError("非融合对照和未融合区间必须保留独立Q尾段")
        full = span(events)
        rows.append(
            dict(
                side=side,
                trace=window["trace"],
                full_us=full["span_us"],
                qb=qb,
                standalone_q=q_tail,
                score_includes_q=fused,
                score=score,
                attention=attention,
                qb_producers=sorted(producers),
                qb_dependency_verified=fused,
                qb_done_to_score_start_us=score["first_us"] - qb["last_us"],
                qb_done_to_attention_start_us=attention["first_us"] - qb["last_us"],
            )
        )
    summary = {}
    for side in ("multitask_current", "multitask_candidate"):
        selected = [r for r in rows if r["side"] == side]
        if len(selected) != 5:
            raise ValueError("必须保留完整五窗")
        fields = {}
        for key in ("full_us", "qb_done_to_score_start_us", "qb_done_to_attention_start_us"):
            values = [r[key] for r in selected]
            fields[key] = dict(
                samples_us=values, min_us=min(values), mean_us=statistics.mean(values), max_us=max(values)
            )
        summary[side] = fields
    return dict(
        result=str(path),
        passed=True,
        history=raw["history"],
        batch=raw["batch"],
        scope="完整L1及共同QB结束到Attention首启区间；融合Score包含Q数学，不能当纯Score比速",
        dependency="实际L1的每个融合Score实例必须包含QB生产者fanin",
        timestamp_note="QB结束至Score首启差值单独记录；不将时间相近代替真实依赖证明",
        summaries=summary,
        windows=rows,
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
