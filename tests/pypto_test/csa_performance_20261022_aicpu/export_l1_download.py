# SPDX-License-Identifier: Apache-2.0
"""导出八档各一次完整 L1，以同次编排的 ring/local ID 还原 task 名称。"""

import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DESTINATION = REPO.parent / "downloads/CSA_TND_MPMD_L1_8cases_20261022_AICPU"


def read(path):
    return json.loads(Path(path).read_text())


def task_map(raw_path):
    """限定已验收 v9 的实际生成编排；遇到其他结构拒绝猜测名称。"""
    names = read(raw_path.with_name("names.json"))["callable_id_to_name"]
    candidates = list((raw_path.parents[1] / "build_output").glob("decode_csa_tnd_*/orchestration/*.cpp"))
    # C05 来自 v11 对照任务：v9 控制组仍有 group_plan，v11 候选已删除。
    candidates = [path for path in candidates if "ext_group_plan" in path.read_text()]
    assert len(candidates) == 1, candidates
    cpp = candidates[0]
    text = cpp.read_text()
    assert text.count("SIMPLER_SCOPE(ScopeMode::MANUAL)") == 1
    manual_start = text.index("SIMPLER_SCOPE(ScopeMode::MANUAL)")
    brace = text.index("{", manual_start)
    depth = 1
    manual_end = brace + 1
    while depth:
        depth += (text[manual_end] == "{") - (text[manual_end] == "}")
        manual_end += 1
    manual_text = text[brace:manual_end]
    assert re.search(r"for \(int64_t group_\w+ = 0; group_\w+ < 8; group_\w+ \+= 1\)", manual_text)
    mixed = {
        name: [int(x.strip()) for x in values.split(",")]
        for name, values in re.findall(r"MixedKernels (mixed_\d+) = \{([^}]+)\};", text)
    }
    records = list(
        re.finditer(
            r"(?m)^( +)TaskOutputTensors task_(\d+)_outs = rt_submit_(aic_task|aiv_task|task)\((\w+), params_t\2\);",
            text,
        )
    )
    assert len(records) in (41, 42)
    automatic, grouped = [], []
    for match in records:
        spaces, site, kind, function = match.groups()
        is_manual = brace < match.start() < manual_end
        assert len(spaces) == (16 if is_manual else 8), "不接受嵌套条件或动态循环里的未知提交"
        ids = mixed[function] if kind == "task" else [int(function)]
        source_names = [names[str(i)] for i in ids]
        if kind == "task":
            assert len(ids) == 3 and ids[1] == ids[2]
            if source_names == ["index_score_topk_cube", "index_score_topk_vector", "index_score_topk_vector"]:
                label = "indexer_score"
            else:
                assert source_names == ["sparse_attn_cube", "sparse_attn_vector", "sparse_attn_vector"]
                label = "sparse_attention"
        else:
            label = {"o_b_group_task": "o_proj_b"}.get(source_names[0], source_names[0])
        record = dict(
            name=label,
            kind=kind,
            function_ids=ids,
            source_names=source_names,
            submit_site=int(site),
            source_line=text.count("\n", 0, match.start()) + 1,
        )
        (grouped if is_manual else automatic).append(record)
    assert [item["name"] for item in grouped] == ["o_proj_a", "o_quant", "o_proj_b"]
    assert automatic[0]["name"] == "hc_widen_rms" and automatic[-1]["name"] == "hc_post"
    # Simpler TaskId=(scope ring<<32)|ring 内单调计数。外层为 ring1，输出 manual scope 为 ring2。
    mapping = {(1 << 32) | i: dict(item) for i, item in enumerate(automatic)}
    for group in range(8):
        for offset, item in enumerate(grouped):
            mapping[(2 << 32) | (group * 3 + offset)] = dict(item, group=group)
    return cpp, mapping


def main():
    report = read(HERE / "PERFORMANCE.json")
    cases = read(HERE / "cases.json")["cases"]
    DESTINATION.mkdir(parents=True, exist_ok=False)
    index = []
    lines = [
        "# TND-MPMD：八档单次完整 L1 泳道",
        "",
        "每档选全部有效样本中最接近该档中位数的一次执行；没有拼接窗口，没有重新上卡。",
        "拖入 Perfetto 或 chrome://tracing。每个条形严格使用 incore begin/end，",
        "起点归零到首个 incore 开始；保留所有实际 task、物理核及任务间隙。",
        "不包含 L1 之外的 AICPU 活动，不画估计的依赖箭头。",
        "",
        "AIC_00～23、AIV_00～47；详情保留原始 core_id、ring/local TaskId 和函数编号。",
        "输出投影标注 group=0～7，Score/Attention 的 AIC 与 AIV 分别显示。",
        "名称来自对应原始运行的生成编排与 Simpler ring 编码规则，未按开始时间猜序。",
        "",
        "| 文件 | 本次 L1 (μs) | 该档中位数 (μs) | task 组数 |",
        "|---|---:|---:|---:|",
    ]
    for case in cases:
        cell = report["cells"][case["id"]]["TND-MPMD-L1"]
        selected = min(
            range(len(cell["samples_us"])), key=lambda i: (abs(cell["samples_us"][i] - cell["median_us"]), i)
        )
        raw_path = Path(cell["raw"][selected])
        raw = read(raw_path)
        rows = raw["aicore_tasks"]
        assert raw["chip_swimlane_level"] == 1
        assert all(len(row) == 6 and 0 <= row[0] < 72 and row[4] >= row[3] > 0 for row in rows)
        cpp, mapping = task_map(raw_path)
        assert {row[1] for row in rows} == set(mapping), "真实任务集合必须完整匹配每个 ring 的提交顺序"
        origin = min(row[3] for row in rows)
        factor = 1e6 / raw["metadata"]["clock_freq_hz"]
        span = (max(row[4] for row in rows) - origin) * factor
        assert abs(span - cell["samples_us"][selected]) < 1e-6
        metadata = dict(
            case=case,
            source_raw=str(raw_path),
            source_cpp=str(cpp),
            source_implementation="d263bc3f2 accepted v9",
            single_invocation=True,
            sample_l1_us=span,
            table_median_us=cell["median_us"],
            selection="closest actual single sample to all-valid-samples median",
            name_mapping="generated orchestration + TMR scope ring and local monotonic TaskId",
            time_boundary="incore begin/end only; normalized to earliest incore start",
            shared_toolchain=report["manifest"]["shared_toolchain"],
        )
        events = []
        for pid, label in ((0, "CSA L1 span"), (1, "AIC"), (2, "AIV")):
            events += [
                dict(ph="M", name="process_name", pid=pid, tid=0, args=dict(name=label)),
                dict(ph="M", name="process_sort_index", pid=pid, tid=0, args=dict(sort_index=pid)),
            ]
        for core in range(72):
            pid, tid = (1, core) if core < 24 else (2, core - 24)
            label = f"{'AIC' if pid == 1 else 'AIV'}_{tid:02d}"
            events += [
                dict(ph="M", name="thread_name", pid=pid, tid=tid, args=dict(name=label)),
                dict(ph="M", name="thread_sort_index", pid=pid, tid=tid, args=dict(sort_index=tid)),
            ]
        for core, task, _, start, end, _ in rows:
            info = mapping[task]
            pid, tid = (1, core) if core < 24 else (2, core - 24)
            assert info["kind"] == "task" or info["kind"] == ("aic_task" if pid == 1 else "aiv_task")
            suffix = f" [group={info['group']}]" if "group" in info else ""
            suffix += f" ({'AIC' if pid == 1 else 'AIV'})" if info["kind"] == "task" else ""
            events.append(
                dict(
                    ph="X",
                    cat="incore",
                    name=info["name"] + suffix,
                    pid=pid,
                    tid=tid,
                    ts=(start - origin) * factor,
                    dur=(end - start) * factor,
                    args=dict(task_id=task, ring=task >> 32, local_id=task & 0xFFFFFFFF, physical_core_id=core, **info),
                )
            )
        events.append(
            dict(
                ph="X",
                cat="summary",
                name=f"{case['id']} complete CSA L1",
                pid=0,
                tid=0,
                ts=0,
                dur=span,
                args=dict(raw_records=len(rows), task_groups=len(mapping)),
            )
        )
        label = f"{case['history_label'] // 1024}K_B{case['active_requests']}_"
        label += "MixedS1-S6_T56" if case["id"] == "C08" else "S6"
        filename = f"{case['id'][1:]}_{label}_TND-MPMD_L1_SingleCSA.json"
        target = DESTINATION / filename
        target.write_text(
            json.dumps(
                dict(traceEvents=events, displayTimeUnit="us", metadata=metadata),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )
        # 全量核内记录和时间都保留，未重复、裁剪或合并实际 task。
        emitted = [event for event in events if event.get("cat") == "incore"]
        assert len(emitted) == len(rows)
        assert Counter(event["args"]["task_id"] for event in emitted) == Counter(row[1] for row in rows)
        assert abs(max(event["ts"] + event["dur"] for event in emitted) - span) < 1e-6
        index.append(
            dict(
                file=str(target),
                **metadata,
                bytes=target.stat().st_size,
                records=len(rows),
                task_groups=len(mapping),
                task_names=mapping,
            )
        )
        lines.append(f"| {filename} | {span:.2f} | {cell['median_us']:.2f} | {len(mapping)} |")
    lines += [
        "",
        "C05 使用 v11 实验中的 v9 控制组记录，算子仍是已接受的 v9，不是 v11 候选。",
        "单次时间与整档中位数可以不同；这里不生成一个虚构的中位数执行。",
    ]
    (DESTINATION / "README.md").write_text("\n".join(lines) + "\n")
    (HERE / "download_manifest.json").write_text(
        json.dumps(dict(directory=str(DESTINATION), cases=index), ensure_ascii=False, indent=2) + "\n"
    )
    print(DESTINATION)
    print({"traces": len(index), "bytes": sum(item["bytes"] for item in index), "all_records_preserved": True})


if __name__ == "__main__":
    main()
