"""复用模型区间提取，保留相邻CSA差值及当前四份JSON下载入口。"""

import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from offline_pd.performance import device_tasks  # noqa: E402


def focus_pair(case, root=ROOT):
    """只展开用户指出的8K/B16第12/14层，用已有记录区分根与设备Worker。"""
    result = {"history": 8192, "batch": 16, "layers": [12, 14], "sides": {}}
    lines = ["", "## 原问题对应的第12/14层", "",
             "| 侧 | step1 前/后 μs | step2 前/后 μs | step3 前/后 μs |",
             "| --- | ---: | ---: | ---: |"]
    for side in ("native", "pto"):
        pairs = [[next(row["body_us"] for row in case[side]["intervals"]
                       if row["step"] == step and row["layer"] == layer)
                  for layer in (12, 14)] for step in range(3)]
        lines.append(f"| {side} | " + " | ".join(f"{a:.2f}/{b:.2f}" for a, b in pairs) + " |")
        detail = {"three_step_pair_us": pairs, "step3": []}
        folder = root / "model/h8192/b16" / side
        tasks = device_tasks(folder, 0)
        for layer in (12, 14):
            interval = next(row for row in case[side]["intervals"] if row["step"] == 2 and row["layer"] == layer)
            selected = [row for row in tasks if interval["start_ns"] <= row["start_ns"]
                        and row["start_ns"] + row["duration_ns"] <= interval["end_ns"]]
            record = {"layer": layer, "span_us": interval["body_us"], "source": str(folder)}
            if side == "native":
                spans = sorted((r["start_ns"], r["start_ns"] + r["duration_ns"]) for r in selected)
                union = []
                for start, end in spans:
                    if union and start <= union[-1][1]:
                        union[-1][1] = max(union[-1][1], end)
                    else:
                        union.append([start, end])
                sums = Counter()
                for row in selected:
                    sums[row["name"].split("_")[0]] += row["duration_ns"] / 1000
                observed = sum(end - start for start, end in union) / 1000
                record.update({"observed_tasks": len(selected), "observed_task_union_us": observed,
                               "outside_task_union_us": interval["body_us"] - observed,
                               "task_duration_sums_us": dict(sums)})
            else:
                for label, prefix in (("root", "simpler_aicpu_kernel_exec_"),
                                      ("worker", "aicore_kernel_mode_")):
                    matched = [row for row in selected if row["name"].startswith(prefix)]
                    if len(matched) != 1:
                        raise ValueError(f"layer{layer}: {label}事件不唯一")
                    record[label + "_us"] = matched[0]["duration_ns"] / 1000
            detail["step3"].append(record)
        result["sides"][side] = detail
    p0, p1 = result["sides"]["pto"]["step3"]
    n0, n1 = result["sides"]["native"]["step3"]
    lines += ["", f"第三步PTO设备Worker为{p0['worker_us']:.2f}→{p1['worker_us']:.2f}μs，"
              f"根调用为{p0['root_us']:.2f}→{p1['root_us']:.2f}μs；变化确实包含设备Worker内部，不能只归给根收尾。",
              f"Native两层各{n0['observed_tasks']}/{n1['observed_tasks']}条已记录任务，区间并集"
              f"{n0['observed_task_union_us']:.2f}→{n1['observed_task_union_us']:.2f}μs，"
              f"并集外间隙{n0['outside_task_union_us']:.2f}→{n1['outside_task_union_us']:.2f}μs。",
              "任务记录包含图内控制事件，并集不是纯Cube/Vector忙时；不能把两侧同位置变慢直接判为同一根因。",
              "本轮没有这两层的逐incore DFX，因此未定位到Score或Sparse；不能据此宣称sync_start已解决问题。",
              "[两层原始统计](pair_detail.json)。"]
    (root / "model/pair_detail.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return lines


def main(root=ROOT, revision="d1f170ff"):
    source = root.parent / "csa_qa_matrix_20260928/report.py"
    spec = importlib.util.spec_from_file_location("model_report", source)
    reporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reporter)
    reporter.ROOT = root
    reporter.REVISION = revision
    reporter.model_report(matrix_label="长短B16")
    data = json.loads((root / "model/model_gap_rank0.json").read_text())
    adjacent = []
    manifest = []
    download = root / "download"
    download.mkdir(exist_ok=True)
    lines = ["# 同一步内相邻CSA的差值", "",
             f"独立rank0三步profile，按本轮算子{revision}分析；不代替正式十步forward。",
             "PTO使用根/Worker并集的body_us，Native为HC_pre到HC_post；首次根调用前的metadata另列。",
             "step从1开始、模型layer从0开始；每行取该侧三步内最大向上跳变，所有相邻对见JSON。",
             "不同层权重/状态不同，差值不能自动解释为调度噪声，旧765/807μs样本也没有被替换。", "",
             "| 档位 | 侧 | step | 相邻layer | 前/后CSA μs | 增量 μs |",
             "| --- | --- | ---: | --- | ---: | ---: |"]
    for case in data["cases"]:
        label = f"{case['history']//1024}K_B{case['batch']}"
        for side in ("native", "pto"):
            pairs = []
            for step in range(3):
                rows = [r for r in case[side]["intervals"] if r["step"] == step]
                if len(rows) != 21 or [r["layer"] for r in rows] != list(range(2, 43, 2)):
                    raise ValueError(f"{label}/{side}/step{step}: CSA层次不完整")
                for a, b in zip(rows, rows[1:]):
                    pairs.append({"step": step + 1, "layers": [a["layer"], b["layer"]],
                                  "before_us": a["body_us"], "after_us": b["body_us"],
                                  "increase_us": b["body_us"] - a["body_us"]})
            largest = max(pairs, key=lambda p: p["increase_us"])
            adjacent.append({"history": case["history"], "batch": case["batch"],
                             "side": side, "largest_increase": largest, "pairs": pairs})
            lines.append(f"| {label} | {side} | {largest['step']} | {largest['layers'][0]}→"
                         f"{largest['layers'][1]} | {largest['before_us']:.2f}/{largest['after_us']:.2f} | "
                         f"{largest['increase_us']:+.2f} |")
            folder = root / "model" / f"h{case['history']}" / f"b{case['batch']}" / side
            exported = json.loads((folder / "profile_export.json").read_text())["exported"]
            entries = [entry for entry in exported if entry["rank"] == "rank0"]
            if len(entries) != 1:
                raise ValueError(f"{folder}: rank0 trace不唯一")
            trace = Path(entries[0]["trace_view"])
            destination = download / f"{len(manifest)+1:02}_{label}_{side}_FullModel_Rank0.json"
            if destination.exists():
                if not os.path.samefile(destination, trace):
                    raise ValueError(f"{destination}: 已有文件来源不同")
            else:
                os.link(trace, destination)
            manifest.append({"file": destination.name, "source": str(trace), "operator": revision,
                             "scope": "真实EP16 rank0独立三步PyTorch profile，不是单层DFX"})
    (root / "model/adjacent_csa.json").write_text(json.dumps(adjacent, ensure_ascii=False, indent=2) + "\n")
    short_case = next(case for case in data["cases"] if case["history"] == 8192 and case["batch"] == 16)
    lines += focus_pair(short_case, root)
    lines += ["", "[全部相邻对](adjacent_csa.json)、[模型分项](MODEL_GAP.md)、",
              "[正式forward](RESULTS.md)、[四份PyTorch JSON](../download/README.md)。"]
    (root / "model/ADJACENT_CSA.md").write_text("\n".join(lines) + "\n")
    (download / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (download / "README.md").write_text(
        f"# {revision}长短B16模型profile\n\n"
        "四份真实EP16 rank0三步PyTorch JSON；未混入旧源码或单卡合成历史DFX。\n\n" +
        "\n".join(f"- [{item['file']}]({item['file']})" for item in manifest) +
        "\n\n[原始路径与口径](manifest.json)、[相邻CSA](../model/ADJACENT_CSA.md)。\n"
    )


if __name__ == "__main__":
    main()
