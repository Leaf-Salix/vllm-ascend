# SPDX-License-Identifier: Apache-2.0
"""按同次构建的静态提交序号解释 L1；不把 callable id 当作 task id。

仅支持当前无分支、无循环提交的 MPMD 编排。TaskId 的低 32 位是 ring 内
提交序号；要求窗口中完整出现 0..N-1，且只含一个 ring。自动匹配要求
提交数量唯一，否则必须用 --orchestration SIDE=PATH 明确指定源码。
incore 时间包含核内等待，不能仅凭 L1 把 task 间隔归因于 AICPU 计算。
"""

import argparse
import json
import math
import re
import statistics
from pathlib import Path


def task_names(path):
    source = path.read_text()
    first_label = re.search(r"// (?:Spmd|Group) ", source)
    submits = list(re.finditer(r"TaskOutputTensors task_\d+_outs = rt_submit_", source))
    if first_label is None or not submits:
        raise ValueError(f"未找到 task 提交：{path}")
    # 提交前计算 launch 大小可以有分支；提交区间本身必须是直线编排。
    submission_region = source[first_label.start() : submits[-1].end()]
    if re.search(r"\b(?:for|while|if)\s*\(", submission_region):
        raise ValueError(f"编排含动态控制流，不能按静态提交序号映射：{path}")
    names = {}
    current = None
    for line in source.splitlines():
        label = re.search(r"// Spmd \S+: (\S+)|// Group (\S+):", line)
        if label:
            current = label.group(1) or label.group(2)
        submit = re.search(r"TaskOutputTensors task_(\d+)_outs = rt_submit_", line)
        if submit:
            ordinal = int(submit.group(1))
            if current is None or ordinal != len(names):
                raise ValueError(f"缺少名称或提交序号不连续：{path}:{line}")
            names[ordinal] = current
            current = None
    if not names:
        raise ValueError(f"未找到 task 提交：{path}")
    return names


def summarize_window(window, names):
    raw = json.loads(Path(window["raw"]).read_text())
    rows = raw["aicore_tasks"]
    meta = raw["metadata"]
    if raw["chip_swimlane_level"] != 1 or not rows:
        raise ValueError("需要非空 L1")
    if len({row[1] >> 32 for row in rows}) != 1:
        raise ValueError("窗口包含多个 ring")
    ordinals = {row[1] & 0xFFFFFFFF for row in rows}
    if ordinals != set(names):
        raise ValueError("L1 task 序号与静态提交不一致")
    if any(len(row) != 6 or row[4] < row[3] or row[3] <= 0 for row in rows):
        raise ValueError("非法 incore 时间戳")
    factor = 1e6 / meta["clock_freq_hz"]
    origin = min(row[3] for row in rows)
    span = (max(row[4] for row in rows) - origin) * factor
    if not math.isclose(span, window["incore_span_us"], abs_tol=1e-7):
        raise ValueError("完整跨度与验收结果不一致")
    stages = []
    for ordinal, name in names.items():
        group = [row for row in rows if row[1] & 0xFFFFFFFF == ordinal]
        stage = {
            "ordinal": ordinal,
            "name": name,
            "begin_us": (min(row[3] for row in group) - origin) * factor,
            "end_us": (max(row[4] for row in group) - origin) * factor,
            "instances": len(group),
            "kinds": {},
        }
        for kind in ("aic", "aiv"):
            durations = [(row[4] - row[3]) * factor for row in group if meta["core_types"][row[0]] == kind]
            if durations:
                stage["kinds"][kind] = {
                    "instances": len(durations),
                    "min_us": min(durations),
                    "mean_us": statistics.mean(durations),
                    "max_us": max(durations),
                }
        stages.append(stage)
    return {"raw": window["raw"], "span_us": span, "stages": stages}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--orchestration", action="append", default=[], metavar="SIDE=PATH")
    args = parser.parse_args()
    result = json.loads((args.run / "result.json").read_text())
    explicit = dict(value.split("=", 1) for value in args.orchestration)
    sources = list(args.run.glob("build_output/decode_csa_tnd_*/orchestration/decode_csa_tnd.cpp"))
    parsed = {path: task_names(path) for path in sources}
    output = {
        "case": result["case"]["name"],
        "result": str((args.run / "result.json").resolve()),
        "method": "单 ring 完整静态提交序号匹配；incore begin/end；等待含在 task 内，不推断 AICPU 忙时",
        "sides": {},
    }
    for side in sorted({w["side"] for w in result["windows"] if w["side"].startswith("mpmd")}):
        windows = [w for w in result["windows"] if w["side"] == side]
        counts = {w["task_groups"] for w in windows}
        if len(counts) != 1:
            raise ValueError(f"{side} 各窗口 task 数不一致")
        if side in explicit:
            source = Path(explicit[side])
            names = task_names(source)
        else:
            matches = [(path, names) for path, names in parsed.items() if len(names) in counts]
            if len(matches) != 1:
                raise ValueError(f"{side} 编排匹配不唯一，请显式传 --orchestration")
            source, names = matches[0]
        records = [summarize_window(window, names) for window in windows]
        means = []
        for ordinal, name in names.items():
            group = [w["stages"][ordinal] for w in records]
            means.append(
                {
                    "name": name,
                    "begin_us": statistics.mean(row["begin_us"] for row in group),
                    "end_us": statistics.mean(row["end_us"] for row in group),
                    "kinds": {
                        kind: {
                            key: statistics.mean(row["kinds"][kind][key] for row in group)
                            for key in ("min_us", "mean_us", "max_us", "instances")
                        }
                        for kind in group[0]["kinds"]
                    },
                }
            )
        output["sides"][side] = {"orchestration": str(source), "windows": records, "mean_stages": means}
        print(f"{side}: {len(names)} tasks, {len(records)} windows")
        for stage in means:
            detail = "; ".join(
                f"{kind} mean/max {values['mean_us']:.2f}/{values['max_us']:.2f}"
                for kind, values in stage["kinds"].items()
            )
            print(f"  {stage['name']:<36} {stage['begin_us']:8.2f} -> {stage['end_us']:8.2f} us; {detail}")
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
