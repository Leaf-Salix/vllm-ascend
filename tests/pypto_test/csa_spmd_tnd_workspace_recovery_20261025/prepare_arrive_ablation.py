# SPDX-License-Identifier: Apache-2.0
"""将到达/等待间的RMS和metadata拆成两项消融，保留原清零/同步协议。"""

import ast
import json
import shutil
import subprocess
from pathlib import Path

from prepare import hashes, replace_once


def main():
    here = Path(__file__).resolve().parent
    source = here / "candidates/init_arrive_overlap_v1"
    original = (source / "vector_schedule.py").read_text()
    report = {"parent": "init_arrive_overlap_v1；底层为f72，数学helper与初始化内容均未改变", "adopted": False}
    for mode in ("rms", "meta"):
        target = here / f"candidates/init_arrive_{mode}_v1"
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        text = original
        if mode == "rms":
            start = text.index("    # metadata仅依赖本次调用不可变的")
            end = text.index("    # 原MIX屏障的释放事件：", start)
            metadata = text[start:end]
            text = text[:start] + text[end:]
            publication = (
                "    if worker == planner:\n"
                "        signal = publish_phase(signal, PHASE_META, core, CORES, generation)\n"
            )
            text = replace_once(text, publication, metadata + publication)
        else:
            rms = "    x_fp32, inv = hc_widen_rms_range(x, x_fp32, inv, widen_tail, worker, VECTORS)\n"
            text = replace_once(text, rms, "")
            generation = "    signal, generation = begin_invocation(signal, core)\n"
            text = replace_once(text, generation, generation + rms)
        text = text.replace(
            "# 独立的RMS/metadata准备可与Cube全局会合并行；所有signal访问仍在释放之后。",
            f"# 只有独立的{mode.upper()}准备与Cube全局会合并行；所有signal访问仍在释放之后。",
        )
        ast.parse(text)
        (target / "vector_schedule.py").write_text(text)
        subprocess.run(["ruff", "format", str(target / "vector_schedule.py")], check=True)
        subprocess.run(["ruff", "check", str(target / "vector_schedule.py")], check=True)
        old, new = hashes(source), hashes(target)
        assert {name for name in old if old[name] != new[name]} == {"vector_schedule.py"}
        report[mode] = {
            "candidate": str(target),
            "source_hashes": new,
            "CPU_gate_pending": True,
            "change": (
                "仅RMS在事件12之后/13之前，metadata恢复在INPUT发布之后"
                if mode == "rms"
                else "仅metadata在事件12之后/13之前，RMS恢复在begin_invocation之后"
            ),
        }
    (here / "init_arrive_ablation_v1.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
