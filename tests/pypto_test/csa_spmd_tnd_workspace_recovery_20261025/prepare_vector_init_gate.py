# SPDX-License-Identifier: Apache-2.0
"""仅初始化写者全局会合，再按核组通知Cube；所有路径仍为单MIX内部初始化。"""

import ast
import json
import shutil
import subprocess
from pathlib import Path

from prepare import hashes, replace_once


def main():
    here = Path(__file__).resolve().parent
    source = here / "candidates/internal_single_f72a"
    target = here / "candidates/vector_init_gate_v1"
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    abi = (target / "runtime_abi.py").read_text()
    abi = replace_once(
        abi,
        "VECTORS = 48\n",
        "VECTORS = 48\n# 初始化沿用原MIX屏障的AIV→AIC事件；业务事件为0..5，全AIV屏障使用14。\n"
        "SIGNAL_INIT_READY_EVENT = 12\n",
    )
    (target / "runtime_abi.py").write_text(abi)
    comment = (
        "    # Both explicit Group members rendezvous before any signal read/write.\n"
        "    # This is hardware FFTS (events 11/12/13), never a GM boot barrier.\n"
    )
    for name in ("cube_schedule.py", "vector_schedule.py"):
        text = (target / name).read_text()
        text = replace_once(text, "\n    VECTORS,\n", "\n    SIGNAL_INIT_READY_EVENT,\n    VECTORS,\n")
        text = replace_once(
            text,
            comment,
            "    # 只有AIV写初始化区；全48个AIV会合后，各自通知本核组AIC。\n"
            "    # Cube收到对应两个AIV的硬件事件，即可确认全局signal清零完成。\n",
        )
        if name == "cube_schedule.py":
            sync = (
                "    pl.system.sync_wait(\n"
                "        SIGNAL_INIT_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC\n"
                "    )\n"
            )
        else:
            sync = (
                "    pl.system.syncall(core_type=pl.KernelType.AIV)\n"
                "    pl.system.sync_set(\n"
                "        SIGNAL_INIT_READY_EVENT, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV\n"
                "    )\n"
            )
        text = replace_once(text, "    pl.system.syncall(core_type=pl.KernelType.MIX)\n", sync)
        ast.parse(text)
        (target / name).write_text(text)
    changed = [str(target / name) for name in ("cube_schedule.py", "vector_schedule.py", "runtime_abi.py")]
    subprocess.run(["ruff", "format", *changed], check=True)
    subprocess.run(["ruff", "check", "--fix", *changed], check=True)
    report = {
        "status": "CPU_gate_pending",
        "candidate": str(target),
        "parent": "internal_single_f72a；未叠加RMS提前候选，单独归因同步路径",
        "source_hashes": hashes(target),
        "adopted": False,
        "protocol": [
            "所有48个AIV保持原全量signal清零与缓存维护/流水完成",
            "48个AIV硬件全局会合（PTO AIVOnly屏障，事件14）",
            "每个AIV发送原模式2本核组就绪事件12，随后可进入业务",
            "每个AIC按原MIX屏障的同一事件12等待本组两个AIV，再做缓存维护/代际读取及业务",
        ],
        "safety_arguments": [
            "AIV不发布事件12，直到全48个初始化写者都完成全量清零；无需AIC参与清零会合",
            "所有AIV信号读取在全AIV屏障之后；所有AIC读取在事件12等待之后",
            "事件12沿用原MIX的AIV→AIC模式/计数关系，不依赖未初始化GM标志",
            "一调用每AIV发送一次、每AIC消费一次，调用完成前配平；事件14由SYNCALL内部配平",
            "所有业务函数保持原数学/顺序；事件0..5与初始化12/14分离",
            "保留24组全占用、sync_start及绑定硬件检查；无独立清零task或外部signal",
        ],
        "risks_to_validate": ["Group中AIVOnly全占用证明", "实际PTOAS事件顺序", "连续图重放及metadata/state正确性"],
    }
    (here / "vector_init_gate_v1.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
