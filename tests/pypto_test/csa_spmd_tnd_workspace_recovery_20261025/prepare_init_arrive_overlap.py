# SPDX-License-Identifier: Apache-2.0
"""拆开AIV初始化屏障的到达/等待，在两者之间执行无signal依赖的准备。"""

import ast
import json
import shutil
import subprocess
from pathlib import Path

from prepare import hashes, replace_once


def main():
    here = Path(__file__).resolve().parent
    source = here / "candidates/internal_single_f72a"
    target = here / "candidates/init_arrive_overlap_v1"
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    abi = replace_once(
        (source / "runtime_abi.py").read_text(),
        "VECTORS = 48\n",
        "VECTORS = 48\n"
        "# 沿用PTO-ISA A2/A3 MIX屏障的原事件；业务事件0..5不变。\n"
        "SIGNAL_INIT_ARRIVE_EVENT = 12\n"
        "SIGNAL_INIT_RELEASE_EVENT = 13\n",
    )
    (target / "runtime_abi.py").write_text(abi)
    text = (source / "vector_schedule.py").read_text()
    text = replace_once(
        text, "\n    VECTORS,\n", "\n    SIGNAL_INIT_ARRIVE_EVENT,\n    SIGNAL_INIT_RELEASE_EVENT,\n    VECTORS,\n"
    )
    text = replace_once(
        text,
        "    # Both explicit Group members rendezvous before any signal read/write.\n"
        "    # This is hardware FFTS (events 11/12/13), never a GM boot barrier.\n",
        "    # 完成全量清零后先发送原MIX屏障的AIV到达事件。Cube仍执行原SYNCALL<MIX>。\n"
        "    # 独立的RMS/metadata准备可与Cube全局会合并行；所有signal访问仍在释放之后。\n",
    )
    text = replace_once(
        text,
        "    pl.system.syncall(core_type=pl.KernelType.MIX)\n"
        "    pl.system.cacheinvalid()\n"
        "    signal, generation = begin_invocation(signal, core)\n",
        "    pl.system.sync_set(\n"
        "        SIGNAL_INIT_ARRIVE_EVENT, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV\n"
        "    )\n",
    )
    start = text.index("    if BALANCED and QUERY_GROUP == 2:\n", text.index("    pl.system.set_ffts(ffts)"))
    end = text.index("    planner = pl.min((tokens + 7) // 8, VECTORS - 1)\n", start)
    input_publication = text[start : text.index("\n    # 公共metadata", start)]
    text = (
        text[:start]
        + (
            "    # metadata仅依赖本次调用不可变的Q/L/positions/slots，不读取signal或HC产物。\n"
            "    # 选择原来的首个空闲AIV；若全部有RMS工作，则仍由47号依次完成。\n"
        )
        + text[end:]
    )
    publication = "        signal = publish_phase(signal, PHASE_META, core, CORES, generation)\n"
    text = replace_once(
        text,
        publication,
        "\n"
        "    # 原MIX屏障的释放事件：必须等到所有48个AIV完成清零，才允许读写signal。\n"
        "    pl.system.sync_wait(\n"
        "        SIGNAL_INIT_RELEASE_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV\n"
        "    )\n"
        "    pl.system.cacheinvalid()\n"
        "    signal, generation = begin_invocation(signal, core)\n"
        + input_publication
        + "\n    if worker == planner:\n"
        + publication,
    )
    ast.parse(text)
    (target / "vector_schedule.py").write_text(text)
    changed = [str(target / name) for name in ("vector_schedule.py", "runtime_abi.py")]
    subprocess.run(["ruff", "format", *changed], check=True)
    subprocess.run(["ruff", "check", "--fix", *changed], check=True)
    assert (target / "cube_schedule.py").read_bytes() == (source / "cube_schedule.py").read_bytes()
    original_hashes, new_hashes = hashes(source), hashes(target)
    assert {k for k in original_hashes if original_hashes[k] != new_hashes[k]} == {
        "vector_schedule.py",
        "runtime_abi.py",
    }
    report = {
        "status": "CPU_gate_pending",
        "candidate": str(target),
        "parent": "internal_single_f72a，未叠加此前无收益的任何候选",
        "source_hashes": new_hashes,
        "adopted": False,
        "protocol": [
            "48个AIV保持原11520 INT32/核全量signal清零、512B槽距及全部缓存维护",
            "清零完成fence之后发送原MIX到达事件12，Cube仍运行未经改动的SYNCALL<MIX>",
            "AIV在等待释放前执行原RMS与纯metadata准备，不访问任何signal",
            "等待原释放事件13后才执行begin_invocation及所有INPUT/META发布",
            "其余业务和数学函数保持原样；没有新增task、外部signal、场景fallback或SDK改动",
        ],
        "safety_arguments": [
            "发送到达事件在独立工作之前，因此Cube会合不必等待RMS/metadata完成",
            "MIX的事件11/12/13及计数关系不变；每个AIV每调用一次到达和一次等待",
            "独立准备输入在本次调用不可变，开始前原全DCache维护仍执行",
            "所有signal读写均在释放事件后；所有数学结果仍经原publish_phase发布",
            "所有24核组全占用、sync_start及设备绑定检查不变",
        ],
        "risks_to_validate": ["PTOAS到达/准备/等待顺序", "同址metadata连续图与状态", "八档严格检查及稳定完整L1收益"],
    }
    (here / "init_arrive_overlap_v1.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
