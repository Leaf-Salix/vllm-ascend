# SPDX-License-Identifier: Apache-2.0
"""仅移除MTE清零边界的全DCache操作，保留硬屏障、逐槽维护和业务发布获取。"""

import json
import shutil
from pathlib import Path

from prepare import hashes, replace_once


def main():
    here = Path(__file__).resolve().parent
    source = here / "candidates/internal_single_f72a"
    target = here / "candidates/selective_init_cache_v1"
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    before = (
        "    # Both explicit Group members rendezvous before any signal read/write.\n"
        "    # This is hardware FFTS (events 11/12/13), never a GM boot barrier.\n"
        "    pl.system.cacheinvalid()\n"
        "    pl.system.fence()\n"
        "    pl.system.syncall(core_type=pl.KernelType.MIX)\n"
        "    pl.system.cacheinvalid()\n"
    )
    after = (
        "    # 初始化只由MTE3写signal；保留完成屏障及全核硬同步。\n"
        "    # begin_invocation逐槽失效后读取；notify/wait亦逐槽维护。\n"
        "    # 此处尚无业务Scalar写入，无需额外清空整份DCache。\n"
        "    # 后续业务publish/wait的缓存维护全部保留。\n"
        "    pl.system.fence()\n"
        "    pl.system.syncall(core_type=pl.KernelType.MIX)\n"
    )
    for name in ("cube_schedule.py", "vector_schedule.py"):
        (target / name).write_text(replace_once((source / name).read_text(), before, after))
    report = {
        "status": "CPU_gate_pending",
        "candidate": str(target),
        "parent": "internal_single_f72a，未叠加其它负结果候选",
        "files": hashes(target),
        "adopted": False,
        "change": "只删除初始化边界四处ENTIRE_DATA_CACHE操作，完整清零、fence和MIX硬屏障不变",
        "coherence_arguments": [
            "本初始化段只用AIV MTE3清零signal，AIC没有初始化GM写入；尚无业务Scalar写入要发布",
            "fence等待清零写回，原全24组MIX硬屏障保留，任何signal读取仍在所有清零完成之后",
            "begin_invocation在读自己phase0前执行单缓存行失效及fence",
            "TNOTIFY Set在store前后均DCCI目标行并DSB；TWAIT native150每次检查前DCCI目标行",
            "所有publish_phase/wait_*对业务数据的完整缓存维护不变；QG2空闲planner获取亦不变",
            "Simpler任务执行前的DCache清理仍在；没有修改共享SDK或核内512B槽布局",
        ],
        "risks_to_validate": ["实际生成物只减少预期四处DCCI", "连续图/同址metadata与state", "全部八档无新增数值回归"],
    }
    (here / "selective_init_cache_v1.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
