# SPDX-License-Identifier: Apache-2.0
"""冻结L2 PMU诊断入口；核内源码不变，插桩读数不能作为正式性能验收。"""

import ast
import json
import shutil
from pathlib import Path

from prepare import replace_once


def main():
    here = Path(__file__).resolve().parent
    frozen = here / "frozen_l2"
    shutil.copytree(here / "frozen_optimization", frozen, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    runner = (frozen / "run_pto.py").read_text()
    runner = replace_once(
        runner,
        '        "scope": "同SDK/同进程/同卡三侧完整L1：外部持久工作区、已保留的内部单SPMD、当前内部单SPMD",\n',
        '        "scope": "仅诊断外部/内部工作区的L2计数，enable_pmu=8；不作优化验收",\n'
        '        "performance_eligible": False,\n'
        '        "diagnostic_only": True,\n'
        '        "pmu_event_type": 8,\n',
    )
    runner = replace_once(
        runner,
        "                    enable_chip_swimlane=1,\n",
        "                    enable_chip_swimlane=1,\n                    enable_pmu=8,\n",
    )
    ast.parse(runner)
    (frozen / "run_pto.py").write_text(runner)
    exporter = (frozen / "complete_l1_v2.py").read_text()
    exporter = replace_once(
        exporter,
        '    (destination / "kernel_config.py").write_bytes(kernel_config.read_bytes())\n',
        '    (destination / "kernel_config.py").write_bytes(kernel_config.read_bytes())\n'
        '    pmu_path = source / "pmu.csv"\n'
        "    if not pmu_path.is_file():\n"
        '        raise ValueError("L2诊断窗口缺少pmu.csv")\n'
        '    (destination / "pmu.csv").write_bytes(pmu_path.read_bytes())\n',
    )
    ast.parse(exporter)
    (frozen / "complete_l1_v2.py").write_text(exporter)
    source = (here / "launch_optimization.sh").read_text().replace("frozen_optimization", "frozen_l2")
    (here / "launch_l2.sh").write_text(source)
    report = {
        "status": "harness_AST_pass_existing_operator_CPU_gates_reused",
        "scope": "L2诊断；PMU插桩的L1不可与无PMU历史值直接计算收益",
        "event_type": 8,
        "event_source": "simpler/src/a2a3/platform/include/common/pmu_profiling.h:PMU_EVENTS_A2A3_L2_CACHE",
        "counters": [
            "write_cache_hit",
            "write_cache_miss_allocate",
            "r0_read_cache_hit",
            "r0_read_cache_miss_allocate",
            "r1_read_cache_hit",
            "r1_read_cache_miss_allocate",
        ],
        "sides": {
            "reference": "b2e0外部持久工作区，诊断harness在构造时full(37)且signal zero_一次",
            "current": "f72a内部工作区，全部初始化和业务在单MIX内",
            "candidate": "仅signal外置的诊断版；其余scratch内部，禁止生产采用",
        },
        "sampling": "同进程三侧轮转、各10窗；保留所有L1原始记录及PMU CSV",
        "interpretation": "先核对72核完整性及计数有效性；计数差异只反映整个MIX，不能直接归因signal或某个阶段",
    }
    (here / "l2_diagnostic.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
