# SPDX-License-Identifier: Apache-2.0
"""用CANN profiler诊断L2；当前Simpler kernel模式明确不支持内建PMU。"""

import ast
import json
import shutil
from pathlib import Path

from prepare import replace_once


def main():
    here = Path(__file__).resolve().parent
    frozen = here / "frozen_cann_l2"
    shutil.copytree(here / "frozen_optimization", frozen, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    runner = (frozen / "run_pto.py").read_text()
    runner = replace_once(
        runner,
        '        "scope": "同SDK/同进程/同卡三侧完整L1：外部持久工作区、已保留的内部单SPMD、当前内部单SPMD",\n',
        '        "scope": "CANN L2Cache诊断，算子与SDK未改；插桩L1不作优化验收",\n'
        '        "performance_eligible": False,\n'
        '        "diagnostic_only": True,\n'
        '        "cann_metric": "L2Cache",\n',
    )
    start = runner.index('                save("paired_complete_l1")\n')
    end = runner.index("                # No host comparisons between the final paired windows.\n", start)
    sampler = runner[start:end]
    sampler = replace_once(
        sampler,
        "                        try:\n                            graphs[side].replay()\n",
        "                        try:\n"
        '                            with torch.profiler.record_function(f"csa_l2/{sample:02d}/{side}"):\n'
        "                                graphs[side].replay()\n",
    )
    setup = """                from torch_npu.profiler import (
                    AiCMetrics, ProfilerActivity, ProfilerLevel, _ExperimentalConfig,
                    profile, tensorboard_trace_handler,
                )
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
                    record_shapes=False,
                    profile_memory=False,
                    with_stack=False,
                    on_trace_ready=tensorboard_trace_handler(str(args.output / "cann_l2")),
                    experimental_config=_ExperimentalConfig(
                        profiler_level=ProfilerLevel.Level1,
                        aic_metrics=AiCMetrics.L2Cache,
                        data_simplification=False,
                    ),
                ):
"""
    instrumented_sampler = "".join("    " + line for line in sampler.splitlines(keepends=True))
    runner = runner[:start] + setup + instrumented_sampler + runner[end:]
    ast.parse(runner)
    (frozen / "run_pto.py").write_text(runner)
    (here / "launch_cann_l2.sh").write_text(
        (here / "launch_optimization.sh").read_text().replace("frozen_optimization", "frozen_cann_l2")
    )
    report = {
        "status": "AST_pass_existing_operators_CPU_gates_reused",
        "scope": "CANN L2Cache诊断，不作为正式L1优化验收",
        "sampling": "同进程旧外部/current内部/仅signal外部三侧轮转各10窗；每个图重放带CPU标签",
        "cann_capture": {"profiler_level": "Level1", "aic_metrics": "L2Cache", "data_simplification": False},
        "raw_artifacts": "runs/<attempt>/cann_l2/，保留原始PROF及导出数据；L1仍保留每窗口原始记录",
        "simpler_pmu_failure": {
            "task_id": "task_20261025_174142_312710121272",
            "result": "runs/h131072_b16_s6_l2_pmu_r1/result.json",
            "reason": (
                "pypto.torch.init无enable_pmu参数；Simpler KernelStaticConfig::validate也明确拒绝PMU。不是零命中计数。"
            ),
        },
        "interpretation": "先验证实际产生非空且可关联到各side的L2计数；只能解释完整运行，不直接归因signal或某阶段",
    }
    (here / "cann_l2_diagnostic.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
