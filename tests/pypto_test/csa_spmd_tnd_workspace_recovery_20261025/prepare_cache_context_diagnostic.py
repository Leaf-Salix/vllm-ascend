# SPDX-License-Identifier: Apache-2.0
"""同一算子对照通常重放、紧邻预热、额外GM读写，诊断缓存上下文敏感性。"""

import ast
import json
import shutil
from pathlib import Path

from prepare import replace_once

SAMPLER = """                records = []
                # 额外GM读写只是一种缓存扰动，不声称硬件L2被完全清空。
                eviction = torch.empty((1024 * 1024 * 1024,), dtype=torch.int8, device=device)
                eviction.zero_()
                protocols = ("usual_pair", "immediate_target_replay", "gm_write_sweep")
                for sample in range(3):
                    for side in graphs:
                        restore(fixture)
                        graphs[side].replay()
                        stream.synchronize()
                save("cache_context")
                combinations = [(protocol, side) for protocol in protocols for side in graphs]
                for sample in range(10):
                    offset = sample % len(combinations)
                    order = combinations[offset:] + combinations[:offset]
                    if sample % 2:
                        order.reverse()
                    for protocol, side in order:
                        restore(fixture)
                        stream.synchronize()
                        if protocol == "immediate_target_replay":
                            graphs[side].replay()
                            stream.synchronize()
                            restore(fixture)
                            stream.synchronize()
                        elif protocol == "gm_write_sweep":
                            eviction.add_(1)
                            stream.synchronize()
                        pypto.torch.begin_dfx()
                        try:
                            graphs[side].replay()
                        finally:
                            pypto.torch.end_dfx()
                        source = args.output / "dfx"
                        if records:
                            source /= f"window_{len(records)}"
                        dest = args.output / f"{protocol}_{sample:02d}_{side}"
                        record = export_l1(source, dest, side, tables[side], init_layout=layouts[side])
                        record.update(protocol=protocol, sample=sample)
                        records.append(record)
"""


def main():
    here = Path(__file__).resolve().parent
    frozen = here / "frozen_cache_context"
    shutil.copytree(here / "frozen_optimization", frozen, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    runner = (frozen / "run_pto.py").read_text()
    runner = replace_once(runner, 'return ("candidate", "current", "reference")', 'return ("candidate", "reference")')
    runner = replace_once(runner, 'return ("reference", "current", "candidate")', 'return ("reference", "candidate")')
    start = runner.index("                for sample in range(3):\n")
    end = runner.index("                # No host comparisons between the final paired windows.\n", start)
    runner = runner[:start] + SAMPLER + runner[end:]
    start = runner.index('            report["statistics_us"] = {}\n')
    end = runner.index('            report["passed"] = True\n', start)
    stats = """            report["protocol_statistics_us"] = {}
            for side in graphs:
                report["protocol_statistics_us"][side] = {}
                for protocol in protocols:
                    selected = [w for w in records if w["side"] == side and w["protocol"] == protocol]
                    assert len(selected) == 10
                    samples = [w["incore_span_us"] for w in selected]
                    report["protocol_statistics_us"][side][protocol] = {
                        "median": statistics.median(samples), "samples": samples,
                    }
"""
    runner = runner[:start] + stats + runner[end:]
    runner = replace_once(
        runner,
        '        "scope": "同SDK/同进程/同卡三侧完整L1：外部持久工作区、已保留的内部单SPMD、当前内部单SPMD",\n',
        '        "scope": "同进程两侧各三种缓存上下文；外部b2e0与内部f72a；不作代码优化验收",\n'
        '        "performance_eligible": False,\n'
        '        "diagnostic_only": True,\n',
    )
    ast.parse(runner)
    (frozen / "run_pto.py").write_text(runner)
    (here / "launch_cache_context.sh").write_text(
        (here / "launch_optimization.sh").read_text().replace("frozen_optimization", "frozen_cache_context")
    )
    report = {
        "status": "AST_pass_existing_operators_CPU_gates_reused",
        "scope": "诊断缓存上下文，不以额外预热结果冒充代码收益",
        "sides": {"reference": "旧外部b2e0", "candidate": "当前内部单SPMD f72a，无候选代码修改"},
        "sampling": "3种协议×2个side按sample轮转/倒序，共60个窗口，各组合10窗",
        "protocols": {
            "usual_pair": "restore并同步后直接测量",
            "immediate_target_replay": "额外完整重放目标后再次restore，保持测量输入相同",
            "gm_write_sweep": "restore后额外读写独立1GiB INT8缓冲；不修改任何算子输入/工作区",
        },
        "limits": [
            "额外重放也改变设备活跃程度，不是纯L2操作",
            "1GiB GM读写不是受保证的L2 flush，只检验外部内存扰动敏感性",
            "内部每调用初始化仍包含于全部被测MIX窗口，外部预热仅作明确标注的诊断",
        ],
    }
    (here / "cache_context_diagnostic.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
