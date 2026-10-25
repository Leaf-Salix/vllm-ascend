# SPDX-License-Identifier: Apache-2.0
"""同进程对照历史候选单侧采样与当前双侧采样，不作为代码优化验收。"""

import ast
import json
import shutil
from pathlib import Path

from prepare import hashes, replace_once

SAMPLER = """                records = []
                # reference=旧保留版，current=当前内部版；candidate仅作历史未优化参考。
                protocols = ("historical_candidate_only", "paired_both_sides")
                blocks = [("reference", protocols[0]), ("current", protocols[1]),
                          ("current", protocols[0]), ("reference", protocols[1])]
                report["diagnostic_blocks"] = blocks
                for target, protocol in blocks:
                    selected = {"warm_reference": graphs["candidate"], "target": graphs[target]}
                    for sample in range(3):
                        order = list(selected) if sample % 2 == 0 else list(reversed(selected))
                        for role in order:
                            restore(fixture)
                            selected[role].replay()
                            stream.synchronize()
                    save("cadence_" + target + "_" + protocol)
                    for sample in range(10):
                        order = list(selected) if sample % 2 == 0 else list(reversed(selected))
                        for role in order:
                            side = "candidate" if role == "warm_reference" else target
                            restore(fixture)
                            stream.synchronize()
                            if role == "warm_reference" and protocol == "historical_candidate_only":
                                selected[role].replay()
                                stream.synchronize()
                                continue
                            pypto.torch.begin_dfx()
                            try:
                                selected[role].replay()
                            finally:
                                pypto.torch.end_dfx()
                            source = args.output / "dfx"
                            if records:
                                source /= f"window_{len(records)}"
                            dest = args.output / f"{protocol}_{target}_{sample:02d}_{role}"
                            record = export_l1(source, dest, side, tables[side], init_layout=layouts[side])
                            record.update(protocol=protocol, measured_target=target, replay_role=role, sample=sample)
                            records.append(record)
"""


def main():
    here = Path(__file__).resolve().parent
    source_dir = (
        here.parent / "csa_spmd_tnd_eight_20261022/runs/sdk_measurement_harness_v1/pypto/deepseek_v4_flash_csa_spmd_tnd"
    )
    target = here / "candidates/historical_unoptimized_reference"
    shutil.copytree(source_dir, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    old = json.loads(
        (here.parent / "csa_spmd_tnd_eight_20261022/runs/h131072_b16_s6_leaf_scale_one_row_v1/result.json").read_text()
    )
    assert hashes(target) == old["source_hashes"]["reference"]
    frozen = here / "frozen_cadence"
    shutil.copytree(here / "frozen_optimization", frozen, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    runner = (frozen / "run_pto.py").read_text()
    start = runner.index("                for sample in range(3):\n")
    end = runner.index("                # No host comparisons between the final paired windows.\n", start)
    runner = runner[:start] + SAMPLER + runner[end:]
    start = runner.index('            report["statistics_us"] = {}\n')
    end = runner.index('            report["passed"] = True\n', start)
    stats = """            report["protocol_statistics_us"] = {}
            for target in ("reference", "current"):
                report["protocol_statistics_us"][target] = {}
                for protocol in protocols:
                    selected = [w for w in records if w["measured_target"] == target
                                and w["protocol"] == protocol and w["replay_role"] == "target"]
                    assert len(selected) == 10
                    samples = [w["incore_span_us"] for w in selected]
                    report["protocol_statistics_us"][target][protocol] = {
                        "median": statistics.median(samples), "samples": samples,
                    }
"""
    runner = runner[:start] + stats + runner[end:]
    runner = replace_once(
        runner,
        '        "scope": "同SDK/同进程/同卡三侧完整L1：外部持久工作区、已保留的内部单SPMD、当前内部单SPMD",\n',
        '        "scope": "仅诊断采样上下文；未优化历史参考只作warm replay或配对参考，不是优化候选",\n'
        '        "performance_eligible": False,\n'
        '        "diagnostic_only": True,\n',
    )
    ast.parse(runner)
    (frozen / "run_pto.py").write_text(runner)
    for name in ("launch_optimization.sh", "compile_optimization.sh"):
        source = (here / name).read_text().replace("frozen_optimization", "frozen_cadence")
        (here / name.replace("optimization", "cadence")).write_text(source)
    report = {
        "scope": "隔离参考是否采DFX对目标读数的影响，不作为代码优化收益",
        "historical_warm_reference_exact_source_hash_match": True,
        "historical_warm_reference_source": str(target),
        "historical_warm_reference_hashes": hashes(target),
        "targets": {"reference": "迁移前保留外部版b2e0e7fea", "current": "当前内部单SPMD f72a1f752"},
        "sampling": "每协议每目标3次warmup，10个有效窗口；两种协议只改变参考是否begin/end DFX，重放/restore顺序相同",
        "limitation": "四个固定顺序块共享同进程；若观察显著差异，须反转块顺序检查时间漂移",
        "status": "CPU_gate_pending",
    }
    (here / "cadence_diagnostic.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
