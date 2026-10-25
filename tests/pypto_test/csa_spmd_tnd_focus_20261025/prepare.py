# SPDX-License-Identifier: Apache-2.0
"""冻结本轮起点及既有测量脚本，适配内部工作区而不改写历史结果。"""

import ast
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def replace_once(source, before, after):
    assert source.count(before) == 1, before
    return source.replace(before, after)


def main():
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    workspace = next(p for p in here.parents if (p / "env-dsv4-0251rc1.sh").is_file())
    previous = here.parent / "csa_spmd_tnd_eight_20261022/runs/sdk_measurement_harness_v2"
    target = here / "frozen"
    target.mkdir()
    for filename in ("scenarios.py", "bsh_entry.py", "outer_profile.py"):
        shutil.copy2(previous / filename, target / filename)
    packages = repo / "vllm_ascend/ops/pypto"
    for package in ("deepseek_v4_flash_csa", "deepseek_v4_flash_csa_spmd_tnd"):
        shutil.copytree(
            packages / package, target / "pypto" / package, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
    source = (previous / "run_pto.py").read_text()
    source = replace_once(
        source,
        "def export_l1(source, destination, side, kernel_config):",
        "def legacy_export_l1(source, destination, side, kernel_config):",
    )
    source = replace_once(source, "\ndef main():\n", "\nfrom complete_l1 import export_l1\n\ndef main():\n")
    source = replace_once(
        source, 'max_seq_len=max(case["seq_lens"]), allocator=allocate,', 'max_seq_len=max(case["seq_lens"]),'
    )
    source = replace_once(
        source,
        "                    enable_chip_swimlane=1,\n",
        "                    ring_heap=tuple(m * 1024 * 1024 for m in (16, 256, 16, 16)),\n"
        "                    ring_task_window=4096,\n"
        "                    enable_chip_swimlane=1,\n",
    )
    source = replace_once(
        source,
        '                        if side == "reference":\n'
        "                            graphs[side].replay()\n"
        "                            stream.synchronize()\n"
        "                            continue\n",
        "",
    )
    source = replace_once(
        source, '            for side in ("candidate",):\n', '            for side in ("reference", "candidate"):\n'
    )
    source = replace_once(
        source,
        '            report["ratio_to_frozen"] = (\n'
        '                report["statistics_us"]["candidate"]["median"] / reference_median\n'
        "                if reference_median is not None else None)\n",
        '            report["ratio_to_frozen"] = (\n'
        '                report["statistics_us"]["candidate"]["median"]\n'
        '                / report["statistics_us"]["reference"]["median"])\n',
    )
    source = source.replace("candidate_l1_with_unmeasured_reference_replays", "paired_complete_l1")
    source = source.replace(
        "TND-SPMD complete 72-role single-MIX L1; SDK-matched comparison",
        "完整TND-SPMD调用L1，含内部signal初始化与72核主MIX；同任务交替对照",
    )
    source = replace_once(
        source,
        '        "enable_chip_swimlane": 1,\n',
        '        "enable_chip_swimlane": 1,\n'
        '        "scratch_guard_scope": "外部cache/state/只读输入保护；内部Simpler heap不伪称有逐tensor红区",\n'
        '        "ring_config": {"heap_mb": [16,256,16,16], "task_window": 4096},\n',
    )
    ast.parse(source)
    (target / "run_pto.py").write_text(source)
    shutil.copy2(previous / "measurement_provenance.py", target / "measurement_provenance.py")
    shutil.copy2(here / "complete_l1.py", target / "complete_l1.py")
    sdk = workspace / "vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_shared_toolchain/manifest.json"
    shutil.copy2(sdk, target / "expected_sdk.json")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    baseline = {
        "reference_kind": "current_internal_workspace_20261025",
        "source_commit": head,
        "files": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((target / "pypto/deepseek_v4_flash_csa_spmd_tnd").glob("*.py"))
        },
    }
    (target / "baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
    manifest = {
        "objective": "相对当前已保留版本再降低完整L1耗时10%，重点128K/B16、B24，最终八档等权几何平均并检查逐档回归",
        "target_ratio": 0.9,
        "baseline": baseline,
        "shared_toolchain": json.loads(sdk.read_text()),
        "frozen": str(target),
        "previous_harness": str(previous),
        "measurement": "每侧10个窗口，同任务交替顺序，完整初始化到主MIX结束；全部样本保留",
        "correctness": "当前冻结版输出/TopK/score/cache/state逐bit比较；历史Native差异不改阈值",
        "historical_timings_eligible": False,
        "reason_for_new_measurement": "公共PyPTO升级且正式入口改为内部工作区，旧SDK结果不得作为新收益分母",
        "candidate_changes": [],
    }
    (here / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
