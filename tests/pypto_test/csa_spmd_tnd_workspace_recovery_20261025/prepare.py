# SPDX-License-Identifier: Apache-2.0
"""冻结同SDK的外部持久工作区、迁移版、当前单SPMD三侧对照。"""

import ast
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def hashes(directory):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.glob("*.py"))}


def main():
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    previous = here.parent / "csa_spmd_tnd_focus_20261025"
    frozen = here / "frozen"
    shutil.copytree(previous / "frozen_v3", frozen, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    package = "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"
    old_commit = subprocess.check_output(["git", "rev-parse", "b2e0e7fea"], cwd=repo, text=True).strip()
    files = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", old_commit, "--", package], cwd=repo, text=True
    )
    old = frozen / "pypto/deepseek_v4_flash_csa_spmd_tnd"
    for name in files.splitlines():
        if name.endswith(".py"):
            (old / Path(name).name).write_bytes(
                subprocess.check_output(["git", "show", f"{old_commit}:{name}"], cwd=repo)
            )
    migration = here / "candidates/internal_separate_4f9b"
    current = here / "candidates/internal_single_f72a"
    shutil.copytree(previous / "frozen/pypto/deepseek_v4_flash_csa_spmd_tnd", migration)
    shutil.copytree(repo / package, current, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    baseline = {
        "reference_kind": "external_persistent_pre_migration_on_current_SDK",
        "source_commit": old_commit,
        "files": hashes(old),
    }
    (frozen / "baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")

    l1 = (previous / "complete_l1_v2.py").read_text()
    l1 = l1.replace('if init_layout != "fused":', 'if init_layout not in ("fused", "persistent_external"):')
    l1 = l1.replace('"初始化布局必须显式声明为separate或fused"', '"须显式声明separate/fused/persistent_external"')
    l1 = replace_once(l1, '        "init_layout": "fused",\n', '        "init_layout": init_layout,\n')
    l1 = replace_once(
        l1,
        '        "scope": "从MIX最早进入到最晚退出，包含核内清零、硬屏障及完整CSA",\n',
        '        "scope": ("从MIX最早进入到最晚退出，包含核内清零、硬屏障及完整CSA"\n'
        '                  if init_layout == "fused" else "外部signal只在绑定准备时清零；本窗口包含完整单MIX业务"),\n',
    )
    l1 = replace_once(
        l1,
        '    if side == "reference" and init_layout != "separate":\n'
        '        raise ValueError("当前冻结对照始终保留独立清零，不能改其边界")\n',
        "",
    )
    (frozen / "complete_l1_v2.py").write_text(l1)

    source = (frozen / "run_pto.py").read_text()
    source = source.replace("import importlib\n", "import importlib\nimport inspect\n")
    source = source.replace('return ("candidate", "reference")', 'return ("candidate", "migration", "reference")')
    source = source.replace('return ("reference", "candidate")', 'return ("reference", "migration", "candidate")')
    source = source.replace("--control-mpmd", "--migration-control").replace("control_mpmd", "migration_control")
    source = source.replace('"mpmd_control"', '"migration"')
    source = source.replace(
        'args.family != "tnd" or args.migration_control or args.baseline_only '
        "or args.state_validation or args.service_validation",
        'args.family != "tnd" or not args.migration_control or args.baseline_only '
        "or args.state_validation or args.service_validation",
    )
    source = source.replace('choices=("separate", "fused")', 'choices=("separate", "fused", "persistent_external")')
    source = replace_once(
        source,
        "    prepare_sides = preparation_sides(args.prepare_order)\n",
        "    prepare_sides = preparation_sides(args.prepare_order)\n"
        '    layouts = {"reference": "persistent_external", "migration": "separate", '
        '"candidate": args.candidate_init_layout}\n',
    )
    source = replace_once(
        source,
        "        args.candidate, frozen_baseline, args.calibrate_baseline)\n",
        "        args.candidate, frozen_baseline, args.calibrate_baseline)\n"
        "    from measurement_provenance import package_hashes\n"
        '    source_hashes["migration"] = package_hashes(args.migration_control)\n'
        '    recovery_manifest = json.loads(Path(__file__).parents[1].joinpath("manifest.json").read_text())\n'
        '    if source_hashes["migration"] != recovery_manifest["sources"]["migration"]["files"]:\n'
        '        raise ValueError("迁移控制源码与冻结清单不符")\n',
    )
    source = source.replace(
        '"reference_commit": manifest["reference_commit"]', '"reference_commit": frozen_baseline["source_commit"]'
    )
    source = source.replace(
        '"scope": "完整TND-SPMD调用L1，含内部signal初始化与72核主MIX；同任务交替对照"',
        '"scope": "同SDK/同进程/同卡三侧完整L1：外部持久工作区、内部独立初始化、当前内部单SPMD"',
    )
    source = replace_once(
        source,
        '        "candidate_init_layout": args.candidate_init_layout,\n',
        '        "candidate_init_layout": args.candidate_init_layout,\n        "layout_by_side": layouts,\n',
    )
    source = replace_once(
        source,
        '                "candidate": (args.candidate, prefix + "_mpmd_tnd_candidate"),\n',
        '                "candidate": (args.candidate, prefix + "_mpmd_tnd_candidate"),\n'
        '                "migration": (args.migration_control, prefix + "_workspace_migration"),\n',
    )
    source = replace_once(
        source,
        "        control_name = (\n"
        '            mount_package(args.migration_control, prefix + "_mpmd_tnd_control") '
        "if args.migration_control is not None else None\n"
        "        )\n",
        '        control_name = mounted["migration"]\n',
    )
    source = source.replace(
        'package_names = {"reference": baseline_name, "candidate": candidate_name}',
        'package_names = {"reference": baseline_name, "migration": control_name, "candidate": candidate_name}',
    )
    source = replace_once(
        source,
        '                    entries[side] = importlib.import_module(package + ".binding").PreparedDynamicCSA(\n'
        "                        native, max_query_tokens=max_query_tokens,\n"
        '                        max_seq_len=max(case["seq_lens"]),\n'
        "                    )\n",
        '                    binding = importlib.import_module(package + ".binding").PreparedDynamicCSA\n'
        '                    extra = {"allocator": allocate} '
        'if "allocator" in inspect.signature(binding).parameters else {}\n'
        "                    entries[side] = binding(\n"
        "                        native, max_query_tokens=max_query_tokens,\n"
        '                        max_seq_len=max(case["seq_lens"]), **extra,\n'
        "                    )\n",
    )
    source = source.replace('for side in ("reference", "candidate"):', "for side in prepare_sides:")
    source = source.replace(
        'init_layout=args.candidate_init_layout if side == "candidate" else "separate"', "init_layout=layouts[side]"
    )
    source = source.replace(
        "                    order = list(graphs) if sample % 2 == 0 else list(reversed(graphs))\n",
        "                    names = list(graphs)\n"
        "                    offset = sample % len(names)\n"
        "                    order = names[offset:] + names[:offset]\n"
        "                    if sample % 2:\n"
        "                        order.reverse()\n",
    )
    source = source.replace('"candidate_l1_with_unmeasured_reference_replays"', '"rotating_three_side_complete_L1"')
    ast.parse(source)
    (frozen / "run_pto.py").write_text(source)
    historical = json.loads((here.parent / "csa_performance_20261023/PERFORMANCE.json").read_text())
    manifest = {
        "objective": "查明内部工作区迁移的完整L1回退并恢复10月23日已保留性能；先B16，最终八档检查",
        "production_start_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "constraints": [
            "正式实现继续使用内部工作区",
            "清零及业务全部在一个SPMD",
            "公共工具链不变",
            "原精度差异单列、无新增回归",
            "详细中文阶段提交，不push",
        ],
        "shared_toolchain": json.loads((frozen / "expected_sdk.json").read_text()),
        "historical_targets_us": {case: row["TND-SPMD"]["median_us"] for case, row in historical["cells"].items()},
        "historical_target_rule": "旧SDK数字是需恢复的历史水平；因果归因只使用当前SDK同进程对照，不能跨SDK计算优化收益",
        "sources": {
            "reference": {
                "source": str(old),
                "commit": old_commit,
                "layout": "persistent_external",
                "files": hashes(old),
            },
            "migration": {
                "source": str(migration),
                "commit": "4f9b563133989900a5ca492583eb6bb5ba5f51ac",
                "layout": "separate",
                "files": hashes(migration),
            },
            "candidate": {"source": str(current), "commit": "f72a1f752", "layout": "fused", "files": hashes(current)},
        },
        "measurement": "每侧3次warmup、10个有效完整L1窗口；轮转三侧顺序，同一process/device；保存全部样本",
        "correctness": (
            "eager/graph/post-L1输出、TopK、score、cache/state逐bit；外部可见保护。"
            "外部scratch加红区，内部scratch不伪称有红区"
        ),
        "status": "CPU_preflight_pending",
    }
    (here / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (here / "current_state.json").write_text(
        json.dumps(
            {
                "status": "investigating",
                "user_instruction": "好好查内部工作区迁移为何变慢，把性能恢复回来",
                "pending_tasks": [],
                "adopted_changes": [],
                "baseline_not_redefined": True,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(frozen)


if __name__ == "__main__":
    main()
