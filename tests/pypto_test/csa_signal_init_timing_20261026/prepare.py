# SPDX-License-Identifier: Apache-2.0
"""冻结f72生产版与精确初始化区间探针，仅测量，不修改生产或共享SDK。"""

import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECOVERY = HERE.parent / "csa_spmd_tnd_workspace_recovery_20261025"
sys.path.insert(0, str(RECOVERY))
from prepare_boot_clock import MAGIC, assert_original_logic, call_name, clock_write, make_package  # noqa: E402


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def package_hashes(path):
    import hashlib

    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(path.glob("*.py"))}


def make_interval_package(source, target):
    make_package(source, target)
    for name, core in (("cube_schedule.py", "worker"), ("vector_schedule.py", "core")):
        path = target / name
        tree = ast.parse(path.read_text())
        func = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
        clean, skip_fence = [], False
        for node in func.body:
            value = getattr(node, "value", None)
            if skip_fence:
                assert call_name(value) == "pl.system.fence"
                skip_fence = False
                continue
            if isinstance(value, ast.Call) and value.args and ast.unparse(value.args[0]) == "diagnostic_clock":
                assert call_name(value) in ("pl.write", "pl.system.cacheinvalid")
                skip_fence = call_name(value) == "pl.system.cacheinvalid"
                continue
            clean.append(node)
        assert not skip_fence
        body, seen = [], []
        for node in clean:
            value = getattr(node, "value", None)
            if call_name(value) == "pl.system.syncall":
                body.append(clock_write(core, 1, MAGIC + 1))
                seen.append("before_sync")
            if isinstance(node, ast.Return):
                body += [
                    clock_write(core, 4, MAGIC + 4),
                    clock_write(core, 5, core),
                    clock_write(core, 6, "generation"),
                ]
                body += ast.parse(
                    f"pl.system.cacheinvalid(diagnostic_clock, [1, 8], [{core}, 0])\npl.system.fence()"
                ).body
            body.append(node)
            if call_name(value) == "pl.system.set_ffts":
                body.append(clock_write(core, 0, MAGIC))
                seen.append("before_clear_or_wait")
            elif call_name(value) == "pl.system.syncall":
                body += [clock_write(core, 2, MAGIC + 2), clock_write(core, 3, MAGIC + 3)]
                seen += ["after_sync", "clock_pair_calibration"]
        assert seen == ["before_clear_or_wait", "before_sync", "after_sync", "clock_pair_calibration"]
        func.body = body
        ast.fix_missing_locations(tree)
        path.write_text("# SPDX-License-Identifier: Apache-2.0\n" + ast.unparse(tree) + "\n")
    files = [str(target / name) for name in ("cube_schedule.py", "vector_schedule.py")]
    subprocess.run(["ruff", "format", *files], check=True)
    return assert_original_logic(source, target)


def main():
    source = RECOVERY / "candidates/internal_single_f72a"
    candidate = HERE / "candidates/init_interval_v1"
    checked = make_interval_package(source, candidate)
    control = HERE / "candidates/current_f72a"
    shutil.copytree(source, control, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    frozen = HERE / "frozen"
    shutil.copytree(RECOVERY / "frozen_boot_clock_v3", frozen, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    reference = frozen / "pypto/deepseek_v4_flash_csa_spmd_tnd"
    for path in control.glob("*.py"):
        shutil.copy2(path, reference / path.name)
    assert package_hashes(reference) == package_hashes(control)
    baseline = {
        "reference_kind": "current_f72_uninstrumented_same_business_and_initialization",
        "source_commit": "f72a1f752bf0d358e3014cbcdcc29dbc18a39779",
        "files": package_hashes(control),
    }
    (frozen / "baseline.json").write_text(json.dumps(baseline, ensure_ascii=False, indent=2) + "\n")
    runner = (frozen / "run_pto.py").read_text()
    runner = replace_once(
        runner, '"reference": "persistent_external", "current": "fused"', '"reference": "fused", "current": "fused"'
    )
    runner = replace_once(
        runner,
        '"scope": "四点硬件时间戳诊断：入函数、初始化后、首段数学后、返回前；不作性能验收",',
        '"scope": "f72清零与syncall区间；排除set_ffts、尾部cacheinvalid及generation，含未插桩同版本对照",',
    )
    runner = (
        replace_once(
            runner,
            'with compilation_probe(args.output / ("instrumentation_" + side)):',
            'with compilation_probe(args.output / ("instrumentation_" + side), enabled=(side == "candidate")):',
        )
        if runner.count('with compilation_probe(args.output / ("instrumentation_" + side)):') == 1
        else runner.replace(
            'with compilation_probe(args.output / ("instrumentation_" + side)):',
            'with compilation_probe(args.output / ("instrumentation_" + side), enabled=(side == "candidate")):',
        )
    )
    runner = replace_once(
        runner,
        '                        record["clock"] = export_entry(entries[side], record)\n',
        '                        if side == "candidate":\n'
        '                            record["clock"] = export_entry(entries[side], record)\n',
    )
    ast.parse(runner)
    (frozen / "run_pto.py").write_text(runner)
    shutil.copy2(HERE / "boot_clock_hooks.py", frozen / "boot_clock_hooks.py")
    for old, new in (("launch_boot_clock.sh", "launch.sh"), ("compile_boot_clock.sh", "compile.sh")):
        text = (
            (RECOVERY / old)
            .read_text()
            .replace("frozen_boot_clock_v3", "frozen")
            .replace("candidates/internal_single_f72a", "candidates/current_f72a")
        )
        (HERE / new).write_text(text)
    report = {
        "purpose": "用户要求统计清零加sync_all耗时；独立测量，不恢复已清除的优化goal",
        "production_revision": baseline["source_commit"],
        "sources": {"current": {"files": package_hashes(control)}, "candidate": {"files": package_hashes(candidate)}},
        "AST_original_logic_verified": checked,
        "case": "128K/B16/S6",
        "samples_per_side": 10,
        "clock_columns": [
            "before_clear_AIV_or_pre_sync_AIC",
            "before_SYNCALL",
            "after_SYNCALL",
            "adjacent_read_calibration",
            "physical_AIC_id_or_minus_one",
            "logical_core",
            "generation",
            "guard_37",
        ],
        "instrumentation": "时间戳先保留为C++局部Scalar，函数末尾统一写出；区间内无诊断GM写回",
        "scope": (
            "包括零Tile生成、MTE3清零及原有前置cacheinvalid/fence和SYNCALL；"
            "排除set_ffts、同步后cacheinvalid、begin_invocation和业务"
        ),
        "status": "prepared_not_compiled",
        "production_changed": False,
        "SDK_changed": False,
    }
    (HERE / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
