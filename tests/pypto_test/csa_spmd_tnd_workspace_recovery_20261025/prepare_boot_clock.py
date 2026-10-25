# SPDX-License-Identifier: Apache-2.0
"""为外部版和内部版增加独立诊断输出，只在隔离进程把占位store变成硬件时间戳。"""

import ast
import copy
import json
import shutil
import subprocess
from pathlib import Path

from prepare import hashes, replace_once

MAGIC = 741852963000


def call_name(node):
    if isinstance(node, ast.Call):
        return ast.unparse(node.func)
    return ""


def clock_write(core, slot, value):
    return ast.parse(f"pl.write(diagnostic_clock, [{core}, {slot}], pl.cast({value}, pl.INT64))").body[0]


def assert_original_logic(source, target):
    """移除明确的探针增量后，四个模块的AST必须与父版本完全一致。"""
    checked = []
    for name in ("entry.py", "cube_schedule.py", "vector_schedule.py", "binding.py"):
        original = ast.parse((source / name).read_text())
        original_returns = {
            node.name: next((stmt.value for stmt in node.body if isinstance(stmt, ast.Return)), None)
            for node in ast.walk(original)
            if isinstance(node, ast.FunctionDef)
        }
        tree = copy.deepcopy(ast.parse((target / name).read_text()))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                node.args.args = [arg for arg in node.args.args if arg.arg != "diagnostic_clock"]
                clean, skip_fence = [], False
                for stmt in node.body:
                    value = getattr(stmt, "value", None)
                    if skip_fence:
                        assert call_name(value) == "pl.system.fence"
                        skip_fence = False
                        continue
                    if (
                        isinstance(value, ast.Call)
                        and value.args
                        and isinstance(value.args[0], ast.Name)
                        and value.args[0].id == "diagnostic_clock"
                    ):
                        assert call_name(value) in ("pl.write", "pl.system.cacheinvalid")
                        skip_fence = call_name(value) == "pl.system.cacheinvalid"
                        continue
                    if name == "binding.py" and isinstance(stmt, ast.Assign):
                        lhs = ast.unparse(stmt.targets[0])
                        if lhs in ("self._diagnostic_clock_storage", "self.buffers['diagnostic_clock']"):
                            continue
                    clean.append(stmt)
                assert not skip_fence
                node.body = clean
                for stmt in node.body:
                    if (
                        isinstance(stmt, ast.Return)
                        and isinstance(stmt.value, ast.Tuple)
                        and isinstance(stmt.value.elts[-1], ast.Name)
                        and stmt.value.elts[-1].id == "diagnostic_clock"
                    ):
                        stmt.value.elts.pop()
                        if not isinstance(original_returns[node.name], ast.Tuple):
                            assert len(stmt.value.elts) == 1
                            stmt.value = stmt.value.elts[0]
            elif isinstance(node, ast.Call):
                node.args = [
                    arg for arg in node.args if not (isinstance(arg, ast.Name) and arg.id == "diagnostic_clock")
                ]
            elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple):
                target_tuple = node.targets[0]
                if isinstance(target_tuple.elts[-1], ast.Name) and target_tuple.elts[-1].id == "diagnostic_clock":
                    target_tuple.elts.pop()
        assert ast.dump(tree) == ast.dump(original), name
        checked.append(name)
    return checked


def make_package(source, target):
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    modules = {
        name: ast.parse((source / name).read_text())
        for name in (
            "entry.py",
            "cube_schedule.py",
            "vector_schedule.py",
            "binding.py",
        )
    }
    positions = {}
    for name in ("entry.py", "cube_schedule.py", "vector_schedule.py"):
        for func in modules[name].body:
            if isinstance(func, ast.FunctionDef):
                positions[func.name] = next(i for i, arg in enumerate(func.args.args) if arg.arg == "QUERY_GROUP")
                annotation = ast.parse("pl.InOut[pl.Tensor[[72, 8], pl.INT64]]", mode="eval").body
                func.args.args.insert(positions[func.name], ast.arg(arg="diagnostic_clock", annotation=annotation))
    for name in ("entry.py", "cube_schedule.py", "vector_schedule.py"):
        tree = modules[name]
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in positions:
                node.args.insert(positions[node.func.id], ast.Name(id="diagnostic_clock", ctx=ast.Load()))
            elif (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id in positions
            ):
                assert len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple)
                node.targets[0].elts.append(ast.Name(id="diagnostic_clock", ctx=ast.Store()))
        for func in tree.body:
            if isinstance(func, ast.FunctionDef):
                for stmt in func.body:
                    if isinstance(stmt, ast.Return):
                        if not isinstance(stmt.value, ast.Tuple):
                            stmt.value = ast.Tuple(elts=[stmt.value], ctx=ast.Load())
                        stmt.value.elts.append(ast.Name(id="diagnostic_clock", ctx=ast.Load()))
        if name in ("cube_schedule.py", "vector_schedule.py"):
            func = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
            core = "worker" if name == "cube_schedule.py" else "core"
            first_math = "hc_project_range" if name == "cube_schedule.py" else "hc_widen_rms_range"
            body, found = [], []
            for node in func.body:
                if isinstance(node, ast.Return):
                    body += [clock_write(core, 3, MAGIC + 3)]
                    body += ast.parse(
                        f"pl.system.cacheinvalid(diagnostic_clock, [1, 8], [{core}, 0])\npl.system.fence()"
                    ).body
                    found.append("end")
                body.append(node)
                value = getattr(node, "value", None)
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == core for t in node.targets):
                    body += [clock_write(core, 0, MAGIC), clock_write(core, 4, MAGIC + 4), clock_write(core, 5, core)]
                    found.append("start")
                elif call_name(value) == "begin_invocation":
                    body += [clock_write(core, 1, MAGIC + 1), clock_write(core, 6, "generation")]
                    found.append("initialized")
                elif call_name(value) == first_math:
                    body += [clock_write(core, 2, MAGIC + 2)]
                    found.append("first_math")
            assert found == ["start", "initialized", "first_math", "end"], (name, found)
            func.body = body
        ast.fix_missing_locations(tree)
    binding = modules["binding.py"]
    cls = next(n for n in binding.body if isinstance(n, ast.ClassDef) and n.name == "PreparedDynamicCSA")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    init.body += ast.parse(
        "self._diagnostic_clock_storage = torch.full((72 * 8 + 16,), 37, dtype=torch.int64, device=device)\n"
        "self.buffers['diagnostic_clock'] = self._diagnostic_clock_storage[8:-8].view(72, 8)"
    ).body
    for name, tree in modules.items():
        ast.fix_missing_locations(tree)
        (target / name).write_text("# SPDX-License-Identifier: Apache-2.0\n" + ast.unparse(tree) + "\n")
    files = [str(target / name) for name in modules]
    subprocess.run(["ruff", "format", *files], check=True)
    # 诊断包保留原导入和全部逻辑；AST重排会暴露原来靠注释压制的格式提示，不改数学。
    assert_original_logic(source, target)
    return hashes(target)


def main():
    here = Path(__file__).resolve().parent
    external = here / "candidates/boot_clock_external_v2"
    internal = here / "candidates/boot_clock_internal_v2"
    packages = {
        "reference": make_package(here / "frozen/pypto/deepseek_v4_flash_csa_spmd_tnd", external),
        "candidate": make_package(here / "candidates/internal_single_f72a", internal),
    }
    frozen = here / "frozen_boot_clock_v2"
    shutil.copytree(here / "frozen_optimization", frozen, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    dest = frozen / "pypto/deepseek_v4_flash_csa_spmd_tnd"
    for source in external.glob("*.py"):
        shutil.copy2(source, dest / source.name)
    baseline = json.loads((frozen / "baseline.json").read_text())
    baseline["files"] = packages["reference"]
    baseline["diagnostic_transform"] = "仅增加独立clock输出与四个插桩点，非性能基线"
    (frozen / "baseline.json").write_text(json.dumps(baseline, ensure_ascii=False, indent=2) + "\n")
    runner = (frozen / "run_pto.py").read_text()
    runner = replace_once(
        runner, "import types\n", "import types\n\nfrom boot_clock_hooks import compilation_probe, export_entry\n"
    )
    runner = replace_once(runner, 'return ("candidate", "current", "reference")', 'return ("candidate", "reference")')
    runner = replace_once(runner, 'return ("reference", "current", "candidate")', 'return ("reference", "candidate")')
    runner = replace_once(runner, 'control_name = mounted["current"]', 'control_name = mounted.get("current")')
    runner = replace_once(
        runner,
        '        "scope": "同SDK/同进程/同卡三侧完整L1：外部持久工作区、已保留的内部单SPMD、当前内部单SPMD",\n',
        '        "scope": "四点硬件时间戳诊断：入函数、初始化后、首段数学后、返回前；不作性能验收",\n'
        '        "performance_eligible": False,\n'
        '        "diagnostic_only": True,\n',
    )
    runner = replace_once(
        runner,
        "                entry(config=options)\n                torch.npu.synchronize()\n",
        '                with compilation_probe(args.output / ("instrumentation_" + side)):\n'
        "                    entry(config=options)\n                torch.npu.synchronize()\n",
    )
    # CPU全后端编译也安装同一后处理，禁止只验证未插桩版本。
    runner = replace_once(
        runner,
        "                    module.decode_csa_spmd.compile(**policy.constexprs(), config=options)\n",
        '                    with compilation_probe(args.output / ("instrumentation_" + side)):\n'
        "                        module.decode_csa_spmd.compile(**policy.constexprs(), config=options)\n",
    )
    marker = (
        '                        records.append(export_l1(source, args.output / f"l1_{sample:02d}_{side}", '
        "side, tables[side], init_layout=layouts[side]))\n"
    )
    runner = replace_once(
        runner,
        marker,
        '                        record = export_l1(source, args.output / f"l1_{sample:02d}_{side}", '
        "side, tables[side], init_layout=layouts[side])\n"
        '                        record["clock"] = export_entry(entries[side], record)\n'
        "                        records.append(record)\n",
    )
    ast.parse(runner)
    (frozen / "run_pto.py").write_text(runner)
    for name in ("launch_optimization.sh", "compile_optimization.sh"):
        (here / name.replace("optimization", "boot_clock")).write_text(
            (here / name).read_text().replace("frozen_optimization", "frozen_boot_clock_v2")
        )
    shutil.copy2(here / "boot_clock_hooks.py", frozen / "boot_clock_hooks.py")
    report = {
        "status": "prepared_not_compiled",
        "sources": {"reference": str(external), "candidate": str(internal)},
        "files": packages,
        "scope": "诊断专用，额外72x8 INT64输出及store会扰动计时，禁止计入优化收益",
        "clock_columns": [
            "function_start",
            "after_initialization",
            "after_hc_first_math",
            "before_return",
            "physical_core",
            "logical_core",
            "generation",
            "guard_37",
        ],
        "safety": "每逻辑核独占64B clock行；额外输出单独分配，首尾64B红区；原数学/清零/同步不变",
    }
    (here / "boot_clock_diagnostic.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
