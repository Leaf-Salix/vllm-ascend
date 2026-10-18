#!/usr/bin/env python3
"""只读核对修复后的分档代码与分配；不将静态范围当成运行时峰值或性能。"""

import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
POLICY = HERE.parent / "attention_local_stats_policy_evidence"


def integer(node):
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.Call) and ast.unparse(node.func) == "pl.const":
        return integer(node.args[0])
    return None


def allocations(directory):
    paths = list((directory / "kernel/passes_dump").glob("*after_AllocateMemoryAddr.py"))
    assert len(paths) == 1, paths
    tree = ast.parse(paths[0].read_text())
    windows = {}
    annotations = 0
    statistics = []
    for stmt in ast.walk(tree):
        if not isinstance(stmt, ast.AnnAssign):
            continue
        annotation = stmt.annotation
        if not isinstance(annotation, ast.Subscript) or ast.unparse(annotation.value) != "pl.Tile":
            continue
        args = annotation.slice.elts
        memory = next((ast.unparse(a) for a in args if ast.unparse(a).startswith("pl.Mem.")), None)
        ref = next((a for a in args if isinstance(a, ast.Call) and ast.unparse(a.func) == "pl.MemRef"), None)
        if memory is None or ref is None:
            continue
        address, size = integer(ref.args[1]), integer(ref.args[2])
        assert address is not None and size is not None
        annotations += 1
        windows.setdefault(memory, set()).add((address, size))
        if "stats_cache" in ast.unparse(stmt.target):
            statistics.append(dict(name=ast.unparse(stmt.target), address=address, bytes=size))
    return {
        "source": str(paths[0]),
        "tile_annotations": annotations,
        "memory": {
            space: dict(unique_windows=len(values), max_declared_end=max(a + s for a, s in values))
            for space, values in sorted(windows.items())
        },
        "statistics_tiles": statistics,
    }


def executable(path):
    return "\n".join(line for line in path.read_text().splitlines() if not line.lstrip().startswith("//"))


def main():
    rows = []
    for strategy in ("long_off", "short_off", "long_on", "short_on", "small_on"):
        new = HERE / f"cpu_{strategy}_v1"
        old = POLICY / f"cpu_{strategy}_v1"
        row = dict(strategy=strategy, before=allocations(old), after=allocations(new))
        result = json.loads((new / "result.json").read_text())
        assert result["passed"] and result["single_mix"] and len(result["symbolic_dimensions"]) == 58
        if strategy.endswith("off"):
            reference = HERE / f"cpu_{strategy}_reference_v1"
            relative = "kernel/ptoas/dynamic_csa_full.cpp"
            row["same_compiler_disabled_equal"] = executable(new / relative) == executable(reference / relative)
            assert row["same_compiler_disabled_equal"]
            assert not row["after"]["statistics_tiles"]
        rows.append(row)
    report = dict(
        scope="原始CPU编译产物的只读AST/代码核对",
        limitation="声明地址末端不是生命周期峰值；本报告不是设备正确性或性能证据",
        normalization="C++仅忽略整行注释；没有修改变量、地址、事件或计算",
        rows=rows,
    )
    (HERE / "CODEGEN_MEMORY_AUDIT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for row in rows:
        print(row["strategy"], "tile annotations", row["before"]["tile_annotations"], "->", row["after"]["tile_annotations"],
              "Vec", row["after"]["memory"].get("pl.Mem.Vec"), "stats", len(row["after"]["statistics_tiles"]))


if __name__ == "__main__":
    main()
