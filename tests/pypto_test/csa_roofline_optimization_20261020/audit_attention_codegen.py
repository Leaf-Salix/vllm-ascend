# SPDX-License-Identifier: Apache-2.0
"""审查完整编译产物中的跨块累加和分配，不将 IR 检查冒充真机证明。"""
import argparse
import ast
import json
import re
from pathlib import Path


def calls(node, name):
    return [
        item for item in ast.walk(node)
        if isinstance(item, ast.Call) and ast.unparse(item.func) == name
    ]


def has_target(loop, prefix):
    return isinstance(loop, ast.For) and any(
        isinstance(item, ast.Name) and item.id.startswith(prefix)
        for item in ast.walk(loop.target)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-output", type=Path, required=True)
    args = parser.parse_args()
    output = args.compile_output
    source = output / "kernel/passes_dump/38_after_AllocateMemoryAddr.py"
    text = source.read_text()
    tree = ast.parse(text)
    cube = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "cube_schedule")
    query = next(n for n in ast.walk(cube) if has_target(n, "qk_t_inline"))
    pv = next(n for n in ast.walk(query) if has_target(n, "pv_sb_inline"))
    accumulations = calls(pv, "pl.tile.matmul_acc")
    pv_stores_inside = calls(pv, "pl.tile.store")
    final_stores = [
        n for n in calls(query, "pl.tile.store")
        if n.args and isinstance(n.args[0], ast.Name)
        and re.match(r"acc[0-3]_inline", n.args[0].id)
    ]
    roots = {}
    for match in re.finditer(
        r"acc([0-3])_inline\w+__ssa_v0: pl.Tile\[\[64, 128\], pl.FP32, "
        r"pl.MemRef\([^,]+, pl.const\((\d+), pl.INT64\), 32768\)", text
    ):
        roots[match[1]] = int(match[2])
    right_addresses = {
        match[1]: int(match[2])
        for match in re.finditer(
            r"right([0-3])_inline\w+__ssa_v0: pl.Tile\[\[128, 128\], pl.BF16, "
            r"pl.MemRef\([^,]+, pl.const\((\d+), pl.INT64\), 32768\)", text
        )
    }
    after_reuse = (output / "kernel/passes_dump/37_after_MemoryReuse.py").read_text()
    boundary = after_reuse.index("    def vector_schedule")
    allocations = {}
    for role, section in (("AIC", after_reuse[:boundary]), ("AIV", after_reuse[boundary:])):
        by_space = {}
        for space, size in re.findall(r"pl.tile.alloc\(pl.Mem.(\w+), (\d+)\)", section):
            by_space[space] = by_space.get(space, 0) + int(size)
        allocations[role] = by_space
    report = {
        "source": str(source),
        "four_acc_addresses": roots,
        "four_right_addresses": right_addresses,
        "matmul_acc_sites_in_block_loop": len(accumulations),
        "init_conditions": [
            ast.unparse(n.args[3] if len(n.args) > 3
                        else next(k.value for k in n.keywords if k.arg == "init_cond"))
            for n in accumulations
        ],
        "pv_store_sites_in_block_loop": len(pv_stores_inside),
        "final_pv_store_sites": len(final_stores),
        "allocated_bytes": allocations,
        "device_execution_proven": False,
        "performance_proven": False,
    }
    report["passed"] = (
        sorted(roots.values()) == [0, 32768, 65536, 98304]
        and len(accumulations) == 4 and not pv_stores_inside and len(final_stores) == 4
        and allocations["AIV"]["Vec"] <= 188416
        and allocations["AIC"]["Acc"] <= 131072
        and allocations["AIC"]["Mat"] <= 524288
    )
    (output / "attention_mechanism_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
