# SPDX-License-Identifier: Apache-2.0
"""Check emitted static tails, production-disabled identity and memory bounds."""

import ast
import json
import re
import runpy
from pathlib import Path

from analyze_priority_phase import executable


def main():
    here = Path(__file__).resolve().parent
    memory = runpy.run_path(str(here.parent / "csa_spmd_tnd_20261020/audit_score_wide_k128.py"))["memory"]
    limits = dict(vec=188416, mat=524288, acc=131072, left=65536, right=65536)
    rows = {}
    for case in ("C04", "C05"):
        build = here / f"runs/compile_output_group_owners_{case}_v2"
        assert json.loads((build / "result.json").read_text())["full_codegen_passed"]
        kernel = build / "kernel"
        old = here / f"runs/compile_output_rectangle_production_{case}_v1/kernel"
        abi = lambda p: json.loads((p / "compiled_meta.json").read_text())["kernel_abi"]
        assert abi(kernel) == abi(old)
        paths = [p.relative_to(kernel) for p in kernel.rglob("*.cpp")]
        assert len(paths) == 4
        for path in paths:
            if case == "C05" or path.parts[0] == "orchestration":
                assert executable((kernel / path).read_text()) == executable((old / path).read_text()), (case, path)
        (allocation,) = (kernel / "passes_dump").glob("*after_AllocateMemoryAddr.py")
        resources = {}
        for fn in ast.walk(ast.parse(allocation.read_text())):
            if not isinstance(fn, ast.FunctionDef) or fn.name not in ("cube_schedule", "vector_schedule"):
                continue
            maxima = {}
            for node in ast.walk(fn):
                if isinstance(node, ast.AnnAssign) and (region := memory(node.annotation)):
                    kind = region[0].strip("'").removeprefix("mem_").split("_")[0]
                    if kind in limits:
                        maxima[kind] = max(maxima.get(kind, 0), region[1] + region[2])
            assert all(value <= limits[kind] for kind, value in maxima.items())
            resources[fn.name] = maxima
        row = dict(root_return_abi_unchanged=True, resources=resources, limits=limits)
        if case == "C05":
            row["all_four_cpp_equal_to_retained_production"] = True
        else:
            ir = (kernel / "ptoas/dynamic_csa_full.pto").read_text()
            lines = ir[: ir.index("func.func @vector_schedule(")].splitlines()
            definitions = {}
            for line in lines:
                if match := re.match(r"\s*(%[\w.$]+) = (.*)", line):
                    definitions[match[1]] = match[2].split(" loc(")[0]
            stores = [line for line in lines if "pto.tstore" in line and '/output_projection_a_stages.py"' in line]
            assert len(stores) == 3
            checked = []
            for line in stores:
                width = int(re.search(r"loc=acc, dtype=f32, rows=96, cols=(\d+)", line)[1])
                assert width in (256, 64, 96)
                destination = re.search(r"outs\((%[\w.$]+)", line)[1]
                view = definitions[destination]
                assert "pto.partition_view %o_r__ssa_v0_view" in view
                valid_rows = re.search(rf"sizes = \[(%[\w.$]+), %c{width}_index\]", view)[1]
                assert definitions[valid_rows].startswith("arith.minsi ") and "%c96_index" in definitions[valid_rows]
                checked.append(dict(width=width, view=view, valid_rows=definitions[valid_rows]))
            assert {row["width"] for row in checked} == {256, 64, 96}
            row["three_static_stores"] = checked
        rows[case] = row
    result = dict(
        candidate="output_group_owners_v2",
        passed=True,
        cases=rows,
        source_contract_tests=29,
        device_execution=False,
        performance_eligible=False,
    )
    with (here / "output_group_owners_v2_codegen.json").open("x") as out:
        out.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({case: row["resources"] for case, row in rows.items()}))


if __name__ == "__main__":
    main()
