# SPDX-License-Identifier: Apache-2.0
"""Verify static N-tail stores, resources, root ABI and exact disabled code."""

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
    report = dict(candidate="output_rectangle_v3", device_execution=False, performance_eligible=False, cases={})
    for case in ("C04", "C05"):
        build = here / f"runs/compile_output_rectangle_{case}_v3"
        kernel = build / "kernel"
        old = here / f"runs/compile_physical_capacity_tier_{case}_v1/kernel"
        assert json.loads((build / "result.json").read_text())["full_codegen_passed"]
        abi = lambda p: json.loads((p / "compiled_meta.json").read_text())["kernel_abi"]
        assert abi(kernel) == abi(old)
        orch = "orchestration/decode_csa_spmd.cpp"
        assert executable((kernel / orch).read_text()) == executable((old / orch).read_text())
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
        row = dict(
            full_codegen_passed=True,
            root_return_abi_and_orchestration_unchanged=True,
            resources=resources,
            limits=limits,
        )
        if case == "C04":
            paths = [p.relative_to(kernel) for p in kernel.rglob("*.cpp")]
            assert len(paths) == 4
            assert all(executable((kernel / p).read_text()) == executable((old / p).read_text()) for p in paths)
            row["all_four_cpp_exactly_equal"] = [str(p) for p in paths]
        else:
            ir = (kernel / "ptoas/dynamic_csa_full.pto").read_text()
            ir = ir[: ir.index("func.func @vector_schedule(")]
            lines = ir.splitlines()
            definitions = {}
            for line in lines:
                if match := re.match(r"\s*(%[\w.$]+) = (.*)", line):
                    definitions[match[1]] = match[2].split(" loc(")[0]
            stores = [line for line in lines if "pto.tstore" in line and '/output_projection_a_stages.py"' in line]
            assert len(stores) == 2
            checked = []
            for line in stores:
                tile = re.search(r"ins\((%[\w.$]+)", line)[1]
                destination = re.search(r"outs\((%[\w.$]+)", line)[1]
                width = int(re.search(r"loc=acc, dtype=f32, rows=160, cols=(\d+)", line)[1])
                assert width in (192, 64)
                view = definitions[destination]
                assert "pto.partition_view %o_r__ssa_v0_view" in view
                rows = re.search(rf"sizes = \[(%[\w.$]+), %c{width}_index\]", view)[1]
                assert definitions[rows].startswith("arith.minsi ") and "%c160_index" in definitions[rows]
                checked.append(dict(tile=tile, width=width, view=view, valid_rows=definitions[rows]))
            assert {item["width"] for item in checked} == {192, 64}
            row["exact_static_N192_N64_output_stores"] = checked
            row["source_contract_checks"] = 29
            row["logical_weight_load_MiB"] = dict(original=128, candidate=64)
            row["logical_activation_load_MiB"] = dict(original=36, candidate=54)
            row["O_A_jobs"] = dict(original=64, candidate=48, candidate_per_AIC=2)
            row["limitations"] = (
                "Logical bytes are not HBM transactions; CPU boundaries are not runtime correctness or gains"
            )
        report["cases"][case] = row
    with (here / "output_rectangle_v3_codegen.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({case: row["resources"] for case, row in report["cases"].items()}))


if __name__ == "__main__":
    main()
