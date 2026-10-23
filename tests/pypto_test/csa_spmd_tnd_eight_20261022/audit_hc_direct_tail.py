# SPDX-License-Identifier: Apache-2.0
"""Verify emitted direct stores retain dynamic row bounds and eliminate scratch I/O."""

import ast
import json
import re
import runpy
from pathlib import Path

from analyze_priority_phase import executable


def main():
    here = Path(__file__).resolve().parent
    build = here / "runs/compile_hc_direct_tail_C01_v2"
    kernel = build / "kernel"
    reference = here / "runs/compile_physical_capacity_tier_C01_v1/kernel"
    assert json.loads((build / "result.json").read_text())["full_codegen_passed"]
    abi = lambda p: json.loads((p / "compiled_meta.json").read_text())["kernel_abi"]
    assert abi(kernel) == abi(reference)
    filename = "orchestration/decode_csa_spmd.cpp"
    assert executable((kernel / filename).read_text()) == executable((reference / filename).read_text())

    # Each generated per-core file also contains the other static function.
    # Compare the complete Cube definition rather than accidentally including
    # the intentionally changed Vector definition in the AIC filename.
    def cube_function(directory):
        source = executable((directory / "kernels/aic/cube_schedule.cpp").read_text())
        return source[
            source.index("static __aicore__ void cube_schedule(") : source.index(
                "static __aicore__ void vector_schedule("
            )
        ]

    assert cube_function(kernel) == cube_function(reference)
    ir = (kernel / "ptoas/dynamic_csa_full.pto").read_text()
    vector = ir[ir.index("func.func @vector_schedule(") :]
    old = (reference / "ptoas/dynamic_csa_full.pto").read_text()
    for name in ("widen_tail", "norm_tail"):
        pattern = rf"%{name}__ssa_v0_view\b"
        assert len(re.findall(pattern, vector)) == 1, "Scratch view must have no consumers"
        assert len(re.findall(pattern, old)) > 1
    definitions = {}
    lines = vector.splitlines()
    for line in lines:
        if match := re.match(r"\s*(%[\w.$]+) = (.*)", line):
            definitions[match[1]] = match[2].split(" loc(")[0]
    source = here / "runs/hc_direct_tail_v2/hc_stages.py"
    positions = {}
    for node in ast.walk(ast.parse(source.read_text())):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == "pl.assemble"
        ):
            positions[node.lineno] = ast.unparse(node.targets[0])
    assert len(positions) == 2
    stores = []
    for number, root in positions.items():
        count = 0
        for index, line in enumerate(lines):
            if "pto.tstore" not in line or f'/hc_stages.py":{number}:' not in line:
                continue
            tile = re.search(r"ins\((%[\w.$]+)", line)[1]
            destination = re.search(r"outs\((%[\w.$]+)", line)[1]
            view = definitions[destination]
            width, expected_root = (512, "x_fp32") if root == "x_hc32_flat" else (256, "normed")
            assert f"pto.partition_view %{expected_root}__ssa_v0_view" in view
            rows = re.search(rf"sizes = \[(%[\w.$]+), %c{width}_index\]", view)[1]
            assert definitions[rows].startswith("arith.minsi ") and "%c8_index" in definitions[rows]
            remaining = re.search(r"arith.minsi (%[\w.$]+)", definitions[rows])[1]
            assert definitions[remaining].startswith("arith.subi ")
            setter = f"pto.set_validshape {tile}, {rows}, %c{width}_index"
            assert any(setter in previous for previous in lines[:index]), (root, tile)
            stores.append(
                dict(
                    root=expected_root,
                    tile=tile,
                    destination=destination,
                    view=view,
                    rows=definitions[rows],
                    remaining=definitions[remaining],
                )
            )
            count += 1
        assert count == (4 if width == 512 else 2), (root, count)
    memory = runpy.run_path(str(here.parent / "csa_spmd_tnd_20261020/audit_score_wide_k128.py"))["memory"]
    (allocation,) = (kernel / "passes_dump").glob("*after_AllocateMemoryAddr.py")
    limits = dict(vec=188416, mat=524288, acc=131072, left=65536, right=65536)
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
    report = dict(
        candidate="hc_direct_tail_v2",
        case_id="C01",
        full_codegen_passed=True,
        root_and_return_abi_unchanged=True,
        complete_cube_function_and_orchestration_unchanged=True,
        scratch_views_unreferenced=True,
        emitted_direct_stores=stores,
        resources=resources,
        limits=limits,
        source_math_tests=6,
        device_execution=False,
        performance_eligible=False,
        limitation="Generated bounds and removed scratch accesses are verified; "
        "actual precision, guards and full L1 remain",
    )
    with (here / "hc_direct_tail_v2_C01_codegen.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(resources))


if __name__ == "__main__":
    main()
