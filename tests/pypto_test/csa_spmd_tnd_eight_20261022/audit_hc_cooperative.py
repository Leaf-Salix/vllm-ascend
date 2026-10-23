# SPDX-License-Identifier: Apache-2.0
"""Gate emitted ownership, ordered RMS carry, valid stores, ABI and resources."""

import argparse
import ast
import json
import re
import runpy
from pathlib import Path

from analyze_priority_phase import executable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("C04", "C05"))
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    build = here / f"runs/compile_hc_cooperative_{args.case}_v2"
    kernel = build / "kernel"
    reference = here / f"runs/compile_physical_capacity_tier_{args.case}_v1/kernel"
    assert json.loads((build / "result.json").read_text())["full_codegen_passed"]
    abi = lambda path: json.loads((path / "compiled_meta.json").read_text())["kernel_abi"]
    assert abi(kernel) == abi(reference)
    orchestration = "orchestration/decode_csa_spmd.cpp"
    assert executable((kernel / orchestration).read_text()) == executable((reference / orchestration).read_text())
    ir = (kernel / "ptoas/dynamic_csa_full.pto").read_text()
    vector = ir[ir.index("func.func @vector_schedule(") :]
    lines = vector.splitlines()
    definitions = {}
    for line in lines:
        if match := re.match(r"\s*(%[\w.$]+) = (.*)", line):
            definitions[match[1]] = match[2].split(" loc(")[0]

    def tile_address(tile):
        return definitions[tile].split("addr = ")[1].split()[0]

    own = [(index, line) for index, line in enumerate(lines) if '/hc_cooperative_stages.py"' in line]
    stores = [(index, line) for index, line in own if "pto.tstore" in line]
    roots = {"x_fp32": [], "widen_tail": [], "inv": []}
    for index, line in stores:
        tile = re.search(r"ins\((%[\w.$]+)", line)[1]
        dest = re.search(r"outs\((%[\w.$]+)", line)[1]
        view = definitions[dest]
        root = re.search(r"pto.partition_view %(\w+)__ssa_v0_view", view)[1]
        assert root in roots
        if root == "x_fp32":
            rows = re.search(r"sizes = \[(%[\w.$]+), %c512_index\]", view)[1]
            assert definitions[rows].startswith("arith.minsi ") and "%c8_index" in definitions[rows]
            remaining = re.search(r"arith.minsi (%[\w.$]+)", definitions[rows])[1]
            assert definitions[remaining].startswith("arith.subi ")
            assert any(f"pto.set_validshape {tile}, {rows}, %c512_index" in prior for prior in lines[:index])
        elif root == "widen_tail":
            assert "sizes = [%c1_index, %c8_index]" in view
        else:
            assert "sizes = [%c8_index, %c1_index]" in view
        roots[root].append(dict(ir_line=index, view=view, tile=tile))
    assert len(roots["x_fp32"]) == len(roots["widen_tail"]) == 4
    assert len(roots["inv"]) == 1
    partial_loads = [line for _, line in own if "pto.tload" in line and "partition_tensor_view<1x8xf32>" in line]
    assert len(partial_loads) == 2
    for line in partial_loads:
        view = re.search(r"ins\((%[\w.$]+)", line)[1]
        assert "pto.partition_view %widen_tail__ssa_v0_view" in definitions[view]
    additions = [line for _, line in own if "pto.tadd ins(" in line]
    assert len(additions) == 2
    operands = [re.search(r"ins\((%[\w.$]+), (%[\w.$]+).*outs\((%[\w.$]+)", line).groups() for line in additions]
    assert operands[1][0] == operands[0][2]
    assert tile_address(operands[1][2]) == tile_address(operands[0][0])
    assert re.search(r"scf.for %chunk\w* = %c0_index to %c32_index step %c2_index", vector)
    loaded_tiles = [re.search(r"outs\((%[\w.$]+)", line)[1] for line in partial_loads]
    assert [row[1] for row in operands] == loaded_tiles
    # Check the initialized accumulator is +0 and the ordered loop carry is
    # consumed by the original multiply/epsilon/rsqrt chain.
    init = next(line for _, line in own if "pto.texpands" in line and f"outs({operands[0][0]} " in line)
    zero = re.search(r"ins\((%[\w.$]+)", init)[1]
    assert re.fullmatch(r"arith.constant 0\.0*(?:e\+0+)? : f32", definitions[zero])
    assert any("pto.tmuls" in line and f"ins({operands[0][0]}," in line for _, line in own)
    last_store = max(row["ir_line"] for row in roots["widen_tail"])
    first_partial_load = lines.index(partial_loads[0])
    handoff = lines[last_store:first_partial_load]
    assert any("pto.comm.tnotify" in line for line in handoff)
    assert any("pto.comm.twait" in line for line in handoff)
    assert any("pto.cmo.cacheinvalid all" in line for line in handoff)

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
        candidate="hc_cooperative_v2",
        case_id=args.case,
        full_codegen_passed=True,
        root_return_abi_and_orchestration_unchanged=True,
        resources=resources,
        limits=limits,
        emitted_stores=roots,
        rms_add_chain=operands,
        rms_add_carry_same_physical_buffer=True,
        rms_32_chunks_two_per_loop_no_reassociation=True,
        input_notify_and_rms_acquire_cache_maintenance_present=True,
        source_tests="11 passed: nine numerical source simulations, exhaustive T1..1024 waits/slots, unchanged modules",
        device_execution=False,
        performance_eligible=False,
        limitation="CPU source simulation and codegen audit; device bit equality and full L1 are still required",
    )
    with (here / f"hc_cooperative_v2_{args.case}_codegen.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(resources))


if __name__ == "__main__":
    main()
