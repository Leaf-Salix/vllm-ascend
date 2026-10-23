# SPDX-License-Identifier: Apache-2.0
"""Audit packed RMS handoff and local views before considering device execution."""

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
    build = here / f"runs/compile_hc_cooperative_{args.case}_v4"
    kernel = build / "kernel"
    old = here / f"runs/compile_physical_capacity_tier_{args.case}_v1/kernel"
    assert json.loads((build / "result.json").read_text())["full_codegen_passed"]
    abi = lambda path: json.loads((path / "compiled_meta.json").read_text())["kernel_abi"]
    assert abi(kernel) == abi(old)
    orch = "orchestration/decode_csa_spmd.cpp"
    assert executable((kernel / orch).read_text()) == executable((old / orch).read_text())
    ir = (kernel / "ptoas/dynamic_csa_full.pto").read_text()
    ir = ir[ir.index("func.func @vector_schedule(") :]
    lines = ir.splitlines()
    definitions = {}
    for line in lines:
        if match := re.match(r"\s*(%[\w.$]+) = (.*)", line):
            definitions[match[1]] = match[2].split(" loc(")[0]
    own = [line for line in lines if '/hc_cooperative_stages.py"' in line]

    def operands(line, kind):
        return re.findall(r"%[\w.$]+", re.search(rf"{kind}\((.*?) :", line)[1])

    def address(tile):
        value = re.search(r"pto.alloc_tile addr = (%[\w.$]+)", definitions[tile])[1]
        return int(re.search(r"arith.constant (\d+) : i64", definitions[value])[1])

    stores = {root: [] for root in ("x_fp32", "widen_tail", "inv")}
    for line in own:
        if "pto.tstore" not in line:
            continue
        (tile,) = operands(line, "ins")
        (dest,) = operands(line, "outs")
        view = definitions[dest]
        root = re.search(r"pto.partition_view %(\w+)__ssa_v0_view", view)[1]
        assert root in stores
        stores[root].append(dict(tile=tile, view=view))
        if root == "x_fp32":
            rows = re.search(r"sizes = \[(%[\w.$]+), %c512_index\]", view)[1]
            assert definitions[rows].startswith("arith.minsi ") and "%c8_index" in definitions[rows]
            assert any(
                f"pto.set_validshape {tile}, {rows}, %c512_index" in prior for prior in lines[: lines.index(line)]
            )
        elif root == "widen_tail":
            assert "sizes = [%c1_index, %c64_index]" in view
        else:
            assert "sizes = [%c8_index, %c1_index]" in view
    assert {root: len(values) for root, values in stores.items()} == {"x_fp32": 4, "widen_tail": 1, "inv": 1}
    partial = next(
        name for name in definitions if name.startswith("%partial_rows_") and "pto.alloc_tile" in definitions[name]
    )
    packet = stores["widen_tail"][0]["tile"]
    assert address(partial) == address(packet)
    assert "rows=8, cols=8" in definitions[partial] and "rows=1, cols=64" in definitions[packet]
    views = [line for line in own if f"pto.subview {partial}[" in line]
    assert len(views) == 4
    assert all("sizes [1, 8] valid [%c1_index, %c8_index]" in line for line in views)
    assert re.search(r"scf.for %local_chunk\w* = %c0_index to %c8_index step %c4_index", ir)
    loads = [line for line in own if "pto.tload" in line and "partition_tensor_view<4x64xf32>" in line]
    assert len(loads) == 1
    (packets,) = operands(loads[0], "outs")
    (view,) = operands(loads[0], "ins")
    assert "pto.partition_view %widen_tail__ssa_v0_view" in definitions[view]
    (extract,) = [line for line in own if "pto.textract" in line]
    ordered, row, column = operands(extract, "ins")
    (part,) = operands(extract, "outs")
    assert column == "%c0_index" and row.startswith("%chunk_")
    assert address(packets) == address(ordered)
    assert "rows=4, cols=64" in definitions[packets] and "rows=32, cols=8" in definitions[ordered]
    assert re.search(r"scf.for %chunk\w* = %c0_index to %c32_index step %c1_index", ir)
    (addition,) = [line for line in own if "pto.tadd ins(" in line]
    total, rhs = operands(addition, "ins")
    (result,) = operands(addition, "outs")
    assert rhs == part and address(result) == address(total)
    assert not address(ordered) <= address(part) < address(ordered) + 1024
    assert not address(ordered) <= address(total) < address(ordered) + 1024
    (sqrt,) = [line for line in own if "pto.trsqrt" in line]
    assert len(operands(sqrt, "ins")) == 2, "Explicit scratch selects the original high-precision mode"
    handoff = lines[
        lines.index(next(line for line in own if "pto.tstore" in line and "1x64xf32" in line)) : lines.index(loads[0])
    ]
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
        candidate="hc_cooperative_v4",
        case_id=args.case,
        full_codegen_passed=True,
        root_return_abi_orchestration_unchanged=True,
        resources=resources,
        limits=limits,
        emitted_stores=stores,
        producer_packet_address=address(packet),
        consumer_packet_address=address(ordered),
        handoff_per_M8="Four 1x64 FP32 stores and one 4x64 strided load; 32 original chunks retained",
        local_views_and_accumulator_carry_checked=True,
        high_precision_rsqrt_scratch=True,
        source_math_and_ownership_checks=11,
        device_execution=False,
        performance_eligible=False,
        limitation="Static/source checks do not prove A3 bit equality or runtime speed; "
        "original v2 tasks remain unchanged",
    )
    with (here / f"hc_cooperative_v4_{args.case}_codegen.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(resources))


if __name__ == "__main__":
    main()
