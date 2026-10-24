# SPDX-License-Identifier: Apache-2.0
"""Audit full C04 pooled-GEMV geometry, GM handoff, resource and binary gates."""

import argparse
import ast
import hashlib
import json
import re
import runpy
from pathlib import Path

from analyze_priority_phase import executable
from audit_score_half_interleave import mma_order, operations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", type=int, default=1)
    args = parser.parse_args()
    assert args.revision > 0
    candidate = f"score_gemv_pool_v{args.revision}"
    here = Path(__file__).resolve().parent
    base = here / "runs/compile_output_rectangle_production_C04_v1/kernel"
    build = here / f"runs/compile_score_gemv_pool_C04_v{args.revision}"
    kernel = build / "kernel"
    assert json.loads((build / "result.json").read_text())["full_codegen_passed"]
    proof = json.loads((here / "runs/score_gemv_pool_probe_device_v1/result.json").read_text())
    assert proof["passed"] and len(proof["cases"]) == 24
    assert all(row["bitwise"] and row["guards"] and row["inputs_unchanged"] for row in proof["cases"])
    sources = [here / "runs" / n for n in ("output_rectangle_production_v1", candidate)]
    changed = {"entry.py", "vector_schedule.py", "indexer_score_stages.py"}
    for path in sources[0].glob("*.py"):
        if path.name not in changed:
            assert path.read_bytes() == (sources[1] / path.name).read_bytes()
    protocols = []
    for path in sources:
        tree = ast.parse((path / "indexer_score_stages.py").read_text())
        protocols.append(
            sorted(
                ast.dump(n)
                for n in ast.walk(tree)
                if isinstance(n, ast.Call) and ast.unparse(n.func).startswith("pl.system.sync_")
            )
        )
    assert protocols[0] == protocols[1]
    native, current = [(p / "ptoas/dynamic_csa_full.pto").read_text() for p in (base, kernel)]
    assert mma_order(current) == ["QK0", "QK1"] * 16
    assert [op for op in mma_order(native) if op.startswith("QK")] == mma_order(current)
    cube_text, vector_text = current.split("func.func @vector_schedule(", 1)
    cube = [line.split(" loc(")[0] for line in cube_text.splitlines()]
    vector = [line.split(" loc(")[0] for line in vector_text.splitlines()]
    gemvs = [line for line in cube if "pto.tgemv ins(" in line]
    assert len(gemvs) == 96
    assert all("loc=left, dtype=f16, rows=1, cols=256" in line for line in gemvs)
    assert all("loc=right, dtype=f16, rows=256, cols=64" in line for line in gemvs)
    assert all("loc=acc, dtype=f32, rows=16, cols=64" in line for line in gemvs)
    assert not any("pto.tmov" in line and line.count("loc=acc") >= 2 for line in cube)
    rights = [line for line in cube if "gemv_right" in line and "pto.alloc_tile" in line]
    assert len(rights) == 96
    assert all("valid_row = %c64_index valid_col = %c64_index" in line for line in rights)
    stores = [line for line in cube if "pto.tstore ins(%split_ws_acc" in line]
    assert len(stores) == 16
    assert all("rows=16, cols=384" in line and "partition_tensor_view<1x384xf32>" in line for line in stores)
    loads = [line for line in vector if "pto.tload" in line and "gemv_panels" in line]
    assert len(loads) == 1 and "rows=16, cols=192" in loads[0]
    gathers = [line for line in vector if "pto.tgather" in line and "gemv_indices" in line]
    assert len(gathers) == 1 and "cols=3072" in gathers[0]
    views = [
        line.strip()
        for line in cube + vector
        if "pto.make_tensor_view" in line and any(name in line for name in ("gemv_packed_transfer", "gemv_transport"))
    ]
    assert len(views) == 2 and all("strides = [%c384_index, %c1_index]" in line for line in views)
    assert len({re.search(r"pto.make_tensor_view (%arg\d+)", line).group(1) for line in views}) == 1
    old_ops, new_ops = operations(native), operations(current)
    assert new_ops["pto.tmatmul"] + new_ops["pto.tmatmul.acc"] == (
        old_ops["pto.tmatmul"] + old_ops["pto.tmatmul.acc"] - 32
    )
    assert new_ops["pto.tstore"] == old_ops["pto.tstore"]
    for op in ("pto.comm.tnotify", "pto.comm.twait", "pto.sync.wait"):
        assert old_ops[op] == new_ops[op], op
    abi = [json.loads((p / "compiled_meta.json").read_text())["kernel_abi"] for p in (base, kernel)]
    assert abi[0] == abi[1]
    orch = "orchestration/decode_csa_spmd.cpp"
    assert executable((base / orch).read_text()) == executable((kernel / orch).read_text())
    memory = runpy.run_path(str(here.parent / "csa_spmd_tnd_20261020/audit_score_wide_k128.py"))["memory"]
    limits = dict(vec=188416, mat=524288, acc=131072, left=65536, right=65536)
    (allocation,) = (kernel / "passes_dump").glob("*after_AllocateMemoryAddr.py")
    resources, accumulators, slots = {}, {}, {}
    for fn in ast.walk(ast.parse(allocation.read_text())):
        if not isinstance(fn, ast.FunctionDef) or fn.name not in ("cube_schedule", "vector_schedule"):
            continue
        maxima = {}
        for node in ast.walk(fn):
            if not isinstance(node, ast.AnnAssign) or not (region := memory(node.annotation)):
                continue
            kind = region[0].strip("'").removeprefix("mem_").split("_")[0]
            if kind in limits:
                maxima[kind] = max(maxima.get(kind, 0), region[1] + region[2])
            if kind != "acc" or not isinstance(node.target, ast.Name):
                continue
            for name in ("split_qk_acc0", "split_qk_acc1", "split_ws_acc"):
                if node.target.id.startswith(name):
                    accumulators.setdefault(name, set()).add((region[1], region[1] + region[2]))
            for query in range(6):
                if node.target.id.startswith(f"gemv_slot{query}"):
                    slots.setdefault(query, set()).add((region[1], region[2]))
        assert all(value <= limits[kind] for kind, value in maxima.items())
        resources[fn.name] = maxima
    assert set(accumulators) == {"split_qk_acc0", "split_qk_acc1", "split_ws_acc"}
    assert all(len(value) == 1 for value in accumulators.values())
    intervals = sorted(next(iter(value)) for value in accumulators.values())
    assert all(left[1] <= right[0] for left, right in zip(intervals, intervals[1:]))
    pool_start, pool_end = next(iter(accumulators["split_ws_acc"]))
    assert pool_end - pool_start == 16 * 384 * 4
    assert set(slots) == set(range(6))
    for query, regions in slots.items():
        assert regions == {(pool_start + query * 16 * 64 * 4, 16 * 64 * 4)}, (query, regions)
    from pypto._kernel_abi import KernelABI
    from pypto.jit._artifact_manifest import BuildKind
    from pypto.runtime._prebuilt import prepare_prebuilt

    prepare_prebuilt(kernel, "a2a3", BuildKind.SINGLE_CHIP, kernel_abi=KernelABI.from_record(abi[1]))
    report = dict(
        candidate=candidate,
        case="C04",
        full_codegen_and_binary_compile_passed=True,
        device_execution=False,
        performance_eligible=False,
        root_abi_unchanged=True,
        orchestration_exact=True,
        resources=resources,
        limits=limits,
        accumulator_intervals={name: sorted(value) for name, value in accumulators.items()},
        gemv_slots={str(query): sorted(value) for query, value in slots.items()},
        actual_GEMV_count=96,
        replaced_WS_MMA_count=32,
        pooled_FIX_stores=16,
        added_FIX_stores=0,
        actual_AIV_packed_load="16x192 FP32 from a 384-column GM view",
        actual_AIV_gather="3072 INT32 indices, derived within safe INT16 range",
        matched_GM_views=views,
        unchanged_QK_MMA_order=True,
        explicit_protocol_source_and_counts_unchanged=True,
        cpp_sha256={
            str(p.relative_to(kernel)): hashlib.sha256(p.read_bytes()).hexdigest() for p in kernel.rglob("*.cpp")
        },
        scope="Full-kernel code generation/resources and binary only; full CSA device correctness/L1 pending",
    )
    with (here / f"{candidate}_codegen.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
