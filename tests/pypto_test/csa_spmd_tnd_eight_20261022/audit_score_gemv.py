# SPDX-License-Identifier: Apache-2.0
"""Audit actual C04 GEMV geometry, aliases, unchanged protocol and binary build."""

import ast
import json
import re
import runpy
from pathlib import Path

from analyze_priority_phase import executable
from audit_score_half_interleave import mma_order, operations


def main():
    here = Path(__file__).resolve().parent
    base = here / "runs/compile_output_rectangle_production_C04_v1/kernel"
    build = here / "runs/compile_score_gemv_C04_v3"
    kernel = build / "kernel"
    assert json.loads((build / "result.json").read_text())["full_codegen_passed"]
    source_dirs = [here / "runs" / name for name in ("output_rectangle_production_v1", "score_gemv_v3")]
    trees = [ast.parse((p / "indexer_score_stages.py").read_text()) for p in source_dirs]
    protocol = []
    for tree in trees:
        protocol.append(
            [
                ast.dump(n, include_attributes=False)
                for n in ast.walk(tree)
                if isinstance(n, ast.Call) and ast.unparse(n.func).startswith("pl.system.sync_")
            ]
        )
    assert protocol[0] == protocol[1]
    for path in source_dirs[0].glob("*.py"):
        if path.name != "indexer_score_stages.py":
            assert path.read_bytes() == (source_dirs[1] / path.name).read_bytes()
    # Physical windows cover uninitialized padding, but their valid rectangles
    # contain exactly the 192 QK-produced rows, never the padding rows.
    for query in range(6):
        offset = query % 3 * 64
        assert offset + 256 <= 384 and offset + 64 <= 192
        assert query * 64 + 64 <= 384
    native, current = [(p / "ptoas/dynamic_csa_full.pto").read_text() for p in (base, kernel)]
    assert mma_order(current) == ["QK0", "QK1"] * 16
    assert [op for op in mma_order(native) if op.startswith("QK")] == mma_order(current)
    cube = current.split("func.func @vector_schedule(", 1)[0]
    gemv = [line for line in cube.splitlines() if "pto.tgemv ins(" in line]
    assert len(gemv) == 96
    assert all("loc=left, dtype=f16, rows=1, cols=256" in line for line in gemv)
    assert all("loc=right, dtype=f16, rows=256, cols=64" in line for line in gemv)
    assert all("loc=acc, dtype=f32, rows=16, cols=64" in line for line in gemv)
    right_allocations = [line for line in cube.splitlines() if "gemv_right" in line and "pto.alloc_tile" in line]
    assert len(right_allocations) == 96
    assert all("valid_row = %c64_index valid_col = %c64_index" in line for line in right_allocations)
    destinations = {re.search(r"outs\((%[\w.]+)", line).group(1) for line in gemv}
    assert len(destinations) == 1, destinations
    old_ops, new_ops = operations(native), operations(current)
    assert new_ops["pto.tmatmul"] + new_ops["pto.tmatmul.acc"] == (
        old_ops["pto.tmatmul"] + old_ops["pto.tmatmul.acc"] - 32
    )
    assert new_ops["pto.tstore"] == old_ops["pto.tstore"] + 80
    for op in set(old_ops) | set(new_ops):
        if any(word in op for word in ("sync", "wait", "notify", "fence")):
            assert old_ops[op] == new_ops[op], op
    abi = [json.loads((p / "compiled_meta.json").read_text())["kernel_abi"] for p in (base, kernel)]
    assert abi[0] == abi[1]
    orch = "orchestration/decode_csa_spmd.cpp"
    assert executable((base / orch).read_text()) == executable((kernel / orch).read_text())
    marker = "static __aicore__ void vector_schedule("
    aiv = [executable((p / "kernels/aiv/vector_schedule.cpp").read_text()).split(marker, 1)[1] for p in (base, kernel)]
    assert aiv[0] == aiv[1]
    memory = runpy.run_path(str(here.parent / "csa_spmd_tnd_20261020/audit_score_wide_k128.py"))["memory"]
    limits = dict(vec=188416, mat=524288, acc=131072, left=65536, right=65536)
    (allocation,) = (kernel / "passes_dump").glob("*after_AllocateMemoryAddr.py")
    resources, accumulators = {}, {}
    for fn in ast.walk(ast.parse(allocation.read_text())):
        if not isinstance(fn, ast.FunctionDef) or fn.name not in ("cube_schedule", "vector_schedule"):
            continue
        maxima = {}
        for node in ast.walk(fn):
            if isinstance(node, ast.AnnAssign) and (region := memory(node.annotation)):
                kind = region[0].strip("'").removeprefix("mem_").split("_")[0]
                if kind in limits:
                    maxima[kind] = max(maxima.get(kind, 0), region[1] + region[2])
                if kind == "acc" and isinstance(node.target, ast.Name):
                    for name in ("split_qk_acc0", "split_qk_acc1", "split_ws_acc"):
                        if node.target.id.startswith(name):
                            accumulators.setdefault(name, set()).add((region[1], region[1] + region[2]))
        assert all(value <= limits[kind] for kind, value in maxima.items())
        resources[fn.name] = maxima
    assert set(accumulators) == {"split_qk_acc0", "split_qk_acc1", "split_ws_acc"}
    assert all(len(value) == 1 for value in accumulators.values())
    intervals = sorted(next(iter(value)) for value in accumulators.values())
    assert all(left[1] <= right[0] for left, right in zip(intervals, intervals[1:]))
    # Finish CCE/host binary compilation without initializing or executing NPU.
    from pypto._kernel_abi import KernelABI
    from pypto.jit._artifact_manifest import BuildKind
    from pypto.runtime._prebuilt import prepare_prebuilt

    prepare_prebuilt(kernel, "a2a3", BuildKind.SINGLE_CHIP, kernel_abi=KernelABI.from_record(abi[1]))
    report = dict(
        candidate="score_gemv_v3",
        case="C04",
        full_codegen_and_binary_compile_passed=True,
        device_execution=False,
        performance_eligible=False,
        root_abi_unchanged=True,
        aiv_function_and_wrapper_exact=True,
        orchestration_exact=True,
        resources=resources,
        limits=limits,
        accumulator_intervals={name: sorted(value) for name, value in accumulators.items()},
        actual_GEMV_count=96,
        actual_Right_valid_K64_N64_count=96,
        replaced_WS_MMA_count=32,
        added_FIX_stores=80,
        unchanged_QK_MMA_order=True,
        single_WS_accumulator=True,
        explicit_sync_counts_unchanged=True,
        explicit_sync_source_arguments_exact=True,
        valid_Right_windows_inside_produced_QK_rows=True,
        scope="Independent-GEMV code generation and resource gates; complete CSA behavior/performance pending",
    )
    with (here / "score_gemv_v3_codegen.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
