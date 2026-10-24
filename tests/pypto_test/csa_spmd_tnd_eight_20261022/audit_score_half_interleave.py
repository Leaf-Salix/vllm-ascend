# SPDX-License-Identifier: Apache-2.0
"""Check the actual N64 issue order, unchanged data movement and memory limits."""

import ast
import collections
import json
import re
import runpy
from pathlib import Path

from analyze_priority_phase import executable


def mma_order(text):
    order = []
    for line in text.split("func.func @vector_schedule(", 1)[0].splitlines():
        if "pto.tmatmul" not in line or "live_half_" in line:
            continue
        for operand, label in (
            ("split_query0", "QK0"),
            ("split_query1", "QK1"),
            ("split_coefficient0", "WS0"),
            ("split_coefficient1", "WS1"),
        ):
            if operand in line:
                order.append(label)
                break
    return order


def operations(text):
    return collections.Counter(re.findall(r"^\s*(?:%[\w.$]+(?:, %[\w.$]+)* = )?(pto\.[\w.]+)\b", text, re.M))


def main():
    here = Path(__file__).resolve().parent
    base = here / "runs/compile_output_rectangle_production_C04_v1/kernel"
    build = here / "runs/compile_score_half_interleave_C04_v1"
    kernel = build / "kernel"
    assert json.loads((build / "result.json").read_text())["full_codegen_passed"]
    native, current = [(p / "ptoas/dynamic_csa_full.pto").read_text() for p in (base, kernel)]
    before, after = mma_order(native), mma_order(current)
    assert before == ["QK0", "QK1"] + ["QK0", "QK1", "WS0", "WS1"] * 15 + ["WS0", "WS1"]
    assert after == ["QK0", "QK1"] + ["QK0", "WS0", "QK1", "WS1"] * 15 + ["WS0", "WS1"]
    assert operations(native) == operations(current)
    abi = lambda p: json.loads((p / "compiled_meta.json").read_text())["kernel_abi"]
    assert abi(base) == abi(kernel)
    identical = []
    for path in ("orchestration/decode_csa_spmd.cpp",):
        assert executable((base / path).read_text()) == executable((kernel / path).read_text()), path
        identical.append(path)
    # Each generated wrapper contains both arch-guarded helpers. Compare the
    # actual AIV function and following wrapper, not the filename's full text.
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
    report = dict(
        candidate="score_half_interleave_v1",
        case="C04",
        full_codegen_passed=True,
        performance_eligible=False,
        device_execution=False,
        root_abi_unchanged=True,
        exactly_identical_cpp=identical,
        aiv_function_and_wrapper_exact=True,
        all_pto_op_counts=dict(operations(current)),
        full_branch_MMA_order_before=before,
        full_branch_MMA_order_after=after,
        accumulator_intervals={name: sorted(value) for name, value in accumulators.items()},
        resources=resources,
        limits=limits,
        source_contract_tests=2,
        scope="Actual issue order and independent Acc addresses; asynchronous overlap and L1 gains unproven",
    )
    with (here / "score_half_interleave_v1_codegen.json").open("x") as out:
        out.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(resources=resources, accumulators=report["accumulator_intervals"])))


if __name__ == "__main__":
    main()
