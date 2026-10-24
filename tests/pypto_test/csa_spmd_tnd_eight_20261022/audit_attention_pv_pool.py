# SPDX-License-Identifier: Apache-2.0
"""Check PV's four original products, Acc column aliases and one FIX store."""

import ast
import collections
import json
import re
import runpy
from pathlib import Path

from analyze_priority_phase import executable
from audit_score_half_interleave import operations


def main():
    here = Path(__file__).resolve().parent
    base = here / "runs/compile_output_rectangle_production_C04_v1/kernel"
    build = here / "runs/compile_attention_pv_pool_C04_v1"
    kernel = build / "kernel"
    assert json.loads((build / "result.json").read_text())["full_codegen_passed"]
    before, after = [(p / "ptoas/dynamic_csa_full.pto").read_text() for p in (base, kernel)]
    pto = [line.split(" loc(")[0] for line in after.splitlines()]
    products = [line for line in pto if "pto.tmatmul" in line and "pv_probability_left" in line]
    assert len(products) == 4
    assert all("pto.tmatmul ins(" in line for line in products)
    assert all("loc=left, dtype=bf16, rows=64, cols=128" in line for line in products)
    assert all("loc=right, dtype=bf16, rows=128, cols=128" in line for line in products)
    old_ops, new_ops = operations(before), operations(after)
    assert new_ops["pto.tmatmul"] == old_ops["pto.tmatmul"]
    assert new_ops["pto.tmatmul.acc"] == old_ops["pto.tmatmul.acc"]
    assert new_ops["pto.tstore"] == old_ops["pto.tstore"] - 3
    stores = [line for line in pto if "pto.tstore" in line and "pv_acc_pool" in line]
    assert len(stores) == 1 and "partition_tensor_view<64x512xf32>" in stores[0]
    abi = [json.loads((p / "compiled_meta.json").read_text())["kernel_abi"] for p in (base, kernel)]
    assert abi[0] == abi[1]
    orch = "orchestration/decode_csa_spmd.cpp"
    assert executable((base / orch).read_text()) == executable((kernel / orch).read_text())
    marker = "static __aicore__ void vector_schedule("
    aiv = [executable((p / "kernels/aiv/vector_schedule.cpp").read_text()).split(marker, 1)[1] for p in (base, kernel)]
    assert aiv[0] == aiv[1]
    memory = runpy.run_path(str(here.parent / "csa_spmd_tnd_20261020/audit_score_wide_k128.py"))["memory"]
    limits = dict(vec=188416, mat=524288, acc=131072, left=65536, right=65536)
    resources, pv_regions = {}, {}
    for label, root in (("retained", base), ("candidate", kernel)):
        (allocation,) = (root / "passes_dump").glob("*after_AllocateMemoryAddr.py")
        names = collections.defaultdict(set)
        resources[label] = {}
        for fn in ast.walk(ast.parse(allocation.read_text())):
            if not isinstance(fn, ast.FunctionDef) or fn.name not in ("cube_schedule", "vector_schedule"):
                continue
            maxima = {}
            for node in ast.walk(fn):
                if isinstance(node, ast.AnnAssign) and (region := memory(node.annotation)):
                    kind = region[0].strip("'").removeprefix("mem_").split("_")[0]
                    if kind in limits:
                        maxima[kind] = max(maxima.get(kind, 0), region[1] + region[2])
                    if isinstance(node.target, ast.Name):
                        for name in ("pv_first_right", "pv_next_right", "pv_first_slot", "pv_slot_acc", "pv_acc_pool"):
                            if node.target.id.startswith(name):
                                names[name].add((region[1], region[2]))
            assert all(value <= limits[kind] for kind, value in maxima.items())
            resources[label][fn.name] = maxima
        pv_regions[label] = {name: sorted(value) for name, value in names.items()}
        assert names["pv_first_right"] | names["pv_next_right"] == {(0, 32768)}
    windows = pv_regions["candidate"]["pv_first_slot"] + pv_regions["candidate"]["pv_slot_acc"]
    assert sorted(set(tuple(x) for x in windows)) == [(0, 32768), (32768, 32768), (65536, 32768), (98304, 32768)]
    assert pv_regions["candidate"]["pv_acc_pool"] == [(0, 131072)]
    sync = {}
    for label, root in (("retained", base), ("candidate", kernel)):
        cpp = (root / "ptoas/dynamic_csa_full.cpp").read_text()
        sync[label] = dict(collections.Counter(re.findall(r"pipe_barrier\((PIPE_\w+)\)", cpp)))
    from pypto._kernel_abi import KernelABI
    from pypto.jit._artifact_manifest import BuildKind
    from pypto.runtime._prebuilt import prepare_prebuilt

    prepare_prebuilt(kernel, "a2a3", BuildKind.SINGLE_CHIP, kernel_abi=KernelABI.from_record(abi[1]))
    report = dict(
        candidate="attention_pv_pool_v1",
        case="C04",
        full_codegen_and_binary_compile_passed=True,
        root_abi_unchanged=True,
        aiv_and_wrapper_exact=True,
        orchestration_exact=True,
        original_PV_math_and_K_order=True,
        static_PV_store_count_before=4,
        static_PV_store_count_after=1,
        resources=resources,
        pv_regions=pv_regions,
        pipe_barrier_counts=sync,
        actual_PV_Right_buffers_before_and_after=1,
        initial_assumption_corrected="The retained PV has two Acc buffers, but one 32KiB Right buffer",
        performance_eligible=False,
        device_execution=False,
    )
    with (here / "attention_pv_pool_v1_codegen.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
