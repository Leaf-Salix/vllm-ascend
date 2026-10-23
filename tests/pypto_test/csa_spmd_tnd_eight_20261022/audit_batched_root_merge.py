# SPDX-License-Identifier: Apache-2.0
"""Audit the qualified format1 root merge in the complete C04 kernel."""

import ast
import collections
import hashlib
import json
import re
import runpy
from pathlib import Path

from analyze_priority_phase import executable
from audit_score_half_interleave import operations


def function_body(source, name):
    text = executable(source)
    start = text.index(f"static __aicore__ void {name}(")
    opening = text.index("{", start)
    depth = 1
    for index in range(opening + 1, len(text)):
        depth += (text[index] == "{") - (text[index] == "}")
        if depth == 0:
            return text[start : index + 1]
    raise AssertionError(f"Unterminated generated function: {name}")


def main():
    here = Path(__file__).resolve().parent
    base = here / "runs/compile_output_rectangle_production_C04_v1/kernel"
    build = here / "runs/compile_batched_root_natural_C04_v1"
    kernel = build / "kernel"
    assert json.loads((build / "result.json").read_text())["full_codegen_passed"]
    probe = json.loads((here / "runs/batched_root_probe_device_v1/result.json").read_text())
    assert probe["passed"] and probe["execution_checks_passed"] and len(probe["cases"]) == 48
    assert "natural" in probe["qualified_variants"]
    before, after = [(p / "ptoas/dynamic_csa_full.pto").read_text() for p in (base, kernel)]
    old_ops, new_ops = operations(before), operations(after)
    assert new_ops["pto.tmrgsort"] == old_ops["pto.tmrgsort"] - 2
    original_roots = [line for line in before.splitlines() if "pto.tmrgsort ins(%batch_sort_new_root" in line]
    assert len(original_roots) == 3 and all("exhausted = false" in line for line in original_roots)
    assert "pto.tmrgsort ins(%batch_sort_new_root" not in after
    for operation in ("pto.tmatmul", "pto.tmatmul.acc", "pto.tstore", "pto.comm.tnotify", "pto.comm.twait"):
        assert old_ops[operation] == new_ops[operation], operation
    root_ops = [line.split(" loc(")[0] for line in after.splitlines() if "pto.tmrgsort ins(%batch_root_flat" in line]
    assert len(root_ops) == 1 and "cols=6144" in root_ops[0] and "%c512_i32" in root_ops[0]
    abi = [json.loads((p / "compiled_meta.json").read_text())["kernel_abi"] for p in (base, kernel)]
    assert abi[0] == abi[1]
    orchestration = "orchestration/decode_csa_spmd.cpp"
    assert executable((base / orchestration).read_text()) == executable((kernel / orchestration).read_text())
    cube = [function_body((p / "kernels/aic/cube_schedule.cpp").read_text(), "cube_schedule") for p in (base, kernel)]
    assert cube[0] == cube[1]
    memory = runpy.run_path(str(here.parent / "csa_spmd_tnd_20261020/audit_score_wide_k128.py"))["memory"]
    limits = dict(vec=188416, mat=524288, acc=131072, left=65536, right=65536)
    resources, sync = {}, {}
    for label, root in (("retained", base), ("candidate", kernel)):
        (allocation,) = (root / "passes_dump").glob("*after_AllocateMemoryAddr.py")
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
            assert all(value <= limits[kind] for kind, value in maxima.items())
            resources[label][fn.name] = maxima
        cpp = (root / "ptoas/dynamic_csa_full.cpp").read_text()
        sync[label] = dict(collections.Counter(re.findall(r"pipe_barrier\((PIPE_\w+)\)", cpp)))
    assert resources["retained"] == resources["candidate"]
    from pypto._kernel_abi import KernelABI
    from pypto.jit._artifact_manifest import BuildKind
    from pypto.runtime._prebuilt import prepare_prebuilt

    prepare_prebuilt(kernel, "a2a3", BuildKind.SINGLE_CHIP, kernel_abi=KernelABI.from_record(abi[1]))
    report = dict(
        candidate="batched_root_natural_v1",
        case="C04",
        full_codegen_and_binary_compile_passed=True,
        root_abi_unchanged=True,
        cube_generated_function_exact=True,
        orchestration_exact=True,
        natural_probe_full_bitwise_cases=48,
        root_format2_calls_before=3,
        root_format1_calls_after=1,
        root_format1_block_words=512,
        root_format1_repeats=3,
        root_pto=root_ops[0].strip(),
        operation_counts={"retained": dict(old_ops), "candidate": dict(new_ops)},
        resources=resources,
        pipe_barrier_counts=sync,
        cpp_sha256={
            str(p.relative_to(kernel)): hashlib.sha256(p.read_bytes()).hexdigest() for p in kernel.rglob("*.cpp")
        },
        performance_eligible=False,
        device_execution=False,
    )
    with (here / "batched_root_natural_v1_codegen.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("operation_counts", "root_pto")}, indent=2))


if __name__ == "__main__":
    main()
