# SPDX-License-Identifier: Apache-2.0
"""Record why this frozen C04 PV prefetch does not overlap with softmax."""

import ast
import hashlib
import json
import runpy
from pathlib import Path

from analyze_priority_phase import executable
from audit_batched_root_merge import function_body


def main():
    here = Path(__file__).resolve().parent
    build = here / "runs/compile_attention_pv_prefetch_C04_v2"
    kernel = build / "kernel"
    base = here / "runs/compile_output_rectangle_production_C04_v1/kernel"
    assert json.loads((build / "result.json").read_text())["full_codegen_passed"]
    abi = [json.loads((p / "compiled_meta.json").read_text())["kernel_abi"] for p in (base, kernel)]
    assert abi[0] == abi[1]
    orch = "orchestration/decode_csa_spmd.cpp"
    assert executable((base / orch).read_text()) == executable((kernel / orch).read_text())
    cube = [function_body((p / "kernels/aic/cube_schedule.cpp").read_text(), "cube_schedule") for p in (base, kernel)]
    assert cube[0] == cube[1]
    memory = runpy.run_path(str(here.parent / "csa_spmd_tnd_20261020/audit_score_wide_k128.py"))["memory"]
    limits = dict(vec=188416, mat=524288, acc=131072, left=65536, right=65536)
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
    cpp = kernel / "ptoas/dynamic_csa_full.cpp"
    lines = cpp.read_text().splitlines()
    # Exact anchors in this immutable output, not a matcher for other kernels.
    anchors = {
        80652: "TLOAD(v10958, v10967);",
        80856: "pipe_barrier(PIPE_ALL);",
        80857: "TLOAD(v11014, v11023);",
        80878: "pipe_barrier(PIPE_ALL);",
        80879: "TLOAD(v11024, v11033);",
        80886: "wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);",
        80887: "TMULS(v11034, v11014, v463);",
        80903: "TROWMAX(v11038, v11036, v11012);",
    }
    assert all(lines[number - 1].strip() == text for number, text in anchors.items())
    report = dict(
        candidate="attention_pv_prefetch_v2",
        case="C04",
        full_codegen_passed=True,
        cce_binary_compile_attempted=False,
        device_execution=False,
        performance_measured=False,
        root_abi_unchanged=True,
        orchestration_exact=True,
        cube_generated_function_exact=True,
        resources=resources,
        limits=limits,
        cpp=str(cpp),
        cpp_sha256=hashlib.sha256(cpp.read_bytes()).hexdigest(),
        line_anchors=anchors,
        excerpts=[
            {"first_line": first, "text": "\n".join(lines[first - 1 : last])}
            for first, last in ((80645, 80656), (80849, 80860), (80874, 80895))
        ],
        intended_PV_DMA_softmax_overlap=False,
        reason=[
            "On the active softmax path PIPE_ALL precedes the Score load, after the earlier PV load",
            "The bias load also follows PIPE_ALL; the first softmax TMULS is later than both barriers",
            "This emitted schedule drains the prefetch before softmax and fails its stated overlap mechanism",
        ],
        decision="Do not submit this frozen candidate to NPU; no performance result is claimed",
        adopted=False,
        limitations=[
            "Phase-level event checks do not replace asynchronous fence or hardware correctness validation",
            "This rejects the emitted schedule, not every possible PV prefetch design",
            "No fences were removed, no SDK was changed, and no new design was started after the stop request",
        ],
    )
    with (here / "attention_pv_prefetch_v2_codegen.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"resources": resources, "overlap": False, "device_execution": False}, indent=2))


if __name__ == "__main__":
    main()
