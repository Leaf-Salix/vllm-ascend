# SPDX-License-Identifier: Apache-2.0
"""Record why the frozen C05 prefetch candidate fails its overlap gate."""

import ast
import hashlib
import json
import runpy
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    root = here / "runs/compile_attention_prefetch_C05_v1"
    kernel = root / "kernel"
    reference = here / "runs/compile_merged_schedule_C05_v1/kernel"
    result = json.loads((root / "result.json").read_text())
    assert result["full_codegen_passed"] and not result["device_execution"]
    actual = json.loads((kernel / "compiled_meta.json").read_text())["kernel_abi"]
    expected = json.loads((reference / "compiled_meta.json").read_text())["kernel_abi"]
    assert actual == expected

    memory = runpy.run_path(str(here.parent / "csa_spmd_tnd_20261020/audit_score_wide_k128.py"))["memory"]
    (allocation,) = (kernel / "passes_dump").glob("*after_AllocateMemoryAddr.py")
    limits = dict(vec=188416, mat=524288, acc=131072, left=65536, right=65536)
    resources = {}
    held_regions = {}
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
            for prefix in ("dma_kv_state_inline1201", "pf_scores_inline1203", "pf_bias_inline1209"):
                if isinstance(node.target, ast.Name) and node.target.id.startswith(prefix):
                    held_regions[prefix] = [region[1], region[1] + region[2]]
        assert all(value <= limits[kind] for kind, value in maxima.items())
        resources[fn.name] = maxima
    assert set(resources) == {"cube_schedule", "vector_schedule"}
    assert len(held_regions) == 3
    dma = held_regions["dma_kv_state_inline1201"]
    for name, (begin, end) in held_regions.items():
        if name != "dma_kv_state_inline1201":
            assert end <= dma[0] or begin >= dma[1]

    cpp = kernel / "ptoas/dynamic_csa_full.cpp"
    lines = cpp.read_text().splitlines()
    # These exact line anchors identify this immutable compilation, not a
    # general synchronization proof or a matcher for future candidates.
    anchors = {
        79154: "TLOAD(v10716, v10725);",
        79176: "TLOAD(v10726, v10736);",
        79352: "TLOAD(v10770, v10778);",
        79367: "set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);",
        79369: "wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);",
        79373: "if (v10781) {",
        79437: "TSTORE(v10796, v1493);",
        79438: "set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);",
        79445: "wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);",
        80566: "TMULS(v11130, v11126, v465);",
    }
    assert all(lines[number - 1].strip() == text for number, text in anchors.items())
    assert "MGATHER<" not in cpp.read_text()
    report = dict(
        candidate="attention_prefetch_v1",
        case_id="C05",
        full_codegen_passed=True,
        root_and_return_abi_unchanged=True,
        resources=resources,
        limits=limits,
        disjoint_held_regions=held_regions,
        score_and_bias_loads_precede_row_dma=True,
        explicit_physical_row_dma_emitted=True,
        intended_kv_dma_vector_overlap_gate_passed=False,
        reason=[
            "Each 32-row DMA chunk unconditionally sets and waits MTE2->V before testing whether zero fill is needed",
            "The half-loop exit waits MTE3->V after the GM ring store, before the first softmax operation",
            "Earlier Score loads and disjoint buffers do not remove these control-flow completion waits",
        ],
        evidence_cpp=str(cpp),
        cpp_sha256=hashlib.sha256(cpp.read_bytes()).hexdigest(),
        line_anchors=anchors,
        excerpts=[
            {"first_line": first, "text": "\n".join(lines[first - 1 : last])}
            for first, last in ((79350, 79377), (79432, 79449), (80545, 80573))
        ],
        decision="Reject this frozen candidate at its declared compile-time mechanism gate; do not submit it to NPU",
        adopted=False,
        device_execution=False,
        performance_eligible=False,
        limitations=[
            "No device precision, liveness or performance test was run for this candidate",
            "This result rejects this emitted schedule, not all possible Attention overlap implementations",
            "It does not establish that dependencies can safely be deleted; "
            "any replacement needs its own alias and reuse proof",
        ],
    )
    with (here / "attention_prefetch_v1_codegen.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"resources": resources, "abi_equal": True, "overlap_gate_passed": False}))


if __name__ == "__main__":
    main()
