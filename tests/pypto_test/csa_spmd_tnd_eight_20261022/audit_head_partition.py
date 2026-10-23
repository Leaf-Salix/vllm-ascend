# SPDX-License-Identifier: Apache-2.0
"""Check actual C01 head-partition allocations, launch ABI and event sites."""

import ast
import json
import re
import runpy
from pathlib import Path

from analyze_priority_phase import executable


def main():
    here = Path(__file__).resolve().parent
    build = here / "runs/compile_head_partition2_C01_v1"
    kernel = build / "kernel"
    reference = here / "runs/compile_physical_capacity_tier_C01_v1/kernel"
    assert json.loads((build / "result.json").read_text())["full_codegen_passed"]
    abi = lambda path: json.loads((path / "compiled_meta.json").read_text())["kernel_abi"]
    assert abi(kernel) == abi(reference)
    orchestration = "orchestration/decode_csa_spmd.cpp"
    assert executable((kernel / orchestration).read_text()) == executable((reference / orchestration).read_text())
    memory = runpy.run_path(str(here.parent / "csa_spmd_tnd_20261020/audit_score_wide_k128.py"))["memory"]
    (allocation,) = (kernel / "passes_dump").glob("*after_AllocateMemoryAddr.py")
    limits = dict(vec=188416, mat=524288, acc=131072, left=65536, right=65536)
    resources, query_tiles, vector_state = {}, [], []
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
            if not isinstance(node.target, ast.Name):
                continue
            annotation = ast.unparse(node.annotation)
            if node.target.id.startswith("qk_q_inline"):
                assert "[[32,512],pl.BF16," in annotation.replace(" ", "") and region[2] == 32768
                query_tiles.append(dict(variable=node.target.id, annotation=annotation, region=region))
            if node.target.id.startswith("running_left_inline"):
                assert "[[16,256],pl.FP32," in annotation.replace(" ", "") and region[2] == 16384
                vector_state.append(dict(variable=node.target.id, annotation=annotation, region=region))
        assert all(value <= limits[kind] for kind, value in maxima.items())
        resources[fn.name] = maxima
    assert set(resources) == {"cube_schedule", "vector_schedule"}
    assert query_tiles and vector_state
    pattern = r"pto\.sync\.(set|wait) <(PIPE_\w+)>, %c(\d+)_index(?: \{ffts_mode = (\d+) : i32\})?"
    new_ir = (kernel / "ptoas/dynamic_csa_full.pto").read_text()
    old_ir = (reference / "ptoas/dynamic_csa_full.pto").read_text()
    assert sorted(re.findall(pattern, new_ir)) == sorted(re.findall(pattern, old_ir))
    report = dict(
        candidate="head_partition2_v1",
        case_id="C01",
        full_codegen_passed=True,
        root_and_return_abi_unchanged=True,
        orchestration_unchanged=True,
        event_sites_unchanged=True,
        resources=resources,
        limits=limits,
        actual_query_tiles=query_tiles,
        actual_vector_state=vector_state,
        cpu_contract_tests=15,
        actual_source_event_masks_checked=6 * 24 * 32,
        active_attention_clusters=dict(reference=6, candidate=12),
        logical_kv_rows_all_valid=dict(reference=6 * 5 * 128, candidate=2 * 6 * 5 * 128),
        external_head_and_ring_pitch=64,
        device_execution=False,
        performance_eligible=False,
        limitation="Ownership, math-expression and event checks do not prove device exactness or speedup; "
        "duplicated KV interfaces are not measured HBM bytes",
    )
    with (here / "head_partition2_v1_C01_codegen.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(resources))


if __name__ == "__main__":
    main()
