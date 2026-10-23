# SPDX-License-Identifier: Apache-2.0
"""Gate actual resources, ABI and first-half DMA overlap before NPU testing."""

import argparse
import ast
import hashlib
import json
import re
import runpy
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("C04", "C05"), required=True)
    parser.add_argument(
        "--candidate", choices=("attention_overlap_v1", "attention_valid_overlap_v1"), default="attention_overlap_v1"
    )
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    stem = args.candidate.removesuffix("_v1")
    build = here / f"runs/compile_{stem}_{args.case}_v1"
    kernel = build / "kernel"
    reference = here / f"runs/compile_merged_schedule_{args.case}_v1/kernel"
    result = json.loads((build / "result.json").read_text())
    assert result["full_codegen_passed"] and not result["device_execution"]
    abi = lambda path: json.loads((path / "compiled_meta.json").read_text())["kernel_abi"]
    assert abi(kernel) == abi(reference)
    memory = runpy.run_path(str(here.parent / "csa_spmd_tnd_20261020/audit_score_wide_k128.py"))["memory"]
    (allocation,) = (kernel / "passes_dump").glob("*after_AllocateMemoryAddr.py")
    limits = dict(vec=188416, mat=524288, acc=131072, left=65536, right=65536)
    resources, held = {}, {}
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
            if isinstance(node.target, ast.Name) and node.target.id.startswith(
                ("dma_kv_state_inline", "qk_scores_half_inline", "qk_probability_inline", "physical_rows_inline")
            ):
                held[node.target.id] = [region[1], region[1] + region[2]]
        assert all(value <= limits[kind] for kind, value in maxima.items())
        resources[fn.name] = maxima
    assert set(resources) == {"cube_schedule", "vector_schedule"}
    dma_regions = [region for name, region in held.items() if name.startswith("dma_kv_state_inline")]
    assert len(dma_regions) == 1
    dma_start, dma_end = dma_regions[0]
    for name, (start, end) in held.items():
        if not name.startswith("dma_kv_state_inline"):
            assert end <= dma_start or start >= dma_end

    cpp = kernel / "ptoas/dynamic_csa_full.cpp"
    lines = cpp.read_text().splitlines()
    index_load = next(i for i, line in enumerate(lines) if "// pto: %physical_rows_inline" in line)
    vector_first = next(i for i in range(index_load, len(lines)) if "TMULS(" in lines[i])
    vector_last = next(i for i in range(vector_first, len(lines)) if "TCVT(" in lines[i])
    if args.candidate == "attention_overlap_v1":
        dma_first = next(i for i in range(vector_last, len(lines)) if "TLOAD(" in lines[i])
        intervening = "\n".join(lines[vector_first : dma_first + 1])
        assert "pipe_barrier(PIPE_ALL)" not in intervening
        assert not re.search(r"wait_flag\(PIPE_V, PIPE_(?:MTE2|S),", intervening)
        assert "wait_flag(PIPE_MTE2, PIPE_S," in "\n".join(lines[vector_last:dma_first])
        order = "Independent softmax first, then first-half row DMA"
    else:
        loads = [i for i in range(index_load, vector_first) if "TLOAD(" in lines[i]]
        assert len(loads) == 4, "Expected indices, Score, bias, then looped row DMA"
        dma_first = loads[-1]
        intervening = "\n".join(lines[dma_first : vector_first + 1])
        assert "pipe_barrier(PIPE_ALL)" not in intervening
        assert "wait_flag(PIPE_MTE3, PIPE_V," not in intervening
        assert "set_flag(PIPE_MTE2, PIPE_V," not in intervening
        assert "set_flag(PIPE_MTE3, PIPE_MTE2," in intervening
        assert "wait_flag(PIPE_MTE3, PIPE_MTE2," in intervening
        assert "ffts_cross_core_sync(PIPE_MTE3," in intervening
        assert "wait_flag(PIPE_MTE2, PIPE_S," in "\n".join(lines[index_load:dma_first])
        order = "Pure DMA and size-dependent KV notification first; independent softmax later"
    assert "main_cache" in "\n".join(lines[dma_first - 18 : dma_first])
    assert all(
        op in "\n".join(lines[vector_first : vector_last + 1]) for op in ("TROWMAX(", "TROWSUM(", "TEXP(", "TCVT(")
    )
    assert "set_flag(PIPE_MTE2, PIPE_S," in "\n".join(lines[index_load:vector_first])
    gate = dict(
        candidate=args.candidate,
        case_id=args.case,
        full_codegen_passed=True,
        root_and_return_abi_unchanged=True,
        resources=resources,
        held_regions=held,
        intended_overlap_path_emitted=True,
        operation_order=order,
        scalar_index_readiness_preserved=True,
        first_vector_line=vector_first + 1,
        last_vector_line=vector_last + 1,
        first_row_dma_line=dma_first + 1,
        cpp=str(cpp),
        cpp_sha256=hashlib.sha256(cpp.read_bytes()).hexdigest(),
        excerpt="\n".join(lines[min(vector_first, dma_first) : max(vector_last, dma_first) + 1]),
        isa_review={
            "pin": "327cd5869f3a7c4d2c6a1b945b2aed06e7665c5d",
            "float_reductions": "TRowMax/TRowSum select TRowReduceOps for FP32; "
            "the V-to-S scalar reduction helpers are integer-only",
            "selected_vector_headers": "TRowReduceOps, TUnaryOp, TRowExpandBinOp, TColExpandBinOp and TCvt "
            "contain no set/wait cross-pipe completion fences",
        },
        device_execution=False,
        performance_eligible=False,
        limitations=[
            "This establishes an emitted overlap path, not measured overlap duration or whole-chain speedup",
            "The second 32-row chunk still waits for the first ring store before reusing its buffer",
            "The source-event DAG does not validate hardware queue capacity, DMA completion or cache behavior",
        ],
    )
    with (here / f"{args.candidate}_{args.case}_codegen.json").open("x") as stream:
        stream.write(json.dumps(gate, indent=2) + "\n")
    print(json.dumps({"case": args.case, "resources": resources, "overlap_path": True, "order": order}))


if __name__ == "__main__":
    main()
