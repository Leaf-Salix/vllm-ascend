# SPDX-License-Identifier: Apache-2.0
"""Check pruning resources, the Cube, and three unchanged resource domains."""

import ast
import hashlib
import json
import re
from pathlib import Path

from audit_borrowed_score_stream import normalized_cpp
from audit_short_half_ub import integer


def main():
    here = Path(__file__).resolve().parent
    cohort = json.loads((here / "cohort_output_balance_v4.json").read_text())
    jobs = {row["task_id"]: row for row in json.loads((here / "jobs.json").read_text())}
    checks, proofs = {}, []
    for label, batch, history, active in (
        ("b28", 28, 131072, True),
        ("b4", 4, 131072, False),
        ("b1", 1, 131072, False),
        ("short_b16", 16, 8192, False),
    ):
        task = next(row["task_id"] for row in cohort["cases"] if (row["batch"], row["history"]) == (batch, history))
        old_files = [
            p
            for p in (Path(jobs[task]["output"]) / "build_output").glob("*/ptoas/dynamic_csa_full.cpp")
            if "PTOAS__WAIT_GE_BACKOFF<150>" in p.read_text()
        ]
        assert len(old_files) == 1
        old = old_files[0]
        kernel = here / f"runs/topk_prefix_{label}_cpu_v97/kernel"
        new = kernel / "ptoas/dynamic_csa_full.cpp"
        host = [root / "orchestration/decode_csa_spmd.cpp" for root in (old.parents[1], kernel)]
        if active:
            pattern = r"AICORE void cube_schedule\([\s\S]*?^\}"
            checks[label + "_cube_cpp_equal"] = (
                re.search(pattern, normalized_cpp(old), re.MULTILINE).group()
                == re.search(pattern, normalized_cpp(new), re.MULTILINE).group()
            )
        else:
            checks[label + "_whole_cpp_equal"] = normalized_cpp(old) == normalized_cpp(new)
        checks[label + "_host_byte_equal"] = host[0].read_bytes() == host[1].read_bytes()
        launch = host[1].read_text()
        checks[label + "_one_mix24"] = launch.count("rt_submit_task(") == 1 and re.findall(
            r"set_block_num\((\d+)\)", launch
        ) == ["24"]
        proofs.append(
            {
                "case": label,
                "active": active,
                "files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (old, new, *host)},
            }
        )
    kernel = here / "runs/topk_prefix_b28_cpu_v97/kernel"
    paths = list((kernel / "passes_dump").glob("*after_AllocateMemoryAddr.py"))
    assert len(paths) == 1
    tree = ast.parse(paths[0].read_text())
    vector = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "vector_schedule")
    compare_calls = [
        n for n in ast.walk(vector) if isinstance(n, ast.Call) and ast.unparse(n.func) == "pl.tile.gather_compare"
    ]
    checks["three_scalar_ge_sites"] = len(compare_calls) == 3 and all(
        any(k.arg == "cmp_mode" and integer(k.value) == 5 for k in n.keywords) for n in compare_calls
    )
    allocations, counts, tuple_outputs = [], [], []
    for node in ast.walk(vector):
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        if "pl.Mem.Vec" not in ast.unparse(node.annotation):
            continue
        if isinstance(node.annotation, ast.Subscript) and ast.unparse(node.annotation.value) == "pl.Tuple":
            assert isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "pl.tile.gather_compare"
            outputs = []
            for index, size in ((0, 8192), (1, 32)):
                unpacked = [
                    n
                    for n in ast.walk(vector)
                    if isinstance(n, ast.AnnAssign)
                    and isinstance(n.value, ast.Subscript)
                    and isinstance(n.value.value, ast.Name)
                    and n.value.value.id == node.target.id
                    and integer(n.value.slice) == index
                ]
                assert len(unpacked) == 1
                refs = [
                    n
                    for n in ast.walk(unpacked[0].annotation)
                    if isinstance(n, ast.Call) and ast.unparse(n.func) == "pl.MemRef"
                ]
                assert len(refs) == 1 and integer(refs[0].args[2]) == size
                outputs.append(unpacked[0].target.id)
            tuple_outputs.append({"tuple": node.target.id, "allocated_outputs": outputs})
            continue
        # gather_compare returns a tuple of two Tiles. Inspect each Tile rather
        # than assuming every annotated assignment owns a single MemRef.
        tiles = [
            n
            for n in ast.walk(node.annotation)
            if isinstance(n, ast.Subscript) and ast.unparse(n.value) == "pl.Tile" and "pl.Mem.Vec" in ast.unparse(n)
        ]
        assert tiles
        for tile in tiles:
            refs = [n for n in ast.walk(tile) if isinstance(n, ast.Call) and ast.unparse(n.func) == "pl.MemRef"]
            assert len(refs) == 1
            ref = refs[0]
            row = {"name": node.target.id, "offset": integer(ref.args[1]), "bytes": integer(ref.args[2])}
            allocations.append(row)
            if node.target.id.startswith("count_inline"):
                counts.append(row)
    checks["count_buffers_aligned32_bytes"] = bool(counts) and all(
        row["bytes"] == 32 and row["offset"] % 32 == 0 for row in counts
    )
    checks["all_compare_tuple_outputs_allocated"] = len(tuple_outputs) == len(compare_calls) == 3
    peak = max(row["offset"] + row["bytes"] for row in allocations)
    checks["vector_fits"] = peak <= 188416
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "proofs": proofs,
        "count_buffers": counts,
        "compare_tuple_outputs": tuple_outputs,
        "vector_peak_end_bytes": peak,
        "active_allocation_pass": str(paths[0]),
        "active_allocation_pass_sha256": hashlib.sha256(paths[0].read_bytes()).hexdigest(),
        "device_execution": False,
        "scope": "CPU/code/resource proof; device precision, ties and full L1 still required",
    }
    (here / "topk_prefix_codegen_audit_v97.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "checks": checks, "peak": peak}, indent=2))
    assert report["passed"], checks


if __name__ == "__main__":
    main()
