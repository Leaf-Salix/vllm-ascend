# SPDX-License-Identifier: Apache-2.0
"""Check the active Q_B resources and unchanged Vector/disabled executables."""

import ast
import hashlib
import json
import re
from pathlib import Path

from audit_borrowed_score_stream import normalized_cpp
from audit_short_half_ub import integer


def main():
    here = Path(__file__).resolve().parent
    checks, proofs = {}, []
    for batch in (1, 8, 16):
        old_root = here / f"runs/borrowed_combined_h131072_b{batch}_cpu_v105/kernel"
        new_root = here / f"runs/qb_wide_b{batch}_cpu_v108b/kernel"
        assert json.loads((new_root.parent / "result.json").read_text())["passed"]
        old, new = [p / "ptoas/dynamic_csa_full.cpp" for p in (old_root, new_root)]
        a, b = normalized_cpp(old), normalized_cpp(new)
        pattern = r"AICORE void vector_schedule\([\s\S]*?^\}"
        checks[f"b{batch}_vector_equal"] = re.search(pattern, a, re.M).group() == re.search(pattern, b, re.M).group()
        host = [p / "orchestration/decode_csa_spmd.cpp" for p in (old_root, new_root)]
        checks[f"b{batch}_host_equal"] = host[0].read_bytes() == host[1].read_bytes()
        if batch == 16:
            checks["disabled_b16_whole_cpp_equal"] = a == b
        else:
            checks[f"b{batch}_candidate_actually_changes_cube"] = a != b
        proofs.append(dict(batch=batch, files={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                              for p in (old, new, *host)}))
    root = here / "runs/qb_wide_b8_cpu_v108b/kernel"
    allocation = next((root / "passes_dump").glob("*after_AllocateMemoryAddr.py"))
    tree = ast.parse(allocation.read_text())
    cube = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "cube_schedule")
    peaks, qb_tiles = {}, []
    capacities = {"Mat": 524288, "Left": 65536, "Right": 65536, "Acc": 131072}
    for node in ast.walk(cube):
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        annotation = ast.unparse(node.annotation)
        memory = next((m for m in capacities if f"pl.Mem.{m}" in annotation), None)
        if memory is None:
            continue
        refs = [n for n in ast.walk(node.annotation) if isinstance(n, ast.Call) and ast.unparse(n.func) == "pl.MemRef"]
        for ref in refs:
            offset, size = integer(ref.args[1]), integer(ref.args[2])
            peaks[memory] = max(peaks.get(memory, 0), offset + size)
            if node.target.id.startswith(("col_acc_inline", "wq_chunk_inline", "qr_chunk_inline", "wq_first_inline")):
                qb_tiles.append(dict(name=node.target.id, memory=memory, offset=offset, size=size,
                                     annotation=annotation))
    checks["whole_cube_resources_fit"] = set(peaks) == set(capacities) and all(
        peaks[m] <= capacities[m] for m in peaks
    )
    checks["m64_n512_int32_accumulator_present"] = any(
        t["memory"] == "Acc" and "[[64, 512], pl.INT32" in t["annotation"] for t in qb_tiles
    )
    right = [t for t in qb_tiles if t["memory"] == "Right" and "[[64, 512], pl.INT8" in t["annotation"]]
    checks["two_32k_right_slots"] = {(t["offset"], t["size"]) for t in right} == {(0, 32768), (32768, 32768)}
    # Cooperative prefix retains physical AIC4..23 and worker%20; the fallback
    # uses all24. Both maps must cover every column block exactly once.
    for workers, physical in ((20, range(4, 24)), (24, range(24))):
        blocks = [block for core in physical for block in range(core % workers, 64, workers)]
        checks[f"unique_n_blocks_{workers}_workers"] = sorted(blocks) == list(range(64))
    checks["int32_dot_cannot_overflow"] = 1024 * 128 * 128 < 2 ** 31
    report = dict(passed=all(checks.values()), checks=checks, proofs=proofs,
                  memory_peak_end_bytes=peaks, qb_allocations=qb_tiles,
                  allocation_pass=str(allocation),
                  allocation_sha256=hashlib.sha256(allocation.read_bytes()).hexdigest(),
                  device_execution=False, performance_measured=False,
                  scope="Full CPU code/resource evidence; device correctness and L1 remain required")
    (here / "qb_wide_codegen_audit_v108b.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "checks": checks, "peaks": peaks}, indent=2))
    assert report["passed"], checks


if __name__ == "__main__":
    main()
