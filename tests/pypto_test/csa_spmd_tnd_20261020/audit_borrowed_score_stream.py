# SPDX-License-Identifier: Apache-2.0
"""Audit current-toolchain WS pool resources and the untouched B1 branch."""

import ast
import hashlib
import json
import re
from pathlib import Path


def normalized_cpp(path):
    return "\n".join(line for line in path.read_text().splitlines()
                     if not line.lstrip().startswith("// pto:"))


def main():
    here = Path(__file__).resolve().parent
    old = here / "runs/output_balance_reuse_v73/b1_candidate/kernel"
    new = here / "runs/borrowed_score_stream_b1_cpu_v87/kernel"
    cpp = [root / "ptoas/dynamic_csa_full.cpp" for root in (old, new)]
    host = [root / "orchestration/decode_csa_spmd.cpp" for root in (old, new)]
    checks = {
        "disabled_b1_cpp_equal": normalized_cpp(cpp[0]) == normalized_cpp(cpp[1]),
        "disabled_b1_host_equal": host[0].read_bytes() == host[1].read_bytes(),
    }
    root = here / "runs/borrowed_score_stream_b16_cpu_v87/kernel"
    allocation_passes = list((root / "passes_dump").glob("*after_AllocateMemoryAddr.py"))
    assert len(allocation_passes) == 1
    tree = ast.parse(allocation_passes[0].read_text())
    cube = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                and node.name == "cube_schedule")
    allocations = []
    for node in ast.walk(cube):
        if not (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                and node.target.id.startswith(("split_qk_acc0_", "split_qk_acc1_", "split_ws_pool_"))
                and node.target.id.endswith("__ssa_v0")):
            continue
        refs = [item for item in ast.walk(node.annotation) if isinstance(item, ast.Call)
                and ast.unparse(item.func) == "pl.MemRef"]
        assert len(refs) == 1
        ref = refs[0]
        values = [ast.literal_eval(item.args[0] if isinstance(item, ast.Call) else item)
                  for item in ref.args[1:3]]
        allocations.append({"name": node.target.id, "root": ast.unparse(ref.args[0]),
                            "offset": values[0], "size": values[1]})
    assert len(allocations) == 3
    assert sorted(row["size"] for row in allocations) == [32768, 49152, 49152]
    assert len({row["root"] for row in allocations}) == 1
    assert max(row["offset"] + row["size"] for row in allocations) <= 131072
    assert all(a["offset"] + a["size"] <= b["offset"] or b["offset"] + b["size"] <= a["offset"]
               for index, a in enumerate(allocations) for b in allocations[index + 1:])
    checks["active_three_acc_allocations_fit_without_overlap"] = True
    launch = (root / "orchestration/decode_csa_spmd.cpp").read_text()
    checks["single_mix24"] = (launch.count("rt_submit_task(") == 1
                              and re.findall(r"set_block_num\((\d+)\)", launch) == ["24"])
    report = {
        "passed": all(checks.values()), "checks": checks, "acc_allocations": allocations,
        "disabled_proof_files": [str(path) for path in cpp + host],
        "disabled_proof_sha256": [hashlib.sha256(path.read_bytes()).hexdigest() for path in cpp + host],
        "device_execution": False,
        "scope": "Native toolchain CPU resource/code audit; actual TND correctness and full L1 still required",
    }
    (here / "borrowed_score_stream_codegen_audit_v87.json").write_text(json.dumps(report, indent=2) + "\n")
    assert report["passed"], checks
    print(json.dumps({"passed": True, "checks": checks, "acc_allocations": allocations}, indent=2))


if __name__ == "__main__":
    main()
