# SPDX-License-Identifier: Apache-2.0
"""Audit the C02 prototype's real table allocation, ABI and event sites."""

import ast
import hashlib
import json
import re
import runpy
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    root = here / "runs/compile_physical_capacity_C02_v1"
    kernel = root / "kernel"
    reference = here / "runs/compile_merged_schedule_C02_v1/kernel"
    assert json.loads((root / "result.json").read_text())["full_codegen_passed"]
    abi = lambda path: json.loads((path / "compiled_meta.json").read_text())["kernel_abi"]
    assert abi(kernel) == abi(reference)
    memory = runpy.run_path(str(here.parent / "csa_spmd_tnd_20261020/audit_score_wide_k128.py"))["memory"]
    (allocation,) = (kernel / "passes_dump").glob("*after_AllocateMemoryAddr.py")
    limits = dict(vec=188416, mat=524288, acc=131072, left=65536, right=65536)
    resources, table_tiles = {}, []
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
            if isinstance(node.target, ast.Name) and node.target.id.startswith("p_table_inline"):
                text = ast.unparse(node.annotation)
                assert "[[1,2080],pl.INT32," in text.replace(" ", "") and region[2] == 8320
                table_tiles.append(dict(variable=node.target.id, annotation=text, region=region))
        assert all(value <= limits[kind] for kind, value in maxima.items())
        resources[fn.name] = maxima
    assert set(resources) == {"cube_schedule", "vector_schedule"} and len(table_tiles) == 1
    pattern = r"pto\.sync\.(set|wait) <(PIPE_\w+)>, %c(\d+)_index(?: \{ffts_mode = (\d+) : i32\})?"
    new_ir = (kernel / "ptoas/dynamic_csa_full.pto").read_text()
    old_ir = (reference / "ptoas/dynamic_csa_full.pto").read_text()
    assert sorted(re.findall(pattern, new_ir)) == sorted(re.findall(pattern, old_ir))
    cpp = kernel / "ptoas/dynamic_csa_full.cpp"
    assert "MGATHER<pto::Coalesce::Row, pto::GatherOOB::Zero>" in cpp.read_text()
    report = dict(
        candidate="physical_capacity_2080_v1",
        case_id="C02",
        full_codegen_passed=True,
        root_and_return_abi_unchanged=True,
        event_sites_unchanged=True,
        resources=resources,
        limits=limits,
        actual_page_table_tile=table_tiles,
        cpp=str(cpp),
        cpp_sha256=hashlib.sha256(cpp.read_bytes()).hexdigest(),
        device_execution=False,
        performance_eligible=False,
        limitation="CPU allocation and code identity only; actual path, device exactness and full L1 remain to verify",
    )
    with (here / "physical_capacity_2080_v1_C02_codegen.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"resources": resources, "page_table_bytes": 8320, "abi_equal": True}))


if __name__ == "__main__":
    main()
