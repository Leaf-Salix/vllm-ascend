# SPDX-License-Identifier: Apache-2.0
"""Gate the query-local candidate on real allocations, ABI and FFTS emission."""

import argparse
import ast
import copy
import hashlib
import json
import re
import runpy
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-dir", type=Path, required=True)
    parser.add_argument("--reference-kernel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    build = json.loads((args.compile_dir / "result.json").read_text())
    assert build["passed"] and build["full_codegen_passed"] and not build["device_execution"]
    kernel = args.compile_dir / "kernel"
    actual = json.loads((kernel / "compiled_meta.json").read_text())["kernel_abi"]
    expected = json.loads((args.reference_kernel / "compiled_meta.json").read_text())["kernel_abi"]
    normalized = copy.deepcopy(actual)
    query = next(p for p in normalized["params"] if p["name"] == "q_heads")
    assert (query["direction"], query["direction_tag"]) == ("InOut", 3)
    query.update(direction="In", direction_tag=1)
    assert normalized == expected, "Unexpected change beyond q_heads access mode"

    memory = runpy.run_path(str(here.parent / "csa_spmd_tnd_20261020/audit_score_wide_k128.py"))["memory"]
    (ir,) = (kernel / "passes_dump").glob("*after_AllocateMemoryAddr.py")
    resources = {}
    limits = dict(vec=188416, mat=524288, acc=131072, left=65536, right=65536)
    for fn in ast.walk(ast.parse(ir.read_text())):
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
    assert set(resources) == {"cube_schedule", "vector_schedule"}

    pattern = r"pto\.sync\.(set|wait) <(PIPE_\w+)>, %c(\d+)_index(?: \{ffts_mode = (\d+) : i32\})?"
    source = (kernel / "ptoas/dynamic_csa_full.pto").read_text()
    baseline = (args.reference_kernel / "ptoas/dynamic_csa_full.pto").read_text()
    before, after = re.findall(pattern, baseline), re.findall(pattern, source)
    new = [event for event in after if event[2] == "6"]
    assert sorted(new) == sorted([("set", "PIPE_MTE3", "6", "2"), ("wait", "PIPE_MTE2", "6", "")])
    assert sorted(event for event in after if event[2] != "6") == sorted(before)
    assert all(0 <= int(event[2]) < 14 for event in after)
    stores = [line for line in source.splitlines() if "pto.tstore" in line and "query_local_stages.py" in line]
    assert len(stores) == 2, stores
    cpp = kernel / "ptoas/dynamic_csa_full.cpp"
    assert cpp.is_file() and "ffts_cross_core_sync" in cpp.read_text()
    report = dict(
        candidate="query_local_v5",
        compile_dir=str(args.compile_dir),
        reference_kernel=str(args.reference_kernel),
        full_codegen_passed=True,
        device_execution=False,
        tensor_parameters=len(actual["params"]),
        packed_slots_shapes_and_return_aliases_unchanged=True,
        declared_access_change={"q_heads": "In -> InOut; AIV writes the existing alias of q_flat"},
        resources=resources,
        added_event_sites=new,
        existing_ffts_sites_unchanged=True,
        query_store_sites=stores,
        cpp_sha256=hashlib.sha256(cpp.read_bytes()).hexdigest(),
        limitations="CPU compilation is not device precision, event-liveness or performance validation",
    )
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "resources": resources, "abi_access_change": report["declared_access_change"]}))


if __name__ == "__main__":
    main()
