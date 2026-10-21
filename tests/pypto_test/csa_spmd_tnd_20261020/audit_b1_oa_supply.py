# SPDX-License-Identifier: Apache-2.0
"""Inspect actual CPU-generated O_A resources and the prefetch dependency window."""

import argparse
import hashlib
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    pto = args.directory / "kernel/ptoas/dynamic_csa_full.pto"
    cpp = pto.with_suffix(".cpp")
    allocations = {}
    operations = []
    pattern = re.compile(
        r"%(oa_\w+?)_inline\w+ = pto.alloc_tile addr = %c(\d+)_i64 .*?"
        r"<loc=(\w+), dtype=(\w+), rows=(\d+), cols=(\d+)"
    )
    for line in pto.read_text().splitlines():
        if "oa_small_supply.py" not in line:
            continue
        if match := pattern.search(line):
            name, address, space, dtype, rows, columns = match.groups()
            allocations[name] = {
                "space": space, "address": int(address), "dtype": dtype,
                "rows": int(rows), "columns": int(columns),
                "bytes": int(rows) * int(columns) * {"bf16": 2, "f32": 4}[dtype],
            }
        if "pto.tmatmul" in line:
            operations.append(line.split(" loc(")[0].strip())
        assert "pto.tmov" not in line, "Unexpected local copy in the supply helper"
    for half in ("first", "second"):
        assert [allocations[f"oa_r_{half}_{k}"]["address"] for k in range(4)] == [0, 32768, 0, 32768]
        assert [allocations[f"oa_l_{half}_{k}"]["address"] for k in range(4)] == [0, 4096, 0, 4096]
    assert len(operations) == 9  # one exclusive init/acc branch plus seven accs
    assert sum("pto.tmatmul ins(" in op for op in operations) == 1
    seed = allocations["oa_seed"]
    assert seed["rows"] == 16 and seed["columns"] == 128 and seed["bytes"] == 8192
    roots = [allocations[f"oa_{kind}_{half}"] for half in ("first", "second") for kind in ("w", "x")]
    intervals = sorted((row["address"], row["address"] + row["bytes"]) for row in roots)
    assert all(left[1] <= right[0] for left, right in zip(intervals, intervals[1:]))
    assert sum(row["bytes"] for row in roots) == 288 * 1024
    assert intervals[-1][1] <= 512 * 1024
    text = cpp.read_text()
    start = text.index("// pto: %oa_seed_")
    end = text.index("TSTORE(", start)
    region = text[start:end]
    first_mma = region.index("TMATMUL(")
    before = region[:first_mma]
    assert before.count("TLOAD(") == 4
    acquired = re.findall(r"wait_flag\(PIPE_MTE2, PIPE_MTE1, EVENT_ID(\d+)\)", before)
    assert acquired == ["0", "1"], acquired
    assert "pipe_barrier(PIPE_ALL)" not in region
    assert "set_flag(PIPE_M, PIPE_MTE1" in region and "wait_flag(PIPE_M, PIPE_MTE1" in region
    result = {
        "passed": True, "device_execution": False,
        "scope": "Actual resources, source dependency window and K order; not measured overlap or speed",
        "pto_sha256": hashlib.sha256(pto.read_bytes()).hexdigest(),
        "cpp_sha256": hashlib.sha256(cpp.read_bytes()).hexdigest(),
        "allocations": allocations,
        "mat_payload_bytes": sum(row["bytes"] for row in roots),
        "mat_highest_address_end": intervals[-1][1],
        "k_chain": "4 iterations * (4+4) increasing K128; only the first product initializes",
        "prefetch": "Both L1 windows issue before first MMA; first-window reads wait only MTE2 events0/1",
        "l0": "Alternating Right0/32768 and Left0/4096; M->MTE1 reuse events retained",
        "tail": "M16 valid rows propagate from tokens; device test adds unwritten-O_A-row sentinel checks",
    }
    path = args.directory / "resource_audit.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("passed", "mat_payload_bytes", "mat_highest_address_end")}))


if __name__ == "__main__":
    main()
