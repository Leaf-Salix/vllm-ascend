#!/usr/bin/env python3
"""只读审查完整CPU产物的Score分配和最初两个panel的实际发射次序。

PTO文件按AIC函数隔离，防止把AIV的同名常量混入。声明地址范围不代表
运行时占用时长；C++语句次序不等于设备引擎已经测得重叠。
"""

import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main():
    directory = HERE / "cpu_long_v1"
    result = json.loads((directory / "result.json").read_text())
    assert result["passed"] and result["single_mix"]
    assert len(result["symbolic_dimensions"]) == 58
    pto = (directory / "kernel/ptoas/dynamic_csa_full.pto").read_text().split("func.func @")[1]
    rows = []
    for line in pto.splitlines():
        if "pto.alloc_tile addr" not in line or not any(name in line for name in (
            "%buf_current_key_", "%buf_prefetched_key_", "%buf_previous_pair_", "%buf_key_l1_pool_"
        )):
            continue
        address = re.search(r"addr = %c(\d+)_i64", line)
        shape = re.search(r"loc=(\w+), dtype=(\w+), rows=(\d+), cols=(\d+)", line)
        assert address and shape, line
        memory, dtype, nrows, ncols = shape.groups()
        byte_count = int(nrows) * int(ncols) * (1 if dtype == "i8" else 2)
        rows.append(dict(name=line.strip().split(" = ")[0], memory=memory, address=int(address[1]), bytes=byte_count))
    assert rows
    key = {r["address"] for r in rows if r["memory"] == "right" and r["bytes"] == 8192}
    ws = {r["address"] for r in rows if r["memory"] == "right" and r["bytes"] == 49152}
    l1 = {(r["address"], r["bytes"]) for r in rows if r["memory"] == "mat"}
    assert key == {0, 49152, 57344} and ws == {0} and l1 == {(163840, 16384)}
    assert all(r["address"] + r["bytes"] <= 65536 for r in rows if r["memory"] == "right")
    qk = sum("pto.tmatmul ins(%buf_query_left_" in line for line in pto.splitlines())
    weighted_sum = sum("pto.tmatmul ins(%buf_coefficient_pair_" in line for line in pto.splitlines())
    assert qk == weighted_sum == 16, (qk, weighted_sum)
    cpp_path = directory / "kernel/ptoas/dynamic_csa_full.cpp"
    cpp = cpp_path.read_text()
    start = cpp.index("// pto: %buf_current_key_")
    end = cpp.index("// pto: %buf_reduced_pair_", start)
    first_line = cpp[:start].count("\n")
    events = []
    for lineno, line in enumerate(cpp[start:end].splitlines(), first_line + 1):
        if any(operation in line for operation in ("TLOAD(", "TEXTRACT(", "TMATMUL(", "set_flag(", "wait_flag(")):
            events.append(dict(line=lineno, source=line.strip()))
    report = dict(
        symbolic_dimensions=58,
        single_mix=True,
        qk_matmul_sites=qk,
        ws_matmul_sites=weighted_sum,
        allocations=rows,
        first_two_panels_cpp_events=events,
        cpp_source=str(cpp_path.resolve()),
        interpretation="Key2在QK0之前TLOAD、之后TEXTRACT；稳定态WS48KiB与两份8KiB Key分离。初始Key的0地址是生命周期复用，不是三份Key并发。",
        limitation="只证明生成物与容量；不证明设备精度、实际重叠时间或完整CSA性能。",
    )
    (HERE / "CODEGEN_AUDIT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("Score audit: 16 QK, 16 WS; Right<=64KiB, Key L1=16KiB; first two panel events saved")


if __name__ == "__main__":
    main()
