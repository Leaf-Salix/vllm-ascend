# SPDX-License-Identifier: Apache-2.0
"""16-head循环状态候选的实际地址/指令检查；不替代设备精度和计时。"""

import argparse
import json
import re
from pathlib import Path

from qk_phase_clock import close_brace, phase_regions


def audit(directory):
    directory = Path(directory)
    ir = (directory / "csa_static_group.pto").read_text().split("func.func @csa_static_aiv", 1)[1]
    ir = ir.split("func.func @", 1)[0]
    constants = {n: int(v) for n, v in re.findall(r"(%\w+) = arith.constant (-?\d+) : i64", ir)}
    tiles = {}
    for line in ir.splitlines():
        m = re.search(r"(%stage_30_\w+) = pto.alloc_tile addr = (%\w+).*"
                      r"!pto.tile_buf<loc=(\w+), dtype=(\w+), rows=(\d+), cols=(\d+)", line)
        if m:
            name, addr, memory, dtype, rows, cols = m.groups()
            if addr not in constants:
                continue
            tiles[name] = {"address": constants[addr], "memory": memory,
                           "dtype": dtype, "shape": [int(rows), int(cols)]}

    def tile(index):
        name = f"%stage_30_v{index}__ssa_v0"
        value = tiles[name]
        if (value["memory"], value["dtype"], value["shape"]) != ("vec", "f32", [16, 256]):
            raise ValueError(f"PV状态规格改变：{name}")
        return name, value

    states = []
    for initial, merged, reset in zip((45, 46, 47, 48), (144, 148, 152, 156), (272, 273, 274, 275)):
        entries = [tile(i) for i in (initial, merged, reset)]
        if len({v["address"] for _, v in entries}) != 1:
            raise ValueError("PV初始化/合并/重置没有复用同一状态槽")
        original, output = entries[0][0], entries[1][0]
        if len(re.findall(r"pto.trowexpandmul ins\(" + re.escape(original) + r",", ir)) != 1:
            raise ValueError("PV旧状态不是唯一一次参与alpha乘法")
        if len([line for line in ir.splitlines() if "pto.tadd ins(" in line
                and f"outs({output} :" in line]) != 1:
            raise ValueError("PV合并结果没有唯一写回原槽")
        states.append({"initial": entries[0][0], "merged": entries[1][0], "reset": entries[2][0],
                       "address": entries[0][1]["address"], "bytes": 16384})
    intervals = sorted((v["address"], v["address"] + v["bytes"]) for v in states)
    if not all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:])):
        raise ValueError("四个PV状态区间重叠")
    if any(v["memory"] == "vec" and v["dtype"] == "f32" and v["shape"] == [32, 256]
           for v in tiles.values()):
        raise ValueError("仍存在旧32-head整状态")

    cpp = (directory / "csa_static_group.cpp").read_text()
    start = re.search(r"^AICORE void csa_static_aiv\([^\n]+\) \{", cpp, re.M)
    body = cpp[start.end():close_brace(cpp, start.end() - 1)]
    regions = phase_regions(body)
    left, right, _ = regions["pv_merge"]
    merge = body[left:right + 1]
    counts = {name: merge.count(token) for name, token in {
        "row_expand_mul": "TROWEXPANDMUL(", "add": "TADD(",
        "full_pipe_barrier": "pipe_barrier(PIPE_ALL)",
        "vector_pipe_barrier": "pipe_barrier(PIPE_V)",
        "tile_insert": "TINSERT", "gm_tile_load": "TLOAD(",
    }.items()}
    if counts["row_expand_mul"] != 8 or counts["tile_insert"]:
        raise ValueError(f"PV分块合并指令不符：{counts}")
    return {"artifact": str(directory.resolve()), "revision": "v6_split_loop_state_16",
            "states": states, "state_intervals_disjoint": True,
            "initial_update_reset_addresses_identical": True,
            "old_full_state_removed": True, "counts": counts,
            "scope": "实际布局/指令检查，不是设备正确性或性能证据"}, merge


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result, merge = audit(args.artifact)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    args.output.with_suffix(".merge.cpp.txt").write_text(merge + "\n")
    print(json.dumps(result, ensure_ascii=False))
