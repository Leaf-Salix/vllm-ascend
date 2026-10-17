# SPDX-License-Identifier: Apache-2.0
"""双AIV分工实际状态地址和指令检查，不代替设备精度/同步验证。"""

import argparse
import json
import re
from pathlib import Path

from qk_phase_clock import close_brace


def audit(directory):
    directory = Path(directory)
    ir = (directory / "csa_static_group.pto").read_text().split("func.func @csa_static_aiv", 1)[1]
    ir = ir.split("func.func @", 1)[0]
    constants = {name: int(value) for name, value in re.findall(r"(%\w+) = arith.constant (-?\d+) : i64", ir)}
    tiles = {}
    for line in ir.splitlines():
        match = re.search(r"(%stage_30_\w+) = pto.alloc_tile addr = (%\w+).*"
                          r"!pto.tile_buf<loc=(\w+), dtype=(\w+), rows=(\d+), cols=(\d+)", line)
        if match:
            name, address, memory, dtype, rows, columns = match.groups()
            if address in constants:
                tiles[name] = {"address": constants[address], "memory": memory,
                               "dtype": dtype, "shape": [int(rows), int(columns)]}
    head_tile = tiles["%stage_30_v36__ssa_v0"]["shape"][0]
    if head_tile not in (8, 16):
        raise ValueError("实际状态head Tile不是8/16")
    state_count = 2 * 64 // head_tile
    state_bytes = head_tile * 256 * 4
    states = []
    for chunk in range(state_count):
        initial = tiles[f"%stage_30_v{36 + chunk}__ssa_v0"]
        if head_tile == 8:
            carried = tiles[f"%stage_30_v{75 + chunk}__phi_v2"]
            valid = tiles[f"%stage_30_v{162 + chunk}__phi_v2"]
            if not (initial == carried == valid):
                raise ValueError("8head初始化/循环回边/有效块状态地址不相同")
        if (initial["memory"], initial["dtype"], initial["shape"]) != ("vec", "f32", [head_tile, 256]):
            raise ValueError("状态Tile规格不符")
        aliases = [name for name, tile in tiles.items() if "__phi_" in name and tile == initial]
        if not aliases:
            raise ValueError("未找到原位状态的分支/循环别名")
        states.append({"chunk": chunk, "address": initial["address"], "bytes": state_bytes,
                       "phi_aliases": aliases})
    intervals = sorted((state["address"], state["address"] + state["bytes"]) for state in states)
    if not all(left[1] <= right[0] for left, right in zip(intervals, intervals[1:])):
        raise ValueError("常驻状态存在重叠")
    additions = []
    for line in ir.splitlines():
        if "pto.tadd ins(" in line:
            match = re.search(r"outs\((%stage_30_\w+) :", line)
            tile = tiles.get(match.group(1)) if match else None
            if tile and tile["shape"] == [head_tile, 256] and tile["dtype"] == "f32":
                additions.append(tile["address"])
    if sorted(additions) != sorted(state["address"] for state in states):
        raise ValueError("PV逐块更新没有一一写回全部状态槽")
    old_inputs = []
    incoming_addresses = []
    for line in ir.splitlines():
        if "pto.trowexpandmul ins(" in line:
            match = re.search(r"pto.trowexpandmul ins\((%stage_30_\w+),", line)
            tile = tiles.get(match.group(1)) if match else None
            if tile and tile["shape"] == [head_tile, 256] and tile["address"] in additions:
                old_inputs.append(tile["address"])
        if "pto.tload ins(" in line:
            match = re.search(r"outs\((%stage_30_\w+) :", line)
            tile = tiles.get(match.group(1)) if match else None
            if tile and tile["shape"] == [head_tile, 256] and tile["dtype"] == "f32":
                incoming_addresses.append(tile["address"])
    if sorted(old_inputs) != sorted(additions):
        raise ValueError("每个状态槽不是唯一一次参与旧值alpha乘法")
    if len(incoming_addresses) != state_count or any(
            start < address + state_bytes and address < end
            for address in incoming_addresses for start, end in intervals):
        raise ValueError("PV incoming Tile覆盖常驻状态或数量不符")
    softmax_heads = {int(value) for line in ir.splitlines()
                     if "pto.trowmax ins(" in line and "%stage_30_" in line
                     for value in re.findall(r"rows=(\d+), cols=128", line)}
    if len(softmax_heads) != 1 or next(iter(softmax_heads)) not in (8, 16, 32):
        raise ValueError("softmax Tile规格不唯一")
    softmax_tile = next(iter(softmax_heads))
    publication_heads = {int(value) for line in ir.splitlines()
                         if "pto.trowexpanddiv ins(" in line and "%stage_30_" in line
                         for value in re.findall(r"rows=(\d+), cols=512", line)}
    if len(publication_heads) != 1 or next(iter(publication_heads)) not in (8, 16):
        raise ValueError("发布Tile规格不唯一")
    publication_tile = next(iter(publication_heads))
    source = (directory / "csa_static_group.cpp").read_text()
    function = re.search(r"^AICORE void csa_static_aiv\([^\n]+\) \{", source, re.M)
    body = source[function.end():close_brace(source, function.end() - 1)]
    segments = []
    for match in re.finditer(r"^  if \([^\n]+\) \{", body, re.M):
        segment = body[match.start():close_brace(body, match.end() - 1) + 1]
        if "// pto: %done_30__tile" in segment:
            segments.append(segment)
    if len(segments) != 1:
        raise ValueError("QK阶段不唯一")
    segment = segments[0]
    expected = {"TROWMAX(": 64 // softmax_tile, "TROWSUM(": 64 // softmax_tile, "TROWEXPANDDIV(": 64 // publication_tile,
                "wait_flag_dev(": 4, "ffts_cross_core_sync(": 4}
    counts = {token: segment.count(token) for token in expected}
    if counts != expected:
        raise ValueError(f"数学或FFTS静态位点不符：{counts}")
    waits = [match.start() for match in re.finditer(r"wait_flag_dev\(", segment)]
    notifications = [match.start() for match in re.finditer(r"ffts_cross_core_sync\(", segment)]
    maxima = [match.start() for match in re.finditer(r"TROWMAX\(", segment)]
    sums = [match.start() for match in re.finditer(r"TROWSUM\(", segment)]
    if not (waits[0] < notifications[0] < notifications[1] < waits[1]
            < waits[2] < notifications[2] < maxima[0] < sums[-1] < notifications[3]
            < waits[3] < segment.index("TROWEXPANDDIV(")):
        raise ValueError("两条AIV的Score/KV/Prob/PV提交顺序改变")
    counts["pipe_barrier(PIPE_ALL)"] = segment.count("pipe_barrier(PIPE_ALL)")
    sizes = {"f32": 4, "i32": 4, "bf16": 2, "f16": 2, "i8": 1, "i64": 8}
    extent = max(tile["address"] + tile["shape"][0] * tile["shape"][1] * sizes[tile["dtype"]]
                 for tile in tiles.values() if tile["memory"] == "vec")
    if extent > 188416:
        raise ValueError("QK显式Tile地址越过可用UB")
    return {"artifact": str(directory.resolve()), "states": states,
            "head_tile": head_tile, "softmax_tile": softmax_tile, "publication_tile": publication_tile,
            "states_disjoint_and_carried_in_place": True, "pv_additions_match_state_addresses": True,
            "old_state_reads_match_updates": True, "incoming_tiles_disjoint_from_state": True,
            "both_role_ffts_source_orders_checked": True,
            "state_bytes": 131072, "qk_explicit_vec_shape_extent_bytes": extent,
            "instruction_sites": counts,
            "scope": "显式Tile地址/指令检查；shape extent不是硬件性能或完整allocator报告"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.artifact)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))
