# SPDX-License-Identifier: Apache-2.0
"""只读核对N32候选的实际地址和生成代码；不把提交顺序当作实测重叠。"""

import argparse
import json
import re
from pathlib import Path

from qk_phase_clock import close_brace


def audit(directory):
    directory = Path(directory)
    mlir = (directory / "csa_static_group.pto").read_text().split("func.func @csa_static_aiv")[0]
    cpp = (directory / "csa_static_group.cpp").read_text()
    constants = {name: int(value) for name, value in
                 re.findall(r"(%\w+) = arith.constant (-?\d+) : i64", mlir)}
    tiles = {}
    for line in mlir.splitlines():
        match = re.search(r"(%\w+) = pto.alloc_tile addr = (%\w+).*"
                          r"!pto.tile_buf<loc=(\w+), dtype=(\w+), rows=(\d+), cols=(\d+)", line)
        if match:
            name, address, memory, dtype, rows, cols = match.groups()
            tiles[name] = {"address": constants[address], "memory": memory, "dtype": dtype,
                           "shape": [int(rows), int(cols)]}

    def operands(memory, dtype, shape):
        return {name: info for name, info in tiles.items()
                if name.startswith("%stage_16_") and info["memory"] == memory
                and info["dtype"] == dtype and info["shape"] == shape}

    qk = operands("acc", "i32", [384, 32])
    products = []
    for line in mlir.splitlines():
        if "pto.tmatmul ins(" in line:
            out = re.search(r"outs\((%\w+)", line)
            if out and out[1] in qk:
                products.append(out[1])
    if len(products) != 32:
        raise ValueError("N32必须包含32个独立QK乘积")
    cpp_names = {}
    for name in products:
        start = cpp.index("// pto: " + name + "\n")
        cpp_names[name] = re.search(r"Tile<[^\n]+> (\w+) = Tile<", cpp[start:])[1]
    pairs = []
    for panel in range(31):
        first, second = products[panel:panel + 2]
        addresses = [qk[first]["address"], qk[second]["address"]]
        if abs(addresses[0] - addresses[1]) < 49152:
            raise ValueError(f"QK面板{panel}的双Acc区域重叠")
        position = cpp.index("TMATMUL(" + cpp_names[first] + ",")
        next_position = cpp.index("TMATMUL(" + cpp_names[second] + ",")
        fix = re.search(r"TINSERT<[^\n]+>\(\w+, " + cpp_names[first] + ",", cpp[position:])
        if not fix or not position < next_position < position + fix.start():
            raise ValueError(f"QK面板{panel}没有在当前FIX之前提交下一QK")
        pairs.append({"panel": panel, "acc_addresses": addresses,
                      "next_qk_before_current_fix": True})
    # 用完整阶段边界，避免v30这样的前缀误匹配v300而只截到阶段后半部。
    end_marker = cpp.index("// pto: %done_16__tile\n")
    opening = list(re.finditer(r"^  if \([^\n]+\) \{", cpp[:end_marker], re.M))[-1]
    segment = cpp[opening.start():close_brace(cpp, opening.end() - 1) + 1]
    right_moves = [line.strip() for line in mlir.splitlines()
                   if "pto.tmov " in line and "%stage_16_" in line
                   and line.count("loc=right") >= 2]
    if right_moves:
        raise ValueError("生成了不支持的Right到Right搬运")
    counts = {"matmul": segment.count("TMATMUL("), "fix_to_l1": segment.count("TINSERT<"),
              "gm_store": segment.count("TSTORE("), "pipe_all": segment.count("pipe_barrier(PIPE_ALL)")}
    if counts["matmul"] != 64 or counts["fix_to_l1"] != 32 or counts["gm_store"] != 32:
        raise ValueError(f"阶段指令数改变：{counts}")
    return {"artifact": str(directory.resolve()), "adjacent_qk_pairs": pairs,
            "all_qk_pairs_disjoint": True, "source_cpp_submission_order_preserved": True,
            "key_right": operands("right", "i8", [128, 32]),
            "ws_right": operands("right", "f16", [384, 32]),
            "ws_acc": operands("acc", "f32", [16, 32]),
            "query_left": operands("left", "i8", [384, 128]),
            "coefficient_left": operands("left", "f16", [16, 384]),
            "right_to_right_moves": right_moves, "stage_cpp_bytes": len(segment.encode()),
            "stage_cpp_counts": counts, "device_validated": False,
            "scope": "实际地址与提交顺序，不证明设备并发或性能收益"}, segment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result, segment = audit(args.artifact)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    args.output.with_suffix(".cpp.txt").write_text(segment + "\n")
    print(json.dumps({key: result[key] for key in ("all_qk_pairs_disjoint",
          "source_cpp_submission_order_preserved", "stage_cpp_counts")}, ensure_ascii=False))
