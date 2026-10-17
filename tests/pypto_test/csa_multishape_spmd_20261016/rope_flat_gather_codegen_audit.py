# SPDX-License-Identifier: Apache-2.0
"""执行前核对三处批量gather的真实Tile形状、行基址链和UB非重叠。"""

import argparse
import collections
import json
import re
from pathlib import Path

from qk_phase_clock import close_brace


def audit(directory):
    path = Path(directory) / "csa_static_group.cpp"
    source = path.read_text()
    fn = re.search(r"^AICORE void csa_static_aiv\([^\n]+\) \{", source, re.M)
    if fn is None:
        raise ValueError("未找到AIV入口")
    body = source[fn.end():close_brace(source, fn.end() - 1)]
    constants = {name: int(value) for name, value in re.findall(
        r"const (?:u?int\d+_t) (v\d+) = (-?\d+);", body)}
    aliases = dict(re.findall(r"uint64_t (v\d+) = \(uint64_t\) (v\d+);", body))
    addresses = {tile: constants[aliases[address]] for tile, address in re.findall(
        r"TASSIGN\((v\d+), (v\d+)\);", body)
        if address in aliases and aliases[address] in constants}
    tiles = {name: (dtype, int(rows), int(cols)) for dtype, rows, cols, name in re.findall(
        r"Tile<TileType::Vec, (float|int32_t), (\d+), (\d+), [^\n;]+?> (v\d+)(?: =|;)", body)}
    operations = {}
    for match in re.finditer(r"(TSHRS|TSHLS|TADD)\((v\d+), (v\d+), (v\d+)\);", body):
        op, dst, src0, src1 = match.groups()
        operations[dst] = (op, src0, src1, match.start())
    sites = []
    for gather in re.finditer(r"TGATHER\((v\d+), (v\d+), (v\d+), (v\d+)\);", body):
        dst, src, index, tmp = gather.groups()
        add = operations.get(index)
        if add is None or add[0] != "TADD":
            continue
        left = operations.get(add[1])
        if left is None or left[0] != "TSHLS" or constants.get(left[2]) != 6:
            continue
        right = operations.get(left[1])
        if right is None or right[0] != "TSHRS" or constants.get(right[2]) != 6:
            raise ValueError("批量gather没有完整的r*64基址链")
        if not right[3] < left[3] < add[3] < gather.start():
            raise ValueError("批量gather索引提交顺序改变")
        shape = tiles.get(dst)
        if shape not in (("float", 1, 1024), ("float", 1, 2048)) or tiles.get(src) != shape:
            raise ValueError("批量gather未产生预期连续FP32 Tile")
        elements = shape[2]
        if tiles.get(index) != ("int32_t", 1, elements) or tiles.get(tmp) != tiles[index]:
            raise ValueError("批量gather索引或scratch形状不符")
        ranges = {name: (addresses[name], addresses[name] + elements * 4)
                  for name in (dst, src, index, tmp)}
        intervals = sorted(ranges.values())
        if any(a[1] > b[0] for a, b in zip(intervals, intervals[1:])):
            raise ValueError("批量gather的同时活跃缓冲区重叠")
        sites.append({"line": source.count("\n", 0, fn.end() + gather.start()) + 1,
                      "elements": elements, "row_shift": 6, "ub_byte_ranges": ranges})
    if collections.Counter(site["elements"] for site in sites) != {1024: 2, 2048: 1}:
        raise ValueError("实际产物没有且仅有三处批量RoPE gather")
    return {"passed": True, "artifact": str(path.resolve()), "sites": sites,
            "scope": "代码生成门禁，不能代替设备精度或性能结果"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.artifact)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))
