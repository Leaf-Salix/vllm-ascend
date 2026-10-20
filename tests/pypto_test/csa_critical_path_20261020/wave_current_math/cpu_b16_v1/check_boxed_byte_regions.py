# SPDX-License-Identifier: Apache-2.0
"""按PTO-ISA GetTileOffset逐元素枚举，核对L1窗口分段及转置别名；不运行设备。"""
import itertools
import json
from pathlib import Path


def isa_offset(layout, rows, cols, row, col):
    # 来源：PTO-ISA common/pto_tile.hpp::GetTileOffset，INT8 inner形状为
    # Nz=(16,32)、Zn=(32,16)，每小块512B。这里逐元素计算，不用闭式分段。
    ir, ic = (16, 32) if layout == "nz" else (32, 16)
    rb, cb, ri, ci = row // ir, col // ic, row % ir, col % ic
    if layout == "nz":
        return (rows // ir * cb + rb) * 512 + ri * ic + ci
    return (cols // ic * rb + cb) * 512 + ci * ir + ri


def segments(layout, rows, cols, row, col, height, width):
    ir, ic = (16, 32) if layout == "nz" else (32, 16)
    if layout == "nz":
        return [(block * rows * ic + row * ic, height * ic)
                for block in range(col // ic, (col + width) // ic)]
    return [(block * cols * ir + col * ir, width * ir)
            for block in range(row // ir, (row + height) // ir)]


def expand(parts):
    return {x for start, size in parts for x in range(start, start + size)}


def main():
    checked = 0
    shapes = ((32, 64), (64, 128), (128, 128), (128, 256), (384, 128))
    for layout in ("nz", "zn"):
        ir, ic = (16, 32) if layout == "nz" else (32, 16)
        for nz_rows, nz_cols in shapes:
            rows, cols = (nz_rows, nz_cols) if layout == "nz" else (nz_cols, nz_rows)
            row_starts = sorted({0, ir, rows - ir})
            col_starts = sorted({0, ic, cols - ic})
            for row, col in itertools.product(row_starts, col_starts):
                for height, width in itertools.product({ir, rows - row}, {ic, cols - col}):
                    oracle = {isa_offset(layout, rows, cols, r, c)
                              for r in range(row, row + height)
                              for c in range(col, col + width)}
                    got = expand(segments(layout, rows, cols, row, col, height, width))
                    assert oracle == got and len(got) == height * width
                    assert min(got) >= 0 and max(got) < rows * cols
                    checked += 1
            # 同地址的转置Tile必须逐元素给出相同字节位置。
            for row, col in itertools.product(range(rows), range(cols)):
                other = "zn" if layout == "nz" else "nz"
                assert isa_offset(layout, rows, cols, row, col) == isa_offset(other, cols, rows, col, row)
    read = segments("zn", 128, 128, 0, 0, 128, 64)
    disjoint_write = segments("nz", 128, 128, 64, 0, 32, 128)
    overlapping_write = segments("nz", 128, 128, 32, 0, 32, 128)
    assert not expand(read) & expand(disjoint_write)
    assert len(expand(read) & expand(overlapping_write)) == 4096
    report = dict(passed=True, windows_checked=checked, layouts=["nz", "zn"],
                  source_reference="PTO-ISA common/pto_tile.hpp::GetTileOffset / getInnerRow / getInnerCol",
                  read_segments=read, disjoint_write_segments=disjoint_write,
                  overlapping_write_segments=overlapping_write,
                  overlap_bytes=4096, device_execution=False)
    Path(__file__).with_name("BOXED_BYTE_REGIONS.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
