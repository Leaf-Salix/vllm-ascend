# SPDX-License-Identifier: Apache-2.0
"""核对主 Q 显式预取的两份 L0 地址和生成代码提交顺序，不推断硬件并发。"""

import argparse
import json
import re
from pathlib import Path


def audit(directory):
    directory = Path(directory)
    mlir = (directory / "csa_static_group.pto").read_text().split("func.func @csa_static_aiv", 1)[0]
    cpp = (directory / "csa_static_group.cpp").read_text()
    constants = {name: int(value) for name, value in
                 re.findall(r"(%\w+) = arith.constant (-?\d+) : i64", mlir)}
    rows = []
    for stage in (5, 8):
        tiles = []
        for line in mlir.splitlines():
            match = re.search(r"(%stage_" + str(stage) + r"_\w+) = pto.alloc_tile addr = (%\w+).*"
                              r"loc=(left|right|acc), dtype=(i8|i32), rows=(\d+), cols=(\d+)", line)
            if not match:
                continue
            name, addr, loc, dtype, nrow, ncol = match.groups()
            start = cpp.index("// pto: " + name + "\n")
            variable = re.search(r"Tile<[^\n]+> (\w+) = Tile<", cpp[start:])[1]
            tiles.append(dict(name=name, offset=constants[addr], loc=loc, dtype=dtype,
                              rows=int(nrow), cols=int(ncol), cpp=variable))
        left = [t for t in tiles if t['loc'] == 'left']
        right = [t for t in tiles if t['loc'] == 'right']
        acc = [t for t in tiles if t['loc'] == 'acc']
        if len(left) != 8 or len(right) != 8 or len(acc) != 1:
            raise ValueError(f"stage {stage} 不是预期的八个 K128 部分积")
        if [t['offset'] for t in left] != [0, 12288] * 4:
            raise ValueError("Left 没有交替使用两份 12KiB 缓冲")
        if [t['offset'] for t in right] != [0, 32768] * 4:
            raise ValueError("Right 没有交替使用两份 32KiB 缓冲")
        positions = []
        for k in range(8):
            call = (f"TMATMUL({acc[0]['cpp']}, " if k == 0 else
                    f"TMATMUL_ACC({acc[0]['cpp']}, {acc[0]['cpp']}, ")
            call += f"{left[k]['cpp']}, {right[k]['cpp']});"
            pos = cpp.index(call)
            if k < 7 and cpp.index(f"TMOV({right[k+1]['cpp']},") >= pos:
                raise ValueError("下一块 Right 没有在当前 matmul 提交前装载")
            positions.append(cpp[:pos].count('\n') + 1)
        rows.append(dict(stage=stage, tiles=tiles, matmul_cpp_lines=positions,
                         two_disjoint_l0_buffers=True,
                         next_right_move_precedes_current_matmul=True))
    return dict(artifact=str(directory.resolve()), phases=rows,
                hardware_overlap_measured=False, device_exact_passed=False,
                scope="两处主 Q 代码的 L0 地址与提交顺序，不等同于设备精度或提速")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.artifact)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print('两处主 Q 的 Left/Right 双缓冲与八段提交顺序通过')
