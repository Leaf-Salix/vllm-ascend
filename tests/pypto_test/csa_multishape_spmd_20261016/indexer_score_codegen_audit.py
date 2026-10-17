# SPDX-License-Identifier: Apache-2.0
"""只读核对M192/N64候选的实际Acc地址与C++提交顺序，不执行设备。"""

import argparse
import json
import re
from pathlib import Path


def audit(directory):
    directory = Path(directory)
    mlir = (directory / "csa_static_group.pto").read_text()
    cpp = (directory / "csa_static_group.cpp").read_text()
    constants = {name: int(value) for name, value in
                 re.findall(r"(%\w+) = arith.constant (-?\d+) : i64", mlir)}
    allocations, products = {}, []
    for line in mlir.splitlines():
        if "pto.alloc_tile" in line and "loc=acc, dtype=i32, rows=192, cols=64" in line:
            match = re.search(r"(%\w+) = pto.alloc_tile addr = (%\w+)", line)
            allocations[match[1]] = constants[match[2]]
        if "pto.tmatmul ins(" in line:
            match = re.search(r"outs\((%\w+) : !pto.tile_buf<loc=acc, dtype=i32, rows=192, cols=64", line)
            if match:
                products.append(match[1])
    if len(products) != 32:
        raise ValueError("当前审计只对应16个N64面板、每面板两个M192乘积")
    pairs = []
    for panel in range(16):
        names = products[panel * 2:panel * 2 + 2]
        offsets = [allocations[name] for name in names]
        if abs(offsets[0] - offsets[1]) < 49152:
            raise ValueError(f"第{panel}组的两份Acc重叠")
        variables = []
        for name in names:
            begin = cpp.index("// pto: " + name + "\n")
            declaration = re.search(r"Tile<[^\n]+> (\w+) = Tile<", cpp[begin:])
            variables.append(declaration[1])
        first = cpp.index("TMATMUL(" + variables[0] + ",")
        second = cpp.index("TMATMUL(" + variables[1] + ",")
        writeback = re.search(r"TINSERT<[^\n]+>\(\w+, " + variables[0] + ",", cpp[first:])
        if not writeback or not first < second < first + writeback.start():
            raise ValueError(f"第{panel}组的提交顺序没有保持")
        pairs.append({"panel": panel, "acc_offsets": offsets, "bytes_each": 49152,
                      "qk_output_cpp": variables, "both_qk_submitted_before_first_fix": True})
    return {"artifact_directory": str(directory.resolve()), "score_acc_pairs": pairs,
            "all_pairs_disjoint": True, "source_and_cpp_submission_order_match": True,
            "hardware_overlap_measured": False,
            "scope": "地址和生成代码提交顺序；不能据此推出真实并发时间或性能"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.artifact)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("16组Acc地址与C++提交顺序检查通过")
