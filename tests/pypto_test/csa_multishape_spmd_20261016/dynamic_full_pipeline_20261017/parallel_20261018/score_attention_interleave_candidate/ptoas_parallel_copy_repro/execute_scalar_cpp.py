# SPDX-License-Identifier: Apache-2.0
"""在CPU执行PTOAS生成的纯Scalar控制流，保留赋值原文，只空实现设备屏障。

这是循环/分支并行赋值的编译正确性回归，不是NPU流水或CSA精度验证。
输入C++含shift_prev、rotate_state、conditional_swap，输出逐例实际值与固定期望。
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cpp", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-bug", action="store_true")
    args = parser.parse_args()
    source = args.cpp.read_text()
    # 只替换设备专用前言；下面三函数的全部控制流/赋值逐字保留。
    body = source[source.index("AICORE void shift_prev("):]
    preamble = """#include <cstdint>
#include <iostream>
#define AICORE
#define __gm__
#define PIPE_ALL 0
enum class cache_line_t {ENTIRE_DATA_CACHE, SINGLE_CACHE_LINE};
using mem_dsb_t = int;
inline void pipe_barrier(int) {}
inline void dcci(void*, cache_line_t) {}
inline void dsb(mem_dsb_t) {}
"""
    harness = """
int main() {
  for (int limit : {0, 1, 2, 3, 5}) {
    int64_t out[3] = {-1, -1, -1};
    shift_prev(out, limit);
    std::cout << "shift " << limit << " " << out[0] << " " << out[1] << "\\n";
    rotate_state(out, limit);
    std::cout << "rotate " << limit << " " << out[0] << " " << out[1] << " " << out[2] << "\\n";
    for (int swap : {0, 1}) {
      conditional_swap(out, limit, swap);
      std::cout << "swap " << limit << " " << swap << " " << out[0] << " " << out[1] << "\\n";
    }
  }
}
"""
    args.output.mkdir(parents=True, exist_ok=False)
    host = args.output / "host.cpp"
    host.write_text(preamble + body + harness)
    binary = args.output / "host"
    compiler = os.environ.get("CXX") or shutil.which("c++")
    if not compiler:
        raise RuntimeError("需要CPU C++编译器执行纯Scalar回归")
    command = [compiler, "-std=c++17", "-O0", str(host), "-o", str(binary)]
    subprocess.run(command, check=True)
    output = subprocess.check_output([str(binary)], text=True)
    (args.output / "stdout.txt").write_text(output)
    rows = []
    rotations = {0: [1, 2, 3], 1: [2, 3, 1], 2: [3, 1, 2]}
    for line in output.splitlines():
        kind, limit, *rest = line.split()
        limit = int(limit)
        values = list(map(int, rest))
        if kind == "shift":
            expected = [limit * 48, max(0, limit - 1) * 48]
        elif kind == "rotate":
            expected = rotations[limit % 3]
        else:
            swap, *values = values
            expected = [22, 11] if swap and limit % 2 else [11, 22]
        rows.append(dict(case=line.split()[:3] if kind == "swap" else [kind, limit],
                         actual=values, expected=expected, passed=values == expected))
    report = dict(source=str(args.cpp), scope="CPU纯Scalar控制流；不验证设备同步/数学性能",
                  command=command, cases=rows, passed=all(row["passed"] for row in rows),
                  failures=sum(not row["passed"] for row in rows))
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"cases": len(rows), "failures": report["failures"], "passed": report["passed"]}))
    if report["passed"] == args.expect_bug:
        raise AssertionError("结果不符合要求的正/负对照")


if __name__ == "__main__":
    main()
