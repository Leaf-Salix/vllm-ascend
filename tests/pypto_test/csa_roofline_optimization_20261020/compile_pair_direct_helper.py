# SPDX-License-Identifier: Apache-2.0
"""仅预检新原语辅助模板的CCEC合法性；不执行、不替代真实PTOAS生成测试。"""
import argparse
import json
from pathlib import Path
import traceback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[4]
    header = root / (
        ".worktrees/ptoas-csa-pair-compact-20261020/"
        "lib/PTO/Transforms/PTOToEmitC/Tensor/GatherCompareScalarSupport.h"
    )
    helper = header.read_text().split('R"cpp(', 1)[1].split(')cpp";', 1)[0]
    source = """#include <cstdint>
#include <pto/pto-inst.hpp>
using namespace pto;
""" + helper + r'''
extern "C" AICORE void kernel_entry(__gm__ int64_t *args) {
#if defined(__DAV_VEC__)
  using S = Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, 1, -1>;
  using D = Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, 1, 2048>;
  using I = Tile<TileType::Vec, int32_t, 1, 2048, BLayout::RowMajor, 1, 2048>;
  using C = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 1>;
  using W = Tile<TileType::Vec, uint8_t, 1, 9216, BLayout::RowMajor, 1, 9216>;
  S src(static_cast<int32_t>(args[0]));
  D values;
  I indices;
  C count;
  W work;
  TASSIGN(src, 0);
  TASSIGN(values, 8192);
  TASSIGN(indices, 16384);
  TASSIGN(work, 24576);
  TASSIGN(count, 33792);
  int32_t offset = static_cast<int32_t>(args[1]);
  PTOAS__COMPACT_COMPARE<D, S, I, W, C, CmpMode::GE>(
      values, src, indices, 0.5f, work, count, offset);
  PTOAS__COMPACT_COMPARE<D, S, I, W, C, CmpMode::GT>(
      values, src, indices, 0.5f, work, count, offset);
  PTOAS__COMPACT_COMPARE<D, S, I, W, C, CmpMode::EQ>(
      values, src, indices, 0.5f, work, count, offset);
#endif
}
'''
    target = args.output / "helper.cpp"
    target.write_text(source)
    report = {"passed": False, "device_run": False, "actual_new_ptoas": False,
              "header": str(header), "scope": "CCEC template compile only"}
    try:
        from pypto.runtime import ensure_pto_isa_root
        from simpler_setup.kernel_compiler import KernelCompiler
        compiler = KernelCompiler("a2a3")
        binary = compiler.compile_incore(str(target), core_type="aiv",
                                        pto_isa_root=ensure_pto_isa_root())
        (args.output / "helper.bin").write_bytes(binary)
        report.update(passed=True, binary_bytes=len(binary))
    except BaseException as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("error", "traceback")},
                     ensure_ascii=False), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
