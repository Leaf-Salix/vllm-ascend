# SPDX-License-Identifier: Apache-2.0
"""CPU 上预检新 helper 的 CCEC 合法性，不用于性能或声称真实后端已通过。

PTOAS 全量构建前，将独立 worktree 中的辅助模板插入现有最小复现 C++，
只替换 compare 调用名字。正式测试仍必须从新 PTOAS 生成，不使用此变换。
"""
import argparse
import json
import traceback
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--cmp-mode", choices=("GT", "GE", "EQ"), default="GT")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    workspace = here.parents[3]
    header = workspace / ".worktrees/ptoas-csa-topk-compact-20261020/lib/PTO/Transforms/PTOToEmitC/Tensor/GatherCompareScalarSupport.h"
    helper = header.read_text().split('R"cpp(', 1)[1].split(')cpp";', 1)[0]
    cpp = (here / "gather_cpu_v3/kernel/kernels/aiv/compare_kernel.cpp").read_text()
    assert cpp.count("  TGATHER<") == 1
    cpp = cpp.replace("  TGATHER<", "  PTOAS__GATHER_COMPARE_SCALAR<")
    cpp = cpp.replace("CmpMode::GT>", f"CmpMode::{args.cmp_mode}>")
    cpp = cpp.replace("using namespace pto;", "using namespace pto;\n" + helper, 1)
    target = args.output / "helper_probe.cpp"
    target.write_text(cpp)
    report = dict(native_compile=False, actual_new_ptoas=False, device_run=False, cmp_mode=args.cmp_mode,
                  helper=str(header), cpp=str(target))
    try:
        from pypto.runtime import ensure_pto_isa_root
        from simpler_setup.kernel_compiler import KernelCompiler
        compiler = KernelCompiler("a2a3")
        raw = compiler.compile_incore(str(target), core_type="aiv",
            pto_isa_root=ensure_pto_isa_root(),
            extra_include_dirs=compiler.get_orchestration_include_dirs("tensormap_and_ringbuffer"))
        report.update(native_compile=True, binary_bytes=len(raw))
    except BaseException as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("error", "traceback")},
                     ensure_ascii=False), flush=True)
    if not report["native_compile"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
