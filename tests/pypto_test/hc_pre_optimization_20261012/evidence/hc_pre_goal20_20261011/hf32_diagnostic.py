"""隔离检查 HC_pre 的 AIC 矩阵乘模式；不是正式 PyPTO 算子入口。

本脚本只在自身进程中修饰生成的 HC_pre AIC wrapper，依赖仓库不落盘修改。
fp32/hf32 两种模式共用完全相同的 PyPTO 源码和调度，分别设置 CTRL[46]。
hf32 同时设置 CTRL[47]，对应 Native 的 SetHF32TransMode(1)。
进入 HC_pre 前保存 CTRL，完成后恢复，不能污染后续任务的矩阵乘模式。
默认只允许 CPU 编译。设备诊断须显式启用，并经 task-submit 调度。
"""

import argparse
import json
import os
import runpy
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fp32", "hf32"), required=True)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--device-diagnostic", action="store_true")
    parser.add_argument("bench_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    bench_args = args.bench_args
    if bench_args and bench_args[0] == "--":
        bench_args = bench_args[1:]
    device_execution = "--build-only" not in bench_args
    if device_execution and not (args.device_diagnostic and os.environ.get("TASK_DEVICE")):
        parser.error("设备诊断需要 --device-diagnostic 且必须经 task-submit 分配设备")
    if args.audit.exists():
        parser.error("诊断清单已存在，请使用新目录")
    driver = args.driver.resolve(strict=True)
    root = Path(__file__).resolve().parents[5]
    tests = root / "vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test"
    sys.path.insert(0, str(tests))
    from dsv4_csa_env import activate

    activate()
    core = root / "pypto/build/python/bindings/pypto_core.cpython-310-aarch64-linux-gnu.so"
    finders = [f for f in sys.meta_path if "pypto.pypto_core" in getattr(f, "known_wheel_files", {})]
    if len(finders) != 1:
        raise RuntimeError("无法唯一定位本进程的 PyPTO editable 扩展映射")
    finders[0].known_wheel_files["pypto.pypto_core"] = str(core.resolve(strict=True))
    from pypto.backend import pto_backend

    original = pto_backend._generate_kernel_wrapper
    records = []

    def generate(func, ptoas_code, **kwargs):
        source = original(func, ptoas_code, **kwargs)
        if not func.name.startswith("hc_") or not func.name.endswith("_aic"):
            return source
        if "TMATMUL(" not in source:
            raise RuntimeError(f"{func.name} 没有预期的矩阵乘，拒绝盲目修改")
        marker = "    // Forward to ptoas-generated function\n"
        if source.count(marker) != 1:
            raise RuntimeError(f"{func.name} 的 wrapper 调用位置不唯一")
        position = source.index(marker) + len(marker)
        call_end = source.index(";\n", position) + 2
        call = source[position:call_end]
        if not call.strip().startswith(func.name + "("):
            raise RuntimeError(f"{func.name} 的 wrapper 调用不匹配")
        mode = (
            "set_ctrl(sbitset1(sbitset1(hc_saved_ctrl, pto::MAD_MODE_BIT), pto::MAD_ROUND_MODE_BIT));"
            if args.mode == "hf32"
            else "set_ctrl(sbitset0(hc_saved_ctrl, pto::MAD_MODE_BIT));"
        )
        prefix = (
            "#if defined(PTO_COMPILE_CUBE) && !defined(__CPU_SIM) && !defined(__COSTMODEL)\n"
            "    // 仅用于模式诊断：与 Native 对齐；保留并恢复调用前的控制寄存器。\n"
            "    const uint64_t hc_saved_ctrl = get_ctrl();\n"
            f"    {mode}\n"
            "#endif\n"
        )
        suffix = (
            "#if defined(PTO_COMPILE_CUBE) && !defined(__CPU_SIM) && !defined(__COSTMODEL)\n"
            "    set_ctrl(hc_saved_ctrl);\n"
            "#endif\n"
        )
        records.append({"function": func.name, "mode": args.mode,
                        "location": "kernel_entry 中包围完整 AIC 函数调用", "restore_ctrl": True})
        return source[:position] + prefix + call + suffix + source[call_end:]

    pto_backend._generate_kernel_wrapper = generate
    status = "FAILED"
    try:
        sys.argv = [str(driver), *bench_args]
        # 与直接执行 driver 一样，使用其相邻的泳道导出模块，避免混用不同工作树版本。
        sys.path.insert(0, str(driver.parent))
        runpy.run_path(str(driver), run_name="__main__")
        if not records:
            raise RuntimeError("没有生成任何模式诊断 AIC，不能记为完成")
        status = "DEVICE_DIAGNOSTIC_PASSED" if device_execution else "COMPILED_ONLY"
    finally:
        pto_backend._generate_kernel_wrapper = original
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        args.audit.write_text(json.dumps({
            "status": status, "device_execution": device_execution,
            "scope": "仅诊断生成代码；不代表原生 PyPTO 接口已支持 HF32",
            "mode": args.mode, "kernels": records,
        }, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
