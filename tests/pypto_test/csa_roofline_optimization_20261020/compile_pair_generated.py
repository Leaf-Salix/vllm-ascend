# SPDX-License-Identifier: Apache-2.0
"""原样编译新PTOAS实际生成的C++，补齐模板预检没有覆盖的调用方类型检查。

仅CPU执行CCEC，不注册或启动NPU kernel，不注入源码，不作为精度/性能证据。
二进制只保留本地；入Git的是调用路径、大小与编译结果。
"""
import argparse
import json
from pathlib import Path
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from pypto.runtime import ensure_pto_isa_root
    from simpler_setup.kernel_compiler import KernelCompiler

    compiler = KernelCompiler("a2a3")
    report = {"passed": False, "device_execution": False, "source_rewritten": False,
              "actual_generated_cpp": True, "kernels": []}
    for role in ("aic", "aiv"):
        for source in sorted((args.artifact / "kernels" / role).glob("*.cpp")):
            item = {"source": str(source), "role": role, "passed": False}
            try:
                binary = compiler.compile_incore(
                    str(source), core_type=role, pto_isa_root=ensure_pto_isa_root(),
                    extra_include_dirs=compiler.get_orchestration_include_dirs(
                        "tensormap_and_ringbuffer"
                    ),
                )
                (args.output / f"{role}_{source.stem}.bin").write_bytes(binary)
                item.update(passed=True, binary_bytes=len(binary))
            except BaseException as exc:
                item.update(error=str(exc), traceback=traceback.format_exc())
            report["kernels"].append(item)
            print(role, source.name, item["passed"], flush=True)
    report["passed"] = bool(report["kernels"]) and all(x["passed"] for x in report["kernels"])
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
