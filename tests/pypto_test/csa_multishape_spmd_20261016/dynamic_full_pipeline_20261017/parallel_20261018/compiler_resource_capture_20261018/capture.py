#!/usr/bin/env python3
"""CPU重编已验证wrapper并保留CCEC中间产物，不运行设备或改变共享编译器。

历史只有raw bin，无法核对真实Scalar寄存器、栈或spill。这里复用正式
KernelCompiler的目标/优化/包含目录，只追加save-temps=obj保存中间产物。
编译结果和指令证据不直接等于性能根因；不从C++变量数推断寄存器溢出。
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--core", choices=("aic", "aiv"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--register-allocation-dump", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from pypto.runtime.kernel_compiler import KernelCompiler
    from pypto.runtime.pto_isa import ensure_pto_isa_root

    report = {"source": str(args.source.resolve()), "core": args.core,
              "device_execution": False, "performance_measured": False,
              "commands": [], "passed": False, "extra_flags": ["-save-temps=obj"]}
    if args.register_allocation_dump:
        report["extra_flags"] += ["-mllvm", "-print-after=greedy"]

    class RetainCompiler(KernelCompiler):
        def _run(self, command, output, label):
            if label == "Incore":
                command = [command[0], *report["extra_flags"], *command[1:]]
            value = subprocess.run(command, cwd=self.project_root, text=True,
                                   capture_output=True, timeout=self._timeout_s)
            (args.output / f"{label}.stdout.log").write_text(value.stdout)
            (args.output / f"{label}.stderr.log").write_text(value.stderr)
            report["commands"].append({"label": label, "argv": command, "cwd": str(self.project_root),
                                       "returncode": value.returncode})
            retained = args.output / "retained"
            retained.mkdir(exist_ok=True)
            for path in output.parent.iterdir():
                if path.is_file() and path.suffix in {".s", ".ll", ".bc", ".o", ".elf", ".su"}:
                    shutil.copyfile(path, retained / path.name)
            if value.returncode:
                raise RuntimeError(f"{label}编译失败：{value.returncode}: {value.stderr[:2500]}")
            if not output.is_file() or output.stat().st_size == 0:
                raise RuntimeError(f"编译器未生成有效输出：{output}")
            return output.read_bytes()

    try:
        compiler = RetainCompiler(platform="a2a3")
        isa = ensure_pto_isa_root()
        report["pto_isa"] = isa
        report["compiler"] = compiler.sdk.ccec.cxx_path
        compiler.compile_incore(str(args.source.resolve()), core_type=args.core,
                                pto_isa_root=isa, runtime_name="tensormap_and_ringbuffer",
                                build_dir=str(args.output))
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        report["artifacts"] = [{"path": str(p), "bytes": p.stat().st_size}
                               for p in sorted((args.output / "retained").glob("*")) if p.is_file()]
        (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({k: report[k] for k in ("passed", "core", "artifacts")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
