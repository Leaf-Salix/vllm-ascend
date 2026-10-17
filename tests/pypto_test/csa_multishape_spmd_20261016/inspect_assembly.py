# SPDX-License-Identifier: Apache-2.0
"""只编译现有 C++ 到汇编，比较融合前后 Scalar 栈访问；不调用 NPU。"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simpler", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--role", choices=("aic", "aiv"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.meta_path[:] = [finder for finder in sys.meta_path
                       if not any("simpler" in str(value) for value in
                                  getattr(finder, "known_source_files", {}).values())]
    sys.path.insert(0, str(args.simpler))
    from simpler_setup.kernel_compiler import KernelCompiler
    compiler = KernelCompiler("a2a3")
    args.output.mkdir(parents=True, exist_ok=True)
    flags = compiler.ccec.get_compile_flags(core_type=args.role)
    include = args.simpler / "build/pto-isa/include"
    command = [compiler.ccec.cxx_path, *flags, f"-I{include}", f"-I{include / 'pto'}",
               *[f"-I{directory}" for directory in compiler.get_incore_include_dirs()],
               *[f"-I{directory}" for directory in
                 compiler.get_orchestration_include_dirs("tensormap_and_ringbuffer")],
               "-o", str(args.output / "kernel.o"), str(args.source.resolve())]
    (args.output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    completed = subprocess.run(command, text=True, capture_output=True, timeout=180)
    (args.output / "compile.log").write_text(completed.stdout + completed.stderr)
    completed.check_returncode()
    objdump = Path(compiler.ccec.cxx_path).parent / "llvm-objdump"
    result = subprocess.run([str(objdump), "-d", str(args.output / "kernel.o")],
                            text=True, capture_output=True, check=True, timeout=60)
    (args.output / "kernel.s").write_text(result.stdout)
    (args.output / "disassemble.log").write_text(result.stderr)
    print(args.output / "kernel.s", flush=True)


if __name__ == "__main__":
    main()
