# SPDX-License-Identifier: Apache-2.0
"""使用现行 CCE 参数生成诊断 C++ 的优化 LLVM IR；仅 CPU，不执行 NPU。"""

import argparse
import json
import subprocess
from pathlib import Path

from simpler_setup.kernel_compiler import KernelCompiler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--licm-dump', action='store_true')
    args = parser.parse_args()
    compiler = KernelCompiler('a2a3')
    flags = compiler.ccec.get_compile_flags(core_type='aic')
    includes = [compiler.project_root / 'build/pto-isa/include',
                compiler.project_root / 'build/pto-isa/include/pto', *compiler.get_incore_include_dirs()]
    extra = ['-mllvm', '-print-after=licm'] if args.licm_dump else ['-S', '-emit-llvm']
    if not args.licm_dump:
        flags = [f for f in flags if f != '-c']
    command = [compiler.ccec.cxx_path, *flags, *extra,
               *(f'-I{path}' for path in includes), '-o', str(args.output), str(args.source)]
    result = subprocess.run(command, cwd=compiler.project_root, text=True, capture_output=True)
    args.output.with_suffix('.compile.log').write_text(result.stdout + result.stderr)
    args.output.with_suffix('.command.json').write_text(json.dumps({
        'command': command, 'cwd': str(compiler.project_root), 'exit_code': result.returncode,
        'device_execution': False}, ensure_ascii=False, indent=2) + '\n')
    if result.returncode:
        raise RuntimeError(result.stderr)
    print(args.output)


if __name__ == '__main__':
    main()
