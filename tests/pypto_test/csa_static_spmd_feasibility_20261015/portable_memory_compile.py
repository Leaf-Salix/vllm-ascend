# SPDX-License-Identifier: Apache-2.0
"""组件侧 CPU 编译驱动：只依赖标准库和已安装的 PyPTO，不加载 vLLM 或 NPU。"""

import argparse
import json
import traceback
from pathlib import Path

import pypto.language as pl
from pypto import ir, passes
from pypto.runtime import RunConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('indexer', 'oproj', 'pair'), default='pair')
    parser.add_argument('--planner', choices=('PYPTO', 'DSA_RP'), default='PYPTO')
    parser.add_argument('--parse-only', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    directories = {'indexer': 'memory_indexer', 'oproj': 'memory_oproj', 'pair': 'memory_pair_compact'}
    source = here / 'static' / directories[args.case] / 'entry.py'
    args.output.mkdir(parents=True, exist_ok=True)
    result = {'source': str(source), 'case': args.case, 'planner': args.planner,
              'parse_passed': False, 'full_codegen_passed': False, 'device_execution': False}
    try:
        program = pl.parse(source.read_text(), filename=str(source))
        result['parse_passed'] = True
        if not args.parse_only:
            compiled = ir.compile(program, **RunConfig(
                platform='a2a3', memory_planner=getattr(passes.MemoryPlanner, args.planner),
                save_kernels=True, save_kernels_dir=str(args.output / 'kernel'), dump_passes=True,
            ).compile_kwargs())
            result.update(full_codegen_passed=True, compiled_directory=str(compiled.output_dir))
    except Exception as exc:
        result.update(error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        (args.output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({k: v for k, v in result.items() if k != 'traceback'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
