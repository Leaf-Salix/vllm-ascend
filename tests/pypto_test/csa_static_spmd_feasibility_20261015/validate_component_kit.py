# SPDX-License-Identifier: Apache-2.0
"""本机核对组件包的文本范围与独立驱动；只做解析，不重复编译或设备测试。"""

import ast
import json
import runpy
import sys
import zipfile
from pathlib import Path

import dsv4_csa_env

dsv4_csa_env.activate()


def main():
    here = Path(__file__).resolve().parent
    root = here.parent / 'results' / here.name
    directory = root / 'component_repro_kit_20261016'
    manifest = json.loads((directory / 'FILES.json').read_text())
    expected = set(manifest['files']) | {'FILES.json'}
    with zipfile.ZipFile(directory.with_suffix('.zip')) as bundle:
        actual = {str(Path(name).relative_to(directory.name)) for name in bundle.namelist()}
        if actual != expected:
            raise AssertionError('压缩包成员与文件清单不一致')
    for relative in expected:
        source = directory / relative
        if source.suffix not in {'.py', '.cpp', '.md', '.json', '.log'}:
            raise AssertionError(f'非预期文本文件：{relative}')
        text = source.read_text()
        if source.suffix == '.py':
            module = ast.parse(text, filename=str(source))
            for node in ast.walk(module):
                if isinstance(node, ast.Import):
                    modules = [v.name for v in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or '']
                else:
                    continue
                if any(name.startswith(('vllm', 'dsv4_csa_env')) for name in modules):
                    raise AssertionError(f'复现包仍依赖本地工程：{relative}')
    previous_argv = sys.argv
    reports = []
    try:
        for case in ('indexer', 'oproj', 'pair'):
            output = root / 'component_kit_parse' / case
            sys.argv = [str(directory / 'portable_memory_compile.py'), '--case', case,
                        '--parse-only', '--output', str(output)]
            runpy.run_path(sys.argv[0], run_name='__main__')
            result = json.loads((output / 'result.json').read_text())
            if not result['parse_passed'] or result['full_codegen_passed'] or result['device_execution']:
                raise AssertionError('独立驱动的解析范围或结果不符')
            reports.append(result)
    finally:
        sys.argv = previous_argv
    report = {'archive_members': len(expected), 'text_only': True,
              'vllm_or_workspace_imports': False, 'portable_driver_parse_cases': reports,
              'compilation_repeated': False, 'device_execution': False, 'checksums_generated': False}
    (here / 'component_kit_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'archive_members': len(expected), 'portable_parse_cases': len(reports),
                      'device_execution': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
