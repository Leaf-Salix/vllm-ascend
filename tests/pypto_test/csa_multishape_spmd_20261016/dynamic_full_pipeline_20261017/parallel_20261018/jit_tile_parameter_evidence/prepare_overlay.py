# SPDX-License-Identifier: Apache-2.0
"""冻结当前隔离依赖的 Python 源；二进制只读链接，不安装、不重建。"""

import json
import shutil
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve().parent
    source = Path('/data/pyptouser/qinchuanyu/pto-eager/.worktrees/pypto-csa-static-fixes-20261016')
    target = here / 'pypto_overlay'
    target.mkdir(exist_ok=False)
    shutil.copytree(
        source / 'python', target / 'python',
        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.so'),
    )
    binaries = []
    for binary in sorted((source / 'python/pypto').glob('*.so')):
        (target / 'python/pypto' / binary.name).symlink_to(binary)
        binaries.append({'source': str(binary), 'bytes': binary.stat().st_size})
    (here / 'reference').mkdir()
    shutil.copy2(source / 'python/pypto/jit/specializer.py', here / 'reference/specializer.py')
    files = []
    for copied in sorted((target / 'python').rglob('*')):
        if copied.is_file() and not copied.is_symlink():
            original = source / copied.relative_to(target)
            if original.read_bytes() != copied.read_bytes():
                raise AssertionError(f'复制内容不一致: {copied}')
            files.append({'path': str(copied.relative_to(target)), 'bytes': copied.stat().st_size})
    report = {
        'source': str(source), 'overlay': str(target),
        'existing_python_fixes_preserved_by_direct_byte_comparison': True,
        'source_files': files, 'read_only_binary_links': binaries,
        'new_cpp_build': False, 'install': False, 'npu': False,
    }
    (here / 'OVERLAY_PREPARATION.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
