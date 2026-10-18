# SPDX-License-Identifier: Apache-2.0
"""只读既有结果，生成补丁/文本归档清单；不导入 PyPTO，不编译或执行设备。"""

import difflib
import json
from pathlib import Path


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    here = Path(__file__).resolve().parent
    source = Path('/data/pyptouser/qinchuanyu/pto-eager/.worktrees/pypto-csa-static-fixes-20261016')
    original = here / 'reference/specializer.py'
    candidate = here / 'pypto_overlay/python/pypto/jit/specializer.py'
    patch = ''.join(difflib.unified_diff(
        original.read_text().splitlines(keepends=True),
        candidate.read_text().splitlines(keepends=True),
        fromfile='a/python/pypto/jit/specializer.py',
        tofile='b/python/pypto/jit/specializer.py',
    ))
    (here / 'jit_inline_tile_annotation.patch').write_text(patch)
    comparisons = []
    for old in sorted((here / 'pypto_reference/python').rglob('*')):
        if old.is_file() and not old.is_symlink() and '__pycache__' not in old.parts:
            relative = old.relative_to(here / 'pypto_reference')
            new = here / 'pypto_overlay' / relative
            comparisons.append({
                'relative_path': str(relative),
                'same_bytes': new.exists() and old.read_bytes() == new.read_bytes(),
                'reference_bytes': old.stat().st_size,
                'candidate_bytes': new.stat().st_size if new.exists() else None,
            })
    write_json(here / 'SOURCE_COMPARISON.json', {
        'method': 'direct byte comparison',
        'compared_files': len(comparisons),
        'different_files': [r for r in comparisons if not r['same_bytes']],
        'original_dependency_specializer_still_equals_frozen_reference':
            (source / 'python/pypto/jit/specializer.py').read_bytes() == original.read_bytes(),
        'npu_execution': False,
        'new_compilation': False,
    })
    cases = ('before_v1', 'after_v1', 'before_v2', 'after_v2', 'state_v1',
             'inout_v1', 'constexpr_v1', 'state_full_v1')
    archive = [
        'README.md', 'collect_archive.py', 'prepare_overlay.py', 'run_cpu.sh',
        'reproduce_tile.py', 'tile_kernels.py', 'state_kernels.py', 'state_full_kernels.py',
        'inout_kernels.py', 'constexpr_kernels.py', 'reference/reproduce_tile_v1.py',
        'jit_inline_tile_annotation.patch',
        'OVERLAY_PREPARATION.json', 'SOURCE_COMPARISON.json', 'LOCAL_ARTIFACTS.json',
        'ARCHIVE_FILES.txt',
    ]
    for case in cases:
        archive.append(case + '.log')
        for name in ('result.json', 'generated_preparse.py', 'traceback.txt', 'pre_pass.py'):
            if (here / case / name).exists():
                archive.append(f'{case}/{name}')
    for case, kernel in (('after_v2', 'tile_kernel'), ('state_full_v1', 'state_kernel')):
        archive.extend([
            f'{case}/kernel/ptoas/{kernel}.pto',
            f'{case}/kernel/ptoas/{kernel}.cpp',
            f'{case}/kernel/compiled_meta.json',
        ])
    archive.append('state_v1/kernel/passes_dump/00_frontend.py')
    (here / 'ARCHIVE_FILES.txt').write_text('\n'.join(archive) + '\n')
    files = []
    # 清单不包含自身，避免文件大小自引用；扩展链接同时记录链接与目标。
    for path in sorted(here.rglob('*')):
        if path.is_file() and path.name != 'LOCAL_ARTIFACTS.json' and '__pycache__' not in path.parts:
            row = {'path': str(path), 'bytes': path.stat().st_size}
            if path.is_symlink():
                row.update(symlink_target=str(path.resolve()), link_bytes=path.lstat().st_size)
            files.append(row)
    write_json(here / 'LOCAL_ARTIFACTS.json', {
        'root': str(here), 'dependency_source': str(source),
        'overlay': str(here / 'pypto_overlay'), 'reference_overlay': str(here / 'pypto_reference'),
        'all_files_except_this_manifest': files,
        'large_dependencies_are_not_in_archive_copy_list': True,
    })
    missing = [name for name in archive if not (here / name).is_file()]
    print(json.dumps({'archive_files': len(archive), 'missing_files': missing,
                      'different_source_files': [r['relative_path'] for r in comparisons if not r['same_bytes']],
                      'inventory_files': len(files)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
