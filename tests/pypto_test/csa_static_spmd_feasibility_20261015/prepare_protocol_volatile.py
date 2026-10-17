# SPDX-License-Identifier: Apache-2.0
"""只改诊断产物的进度标量访问，验证 CCE 是否复用了轮询值；不修改依赖源码。"""

import argparse
import difflib
import json
import re
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--access', choices=('all', 'read', 'write'), default='all')
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    for name in ('protocol_program.py', 'extraction.json'):
        shutil.copy2(args.source / name, args.output / name)
    shutil.copytree(args.source / 'cpu', args.output / 'cpu',
                    ignore=shutil.ignore_patterns('cache', 'dfx_outputs', 'passes_dump',
                                                 '*.o', '*.so', '*.bin'))
    changes = []
    patches = []
    for path in (args.output / 'cpu/kernel').rglob('*.cpp'):
        before = path.read_text()
        if 'static __aicore__ void csa_static_aic(__gm__ int64_t* v1' not in before:
            continue
        # 两个函数的 v1 都是 progress；records/waits 为 v2/v3，不触碰。
        assert 'static __aicore__ void csa_static_aiv(__gm__ int64_t* v1' in before
        # 分开读取/发布，避免把“同时改读写有效”误说成已确定只有读有问题。
        reads = writes = changed = 0
        lines = []
        for line in before.splitlines(keepends=True):
            if '(v1)[' in line:
                is_write = re.match(r'^\s*\(v1\)\[[^\]]+\]\s*=', line) is not None
                assert line.count('(v1)[') == 1, line
                if is_write:
                    writes += 1
                else:
                    reads += 1
                if args.access == 'all' or (args.access == 'write') == is_write:
                    line = line.replace('(v1)[', '(reinterpret_cast<volatile __gm__ int64_t*>(v1))[')
                    changed += 1
            lines.append(line)
        assert reads > 0 and writes > 0 and changed > 0
        after = ''.join(lines)
        path.write_text(after)
        relative = str(path.relative_to(args.output))
        changes.append({'path': relative, 'scalar_progress_accesses': changed,
                        'original_reads': reads, 'original_writes': writes})
        patches.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                           fromfile=relative + '.before', tofile=relative + '.after'))
    changed_roles = {Path(item['path']).parent.name for item in changes}
    assert {'aic', 'aiv'} <= changed_roles, changes
    (args.output / 'volatile_progress.patch').write_text(''.join(patches))
    (args.output / 'diagnostic_variant.json').write_text(json.dumps({
        'source': str(args.source), 'scope': '仅诊断生成 C++ 中 progress 标量访问改 volatile',
        'access': args.access,
        'component_source_changed': False, 'full_static_source_changed': False,
        'same_records_waits_and_poll_budget': True, 'changes': changes,
        'full_csa_validated': False}, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(changes))


if __name__ == '__main__':
    main()
