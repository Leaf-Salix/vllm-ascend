# SPDX-License-Identifier: Apache-2.0
"""复制已有最小证据；大产物/依赖仅记录路径大小，不编译、不执行设备。"""

import json
import shutil
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    target = Path('/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/score_attention_interleave_candidate/mat_left_smoke_evidence')
    target.mkdir(exist_ok=False)
    files = ['README.md', 'archive_evidence.py', 'run_smoke.py', 'run_cpu.sh',
             'mat_left_kernels.py', 'mat_left_kernels_v2.py', 'cpu_v1.log', 'cpu_v2.log']
    for case in ('cpu_v1', 'cpu_v2'):
        for name in ('result.json', 'traceback.txt', 'generated_preparse.py', 'pre_pass.py'):
            if (here / case / name).is_file():
                files.append(case + '/' + name)
    files.extend(['cpu_v2/kernel/ptoas/mat_left_kernel.pto', 'cpu_v2/kernel/ptoas/mat_left_kernel.cpp'])
    for name in files:
        out = target / name
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(here / name, out)
    records = [{'path': str(path), 'bytes': path.stat().st_size}
               for path in sorted(here.rglob('*')) if path.is_file() and '__pycache__' not in path.parts]
    overlay = here.parent / 'jit_tile_parameter_overlay_20261018/pypto_overlay'
    binaries = [{'path': str(path), 'resolved': str(path.resolve()), 'bytes': path.stat().st_size}
                for path in sorted((overlay / 'python/pypto').glob('*.so'))]
    (target / 'LOCAL_ARTIFACTS.json').write_text(json.dumps(
        {'local_root': str(here), 'overlay': str(overlay), 'local_files': records, 'binary_links': binaries},
        ensure_ascii=False, indent=2) + '\n')
    (target / 'ARCHIVE_FILES.txt').write_text('\n'.join(files + ['LOCAL_ARTIFACTS.json', 'ARCHIVE_FILES.txt']) + '\n')
    print(json.dumps({'target': str(target), 'copied_files': len(files)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
