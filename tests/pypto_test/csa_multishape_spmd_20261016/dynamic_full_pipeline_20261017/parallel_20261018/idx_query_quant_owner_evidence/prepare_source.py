"""从正式668复制33源；只替换HAD→Query量化的一处等待，不编译。"""

import json
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
MAIN = Path('/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1')
BASE = '66882634'
PACKAGE = 'vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd'
OLD = '    signal = wait_row_owners(signal, PHASE_IDX_QUERY_HAD, CORES, generation, 0, tokens, 1, GROUPS, 0, 1)\n'
NEW = '''    # HAD每64个head行就是一个token，AIC按token=producer+24*w处理。
    # 本AIV只量化token=worker+48*w，因此所有wave的producer恒为worker%24；
    # 这里用逻辑AIV worker取模，不是同组AIC的worker//2，也不加AIV核编号偏移。
    # 生产者完成本核全部HAD wave后才发布；一次获取覆盖本AIV所有64×128读块。
    # worker>=T时量化循环为空，可跳过获取，但原QUANT空发布仍须保留。
    # 下方系数仍获取全部有效QUANT owner，Score/weights/compact的独立等待不动。
    if worker < tokens:
        signal = wait_producer(signal, PHASE_IDX_QUERY_HAD, worker % GROUPS, CORES, generation)
'''


def main():
    reference, candidate = HERE / 'source_reference', HERE / 'source_candidate'
    if reference.exists() or candidate.exists():
        raise FileExistsError('冻结源码目录已存在；拒绝覆盖')
    paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', BASE, PACKAGE], cwd=MAIN, text=True).splitlines()
    paths = [p for p in paths if p.endswith('.py')]
    assert len(paths) == 33 and all(Path(p).parent.as_posix() == PACKAGE for p in paths)
    reference.mkdir()
    candidate.mkdir()
    for path in paths:
        data = subprocess.check_output(['git', 'show', BASE + ':' + path], cwd=MAIN)
        (reference / Path(path).name).write_bytes(data)
        (candidate / Path(path).name).write_bytes(data)
    path = candidate / 'vector_schedule.py'
    text = path.read_text()
    assert text.count(OLD) == 1
    path.write_text(text.replace(OLD, NEW))
    (HERE / 'SOURCE_PREPARATION.json').write_text(json.dumps({
        'base_revision': BASE, 'file_count_each': 33, 'source_reference': str(reference),
        'source_candidate': str(candidate), 'changed_file': 'vector_schedule.py',
        'scope': '仅HAD→Query量化wait；Cube/RoPE→Had等待及全部数学不动；未编译/上卡',
    }, ensure_ascii=False, indent=2) + '\n')
    print(candidate)


if __name__ == '__main__':
    main()
