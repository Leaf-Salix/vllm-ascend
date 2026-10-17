"""Q_B 三种权重复用设计的有限只读模型；不导入 PyPTO、不编译或上卡。

只复算逻辑请求字节、调用数量与所有权，读取已有真实 PTO。字节不是 HBM
测量，调用数不是时延。正式源、依赖和冻结设备源均不写入，不做内容摘要校验。
"""

import ast
import json
import re
import subprocess
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = Path('/data/pyptouser/qinchuanyu/pto-eager')
MAIN = ROOT / 'vllm-ascend-dsv4-pto-0251rc1'
PKG = MAIN / 'vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd'
QA = ROOT / '.worktrees/csa-ultra-sync-20261018/tests/pypto_test/ultra_qa_small_m16_v2_20261018'
PTO = QA / 'cpu_true_v1/kernel/ptoas/dynamic_csa_full.pto'
CPP = QA / 'cpu_true_v1/kernel/kernels/aic/cube_schedule.cpp'
WORKERS, N, K, K_TILE, M_TILE = 24, 32768, 1024, 256, 128
MIB = 1024 * 1024


def git_source(revision, path):
    return subprocess.check_output(['git', 'show', f'{revision}:{path}'], cwd=MAIN, text=True)


def baseline(tokens):
    mblocks = (tokens + M_TILE - 1) // M_TILE
    nblocks = N // 256
    weight = mblocks * K * N
    qr = nblocks * tokens * K
    return {
        'm': 128, 'n': 256, 'm_blocks': mblocks,
        'gm_weight_bytes': weight, 'gm_qr_bytes': qr,
        'gm_input_bytes': weight + qr,
        'gm_output_bytes': tokens * N * 4,
        'l1_to_right_logical_bytes': weight,
        'l1_to_left_logical_bytes': qr,
        'weight_tload_calls': nblocks * mblocks * 4,
        'qr_tload_calls': nblocks * mblocks * 4,
        'mma_calls': nblocks * mblocks * 4,
        'fix_store_calls': nblocks * mblocks,
        'max_valid_output_elements_per_core': 6 * 256 * tokens,
        'math_mac_count': tokens * K * N,
    }


def two_m(tokens):
    if not 128 < tokens <= 256:
        return None
    d = baseline(tokens)
    d.update(m=128, n=128, gm_weight_bytes=K*N, gm_qr_bytes=256*tokens*K,
             l1_to_right_logical_bytes=K*N, l1_to_left_logical_bytes=256*tokens*K,
             weight_tload_calls=256*4, qr_tload_calls=256*2*4,
             mma_calls=256*2*4, fix_store_calls=256*2,
             max_valid_output_elements_per_core=11*128*tokens)
    d['gm_input_bytes'] = d['gm_weight_bytes'] + d['gm_qr_bytes']
    d['right_reuse_assumption'] = '每 K 的同一个 Right tile 被两个 M 的 matmul 消费；未生成验证'
    return d


def b_resident(tokens):
    if tokens <= 128:
        return None
    d = baseline(tokens)
    d.update(gm_weight_bytes=K*N, weight_tload_calls=N//256)
    d['gm_input_bytes'] = d['gm_weight_bytes'] + d['gm_qr_bytes']
    return d


def qr_b_resident(tokens):
    if not 128 < tokens <= 192:
        return None
    d = baseline(tokens)
    d.update(m=64, m_blocks=3, gm_weight_bytes=K*N, gm_qr_bytes=WORKERS*tokens*K,
             l1_to_right_logical_bytes=3*K*N,
             weight_tload_calls=N//256, qr_tload_calls=WORKERS,
             mma_calls=(N//256)*3*4, fix_store_calls=(N//256)*3)
    d['gm_input_bytes'] = d['gm_weight_bytes'] + d['gm_qr_bytes']
    return d


def coverage():
    # 行与列的分区独立，分别枚举即可证明笛卡尔积上的每个有效元素唯一写。
    for width in (128, 256):
        counts = Counter()
        for worker in range(WORKERS):
            for ni in range(worker, N//width, WORKERS):
                counts.update(range(ni*width, (ni+1)*width))
        assert len(counts) == N and set(counts.values()) == {1}
    for tokens in range(1, 513):
        for rows in (128, 64):
            counts = Counter()
            for t0 in range(0, tokens, rows):
                counts.update(range(t0, t0+min(rows, tokens-t0)))
            assert len(counts) == tokens and set(counts.values()) == {1}
        if 128 < tokens <= 192:
            for row in (0, 64, 128):
                assert row + 64 <= 192
                assert 0 < min(64, tokens-row) <= 64
    return {'tokens_enumerated': [1, 512], 'unique_row_and_column_partitions': True,
            'qr192_m64_physical_windows_fit': True,
            'k_sequence_for_every_accumulator': [0, 256, 512, 768],
            'int32_absolute_dot_upper_bound': 1024*128*128,
            'device_correctness_test': False}


def generated_evidence():
    lines = PTO.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if 'scf.for %qproj_round' in line)
    end = next(i for i in range(start, len(lines)) if 'pto.fence.barrier_all' in lines[i])
    pattern = re.compile(r'pto.alloc_tile addr = %c(\d+)_i64.*loc=(\w+), dtype=(\w+), rows=(\d+), cols=(\d+)')
    allocation = []
    for index in range(start, end):
        match = pattern.search(lines[index])
        if match:
            address, space, dtype, rows, cols = match.groups()
            size = int(rows)*int(cols)*{'i8': 1, 'i32': 4}[dtype]
            allocation.append(dict(line=index+1, space=space, rows=int(rows), cols=int(cols),
                                   address=int(address), bytes=size, end=int(address)+size))
    upper = {}
    for line in lines[:next(i for i, s in enumerate(lines) if 'func.func @vector_schedule' in s)]:
        match = pattern.search(line)
        if match:
            address, space, dtype, rows, cols = match.groups()
            unit = {'i8': 1, 'i16': 2, 'i32': 4, 'i64': 8, 'f16': 2, 'bf16': 2, 'f32': 4}.get(dtype)
            if unit:
                upper[space] = max(upper.get(space, 0), int(address)+int(rows)*int(cols)*unit)
    (HERE/'existing_qb.pto.txt').write_text('\n'.join(f'{i+1}: {lines[i].split(" loc(")[0]}' for i in range(start, end))+'\n')
    cpp_lines = CPP.read_text().splitlines()
    (HERE/'existing_qb.cpp.txt').write_text('\n'.join(f'{i+1}: {cpp_lines[i]}' for i in range(9008, 9445))+'\n')
    return {'pto': str(PTO), 'cpp': str(CPP), 'pto_lines': [start+1, end],
            'allocations': allocation, 'whole_aic_maximum_address_end': upper,
            'qb_l1_two_weight_and_two_qr_live_bytes': 2*65536+2*32768,
            'scope': '真实已有 QA CPU 产物；Q_B 源全文等于正式66882634，不是新候选生成证明'}


def main():
    source = (PKG/'query_projection_stages.py').read_text()
    assert source == git_source('66882634', 'vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/query_projection_stages.py')
    assert source == (QA/'source_candidate/query_projection_stages.py').read_text()
    fn = next(x for x in ast.parse(source).body if isinstance(x, ast.FunctionDef) and x.name == '_q_b_nz_stage')
    source_lines = source.splitlines()
    (HERE/'formal_qb_source.txt').write_text('\n'.join(f'{i+1}: {source_lines[i]}' for i in range(fn.lineno-1, fn.end_lineno))+'\n')
    historical_path = 'vllm_ascend/ops/pypto/deepseek_v4_flash_dspark/q_projection.py'
    old = git_source('2de2baa6', historical_path).splitlines()
    stream = git_source('74be8a6c', historical_path).splitlines()
    (HERE/'historical_qb_source.txt').write_text(
        '2de2baa6:'+historical_path+'\n'+ '\n'.join(f'{i+1}: {old[i]}' for i in range(80, 109))+
        '\n\n74be8a6c:'+historical_path+'\n'+ '\n'.join(f'{i+1}: {stream[i]}' for i in range(84, 119))+'\n')
    cases = []
    for t in (24, 48, 96, 128, 129, 144, 168, 192, 193, 255, 256, 257, 384, 512):
        cases.append({'tokens': t, 'baseline': baseline(t), 'two_m_n128': two_m(t),
                      'b_resident_m128_n256': b_resident(t), 'qr192_b_resident_m64_n256': qr_b_resident(t)})
    evidence = {'base_revision': '66882634', 'no_kernel_compilation_or_device_execution': True,
                'formal_qb_text_equal_to_existing_compiled_source': True,
                'coverage': coverage(), 'cases': cases, 'existing_generated': generated_evidence(),
                'modeled_live_bytes': {
                    'two_m_n128': {'mat': 196608, 'acc': 131072, 'right_one': 32768, 'left_one': 32768},
                    'b_resident_m128_n256': {'mat': 327680, 'acc': 131072, 'right_one': 65536, 'left_one': 32768},
                    'qr192_b_resident_m64_n256': {'mat': 458752, 'acc': 65536, 'right_one': 65536, 'left_one': 16384}},
                'recommendation': '若继续有界实现，优先仅 B 全 K 跨 M 常驻；不是性能采用结论。其它两案本轮不推进。',
                'limitations': ['不把逻辑 GM 字节当作实测 HBM', '不把调用数或有效 MAC 当作时间',
                                '尚无新候选实际 allocation/同步/设备证据',
                                '完整 B 跨 M 常驻是已有原理，当前 M128/QR流水/动态区间的独立消融未找到']}
    (HERE/'evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'passed': True, 'scope': '只读源码+有限整数模型', 'directory': str(HERE),
                      'cases': len(cases), 'whole_aic_maximum_address_end': evidence['existing_generated']['whole_aic_maximum_address_end']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
