"""HC_pre第一阶段8行均分候选的静态分工/地址/生成代码分析，不运行设备。"""
import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'aligned_stage1_rows_v4.py'
BASE = ROOT / 'adopted_l0_interleave_exact96.py'
BUILD = ROOT / 'build_aligned_stage1_rows_v4'
OLD_BUILD = ROOT / 'build_l0_interleave_exact96_v1'
variants = {}
for node in ast.walk(ast.parse(SOURCE.read_text())):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == '_hc_pre_variant':
        args = [ast.literal_eval(x) for x in node.args[7:]]
        variants[args[6].removesuffix('_spmd')] = dict(M=args[0], Kwidth=args[1], K=args[2], static=args[7], rms=args[8])


def variant_for(t):
    return ('hc_small' if t <= 32 else 'hc_medium48' if t <= 48 else
            'hc_medium96' if t < 96 else 'hc_exact96' if t == 96 else
            'hc_large144' if t <= 144 else 'hc_large')


def distribution(t):
    name = variant_for(t)
    p = variants[name]
    m, k = p['M'], p['K']
    blocks = (t + m - 1) // m
    linear_t = blocks * m
    balanced = p['static'] == 0 and p['Kwidth'] == 2048
    groups = (t + 7) // 8
    q, extra = divmod(groups, blocks)
    projection = {}
    rms = {}
    rms_owners = {}
    projection_overlap = 0
    rms_overlap = 0
    chunks = []
    for block in range(blocks):
        begin = (block * q + min(block, extra)) * 8 if balanced else block * m
        owned = (q + (block < extra)) * 8 if balanced else m
        real = max(0, min(owned, t - begin))
        chunks.append(dict(begin=begin, owned_rows=owned, effective_rows=real,
                           AIV_effective_rows=[max(0, min(m // 2, real - lane * (m // 2))) for lane in (0, 1)],
                           cube_valid_rows=owned,
                           cube_ceil16_rows=(owned + 15) // 16 * 16))
        for split in range(k):
            task = block * k + split
            for row in range(begin, begin + owned):
                key = (split, row)
                projection_overlap += key in projection
                projection[key] = task
            # 精确T96每个K段分别写两份RMS；其它分支K/RMS段数一致。
            for rms_sub in range(p['rms'] // k):
                rms_split = split * (p['rms'] // k) + rms_sub
                for lane in (0, 1):
                    row_begin = begin + lane * (m // 2)
                    row_count = max(0, min(m // 2, owned - lane * (m // 2)))
                    for row in range(row_begin, row_begin + row_count):
                        key = (rms_split, row)
                        rms_overlap += key in rms
                        rms[key] = (task, lane, rms_sub)
                        block32 = (rms_split * linear_t + row) // 8
                        rms_owners.setdefault(block32, set()).add((task, lane, rms_sub))
    return dict(
        T=t, variant=name, balanced=balanced, linear_T=linear_t,
        logical_tasks_unchanged=blocks * k,
        before_effective_rows=[max(0, min(m, t - b * m)) for b in range(blocks)],
        after=chunks,
        complete_projection_rows=all((split, row) in projection for split in range(k) for row in range(t)),
        complete_rms_rows=all((split, row) in rms for split in range(p['rms']) for row in range(t)),
        projection_overlapping_rows=projection_overlap,
        rms_overlapping_rows=rms_overlap,
        rms_blocks_shared_by_different_producers=sum(len(owners) > 1 for owners in rms_owners.values()),
        all_writes_inside_allocation=all(0 <= row < linear_t for _, row in projection) and all(0 <= row < linear_t for _, row in rms),
        only_unconsumed_rows_unwritten=all(row >= t for split in range(k) for row in range(linear_t) if (split, row) not in projection),
    )


def normalize(cpp):
    cpp = re.sub(r'/\*.*?\*/', '', cpp, flags=re.S)
    cpp = re.sub(r'//[^\n]*', '', cpp)
    return '\n'.join(x.strip() for x in cpp.splitlines() if x.strip())


codegen = {}
for name in variants:
    cpp = (BUILD / 'build/ptoas' / f'{name}.cpp').read_text()
    base_cpp = (OLD_BUILD / 'build/ptoas' / f'{name}.cpp').read_text()
    item = dict(complete_cpp_equal_to_formal=normalize(cpp) == normalize(base_cpp))
    if name in ('hc_medium96', 'hc_large144', 'hc_large'):
        aic = cpp.split('#if defined(__DAV_VEC__)')[0]
        stores = []
        for call in re.finditer(r'TSTORE\((\w+), (\w+)\);', aic):
            dst, src = call.groups()
            before = aic[:call.start()]
            tile = list(re.finditer(r'Tile<([^>]+)> ' + src + r' = Tile<[^>]+>\(([^;]*)\);', before))[-1]
            fields = [x.strip() for x in tile.group(1).split(',')]
            tensor = list(re.finditer(r'GlobalTensor<float, pto::Shape<([^>]+)>,[^;]*? ' + dst + r' =', before))[-1]
            shape = [x.strip() for x in tensor.group(1).split(',')]
            stores.append(dict(source=src, destination=dst,
                               physical_M=fields[2], physical_N=fields[3],
                               source_valid_N=fields[6], destination_valid_N=shape[4],
                               valid_N_agrees=fields[6] == shape[4] == '24',
                               compact_acc=fields[-1] == 'CompactMode::Normal',
                               source_valid_M_symbol=tile.group(2)))
        item['AIC_projection_stores'] = stores
        item['two_slot_fifo_retained'] = bool(re.search(r'TPipe<[^>]*, 2, 2, false>', cpp))
        item['RMS_write_has_own_valid_window'] = 'rms_write_' in cpp
    codegen[name] = item

cases = [dict(B=b, S=s, **distribution(b*s)) for b in (2,4,8,16,20,24,28) for s in (5,6)]
checks = ('complete_projection_rows', 'complete_rms_rows', 'all_writes_inside_allocation', 'only_unconsumed_rows_unwritten')
report = dict(
    status='STATIC_ANALYSIS_ONLY', candidate=SOURCE.name, parent=BASE.name,
    compile_status=json.loads((BUILD / 'report.json').read_text())['status'],
    device_execution=False,
    scope='14个场景的分工与地址推导，以及六分支生成代码核对；不是精度/性能测试',
    second_stage_source_unchanged=SOURCE.read_text().split('        # 【第二阶段：')[1] == BASE.read_text().split('        # 【第二阶段：')[1],
    cases=cases, codegen=codegen,
    all_address_conditions_hold=all(all(c[key] for key in checks) and c['projection_overlapping_rows'] == 0 and c['rms_overlapping_rows'] == 0 and c['rms_blocks_shared_by_different_producers'] == 0 for c in cases),
    limitations=[
        'FIFO仍是固定物理M，两个AIV仍各占固定半块，尚未消除所有组内AIV负载差。',
        'MMA按硬件粒度执行；valid M变小不能直接换算成比例相同的性能收益。',
        '紧凑Acc行距、动态有效M及输出数值仍需后续设备验证。',
        '不改变原完整L1计时、FP32累加/RMS顺序、精度门禁或20us目标。',
    ],
)
out = ROOT / 'aligned_stage1_rows_static_audit.json'
out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(out)
print('所有静态地址条件成立:', report['all_address_conditions_hold'])
print('第二阶段源码完全相同:', report['second_stage_source_unchanged'])
print('独立保留的原分支:', [name for name, v in codegen.items() if v['complete_cpp_equal_to_formal']])
print('全部Cube写回有效N一致:', all(s['valid_N_agrees'] for v in codegen.values() for s in v.get('AIC_projection_stores', [])))
for c in cases:
    if c['balanced']:
        print(c['B'], c['S'], c['before_effective_rows'], '->', [x['effective_rows'] for x in c['after']])
