"""HC_pre token优先投影布局的静态成本与地址推导；不运行算子/设备测试。"""
import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'token_major_projection_v2.py'
BUILD = ROOT / 'build_token_major_projection_v2'
tree = ast.parse(SOURCE.read_text())
variants = {}
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == '_hc_pre_variant':
        params = [ast.literal_eval(x) for x in node.args[7:]]
        variants[params[6].removesuffix('_spmd')] = {
            'M': params[0], 'K': params[2], 'tail': params[5], 'rms': params[8],
        }


def variant_for(t):
    return ('hc_small' if t <= 32 else 'hc_medium48' if t <= 48 else
            'hc_medium96' if t < 96 else 'hc_exact96' if t == 96 else
            'hc_large144' if t <= 144 else 'hc_large')


def windows(t, tail):
    sink = min(t, 24)
    pairs = (t + 1) // 2
    mix = min(48 - sink, pairs * 8)
    result = []
    for i in range(sink):
        begin = i * (t // sink) + min(i, t % sink)
        end = begin + t // sink + (i < t % sink)
        for pos in range(begin, end, tail):
            result.append(('sink', pos, min(tail, end - pos), 16))
    for i in range(mix):
        begin = (i * (pairs // mix) + min(i, pairs % mix)) * 2
        end = min(t, begin + (pairs // mix + (i < pairs % mix)) * 2)
        if mix > pairs:
            q, extra = divmod(mix, pairs)
            prefix = extra * (q + 1)
            pair = i // (q + 1) if i < prefix else extra + (i - prefix) // q
            begin = min(t, pair * 2)
            end = min(t, begin + 2)
        for pos in range(begin, end, tail):
            result.append(('mix', pos, min(tail, end - pos), 8))
    return result


def permutation(k_count, rows, cols):
    groups = cols // 8
    count = k_count * rows * groups
    aligned = (count + 7) // 8 * 8
    offsets = []
    matches = True
    for raw in range(aligned):
        index = min(raw, count - 1)
        out_k = index >> (rows // 2 - rows // 8 + groups - 1)
        rest = index - (out_k << (rows // 2 - rows // 8 + groups - 1))
        token = rest >> (groups - 1)
        component = rest - (token << (groups - 1))
        token_k = ((token << 3) + (token << 1) + token if k_count == 11 else
                   token << 4 if k_count == 16 else token << 3)
        actual = ((token_k + out_k) << (groups - 1)) + component
        expected_k, rem = divmod(index, rows * groups)
        expected_token, expected_component = divmod(rem, groups)
        expected = (expected_token * k_count + expected_k) * groups + expected_component
        matches = matches and actual == expected
        offsets.append(actual * 32)
    return {
        'K': k_count, 'rows': rows, 'cols': cols,
        'index_count': aligned, 'real_blocks': count,
        'matches_token_to_k_major_mapping': matches,
        'all_offsets_32B_aligned': all(x % 32 == 0 for x in offsets),
        'all_gathered_blocks_in_UB': all(0 <= x and x + 32 <= count * 32 for x in offsets),
        'clamped_extra_index_count': aligned - count,
    }


cases = []
permutations = {}
for b in [2, 4, 8, 16, 20, 24, 28]:
    for s in [5, 6]:
        t = b * s
        name = variant_for(t)
        params = variants[name]
        m, k, rows = params['M'], params['K'], params['tail']
        linear_t = (t + m - 1) // m * m
        # 以128B投影行段为最小所有权单元：不同producer不共享32B块。
        owners = {}
        overlaps = 0
        for task in range(linear_t // m * k):
            token_base = task // k * m
            split = task % k
            for token in range(token_base, token_base + m):
                address = token * k + split
                if address in owners:
                    overlaps += 1
                owners[address] = task % 24
        work = windows(t, rows)
        old_bytes = new_bytes = 0
        bounded = True
        for role, begin, valid, cols in work:
            old_bytes += k * valid * 32 * 4
            new_bytes += k * valid * cols * 4
            bounded = bounded and 0 <= begin and begin + valid <= t <= linear_t
            key = (k, rows, cols)
            permutations.setdefault(key, permutation(*key))
        cases.append({
            'B': b, 'S': s, 'T': t, 'variant': name,
            'linear_T': linear_t, 'K_splits': k, 'tail_rows': rows,
            'all_projection_rows_written_once': overlaps == 0 and len(owners) == linear_t * k,
            'producers_sharing_32B_blocks': overlaps,
            'consumer_valid_windows_in_allocation': bounded,
            'projection_allocation_bytes_unchanged': linear_t * k * 32 * 4,
            'projection_GM_read_bytes_before': old_bytes,
            'projection_GM_read_bytes_candidate': new_bytes,
            'projection_DMA_issues_before': len(work),
            'projection_DMA_issues_3d_candidate': len(work) * k,
            'projection_DMA_issues_token_major': len(work),
            'AIC_fixpipe_stores_before_and_after': linear_t // m * k,
            'AIC_destination_row_stride_fp32': k * 32,
        })

codegen = {}
for name in variants:
    path = BUILD / 'build/ptoas' / f'{name}.cpp'
    cpp = path.read_text()
    streams = {}
    for prefix in ('c', 'm'):
        begin = cpp.index(f'// pto: %{prefix}_role_load_')
        end = cpp.index(f'// pto: %{prefix}_role_total_', begin)
        section = cpp[begin:end]
        calls = re.findall(r'^\s*((?:TLOAD|TGATHERB|TCOLSUM|TSHRS|TSHLS|TFILLPAD|wait_flag|set_flag)[^;]*;)', section, re.M)
        first_shift = section.index('TSHRS(')
        first_consumer_wait = section.index('wait_flag(PIPE_MTE2, PIPE_V')
        streams[prefix] = {
            'two_GM_loads_before_index_math': section[:first_shift].count('TLOAD(') == 2,
            'index_math_before_projection_MTE2_wait': first_shift < first_consumer_wait,
            'instruction_sequence': calls,
        }
    codegen[name] = streams

report = {
    'status': 'STATIC_ANALYSIS_ONLY', 'source': SOURCE.name,
    'compile_status': json.loads((BUILD / 'report.json').read_text())['status'],
    'device_execution': False,
    'scope': '14个场景的所有权、读取边界和搬运次数推导；六分支CPU生成代码审查',
    'assumptions': ['核组数24，用于任务数量推导；算子启动仍查询设备核组数',
                    'DMA issue数量依本地TLoadGm2ubNd2nd实现计算，不是实测耗时',
                    'gatherb块置换次序正确不等于端到端精度已通过'],
    'code_sources': [
        'pto-isa-src/include/pto/common/arch/memory/tload_common.hpp:78',
        'pto-isa-src/include/pto/common/arch/memory/tstore_common.hpp:185',
        'pypto/src/ir/op/tile_ops/gather.cpp:227',
    ],
    'cases': cases, 'permutations': list(permutations.values()), 'codegen': codegen,
    'remaining_costs': ['UB索引构造、一次gatherb、补齐归约结果',
                        'Cube输出行距增大，带步长写回的实际耗时尚未测量'],
    'performance_claim': None, 'precision_claim': None,
}
output = ROOT / 'token_major_projection_static_audit.json'
output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(output)
print('场景数:', len(cases), '分支数:', len(codegen), '块置换配置数:', len(permutations))
print('所有投影写入独占:', all(c['all_projection_rows_written_once'] for c in cases))
print('所有读取边界有效:', all(c['consumer_valid_windows_in_allocation'] for c in cases))
print('所有置换/偏移范围成立:', all(x['matches_token_to_k_major_mapping'] and x['all_gathered_blocks_in_UB'] for x in permutations.values()))
print('所有分支在MTE2等待前构造索引:', all(x['index_math_before_projection_MTE2_wait'] for v in codegen.values() for x in v.values()))
