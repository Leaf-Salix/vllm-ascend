# SPDX-License-Identifier: Apache-2.0
"""T96长Indexer的输入行依赖合同；不改数学、分核或同步发布实现。

quant成员w写query w、w+48的全部64头；系数成员r读6个query，写一请求。
长Indexer在T<144时以leaf*B+request分配给24组，每组只读两个请求。
覆盖1..32叶的并集，不从本轮device长度猜测生产者；三种MIX成员保守地
等待同一组输入，仍保留原有同核先后和FFTS资源复用约束。
"""

import ast
from pathlib import Path


def build_contract(schedule, catalog):
    functions = {n.name: n for n in ast.walk(ast.parse(Path(catalog['body_bundle']).read_text()))
                 if isinstance(n, ast.FunctionDef)}
    checks = {
        'qr_hadamard_quant': [
            'pl.range(qh_quant_worker_inline2061_inline4051__ssa_v0, bs_heads_inline2066_inline4048__ssa_v0 // 64, 48, init_values=(qr_hadamard_i8_inline583__ssa_v0, qr_hadamard_scale_dq_inline582__ssa_v0))',
            'idx_inline2048_inline4038__idx_v0 * 64',
            'pl.tile.store(qh_scale_dq_inline2067_inline4031__tile, [o0_inline2074_inline4039__ssa_v1, 0], qr_hadamard_scale_dq_inline582__iter_v1)',
        ],
        'indexer_head_coefficients': [
            'pl.range(coefficient_worker_inline1288_inline8745__ssa_v0, coefficient_count_inline1283_inline8801__ssa_v0 // 6, 48)',
            'coefficient_pair_inline1289_inline8788__idx_v0 * 6',
            'pl.tile.load(coefficient_scales_inline1285_inline8894__ssa_v0, [coefficient_begin_inline1292_inline8729__ssa_v0, 0], [6, 64], [6, 64], target_memory=pl.Mem.Vec)',
            'pl.tile.store(coefficient_matrix_inline1284_inline8701__ssa_v0, [coefficient_pair_inline1289_inline8788__idx_v0 * 16, 0], coefficients_inline1281_inline8756__ssa_v0)',
        ],
        'indexer_score_topk_native_pair_aic': [
            'pl.range(buf_worker_inline8783__ssa_v0, buf_query_count_inline8792__ssa_v0 // 6 * buf_max_leaves_inline8794__ssa_v1, 24)',
            'buf_query_count_inline8792__ssa_v0 < 144',
            'buf_item_inline8803__idx_v0 % (buf_query_count_inline8792__ssa_v0 // 6) * 6',
            'pl.tile.load(qr_hadamard_i8_inline583__rv_v2, [buf_query_inline8778__phi_v2 * 64, 0], [384, 128], [384, 128], target_memory=pl.Mem.Mat)',
            'pl.tile.load(coefficients_inline8733__ssa_v0, [buf_query_inline8778__phi_v2 // 6 * 16, 0], [16, 384], [16, 384], target_memory=pl.Mem.Mat)',
        ],
    }
    for name, expected in checks.items():
        available = {ast.unparse(n) for n in ast.walk(functions[name])}
        if missing := set(expected) - available:
            raise ValueError(f'Indexer输入分片合同改变：{name}: {missing}')
    calls = schedule['call_bindings']
    ids = {}
    for name in ('qr_hadamard_quant', 'indexer_head_coefficients', 'indexer_score_topk_native_pair'):
        found = [tid for tid, c in calls.items() if c['logical_function'] == name]
        if len(found) != 1:
            raise ValueError('输入依赖不是唯一长Indexer调用：' + name)
        ids[name] = found[0]
    quant, coeff, idx = [ids[n] for n in ('qr_hadamard_quant', 'indexer_head_coefficients',
                                        'indexer_score_topk_native_pair')]
    bindings = {tid: {a['parameter']: a['value'] for a in calls[tid]['arguments']}
                for tid in ids.values()}
    tasks = {t['task_id']: t for t in schedule['tasks']}
    for tid, count in ((quant, 48), (coeff, 16), (idx, 24)):
        if tasks[tid]['block_num'] != count:
            raise ValueError('Indexer输入生产者分核改变')
    if bindings[quant]['bs_heads_inline2066_inline4048__ssa_v0'] != 96 * 64:
        raise ValueError('query/head实际数量改变')
    links = [
        (quant, 'qr_hadamard_i8_inline583__ssa_v0', idx, 'qr_hadamard_i8_inline583__rv_v2', False),
        (coeff, 'coefficients_inline1281_inline8756__ssa_v0', idx, 'coefficients_inline8733__ssa_v0', False),
        (quant, 'qr_hadamard_scale_dq_inline582__ssa_v0', coeff, 'coefficient_scales_inline1285_inline8894__ssa_v0', True),
    ]
    products = {}
    for producer, pn, consumer, cn, reshape in links:
        a, b = bindings[producer][pn], bindings[consumer][cn]
        for key in ('allocation', 'offset', 'dtype', 'layout'):
            if a[key] != b[key]:
                raise ValueError(f'输入依赖不共享同一GM映射：{pn}/{cn}/{key}')
        if reshape:
            # q*64+h 与 (q*64+h,0) 的FP32元素地址相同，不能只比较allocation。
            if (a['shape'], a['strides'], b['shape'], b['strides'], a['offset']) != (
                    [24576, 1], [1, 1], [384, 64], [64, 1], 0):
                raise ValueError('scale平铺视图改变')
        elif any(a[k] != b[k] for k in ('shape', 'strides')):
            raise ValueError('Indexer输入连续行视图改变')
        products[producer, consumer] = {a['allocation']}
    # 独立按query写者枚举：每个query唯一写者且完整256B scale区不共享缓存行。
    written = {w: set(range(w, 96, 48)) for w in range(48)}
    assert sum(map(len, written.values())) == len(set.union(*written.values())) == 96
    quant_for_coeff = {r: sorted({q % 48 for q in range(r * 6, r * 6 + 6)}) for r in range(16)}
    requests = {w: sorted({item % 16 for item in range(w, 16 * 32, 24)}) for w in range(24)}
    quant_for_idx = {w: sorted({q % 48 for r in rs for q in range(r * 6, r * 6 + 6)})
                     for w, rs in requests.items()}
    cases = 0
    for leaf_count in range(1, 33):
        for w in range(24):
            actual_requests = {item % 16 for item in range(w, 16 * leaf_count, 24)}
            reads = {q for r in actual_requests for q in range(r * 6, r * 6 + 6)}
            assert actual_requests <= set(requests[w])
            assert reads <= set.union(*(written[p] for p in quant_for_idx[w]))
            assert len(requests[w]) == 2 and len(quant_for_idx[w]) == 6
            cases += 1
    for r, writers in quant_for_coeff.items():
        assert set(range(r * 6, r * 6 + 6)) <= set.union(*(written[w] for w in writers))
    return {
        'products': products,
        'edges': [
            {'parent': quant, 'consumer': coeff, 'workers': quant_for_coeff,
             'reason': '一个请求的六query scale写者；平铺地址逐元素等价'},
            {'parent': quant, 'consumer': idx, 'workers': quant_for_idx,
             'reason': '32种叶数并集内的两个请求；六个quant成员各写两个query'},
            {'parent': coeff, 'consumer': idx, 'workers': requests,
             'reason': '32种叶数并集内的两个请求系数写者；MIX三成员保持同一输入要求'},
        ],
        'audit': {'source_contracts': checks, 'dynamic_leaf_worker_cases': cases,
                  'max_leaf_count': 32, 'query_rows': 96, 'scale_bytes_per_query': 256,
                  'coefficient_bytes_per_request': 12288, 'same_core_order_retained': True,
                  'input_and_output_math_changed': False, 'device_validated': False},
    }
