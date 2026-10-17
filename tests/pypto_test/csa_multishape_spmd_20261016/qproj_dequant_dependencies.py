# SPDX-License-Identifier: Apache-2.0
"""B16主Q后处理分块的GM映射与完整成员依赖合同；先用于CPU筛选。

只允许原N256/20投影worker和48个后处理worker。数学分块改变时必须重新
枚举每个8行/head块；既有同核顺序、其他task前驱和FFTS组复用全部保留。
"""

import ast
import collections
import copy
import json
from pathlib import Path

from member_schedule import _costs, _estimate, _problem, _resource_parents, _waits, ancestor_sets


def contract(schedule, catalog, head_tile):
    if head_tile not in (4, 8, 16):
        raise ValueError("只分析已枚举的4/8/16-head分块")
    functions = {n.name: n for n in ast.walk(ast.parse(Path(catalog['body_bundle']).read_text()))
                 if isinstance(n, ast.FunctionDef)}
    checks = {
        'qproj_matmul': [
            '(19 - qproj_worker_inline8514__ssa_v0 + 128) // 20',
            'qproj_worker_inline8514__ssa_v0 + qproj_round_inline8515__idx_v0 * 20',
            'qproj_n_idx_inline8522__ssa_v0 * 256',
            'pl.range(0, qproj_t_matmul_inline8513__ssa_v0, 128, init_values=(q_proj_i32_inline7_inline3585__iter_v1,))',
            'pl.min(tile_rows_inline11_inline3587__ssa_v0 - t0_inline8524__idx_v0, 128)',
            'pl.tile.store(col_acc_inline8523__rv_v2, [t0_inline8524__idx_v0, w_col0_inline8519__ssa_v0], q_proj_i32_inline7_inline3585__iter_v3)',
        ],
        'qproj_dequant_rms_nope_rope': [
            '96 <= tile_rows_inline11_inline3587__ssa_v0 and tile_rows_inline11_inline3587__ssa_v0 % 96 == 0',
            'pl.range(dq_worker_inline8595__ssa_v0, (tile_rows_inline11_inline3587__ssa_v0 + 7) // 8 * (64 // dq_head_tile_inline8559__phi_v4), 48)',
            'dq_work_inline8592__idx_v0 % (64 // dq_head_tile_inline8559__phi_v4) * dq_head_tile_inline8559__phi_v4',
            'dq_work_inline8592__idx_v0 // (64 // dq_head_tile_inline8559__phi_v4) * 8',
            'tile_base_inline12_inline3596__idx_v0 + tg_inline8570__ssa_v0',
            'hg_inline8542__ssa_v0 + h_inner_inline8546__idx_v0',
            'h_inline8564__ssa_v0 * 512',
            'pl.tile.load(q_proj_i32_inline7_inline3585__ssa_v6, [tg_inline8570__ssa_v0, h0_inline8544__ssa_v0], [8, 512], [8, 512], target_memory=pl.Mem.Vec)',
            'pl.tile.store(q_nope_bf16_inline8588__ssa_v0, [out_tg_inline8549__ssa_v0, h0_inline8544__ssa_v0], q_flat_inline8548__ssa_v0)',
            'pl.tile.store(q_rope_bf16_inline8596__ssa_v0, [out_tg_inline8549__ssa_v0, h0_inline8544__ssa_v0 + 448], q_flat_inline8548__ssa_v0)',
        ],
        'qk_pv_aic': [
            '(t_dim_inline4335__ssa_v0 - qk_core_inline4347__ssa_v0 + 23) // 24',
            'qk_core_inline4347__ssa_v0 + qk_tick_inline4364__idx_v0 // 5 * 24',
            'pl.tile.gather_row(qk_q_inline4406__iter_v1, q_flat_inline4389__ssa_v0, [0, 0], [qk_t_inline4408__ssa_v0 * 64, 0], [64, 512], transpose=False)',
        ],
    }
    for name, expressions in checks.items():
        present = {ast.unparse(n) for n in ast.walk(functions[name])}
        if missing := set(expressions) - present:
            raise ValueError(f'主Q分片公式改变：{name}: {missing}')
    # 后处理仅写Q；投影仅写INT32临时区。核内Tile写不跨任务，GM Scalar写不允许新增。
    store_targets = {}
    for name, expected in [('qproj_matmul', {'q_proj_i32_inline7_inline3585__iter_v3'}),
                           ('qproj_dequant_rms_nope_rope', {'q_flat_inline8548__ssa_v0'})]:
        calls = [n for n in ast.walk(functions[name]) if isinstance(n, ast.Call)]
        if any(ast.unparse(n.func) in ('pl.tensor.write', 'pl.write', 'pl.tile.scatter_update',
                                      'pl.tile.mscatter', 'pl.tile.gather_row') for n in calls):
            raise ValueError('主Q阶段出现新的写副作用')
        targets = {ast.unparse(n.args[2]) for n in calls if ast.unparse(n.func) == 'pl.tile.store'}
        if targets != expected:
            raise ValueError('主Q阶段GM写目标改变')
        store_targets[name] = sorted(targets)
    ids = {}
    for name in ('qproj_matmul', 'qproj_dequant_rms_nope_rope', 'qk_pv'):
        matching = [tid for tid, call in schedule['call_bindings'].items()
                    if call['logical_function'] == name]
        if len(matching) != 1:
            raise ValueError('不是唯一T96调用：' + name)
        ids[name] = matching[0]
    projection, dequant, qk = [ids[n] for n in ('qproj_matmul', 'qproj_dequant_rms_nope_rope', 'qk_pv')]
    tasks = {t['task_id']: t for t in schedule['tasks']}
    bindings = {tid: {a['parameter']: a['value'] for a in schedule['call_bindings'][tid]['arguments']}
                for tid in ids.values()}
    for tid, count in ((projection, 20), (dequant, 48), (qk, 24)):
        if tasks[tid]['block_num'] != count:
            raise ValueError('主Q或QK逻辑核数改变')
    if not (projection in tasks[dequant]['predecessors'] and dequant in tasks[qk]['predecessors']):
        raise ValueError('待收窄的两条原task依赖不存在')
    if (bindings[projection]['tile_rows_inline11_inline3587__ssa_v0'],
            bindings[projection]['qproj_t_matmul_inline8513__ssa_v0'],
            bindings[dequant]['tile_rows_inline11_inline3587__ssa_v0'],
            bindings[dequant]['tile_base_inline12_inline3596__idx_v0'],
            bindings[qk]['t_dim_inline4335__ssa_v0']) != (96, 128, 96, 0, 96):
        raise ValueError('本合同仅覆盖真实T96/单tile/零基址')
    a = bindings[projection]['q_proj_i32_inline7_inline3585__ssa_v0']
    b = bindings[dequant]['q_proj_i32_inline7_inline3585__ssa_v6']
    if any(a[k] != b[k] for k in ('allocation', 'offset', 'shape', 'strides', 'dtype', 'layout')):
        raise ValueError('INT32 Q生产消费地址不一致')
    if (a['shape'], a['strides'], a['offset'], a['dtype'], a['layout']) != (
            [128, 32768], [32768, 1], 0, 'INT32', 'ND'):
        raise ValueError('INT32 Q布局改变')
    c = bindings[dequant]['q_flat_inline8548__ssa_v0']
    d = bindings[qk]['q_flat_inline4389__ssa_v0']
    if any(c[k] != d[k] for k in ('allocation', 'offset', 'dtype', 'layout')):
        raise ValueError('BF16 Q不共用相同基址')
    if (c['shape'], c['strides'], d['shape'], d['strides'], c['offset'], c['dtype'], c['layout']) != (
            [96, 32768], [32768, 1], [6144, 512], [512, 1], 0, 'BF16', 'ND'):
        raise ValueError('QK的Q reshape不连续')
    shared = {}
    for parent, child, product in ((projection, dequant, a['allocation']), (dequant, qk, c['allocation'])):
        refs = [{v['allocation'] for v in bindings[tid].values()
                 if isinstance(v, dict) and 'allocation' in v} for tid in (parent, child)]
        overlap = refs[0] & refs[1]
        extras = overlap - {product}
        if any(not name.startswith('argument:') for name in extras):
            raise ValueError('收窄边还共享其他临时产物')
        shared[f'{parent}->{child}'] = {'product': product, 'shared_readonly_roots': sorted(extras)}
    # 每个N256块由唯一投影worker写入全部96有效行；每个head由两个N块组成。
    projected = {n: worker for worker in range(20) for n in range(worker, 128, 20)}
    assert len(projected) == 128
    writers, dequant_parents, loops = {}, {}, {}
    for worker in range(48):
        needed, count = set(), 0
        for unit in range(worker, 12 * (64 // head_tile), 48):
            row = unit // (64 // head_tile)
            first = unit % (64 // head_tile) * head_tile
            for head in range(first, first + head_tile):
                if (row, head) in writers:
                    raise ValueError('后处理行/head被重复写入')
                writers[row, head] = worker
                needed.update(projected[head * 2 + part] for part in range(2))
                count += 1
        dequant_parents[worker] = sorted(needed)
        loops[worker] = count
    if len(writers) != 12 * 64 or set(loops.values()) != {16}:
        raise ValueError('后处理数学覆盖或每worker工作量改变')
    qk_parents = {worker: sorted({writers[q // 8, head] for q in range(worker, 96, 24)
                                 for head in range(64)}) for worker in range(24)}
    # 按每个head块的两端验证两个视图的线性地址；地址函数对列是相同的一次式。
    checked = 0
    for query in range(96):
        for head in range(64):
            for column in (0, 447, 448, 511):
                assert query * c['strides'][0] + (head * 512 + column) * c['strides'][1] == (
                    (query * 64 + head) * d['strides'][0] + column * d['strides'][1])
                checked += 1
    return {'projection': projection, 'dequant': dequant, 'qk': qk,
            'dequant_parents': dequant_parents, 'qk_parents': qk_parents,
            'audit': {'head_tile': head_tile, 'source_contracts': checks, 'gm_store_targets': store_targets,
                      'shared_views': shared, 'linear_address_checks': checked,
                      'head_row_blocks_exactly_once': len(writers), 'head_iterations_per_worker': 16,
                      'dequant_producer_counts': sorted({len(x) for x in dequant_parents.values()}),
                      'qk_dequant_parent_counts': sorted({len(x) for x in qk_parents.values()}),
                      'source_head_tile_may_differ_for_cpu_model': True, 'device_validated': False}}


def model(root, output):
    from shape_specialization import catalog

    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    schedule = json.loads((root / 'b16_cached_row_deps_v1/specialization/bindings/h131072_b16_prepared.json').read_text())
    row_audit = json.loads((root / 'b16_cached_row_deps_v1/specialization/control_variants/h131072_b16/row_dependencies/audit.json').read_text())
    nodes, original, _, _, _, _, _, names = _problem(schedule, row_audit)
    durations = _costs(schedule, nodes, root / 'b16_dynamic_l1_cost.json')
    full_original = _resource_parents(nodes, original, names)
    baseline = _estimate(full_original, durations)
    _, old_ancestors = ancestor_sets(full_original)
    members = collections.defaultdict(list)
    for i, row in enumerate(nodes):
        members[row['task']].append(i)
    models = []
    for head_tile in (16, 8, 4):
        proof = contract(schedule, catalog(), head_tile)
        math = copy.deepcopy(original)
        changed = []
        for parent, child, workers in (
                (proof['projection'], proof['dequant'], proof['dequant_parents']),
                (proof['dequant'], proof['qk'], proof['qk_parents'])):
            for i in members[child]:
                expected = set(workers[nodes[i]['logical_block']])
                retained = {p for p in members[parent] if nodes[p]['logical_block'] in expected}
                if len(retained) != len(expected):
                    raise ValueError('逻辑写者没有唯一对应成员')
                math[i].difference_update(members[parent])
                math[i].update(retained)
                changed.append({'node': i, 'parent': parent, 'retained': sorted(retained)})
        parents = _resource_parents(nodes, math, names)
        _, ancestors = ancestor_sets(parents)
        if any(new & ~old for new, old in zip(ancestors, old_ancestors)):
            raise ValueError('收窄意外增加原偏序外的先后')
        candidate_nodes = copy.deepcopy(nodes)
        wait_plan, _ = _waits(candidate_nodes, parents)
        timings = {}
        for extra in (0, 1, 2, 4):
            costs = [value + (extra if nodes[i]['task'] == proof['dequant'] else 0)
                     for i, value in enumerate(durations)]
            timings[str(extra)] = _estimate(parents, costs)
        models.append({'head_tile': head_tile, 'contract': proof['audit'], 'changed_edges': changed,
                       'wait_plan': wait_plan, 'duration_extra_us_per_dequant_worker': timings,
                       'core_order_preserved': True, 'ffts_resource_edges_preserved': 216,
                       'new_partial_order_subset_of_original': True})
    result = {'baseline_member_model_us': baseline, 'models': models,
              'device_executed': False, 'source_head_partition_not_changed': True,
              'scope': 'CPU依赖与成本筛选；包含数学内等待的粗粒度历史成本，不是设备预测或硬件下界'}
    (output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'baseline': baseline, 'models': [{k: v for k, v in r.items()
                       if k in ('head_tile', 'duration_extra_us_per_dequant_worker', 'wait_plan')}
                      for r in models]}, ensure_ascii=False))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    model(Path(__file__).resolve().parent, args.output)
