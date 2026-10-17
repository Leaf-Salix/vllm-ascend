# SPDX-License-Identifier: Apache-2.0
"""同一 AIV 连续执行 HC reduce→split→mix，暂限 T96 的完整8行块。

沿用原三个函数的数学体，只把 reduce 的独立行维从16改为8。GM中间结果仍写回，
两处核内 fence 保证写回/回读，统一Tensor版本传递；不新增Host值或GM分配。
Sinkhorn保持原独立函数，改为等待融合阶段。它只供尾部HCpost消费，不串进norm链。
"""

import ast
import copy
import json
from pathlib import Path


NAMES = ('hc_pre_linear_reduce', 'split_pre_post', 'mix_x_rms_norm')


def ref_key(ref):
    return tuple(ref[k] if not isinstance(ref[k], list) else tuple(ref[k])
                 for k in ('allocation', 'shape', 'strides', 'offset', 'dtype', 'layout', 'manual_dep'))


class Rename(ast.NodeTransformer):
    def __init__(self, mapping):
        self.mapping = mapping

    def visit_Name(self, node):
        return ast.copy_location(ast.Name(id=self.mapping.get(node.id, node.id), ctx=node.ctx), node)


class ReduceRows(ast.NodeTransformer):
    def visit_Constant(self, node):
        return ast.copy_location(ast.Constant(8), node) if node.value == 16 else node


def fuse_front(schedule, anchor, catalog, output, cost_profile):
    if schedule['root_arguments']['x_hc']['shape'][0] != 96:
        raise ValueError('本候选只验证T96，不能未经验证处理共享尾行缓冲')
    if cost_profile is None:
        raise ValueError('融合调度需要同规格原始成员成本')
    module = ast.parse(Path(catalog['body_bundle']).read_text())
    functions = {n.name: n for n in ast.walk(module) if isinstance(n, ast.FunctionDef)}
    tasks = {t['task_id']: t for t in schedule['tasks']}
    ids = [next(tid for tid, t in tasks.items() if name in t['kernel_names']) for name in NAMES]
    reduce_id, split_id, fused_id = ids
    if [tasks[tid]['block_num'] for tid in ids] != [6, 12, 12]:
        raise ValueError('原6/12/12分片契约改变')
    old_functions = [copy.deepcopy(functions[name]) for name in NAMES]
    reducer, splitter, mixer = old_functions
    if sum(isinstance(n, ast.Constant) and n.value == 16 for n in ast.walk(reducer)) != 9:
        raise ValueError('reduce的16行尺寸常量契约改变')
    text = ast.unparse(mixer)
    for contract in ['pl.pipeline(16, stage=2', 'mix_db_inline518__idx_v0 * 256',
                     'pl.tile.rsqrt(', 'pl.tile.adds(mean_inline3545__tile, 1e-06)']:
        if contract not in text:
            raise ValueError('mix原归约契约改变：' + contract)
    reducer = ReduceRows().visit(reducer)
    calls = {tid: copy.deepcopy(schedule['call_bindings'][tid]) for tid in ids}
    # 先保留mix的参数名；同一GM视图用一个formal，避免生产/消费版本形成不明别名。
    arguments, params, formal, seen_refs = [], [], {}, {}
    for name, tid in [(NAMES[2], fused_id), (NAMES[0], reduce_id), (NAMES[1], split_id)]:
        bound = {a['parameter']: a for a in calls[tid]['arguments']}
        for param in functions[name].args.args:
            if param.arg == 'original_block_idx':
                formal[name, param.arg] = param.arg
                if not any(p.arg == param.arg for p in params):
                    params.append(copy.deepcopy(param))
                continue
            arg = bound[param.arg]
            value = arg['value']
            key = ref_key(value) if isinstance(value, dict) and 'allocation' in value else None
            canonical = seen_refs.get(key, param.arg) if key is not None else param.arg
            formal[name, param.arg] = canonical
            if any(p.arg == canonical for p in params):
                continue
            added = copy.deepcopy(param)
            # 该三阶段的临时Tensor可能先写后读，方向标成InOut；原只读权重仍只读。
            if key is not None:
                seen_refs[key] = canonical
                if value['allocation'].startswith('temporary_'):
                    annotation = added.annotation
                    if ast.unparse(annotation.value) in ('pl.Out', 'pl.InOut'):
                        annotation = annotation.slice
                    added.annotation = ast.Subscript(value=ast.parse('pl.InOut', mode='eval').body,
                                                     slice=annotation, ctx=ast.Load())
            params.append(added)
            arguments.append(copy.deepcopy(arg))

    bodies, local_maps = [], {}
    for index, (name, fn) in enumerate(zip(NAMES, [reducer, splitter, mixer])):
        if not isinstance(fn.body[-1], ast.Return):
            raise ValueError('融合前函数返回结构改变')
        assigned = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        mapping = {x: f'hcf{index}_{x}' for x in assigned}
        mapping.update({p.arg: formal[name, p.arg] for p in fn.args.args})
        local_maps[name] = mapping
        # 明确使用上一步的store/loop结果Tensor，避免依赖旧formal的隐式别名追踪。
        if index == 1:
            mapping['mixes_raw_inline448__ssa_v1'] = local_maps[NAMES[0]]['mixes_raw_inline448__tile']
        if index == 2:
            mapping['pre_val_store__rv_v2'] = local_maps[NAMES[1]]['pre_val_store__rv_v2']
        bodies.extend(Rename(mapping).visit(copy.deepcopy(stmt)) for stmt in fn.body[1:-1])
        if index < 2:
            bodies.extend(ast.parse('pl.system.fence()').body)
    fused = functions[NAMES[2]]
    fused.args.args = params
    fused.body = [copy.deepcopy(fused.body[0]), *bodies, copy.deepcopy(fused.body[-1])]
    ast.fix_missing_locations(module)

    fused_parents = sorted(set().union(*(set(tasks[tid]['predecessors']) for tid in ids)) - set(ids))
    redirected = []
    for tid, task in tasks.items():
        if tid in ids:
            continue
        before = task['predecessors']
        after = sorted({fused_id if parent in ids else parent for parent in before})
        if before != after:
            redirected.append({'task':tid, 'before':before, 'after':after})
        task['predecessors'] = after
    tasks[fused_id]['predecessors'] = fused_parents
    schedule['tasks'] = [t for t in schedule['tasks'] if t['task_id'] not in (reduce_id, split_id)]
    schedule['call_bindings'][fused_id]['arguments'] = arguments
    schedule['call_bindings'][fused_id]['options']['name_hint'] = 'hc_reduce_split_mix_fused'
    for tid in (reduce_id, split_id):
        del schedule['call_bindings'][tid]
    new_anchor = copy.deepcopy(anchor)
    for seqs in (schedule['per_core_sequences'], new_anchor['per_core_sequences']):
        for core, rows in seqs.items():
            seqs[core] = [r for r in rows if r['task'] not in (reduce_id, split_id)]
    # 原数学成本仅为重新分配提供估计；融合核本身没有实测成本，明确标记来源。
    costs = json.loads(Path(cost_profile).read_text())
    if costs['kind'] != 'same_shape_dynamic_l1_cost':
        raise ValueError('只接受原动态同规格成本，不混用插桩结果')
    cost_rows = {r['task']: r for r in costs['task_slots'] if r['task'] in ids}
    if set(cost_rows) != set(ids):
        raise ValueError('缺少被融合的原成员成本')
    total = sum(cost_rows[tid]['median_us'] for tid in ids)
    costs['task_slots'] = [r for r in costs['task_slots'] if r['task'] not in (reduce_id, split_id)]
    next(r for r in costs['task_slots'] if r['task'] == fused_id)['median_us'] = total
    costs['kind'] = 'same_shape_subtask_cost_sum'
    costs['note'] = '融合阶段用原三阶段中位成本之和估计，未测融合核，不作为性能证据'
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source = output/'core_bodies_hc_front_fusion.py'
    source.write_text('# Copyright (c) PyPTO Contributors.\n'
                      '# CANN Open Software License Agreement Version 2.0; see repository LICENSE.\n'
                      + ast.unparse(module) + '\n')
    cost_path = output/'subtask_cost_sum.json'
    cost_path.write_text(json.dumps(costs, ensure_ascii=False, indent=2) + '\n')
    writes = [row for worker in range(12) for row in range(worker*8, worker*8+8)]
    assert sorted(writes) == list(range(96)) and len(set(writes)) == 96
    audit = {'fused_tasks':dict(zip(NAMES, ids)), 'fused_members':12,
             'reduce_rows_before':16, 'reduce_rows_after':8, 'unique_rows_covered':96,
             'partial_sum_order':[0,1,2,3], 'rms_chunks':16, 'rms_chunk_width':256,
             'local_fences_added':2, 'global_stages_removed':2,
             'fused_predecessors':fused_parents, 'redirected_consumers':redirected,
             'sinkhorn_separate':True, 'root_and_gm_allocations_unchanged':True,
             'intermediate_gm_stores_retained':True,
             'tensor_versions_forwarded_between_substages':True,
             'fused_cost_is_estimate_not_measurement':True,
             'device_exact_passed':False}
    (output/'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    result = copy.deepcopy(catalog)
    result['body_bundle'] = str(source)
    return result, new_anchor, cost_path, audit
