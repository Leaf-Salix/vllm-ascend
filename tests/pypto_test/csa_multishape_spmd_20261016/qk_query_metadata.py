# SPDX-License-Identifier: Apache-2.0
"""每 query 一次准备512个压缩KV地址，四个块复用本核UB元数据。

在tick%5==0且query有效时准备，不依赖第一个压缩块一定有效。原始负索引
和物理行分别以两个Tile携带到下一tick；三槽KV流水、数学与FFTS全部保持。
不会跨query复用旧值，也不会把元数据传回Host。
"""

import ast
import copy
import json
from pathlib import Path


def use_query_metadata(schedule, catalog, output):
    module = ast.parse(Path(catalog['body_bundle']).read_text())
    fn = next(n for n in ast.walk(module) if isinstance(n, ast.FunctionDef) and n.name == 'qk_pv_aiv')

    def unchanged_calls(function):
        names = {'pl.tile.gather_row', 'pl.tile.store', 'pl.tile.matmul', 'pl.tile.matmul_acc',
                 'pl.tile.exp', 'pl.tile.row_max', 'pl.tile.row_sum', 'pl.tile.row_expand_div',
                 'pl.tile.row_expand_mul', 'pl.tile.col_expand_mul', 'pl.system.sync_set',
                 'pl.system.sync_wait'}
        return sorted(ast.dump(n) for n in ast.walk(function)
                      if isinstance(n, ast.Call) and ast.unparse(n.func) in names)

    protected = unchanged_calls(fn)
    loop = next(n for n in fn.body if isinstance(n, ast.For)
                and ast.unparse(n.target).startswith('(qk_tick_inline4279__idx_v0,'))
    branch = next(n for n in ast.walk(loop) if isinstance(n, ast.If)
                  and any(isinstance(s, ast.Assign) and ast.unparse(s.targets[0]) == 'qk_meta_start'
                          for s in n.orelse))
    prefix = branch.orelse[1:10]
    expected = ['qk_meta_start', 'qk_meta_raw', 'qk_meta_nonnegative', 'qk_meta_page_columns',
                'qk_meta_table', 'qk_meta_scratch', 'qk_meta_pages', 'qk_meta_remainders',
                'qk_meta_physical_rows']
    if [ast.unparse(s.targets[0]) for s in prefix] != expected:
        raise ValueError('原每块metadata准备结构改变')
    load_table = prefix[4].value
    padded, columns = ast.literal_eval(load_table.args[2])[1], ast.literal_eval(load_table.args[3])[1]
    initial = ast.parse('''
qk_query_raw_init = pl.tile.full([1, 512], dtype=pl.INT32, value=0)
qk_query_rows_init = pl.tile.full([1, 512], dtype=pl.INT32, value=0)
''').body
    position = fn.body.index(loop)
    fn.body[position:position] = initial
    for name in ('qk_query_raw_iter', 'qk_query_rows_iter'):
        loop.target.elts[1].elts.append(ast.Name(id=name, ctx=ast.Store()))
    initial_values = next(k.value for k in loop.iter.keywords if k.arg == 'init_values')
    for name in ('qk_query_raw_init', 'qk_query_rows_init'):
        initial_values.elts.append(ast.Name(id=name, ctx=ast.Load()))

    prepare = ast.parse(f'''
if qk_tick_inline4279__idx_v0 < qk_block_count_inline4405__ssa_v0 and qk_sb_inline4367__ssa_v1 == 0:
    qk_query_raw_loaded = pl.tile.load(cmp_sparse_indices_inline4329__rv_v2,
        [qk_t_inline4408__ssa_v1, 0], [1, 512], [1, 512], target_memory=pl.Mem.Vec)
    qk_query_safe = pl.tile.maximums(qk_query_raw_loaded, 0)
    qk_query_columns = pl.tile.shrs(qk_query_safe, 5)
    qk_query_table = pl.tile.load(cmp_block_table__ssa_v0,
        [qk_b_inline4274__ssa_v0, 0], [1, {padded}], [1, {columns}], target_memory=pl.Mem.Vec)
    qk_query_scratch = pl.tile.create([1, 512], dtype=pl.INT32, target_memory=pl.Mem.Vec)
    qk_query_pages = pl.tile.gather(qk_query_table, qk_query_columns, qk_query_scratch)
    qk_query_remainders = pl.tile.sub(qk_query_safe, pl.tile.shls(qk_query_columns, 5))
    qk_query_physical = pl.tile.add(pl.tile.shls(qk_query_pages, 5), qk_query_remainders)
    qk_query_raw, qk_query_rows = pl.yield_(qk_query_raw_loaded, qk_query_physical)
else:
    qk_query_raw, qk_query_rows = pl.yield_(qk_query_raw_iter, qk_query_rows_iter)
''').body
    if [ast.unparse(n.target) for n in loop.body[:3]] != [
            'qk_t_inline4408__ssa_v1', 'qk_b_inline4274__ssa_v0', 'qk_sb_inline4367__ssa_v1']:
        raise ValueError('QK tick坐标绑定改变')
    loop.body[3:3] = prepare
    # 只保留块起始列。逐行读取新512列Tile中的相同列，KV DMA参数不变。
    branch.orelse[1:10] = [prefix[0]]
    changed = 0
    for call in ast.walk(loop):
        if (isinstance(call, ast.Call) and ast.unparse(call.func) == 'pl.tile.read'
                and isinstance(call.args[0], ast.Name)
                and call.args[0].id in {'qk_meta_raw', 'qk_meta_physical_rows'}):
            call.args[0] = ast.Name(id={'qk_meta_raw': 'qk_query_raw',
                                      'qk_meta_physical_rows': 'qk_query_rows'}[call.args[0].id], ctx=ast.Load())
            call.args[1].elts[1] = ast.parse('qk_meta_start + qk_row_inline4264__idx_v0', mode='eval').body
            changed += 1
    if changed != 2:
        raise ValueError('应当只改变两处Scalar读取的元数据Tile索引')
    result = loop.body[-1]
    if not isinstance(result, ast.Assign) or ast.unparse(result.value.func) != 'pl.yield_':
        raise ValueError('QK tick返回状态改变')
    for target, value in [('qk_query_raw_final', 'qk_query_raw'), ('qk_query_rows_final', 'qk_query_rows')]:
        result.targets[0].elts.append(ast.Name(id=target, ctx=ast.Store()))
        result.value.args.append(ast.Name(id=value, ctx=ast.Load()))
    if unchanged_calls(fn) != protected:
        raise ValueError('数学、KV搬运或FFTS被意外修改')
    ast.fix_missing_locations(module)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    path = output / 'core_bodies_qk_query_metadata.py'
    path.write_text('# Copyright (c) PyPTO Contributors.\n'
                    '# CANN Open Software License Agreement Version 2.0; see repository LICENSE.\n'
                    + ast.unparse(module) + '\n')
    audit = {'metadata_entries_per_query': 512, 'persistent_ub_bytes_per_aiv': 4096,
             'table_columns': columns, 'table_ub_columns': padded,
             'table_reads_per_query_before_max': 4, 'table_reads_per_query_after': 1,
             'address_math_entries_per_query_before_max_per_aiv': 256,
             'address_math_entries_per_query_after_per_aiv': 512,
             'math_dma_ffts_unchanged': True, 'negative_index_branch_preserved': True,
             'prepare_independent_of_first_block_validity': True,
             'new_gm_allocations': 0, 'host_metadata_readback': False, 'device_validated': False}
    (output / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    updated = copy.deepcopy(catalog)
    updated['body_bundle'] = str(path)
    return updated, audit
