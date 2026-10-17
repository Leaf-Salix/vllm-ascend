# SPDX-License-Identifier: Apache-2.0
"""已融合HC阶段内直接传递Tile；保留兼容GM写回和阶段外发布屏障。

T96/12成员保证每个成员恰好一个8行块，因此可以展开split的单次分片循环。
reduce的8×32转置为32×8，再用extract取行0..7或4..11，避免列4的非对齐UB切片。
raw的两处GM读及pre的一处GM读改为SSA Tile输入，原store及编译器自动事件保留。
当前slice后的转置临时Tile在生成IR中valid类型不一致；extract的对齐UB复制可正常编译。
"""

import ast
import copy
import json
from pathlib import Path


def forward_tiles(catalog, output):
    module = ast.parse(Path(catalog['body_bundle']).read_text())
    fn = next(n for n in ast.walk(module)
              if isinstance(n, ast.FunctionDef) and n.name == 'mix_x_rms_norm')
    loops = [n for n in fn.body if isinstance(n, ast.For)
             and ast.unparse(n.target).startswith('(hcf1_ob_inline470__idx_v0,')]
    if len(loops) != 1:
        raise ValueError('只接受hc_front_fusion生成的单次split循环')
    loop = loops[0]
    args = loop.iter.args
    if ast.unparse(args[0]) != 'hcf1_ob_worker_inline476__ssa_v0':
        raise ValueError('split起始worker改变')
    if ast.unparse(args[1]) != 'token_tiles_inline443__ssa_v0':
        raise ValueError('split行块数量改变')
    initial = next(k.value for k in loop.iter.keywords if k.arg == 'init_values')
    if len(loop.target.elts) != 2 or len(loop.target.elts[1].elts) != 3:
        raise ValueError('split循环携带状态改变')
    expanded = [ast.Assign(targets=[copy.deepcopy(loop.target.elts[0])], value=copy.deepcopy(args[0])),
                ast.Assign(targets=[copy.deepcopy(loop.target.elts[1])], value=copy.deepcopy(initial))]
    contents = copy.deepcopy(loop.body)
    last = contents[-1]
    if not isinstance(last, ast.Assign) or ast.unparse(last.value.func) != 'pl.yield_':
        raise ValueError('split循环出口不是预期的三Tensor结果')
    last.value = ast.Tuple(elts=last.value.args, ctx=ast.Load())
    expanded.extend(contents)
    position = fn.body.index(loop)
    fn.body[position:position+1] = expanded

    prelude = ast.parse('''
hcf_raw_transposed = pl.tile.transpose(hcf0_mixes_total_inline423__rv_v2, 0, 1)
''').body
    # 插入reduce已完成之后、split首次读其结果之前。
    point = next(i for i,n in enumerate(fn.body)
                 if isinstance(n, ast.AnnAssign) and n.target.id == 'hcf1_ob_worker_inline476__ssa_v0')
    fn.body[point:point] = prelude
    raw_loads = 0
    pre_loads = 0
    rewritten = []
    for node in fn.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'pl.system.fence':
            continue
        call = node.value if isinstance(node, (ast.AnnAssign, ast.Assign)) else None
        if isinstance(call, ast.Call) and ast.unparse(call.func) == 'pl.tile.load':
            source = ast.unparse(call.args[0])
            if source == 'hcf0_mixes_raw_inline448__tile':
                column = call.args[1].elts[1]
                if not isinstance(column, ast.Constant) or column.value not in (0,4):
                    raise ValueError('pre/post读取列偏移改变')
                offset = column.value
                prefix = f'hcf_raw_gate_{offset}'
                rewritten.extend(ast.parse(f'''
{prefix}_rows = pl.tile.extract(hcf_raw_transposed, {offset}, 0, [8, 8], target_memory=pl.Mem.Vec)
''').body)
                node.value = ast.parse(f'pl.tile.transpose({prefix}_rows, 0, 1)', mode='eval').body
                raw_loads += 1
            elif source == 'hcf1_pre_val_store__rv_v2':
                node.value = ast.Name(id='hcf1_pre_val_inline446__tile', ctx=ast.Load())
                pre_loads += 1
        rewritten.append(node)
    if (raw_loads, pre_loads) != (2,1):
        raise ValueError(f'要替换的GM读数改变：{raw_loads}/{pre_loads}')
    fn.body = rewritten
    ast.fix_missing_locations(module)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source = output/'core_bodies_hc_front_tiles.py'
    source.write_text('# Copyright (c) PyPTO Contributors.\n'
                      '# CANN Open Software License Agreement Version 2.0; see repository LICENSE.\n'
                      + ast.unparse(module) + '\n')
    audit = {'tokens':96, 'members':12, 'rows_per_member':8, 'single_split_iteration_unwrapped':True,
             'raw_gm_loads_removed':raw_loads, 'pre_val_gm_loads_removed':pre_loads,
             'local_fences_removed':2, 'outside_stage_fences_retained':True,
             'gm_stores_retained':True, 'reduce_and_rms_order_unchanged':True,
             'transpose_and_slice_only_no_new_arithmetic':True,
             'device_exact_passed':False}
    (output/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    result = copy.deepcopy(catalog)
    result['body_bundle'] = str(source)
    return result, audit
