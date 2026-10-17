# SPDX-License-Identifier: Apache-2.0
"""HC mix保持原8行/16×K256流水，将BF16块另存本核UB供norm复用。

与此前48成员整行算法不同：成员数仍12，全部数学AST与归约顺序保持，
只增加一个128×256 BF16驻留Tile和每块assemble，将norm的GM读换成UB提取。
每个8×256块占连续8行，避免向宽4096列的UB做窄列插入。
原mixed GM写保留；不引入设备Scalar求和或整行FP32中间结果。
"""

import ast
import copy
import json
from pathlib import Path


def use_chunk_buffer(schedule, catalog, output):
    if schedule['root_arguments']['x_hc']['shape'][0] != 96:
        raise ValueError('HC分块驻留当前只证明T96十二个完整8行块')
    module = ast.parse(Path(catalog['body_bundle']).read_text())
    fn = next(n for n in ast.walk(module) if isinstance(n, ast.FunctionDef)
              and n.name == 'mix_x_rms_norm')
    loops = [n for n in fn.body if isinstance(n, ast.For)
             and isinstance(n.iter, ast.Call) and ast.unparse(n.iter.func) == 'pl.pipeline']
    if len(loops) != 2:
        raise ValueError('HC原mix/norm两个流水改变')
    mix, norm = loops
    if ast.unparse(mix.target) != '(mix_db_inline518__idx_v0, (sq_sum_inline530__iter_v1, x_mixed_inline534__iter_v1))':
        raise ValueError('HC mix循环变量改变')
    if ast.unparse(norm.iter.args[0]) != '16':
        raise ValueError('HC norm分片改变')
    pos = fn.body.index(mix)
    fn.body[pos:pos] = ast.parse('''
hc_mixed_ub_init = pl.tile.create([128, 256], dtype=pl.BF16, target_memory=pl.Mem.Vec)
''').body
    mix.target.elts[1].elts.append(ast.Name(id='hc_mixed_ub_iter',ctx=ast.Store()))
    initial = next(k for k in mix.iter.keywords if k.arg == 'init_values')
    initial.value.elts.append(ast.Name(id='hc_mixed_ub_init',ctx=ast.Load()))
    result = mix.body[-1]
    if not isinstance(result,ast.Assign) or ast.unparse(result.value.func) != 'pl.yield_':
        raise ValueError('HC mix返回状态改变')
    # 每次只写已算出的256列。循环结束后16块精确覆盖4096列，才供norm读取。
    mix.body[-1:-1] = ast.parse('''
hc_mixed_ub_next = pl.tile.assemble(hc_mixed_ub_iter, y_bf16_inline533__tile, [mix_db_inline518__idx_v0 * 8, 0])
''').body
    result.targets[0].elts.append(ast.Name(id='hc_mixed_ub_result',ctx=ast.Store()))
    result.value.args.append(ast.Name(id='hc_mixed_ub_next',ctx=ast.Load()))
    reads = [n for n in norm.body if isinstance(n,ast.AnnAssign)
             and isinstance(n.value,ast.Call) and ast.unparse(n.value.func) == 'pl.tile.load'
             and ast.unparse(n.value.args[0]) == 'x_mixed_inline534__rv_v2']
    if len(reads) != 1:
        raise ValueError('HC norm的mixed回读不唯一')
    reads[0].value = ast.parse('pl.tile.extract(hc_mixed_ub_result, norm_db_inline517__idx_v0 * 8, 0, [8, 256], target_memory=pl.Mem.Vec)', mode='eval').body
    task = [t for t in schedule['tasks'] if {x for x in t['kernel_names'] if x} == {'mix_x_rms_norm'}]
    if len(task) != 1 or task[0]['block_num'] != 12:
        raise ValueError('本候选保持十二个HC mix成员')
    ast.fix_missing_locations(module)
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    source = output/'core_bodies_hc_mix_chunk_buffer.py'
    source.write_text('# Copyright (c) PyPTO Contributors.\n'
                      '# CANN Open Software License Agreement Version 2.0; see repository LICENSE.\n'
                      + ast.unparse(module)+'\n')
    audit={'tokens':96,'members':12,'rows_per_member':8,'resident_bf16_bytes':8*4096*2,
           'resident_physical_shape':[128,256],'layout':'16个连续8×256块，按块优先存放',
           'original_mixed_gm_stores_retained':True,'removed_mixed_gm_read_bytes':96*4096*2,
           'math_order_and_original_pipeline_retained':True,'extra_operations':['tile.create','tile.assemble','tile.extract'],
           'new_root_arguments':0,'new_gm_allocations':0,'device_validated':False}
    (output/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    result_catalog=copy.deepcopy(catalog)
    result_catalog['body_bundle']=str(source)
    return result_catalog,audit
