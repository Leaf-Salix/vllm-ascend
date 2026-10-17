# SPDX-License-Identifier: Apache-2.0
"""将Indexer scale整缓存行回写按物理页分配给8个AIV。

原单AIV对每个有效compact行读64字节scale区、修改一个FP16、写回64字节。
不能按request直接分核：不同request可能指向同一个物理页，整行RMW会互相覆盖。
这里owner=physical_page%8，每个owner按原compact顺序扫描，只有它能更新该页。
保留原动态位置/页/offset判定，不从Host读取device metadata；跨页的scale区不重叠。
"""

import ast
import copy
import json
from pathlib import Path


def prove_ownership(page_count, workers):
    # Native每页4160字节：4096字节key，64字节scale。页基址和scale区都64字节对齐。
    lines = [(p * 4160 + 4096) // 64 for p in range(page_count)]
    if len(set(lines)) != page_count or any((p * 4160 + 4096) % 64 for p in range(page_count)):
        raise ValueError('物理页scale缓存行重叠或未对齐')
    # 刻意有共享页、重复offset和逆序页，验证完整写序列和重复写的最终覆盖顺序。
    writes = [(p, k % 32, k) for k,p in enumerate([0,9,1,9,0,2,9,17,1,0,17,2,3,9])]
    owners = [[w for w in writes if w[0] % workers == owner] for owner in range(workers)]
    for page in {w[0] for w in writes}:
        assert [w for w in owners[page % workers] if w[0] == page] == [w for w in writes if w[0] == page]
    return {'page_count':page_count, 'unique_scale_cache_lines':len(set(lines)),
            'cache_line_bytes':64, 'one_writer_per_physical_page':True,
            'aliased_request_order_preserved':True, 'model_is_not_device_proof':True}


def use_page_owners(schedule, catalog, output, workers=8):
    if workers != 8 or schedule['root_arguments']['x_hc']['shape'][0] != 96:
        raise ValueError('本候选先验证T96、8个scale写回worker')
    root = schedule['root_arguments']['idx_native_kv_cache']
    if root['shape'][1:] != [4160] or root['strides'] != [4160,1] or root['offset'] != 0:
        raise ValueError('Native Key4096+Scale64的整页布局改变')
    proof = prove_ownership(root['shape'][0], workers)
    module = ast.parse(Path(catalog['body_bundle']).read_text())
    fn = next(n for n in ast.walk(module) if isinstance(n,ast.FunctionDef) and n.name=='idx_kv_scale_commit')
    if any(p.arg=='original_block_idx' for p in fn.args.args):
        raise ValueError('原scale回写已有worker参数，需要重新核对')
    fn.args.args.append(ast.arg(arg='original_block_idx', annotation=ast.parse('pl.Scalar[pl.INDEX]',mode='eval').body))
    matches=[]
    for node in ast.walk(fn):
        if not isinstance(node,ast.If):continue
        if any(isinstance(n,ast.Call) and ast.unparse(n.func)=='pl.tile.write' for n in ast.walk(node)):
            if 'native_page_v1_inline3986__tile' in ast.unparse(node.test):matches.append(node)
    if len(matches)!=1:
        raise ValueError('原Native页有效性检查不唯一')
    owner=ast.parse('pl.cast(native_page_v1_inline3986__tile, pl.INDEX) % 8 == original_block_idx',mode='eval').body
    matches[0].test=ast.BoolOp(op=ast.And(),values=[matches[0].test,owner])
    # 不改循环次序、load/write/store与cast，只在原有效性条件中添加唯一写者过滤。
    task=next(t for t in schedule['tasks'] if 'idx_kv_scale_commit' in t['kernel_names'])
    if task['block_num']!=1:raise ValueError('原scale不是单writer')
    task.update(block_num=workers,member_count=workers)
    schedule['call_bindings'][task['task_id']]['block_num']=workers
    ast.fix_missing_locations(module)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    source=output/'core_bodies_indexer_scale_page_owner.py'
    source.write_text('# Copyright (c) PyPTO Contributors.\n'
                      '# CANN Open Software License Agreement Version 2.0; see repository LICENSE.\n'
                      +ast.unparse(module)+'\n')
    audit={'workers':workers,'owner_rule':'physical_page % 8','original_loop_order_retained':True,
           'original_load_modify_store_retained':True,'host_metadata_read_added':False,
           'root_and_gm_layout_unchanged':True,'coverage':proof,
           'device_exact_passed':False,'duration_scale_assumption':0.5,
           'duration_assumption_is_not_measured':True}
    (output/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    result=copy.deepcopy(catalog);result['body_bundle']=str(source)
    return result,audit
