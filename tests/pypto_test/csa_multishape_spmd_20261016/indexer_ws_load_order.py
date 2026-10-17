# SPDX-License-Identifier: Apache-2.0
"""CPU候选：先发当前QK，再加载上块WS右矩阵，不改任何数学节点。

当前展开版本已具备QK(current)/WS(previous)流水，不能重复实现后称新增。
这里保留首对原顺序，只交换后14处独立的Right extract与QK matmul，检查实际缓冲和同步是否
允许Cube计算与MTE1加载重叠。若Right被复用而引入M→MTE1等待，则不视为收益。
"""

import ast
import collections
import copy
import json
from pathlib import Path


def reorder(catalog, output):
    module = ast.parse(Path(catalog['body_bundle']).read_text())
    functions = {n.name: n for n in ast.walk(module) if isinstance(n, ast.FunctionDef)}
    before = {name: ast.dump(fn) for name, fn in functions.items()}
    fn = functions['indexer_score_topk_native_pair_aic']
    item = next(n for n in fn.body if isinstance(n, ast.For)
                and ast.unparse(n.target).startswith('buf_item'))
    leaf = item.body[-1]
    loop = leaf.body[8]
    if not isinstance(loop, ast.For) or len(loop.body) != 434:
        raise ValueError('要求原T96的16个N64展开面板')
    original_statements = collections.Counter(ast.dump(n) for n in loop.body)
    original_body = list(loop.body)
    changes = []
    retained_first = []
    if ast.unparse(loop.body[-5].target) != 'buf_last_pair_inline8679__ssa_v0':
        raise ValueError('原末块排空结构改变')
    original_drain = [ast.dump(n) for n in loop.body[-5:]]
    for i in range(len(loop.body) - 2):
        load, qk, ws = original_body[i:i+3]
        if not all(isinstance(n, ast.AnnAssign) and isinstance(n.value, ast.Call)
                   for n in (load, qk, ws)):
            continue
        if (ast.unparse(load.value.func) != 'pl.tile.extract'
                or not load.target.id.startswith('buf_previous_pair_')):
            continue
        if (ast.unparse(qk.value.func) != 'pl.tile.matmul'
                or ast.unparse(ws.value.func) != 'pl.tile.matmul'
                or ast.unparse(ws.value.args[1]) != load.target.id
                or ast.unparse(load.value.args[3]) != '[384, 64]'):
            raise ValueError('前块WS加载与当前QK结构不符')
        qk_reads = {n.id for n in ast.walk(qk.value) if isinstance(n, ast.Name)}
        load_reads = {n.id for n in ast.walk(load.value) if isinstance(n, ast.Name)}
        if load.target.id in qk_reads or qk.target.id in load_reads:
            raise ValueError('交换的两个节点存在值依赖')
        if load.target.id == 'buf_previous_pair_inline8683__ssa_v1':
            retained_first.append(i)
            continue
        changes.append({'position': i, 'right_load': load.target.id, 'qk': qk.target.id})
        loop.body[i:i+2] = [qk, load]
    if (len(changes) != 14 or len(retained_first) != 1
            or [ast.dump(n) for n in loop.body[-5:]] != original_drain):
        raise ValueError(f'预期保留首对、交换14处且原排空不变，实际交换{len(changes)}处')
    if original_statements != collections.Counter(ast.dump(n) for n in loop.body):
        raise ValueError('不允许增加、删除或修改节点内容')
    if [name for name, node in functions.items() if ast.dump(node) != before[name]] != [fn.name]:
        raise ValueError('改变了其他核体')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    target = output/'core_bodies_indexer_ws_load_order.py'
    target.write_text('# Copyright (c) PyPTO Contributors.\n'
                      '# CANN Open Software License Agreement Version 2.0; see repository LICENSE.\n'
                      + ast.unparse(ast.fix_missing_locations(module)) + '\n')
    audit = {'changed_function': fn.name, 'swaps': changes,
             'retained_first_pair_position': retained_first[0],
             'all_original_statements_retained_verbatim_ast': True,
             'no_data_dependencies_between_swapped_nodes': True,
             'other_functions_unchanged': True, 'physical_overlap_proven': False,
             'device_validated': False, 'scope': '只改变源提交顺序，必须审查实际Right分配/同步'}
    (output/'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
    result = copy.deepcopy(catalog)
    result['body_bundle'] = str(target)
    return result, audit
