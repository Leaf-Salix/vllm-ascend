# SPDX-License-Identifier: Apache-2.0
"""N80的WS改K128三段，缩小Right占用并恢复下一Key预取。

已有N80版本的WS Right为384×80 FP16=60KiB，不能同时保留10KiB Key。
本候选每个WS Right为128×80=20KiB，两块WS与两块Key合计60KiB。
保持QK M384/N80、13个panel、原叶与FFTS、AIV TopK，WS按0/128/256顺序累加。
K分段改变了matmul指令边界，必须重新过原exact门槛，不能仅凭数学等价认定通过。
"""

import ast
import copy
import json
from pathlib import Path

from indexer_n80 import use_indexer_n80


def use_ws128(schedule, catalog, output):
    output = Path(output)
    updated, base_audit = use_indexer_n80(schedule, catalog, output / 'base_n80')
    module = ast.parse(Path(updated['body_bundle']).read_text())
    fn = next(n for n in ast.walk(module) if isinstance(n, ast.FunctionDef)
              and n.name == 'indexer_score_topk_native_pair_aic')
    item = next(n for n in fn.body if isinstance(n, ast.For)
                and ast.unparse(n.target).startswith('buf_item'))
    leaf = item.body[-1]
    coefficient = next(n for n in leaf.body if isinstance(n, ast.Assign)
                       and ast.unparse(n.targets[0]) == 'n80_coef')
    pos = leaf.body.index(coefficient)
    leaf.body[pos:pos + 1] = ast.parse('\n'.join(
        f'n80_coef_chunk{k} = pl.tile.extract(n80_coef_l1, 0, {k * 128}, [16, 128], target_memory=pl.Mem.Left)'
        for k in range(3))).body
    loop = next(n for n in leaf.body if isinstance(n, ast.For)
                and ast.unparse(n.target) == 'n80_step')
    # 只改源码提交顺序；实际是否分配成独立Right槽仍须检查PTOAS产物。
    for panel in range(1, 13):
        key = next(n for n in loop.body if isinstance(n, ast.Assign)
                   and ast.unparse(n.targets[0]) == f'n80_key_{panel}')
        loop.body.remove(key)
        qk = next(n for n in loop.body if isinstance(n, ast.Assign)
                  and ast.unparse(n.targets[0]) == f'n80_qk_{panel - 1}')
        loop.body.insert(loop.body.index(qk), key)
    for panel in range(13):
        old = next(n for n in loop.body if isinstance(n, ast.Assign)
                   and ast.unparse(n.targets[0]) == f'n80_ws_right_{panel}')
        start = loop.body.index(old)
        old_mm = loop.body[start + 1]
        if ast.unparse(old_mm.targets[0]) != f'n80_ws_{panel}':
            raise ValueError('原WS加载与matmul不相邻')
        loop.body[start:start + 2] = ast.parse(f'''
n80_ws_right_{panel}_0 = pl.tile.extract(n80_score_{panel}, 0, 0, [128, 80], target_memory=pl.Mem.Right)
n80_ws_right_{panel}_1 = pl.tile.extract(n80_score_{panel}, 128, 0, [128, 80], target_memory=pl.Mem.Right)
n80_ws_acc_{panel}_0 = pl.tile.matmul(n80_coef_chunk0, n80_ws_right_{panel}_0)
n80_ws_right_{panel}_2 = pl.tile.extract(n80_score_{panel}, 256, 0, [128, 80], target_memory=pl.Mem.Right)
n80_ws_acc_{panel}_1 = pl.tile.matmul_acc(n80_ws_acc_{panel}_0, n80_coef_chunk1, n80_ws_right_{panel}_1)
n80_ws_{panel} = pl.tile.matmul_acc(n80_ws_acc_{panel}_1, n80_coef_chunk2, n80_ws_right_{panel}_2)
''').body
    if sum(isinstance(n, ast.Assign) and ast.unparse(n.targets[0]).startswith('n80_qk_')
           for n in loop.body) != 13:
        raise ValueError('QK panel数量改变')
    ast.fix_missing_locations(module)
    path = output / 'core_bodies_indexer_n80_ws128.py'
    path.write_text('# Copyright (c) PyPTO Contributors.\n'
                    '# CANN Open Software License Agreement Version 2.0; see repository LICENSE.\n'
                    + ast.unparse(module) + '\n')
    audit = {'base_n80_audit': base_audit, 'ws_k_chunks': [128, 128, 128],
             'ws_k_order': [0, 128, 256], 'ws_matmul_boundaries_changed': True,
             'qk_panels_unchanged': True, 'leaf_plan_and_ffts_unchanged': True,
             'topk_aiv_unchanged': True, 'root_and_gm_scratch_unchanged': True,
             'next_key_right_submitted_before_current_qk': True,
             'theoretical_live_bytes': {'left': 61440, 'right': 61440, 'acc': 128000},
             'capacity_is_not_physical_allocation_proof': True, 'exact_equivalence_validated': False}
    (output / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    result = copy.deepcopy(updated)
    result['body_bundle'] = str(path)
    return result, audit
