# SPDX-License-Identifier: Apache-2.0
"""QK全部五块有效时，把重复的有效块判断移到循环外。

三个物理成员都从同一只读plan计算条件。任意一个query/块无效，就完整执行
原循环。快路径只消除已证明恒真的外层块判断，不删稀疏行有效性/尾部判断，
不改变FFTS次数、KV搬运、softmax、PV和输出。前置plan依赖保持原样。
"""

import ast
import copy
import json
from pathlib import Path


def use_all_valid(schedule, catalog, output):
    module = ast.parse(Path(catalog['body_bundle']).read_text())
    audits = {}
    for role, read_count in [('aic', 2), ('aiv', 6)]:
        fn = next(n for n in ast.walk(module) if isinstance(n, ast.FunctionDef) and n.name == 'qk_pv_' + role)
        loops = [n for n in fn.body if isinstance(n, ast.For)]
        if len(loops) != 1:
            raise ValueError('QK预期只有一个顶层tick循环')
        original_loop = loops[0]
        original_loop_ast = ast.dump(original_loop)
        fast = copy.deepcopy(original_loop)
        reads = {}
        for n in ast.walk(fast):
            if (isinstance(n, ast.AnnAssign) and isinstance(n.value, ast.Call)
                    and ast.unparse(n.value.func) == 'pl.tensor.read'
                    and ast.unparse(n.value.args[0]) == 'valid_block_mask_inline4413__ssa_v0'):
                reads[ast.unparse(n.target)] = n
        if len(reads) != read_count:
            raise ValueError('原块标记读取数发生变化')

        class Fold(ast.NodeTransformer):
            folded = 0

            def visit_AnnAssign(self, node):
                if ast.unparse(node.target) in reads:
                    return None
                return self.generic_visit(node)

            def visit_If(self, node):
                names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)} & reads.keys()
                if not names:
                    return self.generic_visit(node)
                if len(names) != 1 or ast.unparse(node.test) != f'0 < pl.cast({next(iter(names))}, pl.INDEX)':
                    raise ValueError('不能证明被替换的判断是valid_mask>0')
                self.folded += 1
                node = self.generic_visit(node)
                body = node.body
                last = body[-1]
                if isinstance(last, (ast.Assign, ast.AnnAssign)) and isinstance(last.value, ast.Call) and ast.unparse(last.value.func) == 'pl.yield_':
                    args = last.value.args
                    last.value = args[0] if len(args) == 1 else ast.Tuple(elts=args, ctx=ast.Load())
                return body

        fold = Fold()
        fast = fold.visit(fast)
        if fold.folded != read_count or any(isinstance(n, ast.Name) and n.id in reads for n in ast.walk(fast)):
            raise ValueError('块判断未全部消除或临时变量仍有其他用途')
        # 不以Tensor内容做Host选择；三成员在核内检查同一组query、五个块。
        scan = ast.parse('''
qk_all_initial = pl.const(1, pl.INDEX)
for qk_check_query, (qk_all_query_iter,) in pl.range(qk_query_count_inline4331__ssa_v0, init_values=(qk_all_initial,)):
    for qk_check_block, (qk_all_block_iter,) in pl.range(5, init_values=(qk_all_query_iter,)):
        qk_check_mask = pl.cast(pl.tensor.read(valid_block_mask_inline4413__ssa_v0,
            [qk_core_inline4347__ssa_v0 + qk_check_query * 24, qk_check_block]), pl.INDEX)
        qk_all_block_result = pl.yield_(pl.min(qk_all_block_iter, qk_check_mask))
    qk_all_result = pl.yield_(qk_all_block_result)
''').body
        branch = ast.If(test=ast.parse('qk_all_result > 0', mode='eval').body,
                        body=[fast], orelse=[original_loop])
        pos = fn.body.index(original_loop)
        fn.body[pos:pos + 1] = scan + [branch]
        if ast.dump(branch.orelse[0]) != original_loop_ast:
            raise AssertionError('回退循环必须保持原AST')
        audits[role] = {'removed_static_mask_reads_in_fast_loop': read_count,
                        'removed_static_mask_guards': fold.folded,
                        'fallback_loop_ast_unchanged': True,
                        'does_not_remove_inner_sparse_row_or_tail_guards': True}
    ast.fix_missing_locations(module)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    path = output / 'core_bodies_qk_all_valid.py'
    path.write_text('# Copyright (c) PyPTO Contributors.\n'
                    '# CANN Open Software License Agreement Version 2.0; see repository LICENSE.\n'
                    + ast.unparse(module) + '\n')
    audit = {'roles': audits, 'device_flag_scan_per_group': 'query_count × 5',
             'all_three_members_read_identical_immutable_plan': True,
             'existing_plan_dependencies_retained': True, 'new_host_readback': False,
             'new_gm_buffers': 0, 'device_validated': False}
    (output / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    result = copy.deepcopy(catalog)
    result['body_bundle'] = str(path)
    return result, audit
