# SPDX-License-Identifier: Apache-2.0
"""准备完整动态CSA的Key-major Score候选；只转置两次矩阵与内部交接布局。

原QK=[G*64,128]@[128,N]，WS=[16,G*64]@[G*64,N]。
候选QK=[N,128]@[128,G*64]，WS=[N,G*64]@[G*64,16]。
INT8 dot、FIX 2^-10/ReLU/FP16边界和WS的K顺序均保留；FP16中间值
改走Left非转置读取。G6/N64时Left两Key共16KiB加WS48KiB，Right的
Query48KiB加系数12KiB，仍在64KiB以内。AIV把[key,8]交接Tile转回
原query-major再做同一scale和TopK；是否精确及是否提速必须真机验证。
"""

import ast
import difflib
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
SOURCE = REPO / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"
CONDITION = "query_group_size == 6 and qk_panel_cols == 64 and score_tile == 1024 and key_prefetch_panels == 1 and key_prefetch_to_l0 and balance_leaves"


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"变换点不唯一：{old[:90]}")
    return text.replace(old, new, 1)


def simplify_key_major(candidate):
    """专用helper由wrapper限定prefetch1/toL0；去掉不可能执行的分支。

    Left create缺TileView而extract带row_major，不能用空Tile占位后重绑。
    直接以已实际装入的首Key作为初值；下一Key的初值只作同类型别名，
    真正参与后继计算前仍必须经过原先的有效预取分支。
    """
    class Fold(ast.NodeTransformer):
        def visit_If(self,node):
            names={n.id for n in ast.walk(node.test) if isinstance(n,ast.Name)}
            if names and names <= {'key_prefetch_panels','key_prefetch_to_l0'}:
                value=eval(compile(ast.Expression(node.test),'<strategy>','eval'),
                           {'__builtins__':{}}, {'key_prefetch_panels':1,'key_prefetch_to_l0':True})
                out=[]
                for child in (node.body if value else node.orelse):
                    item=self.visit(child)
                    if item is not None:out.extend(item if isinstance(item,list) else [item])
                return out
            return self.generic_visit(node)

        def visit_Assign(self,node):
            if len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
                name=node.targets[0].id
                if name in ('buf_current_key','buf_prefetched_key') and isinstance(node.value,ast.Call):
                    if ast.unparse(node.value.func)=='pl.tile.create':return None
                    if name=='buf_current_key' and ast.unparse(node.value.func)=='pl.tile.extract':
                        return [node,ast.parse('buf_prefetched_key = buf_current_key').body[0]]
            return self.generic_visit(node)

    tree=Fold().visit(ast.parse(candidate));ast.fix_missing_locations(tree)
    result=ast.unparse(tree)+'\n'
    result=result.replace('    for buf_item in pl.range(',
        '    # 按真实device计划领取group/leaf；空叶不产生事件。\n    for buf_item in pl.range(',1)
    result=result.replace('            for buf_score_step in pl.range(',
        '            # GM仍为两槽；覆盖前等双AIV ACK，末叶必须排空。\n            for buf_score_step in pl.range(',1)
    result=result.replace('                for buf_qk_panel in pl.unroll(',
        '                # Key双Left与前面板WS Left同时存活；预取一面板，QK/WS交错。\n                for buf_qk_panel in pl.unroll(',1)
    return result


def main():
    for label in ('source_reference','source_candidate'):
        target=HERE/label;target.mkdir(exist_ok=False)
        for path in SOURCE.glob('*.py'):
            (target/path.name).write_bytes(path.read_bytes())
    file=HERE/'source_candidate/indexer_score_stages.py'
    source=file.read_text()
    tree=ast.parse(source)
    cube=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='indexer_score_cube_stage')
    lines=source.splitlines(keepends=True)
    start=cube.decorator_list[0].lineno-1;end=cube.end_lineno
    original=''.join(lines[start:end])
    signature=''.join(lines[start:cube.body[0].lineno-1])
    candidate=original.replace('def indexer_score_cube_stage(', 'def _indexer_score_cube_key_major_stage(',1)
    candidate=candidate.replace('"""AIC生产FP32 score双槽；须与两个vector_lane同MIX并发，外层/运行时已设置同一FFTS域。"""',
        '"""Key-major AIC：两矩阵转置，WS用Left；GM仍双槽READY/ACK。\n\n'
        '    输入T/B、device页表、有效叶与尾组全部沿用；只改变固定数学tile的\n'
        '    M/N方向。query与系数常驻Right，两Key与WS共用Left的不同区域。\n'
        '    QK的INT8点积、FIX缩放/ReLU/FP16位置和WS的K累加顺序保持。\n'
        '    每槽GM布局[score_tile,8]，只写前G列；两AIV读同一槽并恢复原\n'
        '    query-major后继续原排序，收到两个ACK后才可覆盖。未验证收益。\n'
        '    """')
    candidate=replace_once(candidate,
        'buf_worker * 2 * query_group_size + buf_score_step % 2 * query_group_size',
        '(buf_worker * 2 + buf_score_step % 2) * score_tile')
    candidate=replace_once(candidate,
        'pl.tile.move(buf_query_vector, target_memory=pl.MemorySpace.Left)',
        'pl.tile.move(pl.tile.transpose_view(buf_query_vector), target_memory=pl.MemorySpace.Right)')
    candidate=replace_once(candidate,
        'pl.tile.move(buf_coefficients_l1, target_memory=pl.MemorySpace.Left)',
        'pl.tile.move(pl.tile.transpose_view(buf_coefficients_l1), target_memory=pl.MemorySpace.Right)')
    candidate=candidate.replace('[query_group_size * IDX_N_HEADS, qk_panel_cols], dtype=pl.FP16',
                                '[qk_panel_cols, query_group_size * IDX_N_HEADS], dtype=pl.FP16')
    candidate=candidate.replace('[IDX_HEAD_DIM, qk_panel_cols], dtype=pl.INT8, target_memory=pl.MemorySpace.Right',
                                '[qk_panel_cols, IDX_HEAD_DIM], dtype=pl.INT8, target_memory=pl.MemorySpace.Left')
    # Key的L1仍为[2*N,128]原页顺序；提取时不再做转置view。
    pattern=re.compile(r'pl.tile.transpose_view\(buf_key_l1_pool\),\s*0,\s*([^\n]+),\s*'
                       r'\[IDX_HEAD_DIM, qk_panel_cols\],\s*target_memory=pl.MemorySpace.Right,')
    candidate,count=pattern.subn(lambda m:'buf_key_l1_pool,\n                            '+m[1]+',\n                            0,\n                            [qk_panel_cols, IDX_HEAD_DIM],\n                            target_memory=pl.MemorySpace.Left,',candidate)
    assert count==3,count
    candidate=candidate.replace('pl.tile.transpose_view(buf_panel_keys), target_memory=pl.MemorySpace.Right',
                                'buf_panel_keys, target_memory=pl.MemorySpace.Left')
    pattern=re.compile(r'(buf_previous_l1,\s*0,\s*0,\s*)\[query_group_size \* IDX_N_HEADS, qk_panel_cols\],\s*target_memory=pl.MemorySpace.Right,')
    candidate,count=pattern.subn(r'\1[qk_panel_cols, query_group_size * IDX_N_HEADS],\n                            target_memory=pl.MemorySpace.Left,',candidate)
    assert count==2,count
    candidate=candidate.replace('pl.tile.matmul(buf_query_left, buf_key_panel)',
                                'pl.tile.matmul(buf_key_panel, buf_query_left)')
    candidate=candidate.replace('pl.tile.matmul(buf_coefficient_pair, buf_previous_pair)',
                                'pl.tile.matmul(buf_previous_pair, buf_coefficient_pair)')
    candidate=candidate.replace('pl.tile.matmul(buf_coefficient_pair, buf_last_pair)',
                                'pl.tile.matmul(buf_last_pair, buf_coefficient_pair)')
    candidate=candidate.replace('pl.set_validshape(buf_reduced_pair, query_group_size, qk_panel_cols)',
                                'pl.set_validshape(buf_reduced_pair, qk_panel_cols, query_group_size)')
    candidate=candidate.replace('pl.set_validshape(buf_last_scores, query_group_size, qk_panel_cols)',
                                'pl.set_validshape(buf_last_scores, qk_panel_cols, query_group_size)')
    candidate=candidate.replace('[buf_transfer_row, buf_panel_col - qk_panel_cols]',
                                '[buf_transfer_row + buf_panel_col - qk_panel_cols, 0]')
    candidate=candidate.replace('[buf_transfer_row, score_tile - qk_panel_cols]',
                                '[buf_transfer_row + score_tile - qk_panel_cols, 0]')
    for old,new in [('buf_query_left','buf_query_right'),('buf_coefficient_pair','buf_coefficient_right'),
                    ('buf_previous_pair','buf_previous_left'),('buf_last_pair','buf_last_left')]:
        candidate=re.sub(r'\b'+old+r'\b',new,candidate)
    candidate=simplify_key_major(candidate)
    # wrapper沿用同一公开helper ABI，谓词完全由策略constexpr构成；不看实值B/S。
    args=', '.join(a.arg for a in cube.args.args)
    dispatch=signature+'    """按Score数学策略选择交接布局，不改变动态device工作范围。"""\n'
    dispatch+=f'    if {CONDITION}:\n        result = _indexer_score_cube_key_major_stage({args})\n'
    dispatch+=f'    else:\n        result = _indexer_score_cube_query_major_stage({args})\n    return result\n'
    legacy=original.replace('def indexer_score_cube_stage(', 'def _indexer_score_cube_query_major_stage(',1)
    modified=''.join(lines[:start])+legacy+'\n\n'+candidate+'\n\n'+dispatch+'\n'+''.join(lines[end:])
    old='''                    single_scores = pl.load(
                        buf_score_transfer, [single_transfer_row, 0], [query_group_size // 2, score_tile]
                    )'''
    new=f'''                    if {CONDITION}:
                        # Key-major GM每槽[1024,8]；一次32KiB读取和片上转置，
                        # 再取本lane的3条query，后续scale/sort/merge逐字保持。
                        key_major_row = (buf_worker * 2 + single_step % 2) * score_tile
                        key_major = pl.load(buf_score_transfer, [key_major_row, 0], [score_tile, 8])
                        query_major = pl.transpose(key_major, axis1=0, axis2=1)
                        single_scores = pl.tile.extract(query_major,
                            buf_score_lane * (query_group_size // 2), 0,
                            [query_group_size // 2, score_tile], target_memory=pl.MemorySpace.Vec)
                    else:
                        single_scores = pl.load(
                            buf_score_transfer, [single_transfer_row, 0], [query_group_size // 2, score_tile]
                        )'''
    modified=replace_once(modified,old,new)
    intro='''    【Key-major候选】G6/N64/1024/prefetch1/toL0/balanced区间的QK和WS
    同时转置，WS从Left读取FP16中间值。双槽GM改为[24*2*1024,8]，每槽
    前6列为有效query分数；AIV经片上转置回原布局后做同一TopK。其它策略
    沿用原布局。此候选尚未设备验证，不能将容量/CPU通过当性能结论。
    阅读新增的_cube_key_major_stage及原公开wrapper；_cube_query_major_stage
    保留原数学供其它区间。真实T/B、有效叶、页表及尾组仍在核内计算。
\n'''
    modified=modified.replace('阅读顺序与范围\n', '阅读顺序与范围\n'+intro,1)
    ast.parse(modified);file.write_text(modified)
    bind=HERE/'source_candidate/binding.py';b=bind.read_text()
    # 同一策略谓词，绝不按测试batch白名单决定工作区。
    oldshape='"indexer_transfer": ((24 * 2 * group, self.policy.score_tile), torch.float32),'
    newshape='''"indexer_transfer": (
                (24 * 2 * self.policy.score_tile, 8)
                if (group == 6 and self.policy.qk_panel == 64 and self.policy.score_tile == 1024
                    and self.policy.prefetch == 1 and self.policy.constexprs()["TO_L0"] and self.policy.balanced)
                else (24 * 2 * group, self.policy.score_tile), torch.float32),'''
    b=replace_once(b,oldshape,newshape);ast.parse(b);bind.write_text(b)
    diffs=[]
    for name in ('indexer_score_stages.py','binding.py'):
        before=(HERE/'source_reference'/name).read_text();after=(HERE/'source_candidate'/name).read_text()
        diffs.extend(difflib.unified_diff(before.splitlines(keepends=True),after.splitlines(keepends=True),fromfile='formal/'+name,tofile='candidate/'+name))
    (HERE/'candidate.diff').write_text(''.join(diffs))
    (HERE/'PREPARATION.json').write_text(json.dumps(dict(source=str(SOURCE),changed=['indexer_score_stages.py','binding.py'],
        device_validated=False,score_transfer_bytes=dict(original=24*2*6*1024*4,candidate=24*2*1024*8*4),
        dynamic_inputs_preserved=True,policy_predicate=CONDITION),ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    main()
