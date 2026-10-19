# SPDX-License-Identifier: Apache-2.0
"""生成源级动态G6 Score候选：QK双48KiB Acc早FIX，WS分K但每panel只STORE一次。

原几何/页保护/预取/FFTS和AIV函数逐字保留。为独立对照保留原helper，
仅在已有长G6/N64/1024区间分支到新helper，不包含batch或history测试点。
"""

import ast
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent


def once(source, old, new):
    assert source.count(old)==1, old[:160]
    return source.replace(old,new)


def qk_pair(indent):
    out=''
    for half in (0,1):
        out+=f'''{indent}split_qk{half} = pl.tile.matmul_acc(
{indent}    split_qk_acc{half}, split_query{half}, buf_key_panel, init_cond=True,
{indent})
{indent}# 第一份FIX位于第二份QK之前；两份Acc明确独立才有重叠机会。
{indent}split_current{half} = pl.tile.assemble(
{indent}    split_current{half}, split_qk{half}, [0, 0],
{indent}    pre_quant=1.0 / BUFFERED_SCORE_SCALE, pre_relu=True,
{indent})
'''
    return out


def ws_pair(indent, suffix):
    return f'''{indent}split_ws_first{suffix} = pl.tile.matmul_acc(
{indent}    split_ws_acc, split_coefficient0, split_right0{suffix}, init_cond=True,
{indent})
{indent}split_ws_done{suffix} = pl.tile.matmul_acc(
{indent}    split_ws_first{suffix}, split_coefficient1, split_right1{suffix},
{indent})
'''


def right_pair(indent, suffix):
    return ''.join(f'''{indent}split_right{half}{suffix} = pl.tile.extract(
{indent}    split_previous{half}, 0, 0, [SPLIT_QK_ROWS, qk_panel_cols],
{indent}    target_memory=pl.MemorySpace.Right,
{indent})
''' for half in (0,1))


def main():
    base=HERE/'source_reference';target=HERE/'source_score_split_store'
    target.mkdir(exist_ok=False)
    for p in base.glob('*.py'):
        (target/p.name).write_bytes(p.read_bytes())
    path=target/'indexer_score_stages.py';source=path.read_text()
    tree=ast.parse(source)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='indexer_score_cube_stage')
    lines=source.splitlines(keepends=True)
    begin=min(d.lineno for d in fn.decorator_list)-1;end=fn.end_lineno
    original=''.join(lines[begin:end])
    split=original.replace('def indexer_score_cube_stage(', 'def _indexer_score_cube_split_store(')
    split=once(split, '    """AIC生产FP32 score双槽；须与两个vector_lane同MIX并发，外层/运行时已设置同一FFTS域。"""',
               '''    """长G6/N64：两半QK早FIX；原完整16行系数分K累加后一次写回。

    本helper保留原group/leaf/device页检查、尾query安全填充和双槽协议。
    QK两份48KiB Acc及WS一份4KiB跨panel保留，各次init_cond=True重新初始化。
    两次WS按原K顺序累计到同一Acc，不能把两个三query结果分别写回。
    是否减少等待、保持舍入和产生完整收益，分别由代码审计与真机检查决定。
    """''')
    old='''            buf_query_left = pl.tile.move(buf_query_vector, target_memory=pl.MemorySpace.Left)
            buf_coefficient_pair = pl.tile.move(buf_coefficients_l1, target_memory=pl.MemorySpace.Left)
'''
    new='''            # 两份query共48KiB Left，两份完整16行系数共12KiB Left。
            # 系数按K截成两半，仍保留六个query的原行号与零项，尾组语义不变。
'''
    for half in (0,1):
        new+=f'''            split_query{half} = pl.tile.extract(
                buf_query_vector, {half} * SPLIT_QK_ROWS, 0, [SPLIT_QK_ROWS, IDX_HEAD_DIM],
                target_memory=pl.MemorySpace.Left,
            )
            split_coefficient{half} = pl.tile.extract(
                buf_coefficients_l1, 0, {half} * SPLIT_QK_ROWS, [NATIVE_QLI_WEIGHT_ROWS, SPLIT_QK_ROWS],
                target_memory=pl.MemorySpace.Left,
            )
            split_qk_acc{half} = pl.tile.create(
                [SPLIT_QK_ROWS, qk_panel_cols], dtype=pl.INT32, target_memory=pl.MemorySpace.Acc,
            )
'''
    new+='''            # 独立WS句柄避免FIX仍读QK时又被WS覆盖；实际地址必须审计。
            split_ws_acc = pl.tile.create(
                [NATIVE_QLI_WEIGHT_ROWS, qk_panel_cols], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc,
            )
'''
    split=once(split,old,new)
    split=once(split,'''                buf_previous_l1 = pl.tile.create(
                    [query_group_size * IDX_N_HEADS, qk_panel_cols], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat
                )
''',''.join(f'''                split_previous{half} = pl.tile.create(
                    [SPLIT_QK_ROWS, qk_panel_cols], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat,
                )
''' for half in (0,1)))
    start=split.index('                    buf_scores_l1 = pl.tile.create(')
    stop=split.index('                    if key_prefetch_panels > 0:',start)
    math=''.join(f'''                    split_current{half} = pl.tile.create(
                        [SPLIT_QK_ROWS, qk_panel_cols], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat,
                    )
''' for half in (0,1))+'''                    if buf_qk_panel > 0:
                        # 先备齐上一panel的两个24KiB Right；与两份8KiB Key
                        # 同时存在恰为64KiB，不额外创建第三份Key或WS全K Right。
'''+right_pair('                        ','')+qk_pair('                        ')+ws_pair('                        ','')+'''                        # 每panel仅一次原六行写回，保留1024列FIFO布局和顺序。
                        pl.store(
                            pl.set_validshape(split_ws_done, query_group_size, qk_panel_cols),
                            [buf_transfer_row, buf_panel_col - qk_panel_cols], buf_score_transfer,
                        )
                    else:
'''+qk_pair('                        ')+'''                    split_previous0 = split_current0
                    split_previous1 = split_current1
'''
    split=split[:start]+math+split[stop:]
    start=split.index('                buf_last_pair = pl.tile.extract(')
    stop=split.index('                pl.system.sync_set(SCORE_READY_EVENT',start)
    split=split[:start]+right_pair('                ','_last')+ws_pair('                ','_last')+'''                pl.store(
                    pl.set_validshape(split_ws_done_last, query_group_size, qk_panel_cols),
                    [buf_transfer_row, score_tile - qk_panel_cols], buf_score_transfer,
                )
'''+split[stop:]
    # 包装器的签名来自原helper，根入参与全部caller不变；分支仅由原constexpr控制。
    signature=original[:original.index('    """')]
    arg_names=[a.arg for a in fn.args.args]
    args='\n'.join('            '+name+',' for name in arg_names)
    wrapper=signature+'''    """只在既有长G6/N64/1024资源组合试验双QK早FIX；其它组合保留原helper。"""
    if balance_leaves and query_group_size == 6 and qk_panel_cols == 64 and score_tile == 1024:
        result = _indexer_score_cube_split_store(
'''+args+'''
        )
    else:
        result = _indexer_score_cube_original(
'''+args+'''
        )
    return result
'''
    old_helper=original.replace('def indexer_score_cube_stage(', 'def _indexer_score_cube_original(')
    source=''.join(lines[:begin])+old_helper+'\n\n\n'+split+'\n\n\n'+wrapper+''.join(lines[end:])
    source=once(source,'NATIVE_QLI_WEIGHT_ROWS = 16\n','NATIVE_QLI_WEIGHT_ROWS = 16\nSPLIT_QK_ROWS = 3 * IDX_N_HEADS  # 既有G6一半的数学M，非batch专用shape。\n')
    doc_end=source.index('"""',source.index('"""')+3)
    source=source[:doc_end]+'''
候选总览
    原M=G*64是逻辑query/head域。长G6/N64候选将其按行拆成两M192，
    以独立Acc支持FIX0与QK1重叠；WS保留完整六query系数，按K192两段
    累计到同一FP32 Acc并只做一次原六行写回。其它资源区间走原函数。
    与旧双M192各三行STORE不同，不能仅以双Acc存在当作新收益依据。
    全部T/B/历史/页表和合法尾组仍动态，当前仅独立候选，尚无通过声明。
'''+source[doc_end:]
    ast.parse(source)
    path.write_text(source)
    # 除私有helper名外，原Cube函数完整AST保持；AIV生产消费函数逐字未修改。
    old_node=ast.parse(old_helper).body[0];old_node.name=fn.name
    assert ast.dump(old_node,include_attributes=False)==ast.dump(fn,include_attributes=False)
    original_vector=(base/path.name).read_text().split('def indexer_score_vector_lane_stage(',1)[1]
    assert source.split('def indexer_score_vector_lane_stage(',1)[1]==original_vector
    report=dict(base='source_reference/E49',changed=['indexer_score_stages.py'],
                selection='BALANCED and G6/N64/1024',qk_acc_bytes_each=49152,qk_acc_count=2,
                ws_acc_bytes=4096,expected_left_bytes=61440,expected_right_bytes=65536,
                expected_store_sites_per_1024=16,original_aiv_unchanged=True,
                original_fallback_ast_equal=True,formal_changed=False,device_validated=False)
    (HERE/'SCORE_SPLIT_STORE_PREPARE.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(report)


if __name__=='__main__':
    main()
