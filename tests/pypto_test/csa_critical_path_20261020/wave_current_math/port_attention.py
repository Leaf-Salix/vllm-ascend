# SPDX-License-Identifier: Apache-2.0
"""从当前完整Attention抽出AIV一拍，保留physical/MGATHER和本地统计。

只将全局query首项从物理核号改成wave的query_first。GM三槽、mask、事件、
softmax、在线归并和最终BF16/逆RoPE数学均取当前函数原文。新的统计Tile
作为第五份跨拍状态显式携带；不靠额外GM往返伪装成本地统计。
"""

import ast
import json
from pathlib import Path
import sys
import textwrap

from source_utils import block,function,header,replace_function,rewrite_calls,signature

HERE=Path(__file__).resolve().parent
REF=HERE.parent/'source_after_score_split'
TARGET=HERE/'source_candidate'
STATS_TYPE='pl.Tile[[QK_TRANSFER_SLOTS * 2, H // 2], pl.FP32, pl.MemorySpace.Vec]'


def main():
    current=(REF/'attention_stream_stages.py').read_text()
    fn=function(current,'attention_qk_pv_vector_lane_stage')
    loop,=[n for n in ast.walk(fn) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple)
           and isinstance(n.target.elts[0],ast.Name) and n.target.elts[0].id=='qk_tick']
    last=loop.body[-1]
    assert isinstance(last,ast.Assign) and isinstance(last.value,ast.Call) and ast.unparse(last.value.func)=='pl.yield_'
    assert len(last.value.args)==5
    code=block(current,loop.body[:-1])
    # 仅全局query位置作wave平移。物理环槽qk_core*3和lane分工不变。
    translations=code.count('qk_core + ')
    assert translations==4,translations
    code=code.replace('qk_core + ','query_first + ')
    state_values=[ast.unparse(a) for a in last.value.args]
    code+='\nreturn kv_transfer, probability_transfer, mi_transfer, li_transfer, o_packed_heads, '+', '.join(state_values)+'\n'
    path=TARGET/'attention_tick_stages.py';source=path.read_text();old=function(source,'attention_vector_tick')
    args=signature(source,'attention_vector_tick');at=next(i for i,a in enumerate(args) if a[0]=='CORE_GROUPS')
    args[at:at]=[('stats_cache_iter',STATS_TYPE,None),('physical_mode','pl.Scalar[pl.INT32]',None)]
    args.extend([('LOCAL_ATTENTION_STATS','pl.constexpr','False'),('PLAN_PHYSICAL_ROWS','pl.constexpr','False')])
    returns=[ast.unparse(v) for v in old.returns.slice.elts]+[STATS_TYPE]
    doc='''"""当前完整Attention的一拍：physical plan/MGATHER和local mi/li真实生效。

    前置wave获取负责plan/cache可见性；这里只平移全局query首项，GM环仍
    绑定物理核。physical_mode与plan使用同一容量谓词；stats_cache_iter是
    本lane三槽UB统计环，随四份在线状态一起返回，保持写后两拍读取顺序。
    本函数不发布wave phase，不删除原有效块/尾拍/FFT​​S与最终输出数学。
    """
'''
    new=header('attention_vector_tick',args,'tuple['+', '.join(returns)+']')+textwrap.indent(doc+code,'    ')
    source=replace_function(source,'attention_vector_tick',new)
    path.write_text(source)

    # 将两个策略常量一路传入实际被调用的wave，不只在policy中保留名字。
    sys.path.insert(0,str(HERE.parent));import prepare_qb_stream_reuse as wiring
    old_flag=wiring.FLAG
    try:
        for flag in ('LOCAL_ATTENTION_STATS','PLAN_PHYSICAL_ROWS'):
            wiring.FLAG=flag
            path=TARGET/'interleave_vector_wave.py';path.write_text(wiring.wire(path.read_text(),('interleave_vector_wave_g6',),()))
            path=TARGET/'vector_schedule.py';path.write_text(wiring.wire(path.read_text(),(),('interleave_vector_wave_g6',)))
    finally:wiring.FLAG=old_flag
    path=TARGET/'interleave_vector_wave.py';source=path.read_text()
    anchor='import pypto.language as pl\n';assert source.count(anchor)==1
    source=source.replace(anchor,anchor+'\nfrom .attention_plan_stages import physical_plan_supported\nfrom .attention_tick_stages import QK_TRANSFER_SLOTS, BLOCK_SIZE\n',1)
    marker='    initial_step = pl.cast(0, pl.INDEX)\n';assert source.count(marker)==1
    source=source.replace(marker,'''    # 同当前完整Attention：每个lane独占三槽统计环；首个有效softmax才写入，
    # 末尾三拍已排空后才能切下一wave。无效块始终不读取未写统计。
    initial_stats = pl.create_tile([QK_TRANSFER_SLOTS * 2, H // 2], dtype=pl.FP32)
    physical_mode = physical_plan_supported(
        pl.tensor.dim(cmp_block_table, 1), pl.tensor.dim(cmp_kv_flat, 0) // BLOCK_SIZE,
        PLAN_PHYSICAL_ROWS,
    )
'''+marker)
    replacements=(
        ('m_iter, l_iter, left_iter, right_iter, roots_iter, segments_iter,','m_iter, l_iter, left_iter, right_iter, stats_iter, roots_iter, segments_iter,'),
        ('initial_m, initial_l, initial_left, initial_right, initial_roots, initial_segments,','initial_m, initial_l, initial_left, initial_right, initial_stats, initial_roots, initial_segments,'),
        ('tick_m, tick_l, tick_left, tick_right,','tick_m, tick_l, tick_left, tick_right, tick_stats,'),
        ('m_after, l_after, left_after, right_after = pl.yield_(tick_m, tick_l, tick_left, tick_right)',
         'm_after, l_after, left_after, right_after, stats_after = pl.yield_(tick_m, tick_l, tick_left, tick_right, tick_stats)'),
        ('m_after, l_after, left_after, right_after = pl.yield_(m_iter, l_iter, left_iter, right_iter)',
         'm_after, l_after, left_after, right_after, stats_after = pl.yield_(m_iter, l_iter, left_iter, right_iter, stats_iter)'),
        ('final_m, final_l, final_left, final_right, final_roots, final_segments,','final_m, final_l, final_left, final_right, final_stats, final_roots, final_segments,'),
        ('m_after, l_after, left_after, right_after, roots_after, segments_after,','m_after, l_after, left_after, right_after, stats_after, roots_after, segments_after,'),
    )
    for before,after in replacements:
        assert source.count(before)==1,(before,source.count(before))
        source=source.replace(before,after)
    source,calls=rewrite_calls(source,'attention_vector_tick',lambda a:
        a[:-3]+['stats_iter','physical_mode']+a[-3:]+['LOCAL_ATTENTION_STATS','PLAN_PHYSICAL_ROWS'])
    path.write_text(source)
    for p in TARGET.glob('*.py'):ast.parse(p.read_text())
    report=dict(attention_tick_from_current=True,query_origin_translations=translations,
                active_mgather=True,active_local_stats_loop_state=True,
                physical_predicate_once_per_wave=True,math_state_count=5,
                tick_call_sites=calls,cube_two_l1_slots_unchanged=True,
                compiled=False,device_executed=False,formal_modified=False)
    (HERE/'PORT_ATTENTION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(report)


if __name__=='__main__':main()
